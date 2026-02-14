import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import {
  Play,
  Pause,
  Download,
  RefreshCw,
  Clock,
  TrendingUp,
  Database,
  CheckCircle,
  Settings,
  FileText,
  Zap,
  Trash2,
  ChevronDown,
  ChevronUp,
  RotateCcw,
} from 'lucide-react';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import Progress from '../components/ui/Progress';
import Badge from '../components/ui/Badge';
import Input, { Select } from '../components/ui/Input';
import { SkeletonStatsCard, SkeletonProgress } from '../components/ui/Skeleton';
import { useToast } from '../components/ui/Toast';
import Modal from '../components/ui/Modal';
import { AnimatedSection } from '../hooks/useIntersectionObserver';
import api from '../services/api';

const statusVariant = (status) => {
  if (status === 'running') return 'success';
  if (status === 'completed') return 'info';
  if (status === 'stopped') return 'warning';
  if (status === 'failed') return 'error';
  return 'default';
};

const formatTime = (seconds) => {
  const safe = Math.max(0, Number(seconds || 0));
  const h = Math.floor(safe / 3600);
  const m = Math.floor((safe % 3600) / 60);
  const s = safe % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

const Dashboard = () => {
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { toast } = useToast();

  const [isLoading, setIsLoading] = useState(true);
  const [jobs, setJobs] = useState([]);
  const [domains, setDomains] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportFormat, setExportFormat] = useState('jsonl');
  const [selectedJobForExport, setSelectedJobForExport] = useState(null);
  const [isConfigOpen, setIsConfigOpen] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [clockTick, setClockTick] = useState(Date.now());
  const [generationConfig, setGenerationConfig] = useState({
    domain: 'financial',
    targetCount: 1000,
    batchSize: 25,
    outputFormat: 'jsonl',
    provider: 'mock',
    parseMode: 'qa',
    prompt: '',
    domainId: '',
    domainDescription: '',
    topicsInput: '',
  });

  const lastEventIdRef = useRef(0);
  const pollFallbackRef = useRef(null);
  const sseRef = useRef(null);

  const refreshJobs = useCallback(async () => {
    const data = await api.listJobs({ limit: 100 });
    const list = Array.isArray(data.jobs) ? data.jobs : [];
    setJobs(list);
    setActiveJobId((prev) => {
      if (prev && list.some((job) => job.id === prev)) {
        return prev;
      }
      const running = list.find((job) => ['queued', 'running'].includes(job.status));
      return running?.id || list[0]?.id || null;
    });
  }, []);

  const refreshJob = useCallback(async (jobId) => {
    if (!jobId) return;
    const job = await api.getJobStatus(jobId);
    setJobs((prev) => {
      const idx = prev.findIndex((item) => item.id === job.id);
      if (idx === -1) return [job, ...prev];
      const next = [...prev];
      next[idx] = job;
      return next;
    });
  }, []);

  const refreshDomains = useCallback(async () => {
    const data = await api.listDomains();
    setDomains(Array.isArray(data.domains) ? data.domains : []);
  }, []);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        await Promise.all([refreshJobs(), refreshDomains()]);
      } catch (error) {
        if (mounted) {
          toast.error(error.message || 'Failed to load dashboard state');
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, [refreshJobs, refreshDomains, toast]);

  useEffect(() => {
    const timer = setInterval(() => setClockTick(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const templateId = searchParams.get('template');
    if (!templateId) return;

    let alive = true;
    const applyTemplate = async () => {
      try {
        const template = await api.getTemplate(templateId);
        if (!alive) return;
        setGenerationConfig((prev) => ({
          ...prev,
          domain: template.category || prev.domain,
          domainDescription: template.description || '',
          topicsInput: Array.isArray(template.topics) ? template.topics.join(', ') : '',
          domainId: '',
        }));
        toast.info(`Template loaded: ${template.name}`);
      } catch (error) {
        if (alive) toast.error(error.message || 'Failed to load template');
      }
    };

    applyTemplate();
    return () => {
      alive = false;
    };
  }, [searchParams, toast]);

  useEffect(() => {
    const domainId = location.state?.domainId;
    if (!domainId) return;
    setGenerationConfig((prev) => ({
      ...prev,
      domain: 'custom',
      domainId,
    }));
    toast.success('Saved domain selected for generation');
  }, [location.state, toast]);

  useEffect(() => {
    if (!activeJobId) return undefined;

    const closeSse = () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
      if (pollFallbackRef.current) {
        clearInterval(pollFallbackRef.current);
        pollFallbackRef.current = null;
      }
    };

    const startPollingFallback = () => {
      if (pollFallbackRef.current) return;
      pollFallbackRef.current = setInterval(() => {
        refreshJob(activeJobId).catch(() => {});
      }, 2000);
    };

    closeSse();

    try {
      const es = api.streamJobEvents(activeJobId, lastEventIdRef.current);
      sseRef.current = es;

      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed?.eventId) {
            lastEventIdRef.current = parsed.eventId;
          }
        } catch {
          // no-op
        }
        refreshJob(activeJobId).catch(() => {});
      };

      es.onerror = () => {
        closeSse();
        startPollingFallback();
      };
    } catch {
      startPollingFallback();
    }

    refreshJob(activeJobId).catch(() => {});

    return closeSse;
  }, [activeJobId, refreshJob]);

  const activeJob = useMemo(
    () => jobs.find((job) => job.id === activeJobId) || null,
    [jobs, activeJobId]
  );

  const generatedCount = activeJob?.generatedCount || 0;
  const targetCount = activeJob?.targetCount || generationConfig.targetCount;
  const isGenerating = activeJob ? ['queued', 'running'].includes(activeJob.status) : false;
  const ratePerMinute = activeJob?.rateItemsPerSec ? Math.round(activeJob.rateItemsPerSec * 60) : 0;
  const elapsedSeconds =
    activeJob?.startedAt
      ? Math.max(0, Math.floor((clockTick - new Date(activeJob.startedAt).getTime()) / 1000))
      : 0;
  const qualityScore =
    generatedCount > 0
      ? ((generatedCount - (activeJob?.invalidCount || 0)) / generatedCount) * 100
      : 100;

  const handleStartGeneration = useCallback(async () => {
    setIsStarting(true);
    try {
      const topics = generationConfig.topicsInput
        .split(',')
        .map((topic) => topic.trim())
        .filter(Boolean);

      const payload = {
        domain: generationConfig.domain,
        targetCount: Number(generationConfig.targetCount),
        batchSize: Number(generationConfig.batchSize),
        outputFormat: generationConfig.outputFormat,
        provider: generationConfig.provider,
        parseMode: generationConfig.parseMode,
      };

      if (generationConfig.prompt.trim()) payload.prompt = generationConfig.prompt.trim();
      if (generationConfig.domainId) payload.domainId = generationConfig.domainId;
      if (generationConfig.domainDescription.trim()) {
        payload.domainDescription = generationConfig.domainDescription.trim();
      }
      if (topics.length > 0) payload.topics = topics;

      const response = await api.startGeneration(payload);
      await refreshJobs();
      setActiveJobId(response.jobId);
      toast.success(`Job queued: ${response.jobId}`);
    } catch (error) {
      toast.error(error.message || 'Failed to start generation');
    } finally {
      setIsStarting(false);
    }
  }, [generationConfig, refreshJobs, toast]);

  const handleStopGeneration = useCallback(async () => {
    if (!activeJobId) return;
    try {
      await api.stopJob(activeJobId);
      await refreshJob(activeJobId);
      toast.warning('Stop requested');
    } catch (error) {
      toast.error(error.message || 'Failed to stop job');
    }
  }, [activeJobId, refreshJob, toast]);

  const handleReset = useCallback(() => {
    setActiveJobId(null);
  }, []);

  const handleExport = useCallback(
    (job = null) => {
      const candidate =
        job || jobs.find((item) => item.status === 'completed') || jobs.find((item) => item.status === 'stopped');
      if (!candidate) {
        toast.error('No completed/stopped job available for download');
        return;
      }
      setSelectedJobForExport(candidate);
      setExportFormat(candidate.outputFormat || 'jsonl');
      setShowExportModal(true);
    },
    [jobs, toast]
  );

  const handleDownload = useCallback(() => {
    if (!selectedJobForExport) {
      toast.error('No job selected');
      return;
    }

    const downloadUrl = api.getDownloadUrl(selectedJobForExport.id, exportFormat);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${selectedJobForExport.id}.${exportFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    toast.success(`Downloading ${selectedJobForExport.id}.${exportFormat}`);
    setShowExportModal(false);
  }, [selectedJobForExport, exportFormat, toast]);

  const handleDeleteJob = useCallback(
    async (jobId) => {
      try {
        await api.deleteJob(jobId);
        await refreshJobs();
        toast.success('Job deleted');
      } catch (error) {
        toast.error(error.message || 'Failed to delete job');
      }
    },
    [refreshJobs, toast]
  );

  const handleRetryJob = useCallback(
    async (jobId) => {
      try {
        await api.retryJob(jobId);
        await refreshJobs();
        setActiveJobId(jobId);
        toast.success('Job queued for retry');
      } catch (error) {
        toast.error(error.message || 'Failed to retry job');
      }
    },
    [refreshJobs, toast]
  );

  const domainOptions = [
    { value: 'financial', label: 'Financial Education' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'legal', label: 'Legal' },
    { value: 'technology', label: 'Technology' },
    { value: 'science', label: 'Science' },
    { value: 'education', label: 'Education' },
    { value: 'custom', label: 'Custom' },
  ];

  const formatOptions = [
    { value: 'jsonl', label: 'JSONL' },
    { value: 'csv', label: 'CSV' },
    { value: 'json', label: 'JSON' },
  ];

  const providerOptions = [
    { value: 'mock', label: 'Mock (Dev/CI)' },
    { value: 'openai', label: 'OpenAI' },
    { value: 'huggingface', label: 'Hugging Face' },
  ];

  const parseModeOptions = [
    { value: 'qa', label: 'Q&A' },
    { value: 'text', label: 'Text' },
    { value: 'json', label: 'Structured JSON' },
  ];

  const domainSelectionOptions = [
    { value: '', label: 'None' },
    ...domains.map((domain) => ({ value: domain.id, label: domain.name || domain.id })),
  ];

  const statCards = [
    {
      label: 'Generated',
      value: generatedCount.toLocaleString(),
      subtext: `of ${targetCount.toLocaleString()} target`,
      icon: <Database className="w-5 h-5" />,
      iconColor: 'text-purple-400',
    },
    {
      label: 'Speed',
      value: isGenerating ? ratePerMinute : '--',
      subtext: 'items / minute',
      icon: <TrendingUp className="w-5 h-5" />,
      iconColor: 'text-emerald-400',
    },
    {
      label: 'Elapsed Time',
      value: formatTime(elapsedSeconds),
      subtext: 'active job runtime',
      icon: <Clock className="w-5 h-5" />,
      iconColor: 'text-blue-400',
      mono: true,
    },
    {
      label: 'Quality Score',
      value: `${qualityScore.toFixed(1)}%`,
      subtext: 'valid output ratio',
      icon: <CheckCircle className="w-5 h-5" />,
      iconColor: 'text-emerald-400',
    },
  ];

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <AnimatedSection animation="fade-down" className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Generation Dashboard</h1>
          <p className="text-gray-400">Durable job queue with real worker progress and downloadable artifacts.</p>
        </AnimatedSection>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {isLoading ? (
            <>
              <SkeletonStatsCard />
              <SkeletonStatsCard />
              <SkeletonStatsCard />
              <SkeletonStatsCard />
            </>
          ) : (
            statCards.map((stat, index) => (
              <AnimatedSection key={index} animation="fade-up" delay={index * 50}>
                <Card>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-400 font-medium">{stat.label}</span>
                    <div className={`p-2 rounded-lg bg-slate-800/50 ${stat.iconColor}`}>{stat.icon}</div>
                  </div>
                  <div className={`text-2xl font-bold mb-1 ${stat.mono ? 'font-mono' : ''}`}>{stat.value}</div>
                  <div className="text-xs text-gray-400">{stat.subtext}</div>
                </Card>
              </AnimatedSection>
            ))
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            {isLoading ? (
              <SkeletonProgress />
            ) : (
              <AnimatedSection animation="fade-up">
                <Card>
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Generation Progress</h2>
                    <Badge variant={statusVariant(activeJob?.status || 'idle')} dot pulsing={isGenerating}>
                      {activeJob?.status || 'idle'}
                    </Badge>
                  </div>

                  <Progress
                    value={generatedCount}
                    max={targetCount}
                    size="lg"
                    showLabel
                    className="mb-6"
                  />

                  <div className="flex flex-wrap gap-3">
                    {!isGenerating ? (
                      <Button
                        onClick={handleStartGeneration}
                        isLoading={isStarting}
                        leftIcon={<Play className="w-5 h-5" />}
                      >
                        Start Generation
                      </Button>
                    ) : (
                      <Button
                        onClick={handleStopGeneration}
                        variant="secondary"
                        leftIcon={<Pause className="w-5 h-5" />}
                        className="bg-orange-500/20 border-orange-500/30 hover:bg-orange-500/30"
                      >
                        Stop Job
                      </Button>
                    )}

                    <Button onClick={() => refreshJobs()} variant="secondary" leftIcon={<RefreshCw className="w-5 h-5" />}>
                      Refresh
                    </Button>

                    <Button onClick={handleReset} variant="secondary" leftIcon={<RotateCcw className="w-5 h-5" />}>
                      Clear Selection
                    </Button>

                    <Button
                      disabled={!jobs.some((job) => ['completed', 'stopped'].includes(job.status))}
                      variant="secondary"
                      leftIcon={<Download className="w-5 h-5" />}
                      onClick={() => handleExport()}
                    >
                      Export
                    </Button>
                  </div>
                </Card>
              </AnimatedSection>
            )}

            <AnimatedSection animation="fade-up" delay={100}>
              <Card>
                <h2 className="text-xl font-semibold mb-4">Durable Job History</h2>

                {jobs.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-700/50 flex items-center justify-center">
                      <FileText className="w-8 h-8 text-gray-500" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No jobs yet</h3>
                    <p className="text-sm text-gray-500 mb-6 max-w-sm mx-auto">
                      Start your first generation job. History persists across API and worker restarts.
                    </p>
                    <Button onClick={handleStartGeneration} size="sm" leftIcon={<Play className="w-4 h-4" />}>
                      Start First Generation
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {jobs.map((job) => (
                      <div
                        key={job.id}
                        className={`flex items-center justify-between p-4 rounded-xl border transition-colors ${
                          activeJobId === job.id
                            ? 'bg-purple-500/10 border-purple-500/30'
                            : 'bg-slate-700/30 border-slate-700 hover:bg-slate-700/50'
                        }`}
                      >
                        <button
                          type="button"
                          className="flex-1 text-left"
                          onClick={() => setActiveJobId(job.id)}
                        >
                          <div className="font-medium capitalize">{job.domain} Dataset</div>
                          <div className="text-sm text-gray-400">{new Date(job.createdAt).toLocaleString()}</div>
                          <div className="text-xs text-gray-500 mt-1">
                            {job.generatedCount?.toLocaleString() || 0} / {job.targetCount?.toLocaleString() || 0}
                          </div>
                        </button>

                        <div className="flex items-center gap-2 ml-4">
                          <Badge size="sm" variant={statusVariant(job.status)}>
                            {job.status}
                          </Badge>

                          {['failed', 'stopped'].includes(job.status) && (
                            <button
                              onClick={() => handleRetryJob(job.id)}
                              className="p-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
                              title="Retry job"
                            >
                              <RefreshCw className="w-4 h-4" />
                            </button>
                          )}

                          {['completed', 'stopped'].includes(job.status) && (
                            <button
                              onClick={() => handleExport(job)}
                              className="p-2 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors"
                              title="Download dataset"
                            >
                              <Download className="w-4 h-4" />
                            </button>
                          )}

                          {!['queued', 'running'].includes(job.status) && (
                            <button
                              onClick={() => handleDeleteJob(job.id)}
                              className="p-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                              title="Delete job"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </AnimatedSection>
          </div>

          <div className="space-y-6">
            <AnimatedSection animation="fade-left" delay={200}>
              <Card>
                <div
                  className="flex items-center justify-between mb-6 cursor-pointer"
                  onClick={() => setIsConfigOpen(!isConfigOpen)}
                >
                  <div className="flex items-center space-x-2">
                    <Settings className="w-5 h-5 text-purple-400" />
                    <h2 className="text-xl font-semibold">Configuration</h2>
                  </div>
                  <button className="text-gray-400 hover:text-white transition-colors">
                    {isConfigOpen ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </button>
                </div>

                {isConfigOpen && (
                  <div className="space-y-4 animate-fade-in">
                    <Select
                      label="Domain"
                      value={generationConfig.domain}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({
                          ...prev,
                          domain: e.target.value,
                          domainId: e.target.value === 'custom' ? prev.domainId : '',
                        }))
                      }
                      options={domainOptions}
                    />

                    <Select
                      label="Saved Domain (optional)"
                      value={generationConfig.domainId}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({
                          ...prev,
                          domainId: e.target.value,
                          domain: e.target.value ? 'custom' : prev.domain,
                        }))
                      }
                      options={domainSelectionOptions}
                    />

                    <Input
                      label="Target Count"
                      type="number"
                      value={generationConfig.targetCount}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({
                          ...prev,
                          targetCount: Number(e.target.value) || 0,
                        }))
                      }
                      min="100"
                      max="100000"
                    />

                    <Input
                      label="Batch Size"
                      type="number"
                      value={generationConfig.batchSize}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({
                          ...prev,
                          batchSize: Number(e.target.value) || 0,
                        }))
                      }
                      min="1"
                      max="50"
                    />

                    <Select
                      label="Output Format"
                      value={generationConfig.outputFormat}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({ ...prev, outputFormat: e.target.value }))
                      }
                      options={formatOptions}
                    />

                    <Select
                      label="Provider"
                      value={generationConfig.provider}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({ ...prev, provider: e.target.value }))
                      }
                      options={providerOptions}
                    />

                    <Select
                      label="Parse Mode"
                      value={generationConfig.parseMode}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({ ...prev, parseMode: e.target.value }))
                      }
                      options={parseModeOptions}
                    />

                    <Input
                      label="Topics (comma separated)"
                      value={generationConfig.topicsInput}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({ ...prev, topicsInput: e.target.value }))
                      }
                      placeholder="Investing, budgeting, credit score"
                    />

                    <Input
                      label="Domain Description"
                      value={generationConfig.domainDescription}
                      onChange={(e) =>
                        setGenerationConfig((prev) => ({ ...prev, domainDescription: e.target.value }))
                      }
                      placeholder="Optional description"
                    />

                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-gray-400">Prompt (optional)</label>
                      <textarea
                        rows={4}
                        value={generationConfig.prompt}
                        onChange={(e) =>
                          setGenerationConfig((prev) => ({ ...prev, prompt: e.target.value }))
                        }
                        placeholder="Optional explicit prompt (server derives prompt when omitted and domainId is set)"
                        className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500 hover:border-slate-500 transition-all placeholder-slate-400"
                      />
                    </div>
                  </div>
                )}
              </Card>
            </AnimatedSection>

            <AnimatedSection animation="fade-left" delay={300}>
              <Card variant="gradient">
                <div className="flex items-center space-x-2 mb-4">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  <h3 className="font-semibold">Operational Notes</h3>
                </div>
                <ul className="space-y-3 text-sm text-gray-300">
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>Progress comes from worker updates, not client simulation.</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>Job history is durable in SQLite across service restarts.</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>Supported output formats this cycle: JSONL, CSV, JSON.</span>
                  </li>
                </ul>
              </Card>
            </AnimatedSection>
          </div>
        </div>
      </div>

      <Modal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        title="Export Dataset"
      >
        <div className="space-y-6">
          <p className="text-gray-400">Choose a format to download the generated artifact.</p>

          <div className="space-y-3">
            {[
              { value: 'jsonl', label: 'JSONL', description: 'Recommended for ML training pipelines' },
              { value: 'csv', label: 'CSV', description: 'Compatible with spreadsheets and tabular tooling' },
              { value: 'json', label: 'JSON', description: 'Single JSON array output' },
            ].map((format) => (
              <button
                key={format.value}
                onClick={() => setExportFormat(format.value)}
                className={`
                  w-full p-4 rounded-xl border text-left transition-all
                  ${
                    exportFormat === format.value
                      ? 'bg-purple-500/20 border-purple-500/50 text-white'
                      : 'bg-slate-700/30 border-slate-600 text-gray-300 hover:border-slate-500'
                  }
                `}
              >
                <div className="font-medium">{format.label}</div>
                <div className="text-sm text-gray-400">{format.description}</div>
              </button>
            ))}
          </div>

          {selectedJobForExport && (
            <div className="p-4 bg-slate-700/30 rounded-xl">
              <div className="text-sm text-gray-400">Selected Job</div>
              <div className="font-medium">{selectedJobForExport.id}</div>
              <div className="text-sm text-gray-500 capitalize">{selectedJobForExport.domain} domain</div>
            </div>
          )}

          <div className="flex justify-end space-x-3">
            <Button variant="secondary" onClick={() => setShowExportModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleDownload} leftIcon={<Download className="w-4 h-4" />}>
              Download {exportFormat.toUpperCase()}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Dashboard;
