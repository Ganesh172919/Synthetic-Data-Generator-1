import { useState, useEffect, useCallback } from 'react';
import { 
  Play, Pause, Download, RefreshCw, Clock, 
  TrendingUp, Database, CheckCircle, AlertCircle,
  Settings, FileText, Zap, AlertTriangle
} from 'lucide-react';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';
import Progress from '../components/ui/Progress';
import Badge from '../components/ui/Badge';
import { Select } from '../components/ui/Input';
import Input from '../components/ui/Input';
import { SkeletonStatsCard, SkeletonProgress } from '../components/ui/Skeleton';
import { useToast } from '../components/ui/Toast';
import { AnimatedSection } from '../hooks/useIntersectionObserver';

/**
 * Dashboard Component
 * 
 * The main control center for dataset generation with real-time
 * progress monitoring, job management, and configuration.
 * 
 * UX Improvements:
 * - Skeleton loaders during initial load
 * - Improved empty states with guidance
 * - Toast notifications for feedback
 * - Better visual hierarchy
 * - Responsive stat cards
 */
const Dashboard = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [generationConfig, setGenerationConfig] = useState({
    domain: 'financial',
    targetCount: 1000,
    batchSize: 25,
    outputFormat: 'jsonl'
  });
  const [stats, setStats] = useState({
    generated: 0,
    rate: 0,
    elapsed: 0,
    quality: 99.2
  });
  const [jobs, setJobs] = useState([]);
  const { toast } = useToast();

  // Simulate initial loading
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1000);
    return () => clearTimeout(timer);
  }, []);

  // Simulate real-time generation
  useEffect(() => {
    let interval;
    if (isGenerating && stats.generated < generationConfig.targetCount) {
      interval = setInterval(() => {
        setStats(prev => {
          const newGenerated = Math.min(prev.generated + Math.floor(Math.random() * 10) + 5, generationConfig.targetCount);
          
          // Completion notification
          if (newGenerated >= generationConfig.targetCount && prev.generated < generationConfig.targetCount) {
            toast.success('Generation Complete!', {
              title: 'Success',
              duration: 5000
            });
            setIsGenerating(false);
            setJobs(prevJobs => prevJobs.map((job, i) => 
              i === 0 ? { ...job, status: 'completed' } : job
            ));
          }
          
          return {
            ...prev,
            generated: newGenerated,
            rate: Math.floor(Math.random() * 30) + 140,
            elapsed: prev.elapsed + 1,
            quality: 99.2 + (Math.random() * 0.6 - 0.3)
          };
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isGenerating, stats.generated, generationConfig.targetCount, toast]);

  const handleStartGeneration = useCallback(async () => {
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(generationConfig)
      });
      const data = await response.json();
      
      setJobs(prev => [{
        id: data.jobId || `job-${Date.now()}`,
        domain: generationConfig.domain,
        target: generationConfig.targetCount,
        status: 'running',
        created: new Date().toLocaleString()
      }, ...prev]);
      
      setIsGenerating(true);
      setStats({ generated: 0, rate: 0, elapsed: 0, quality: 99.2 });
      
      toast.success(`Started generating ${generationConfig.targetCount.toLocaleString()} items`, {
        title: 'Generation Started'
      });
    } catch (error) {
      console.error('Failed to start generation:', error);
      
      // Fallback for demo - start anyway
      setJobs(prev => [{
        id: `job-${Date.now()}`,
        domain: generationConfig.domain,
        target: generationConfig.targetCount,
        status: 'running',
        created: new Date().toLocaleString()
      }, ...prev]);
      
      setIsGenerating(true);
      setStats({ generated: 0, rate: 0, elapsed: 0, quality: 99.2 });
      
      toast.info('Running in demo mode', {
        title: 'Demo Mode'
      });
    }
  }, [generationConfig, toast]);

  const handleStopGeneration = useCallback(() => {
    setIsGenerating(false);
    if (jobs.length > 0) {
      setJobs(prev => prev.map((job, i) => 
        i === 0 && job.status === 'running' ? { ...job, status: 'paused' } : job
      ));
    }
    toast.warning('Generation paused', {
      title: 'Paused'
    });
  }, [jobs.length, toast]);

  const handleReset = useCallback(() => {
    setStats({ generated: 0, rate: 0, elapsed: 0, quality: 99.2 });
    setIsGenerating(false);
    toast.info('Progress reset', {
      title: 'Reset'
    });
  }, [toast]);

  const formatTime = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const domainOptions = [
    { value: 'financial', label: 'Financial Education' },
    { value: 'healthcare', label: 'Healthcare' },
    { value: 'legal', label: 'Legal' },
    { value: 'technology', label: 'Technology' },
    { value: 'science', label: 'Science' },
    { value: 'custom', label: 'Custom Domain' }
  ];

  const formatOptions = [
    { value: 'jsonl', label: 'JSONL' },
    { value: 'csv', label: 'CSV' },
    { value: 'parquet', label: 'Parquet' }
  ];

  // Stat card data
  const statCards = [
    {
      label: 'Generated',
      value: stats.generated.toLocaleString(),
      subtext: `of ${generationConfig.targetCount.toLocaleString()} target`,
      icon: <Database className="w-5 h-5" />,
      iconColor: 'text-purple-400'
    },
    {
      label: 'Speed',
      value: isGenerating ? stats.rate : '--',
      subtext: 'pairs / minute',
      icon: <TrendingUp className="w-5 h-5" />,
      iconColor: 'text-emerald-400'
    },
    {
      label: 'Elapsed Time',
      value: formatTime(stats.elapsed),
      subtext: 'running time',
      icon: <Clock className="w-5 h-5" />,
      iconColor: 'text-blue-400',
      mono: true
    },
    {
      label: 'Quality Score',
      value: `${stats.quality.toFixed(1)}%`,
      subtext: 'validation rate',
      icon: <CheckCircle className="w-5 h-5" />,
      iconColor: 'text-emerald-400'
    }
  ];

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <AnimatedSection animation="fade-down" className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Generation Dashboard</h1>
          <p className="text-gray-400">Monitor and control your synthetic data generation</p>
        </AnimatedSection>

        {/* Stats Cards */}
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
                <Card className="group hover:border-purple-500/30">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-400">{stat.label}</span>
                    <div className={`${stat.iconColor} group-hover:scale-110 transition-transform`}>
                      {stat.icon}
                    </div>
                  </div>
                  <div className={`text-3xl font-bold ${stat.mono ? 'font-mono' : ''}`}>
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-500">{stat.subtext}</div>
                </Card>
              </AnimatedSection>
            ))
          )}
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Generation Control */}
          <div className="lg:col-span-2 space-y-6">
            {/* Progress */}
            {isLoading ? (
              <SkeletonProgress />
            ) : (
              <AnimatedSection animation="fade-up">
                <Card>
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-semibold">Generation Progress</h2>
                    <Badge 
                      variant={isGenerating ? 'success' : 'default'} 
                      dot 
                      pulsing={isGenerating}
                    >
                      {isGenerating ? 'Running' : stats.generated > 0 ? 'Paused' : 'Idle'}
                    </Badge>
                  </div>
                  
                  <Progress 
                    value={stats.generated} 
                    max={generationConfig.targetCount}
                    size="lg"
                    showLabel
                    className="mb-6"
                  />

                  <div className="flex flex-wrap gap-3">
                    {!isGenerating ? (
                      <Button
                        onClick={handleStartGeneration}
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
                        Pause Generation
                      </Button>
                    )}
                    
                    <Button
                      onClick={handleReset}
                      variant="secondary"
                      leftIcon={<RefreshCw className="w-5 h-5" />}
                    >
                      Reset
                    </Button>
                    
                    <Button
                      disabled={stats.generated === 0}
                      variant="secondary"
                      leftIcon={<Download className="w-5 h-5" />}
                    >
                      Export
                    </Button>
                  </div>
                </Card>
              </AnimatedSection>
            )}

            {/* Recent Jobs */}
            <AnimatedSection animation="fade-up" delay={100}>
              <Card>
                <h2 className="text-xl font-semibold mb-4">Recent Jobs</h2>
                
                {jobs.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-700/50 flex items-center justify-center">
                      <FileText className="w-8 h-8 text-gray-500" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No generation jobs yet</h3>
                    <p className="text-sm text-gray-500 mb-6 max-w-sm mx-auto">
                      Start your first generation to see your job history here. 
                      Jobs are saved so you can track progress and download results.
                    </p>
                    <Button
                      onClick={handleStartGeneration}
                      size="sm"
                      leftIcon={<Play className="w-4 h-4" />}
                    >
                      Start First Generation
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {jobs.map((job) => (
                      <div 
                        key={job.id} 
                        className="flex items-center justify-between p-4 bg-slate-700/30 rounded-xl hover:bg-slate-700/50 transition-colors"
                      >
                        <div className="flex items-center space-x-4">
                          <div className={`w-2.5 h-2.5 rounded-full ${
                            job.status === 'running' ? 'bg-emerald-400 animate-pulse' :
                            job.status === 'completed' ? 'bg-blue-400' :
                            job.status === 'paused' ? 'bg-amber-400' : 'bg-gray-400'
                          }`} />
                          <div>
                            <div className="font-medium capitalize">{job.domain} Dataset</div>
                            <div className="text-sm text-gray-400">{job.created}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">{job.target.toLocaleString()} items</div>
                          <Badge 
                            size="sm"
                            variant={
                              job.status === 'running' ? 'success' :
                              job.status === 'completed' ? 'info' :
                              job.status === 'paused' ? 'warning' : 'default'
                            }
                          >
                            {job.status}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </AnimatedSection>
          </div>

          {/* Configuration Panel */}
          <div className="space-y-6">
            <AnimatedSection animation="fade-left" delay={200}>
              <Card>
                <div className="flex items-center space-x-2 mb-6">
                  <Settings className="w-5 h-5 text-purple-400" />
                  <h2 className="text-xl font-semibold">Configuration</h2>
                </div>
                
                <div className="space-y-4">
                  <Select
                    label="Domain"
                    value={generationConfig.domain}
                    onChange={(e) => setGenerationConfig({...generationConfig, domain: e.target.value})}
                    options={domainOptions}
                  />

                  <Input
                    label="Target Count"
                    type="number"
                    value={generationConfig.targetCount}
                    onChange={(e) => setGenerationConfig({...generationConfig, targetCount: parseInt(e.target.value) || 0})}
                    min="100"
                    max="100000"
                    step="100"
                  />

                  <Input
                    label="Batch Size"
                    type="number"
                    value={generationConfig.batchSize}
                    onChange={(e) => setGenerationConfig({...generationConfig, batchSize: parseInt(e.target.value) || 0})}
                    min="5"
                    max="50"
                    step="5"
                  />

                  <Select
                    label="Output Format"
                    value={generationConfig.outputFormat}
                    onChange={(e) => setGenerationConfig({...generationConfig, outputFormat: e.target.value})}
                    options={formatOptions}
                  />
                </div>
              </Card>
            </AnimatedSection>

            {/* Quick Tips */}
            <AnimatedSection animation="fade-left" delay={300}>
              <Card variant="gradient">
                <div className="flex items-center space-x-2 mb-4">
                  <Zap className="w-5 h-5 text-yellow-400" />
                  <h3 className="font-semibold">Quick Tips</h3>
                </div>
                <ul className="space-y-3 text-sm text-gray-300">
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>Use batch size 25 for optimal speed on T4 GPU</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>Enable auto-save for long generation jobs</span>
                  </li>
                  <li className="flex items-start space-x-3">
                    <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                    <span>JSONL format is recommended for ML training</span>
                  </li>
                </ul>
              </Card>
            </AnimatedSection>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
