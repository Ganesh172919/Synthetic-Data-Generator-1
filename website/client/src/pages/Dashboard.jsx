import { useState, useEffect } from 'react';
import { 
  Play, Pause, Download, RefreshCw, Clock, 
  TrendingUp, Database, CheckCircle, AlertCircle,
  Settings, FileText, Zap
} from 'lucide-react';

const Dashboard = () => {
  const [isGenerating, setIsGenerating] = useState(false);
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

  // Simulate real-time generation
  useEffect(() => {
    let interval;
    if (isGenerating && stats.generated < generationConfig.targetCount) {
      interval = setInterval(() => {
        setStats(prev => ({
          ...prev,
          generated: Math.min(prev.generated + Math.floor(Math.random() * 10) + 5, generationConfig.targetCount),
          rate: Math.floor(Math.random() * 30) + 140,
          elapsed: prev.elapsed + 1,
          quality: 99.2 + (Math.random() * 0.6 - 0.3)
        }));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isGenerating, stats.generated, generationConfig.targetCount]);

  const handleStartGeneration = async () => {
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(generationConfig)
      });
      const data = await response.json();
      
      setJobs(prev => [{
        id: data.jobId,
        domain: generationConfig.domain,
        target: generationConfig.targetCount,
        status: 'running',
        created: new Date().toLocaleString()
      }, ...prev]);
      
      setIsGenerating(true);
      setStats({ generated: 0, rate: 0, elapsed: 0, quality: 99.2 });
    } catch (error) {
      console.error('Failed to start generation:', error);
    }
  };

  const handleStopGeneration = () => {
    setIsGenerating(false);
    if (jobs.length > 0) {
      setJobs(prev => prev.map((job, i) => 
        i === 0 ? { ...job, status: 'paused' } : job
      ));
    }
  };

  const formatTime = (seconds) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const progress = (stats.generated / generationConfig.targetCount) * 100;

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Generation Dashboard</h1>
          <p className="text-gray-400">Monitor and control your synthetic data generation</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-400">Generated</span>
              <Database className="w-5 h-5 text-purple-400" />
            </div>
            <div className="text-3xl font-bold">{stats.generated.toLocaleString()}</div>
            <div className="text-sm text-gray-400">of {generationConfig.targetCount.toLocaleString()} target</div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-400">Speed</span>
              <TrendingUp className="w-5 h-5 text-green-400" />
            </div>
            <div className="text-3xl font-bold">{stats.rate}</div>
            <div className="text-sm text-gray-400">pairs / minute</div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-400">Elapsed Time</span>
              <Clock className="w-5 h-5 text-blue-400" />
            </div>
            <div className="text-3xl font-bold font-mono">{formatTime(stats.elapsed)}</div>
            <div className="text-sm text-gray-400">running time</div>
          </div>

          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <span className="text-gray-400">Quality Score</span>
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            </div>
            <div className="text-3xl font-bold">{stats.quality.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">validation rate</div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Generation Control */}
          <div className="lg:col-span-2 space-y-6">
            {/* Progress */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Generation Progress</h2>
                <span className={`px-3 py-1 rounded-full text-sm ${
                  isGenerating 
                    ? 'bg-green-500/20 text-green-400' 
                    : 'bg-gray-500/20 text-gray-400'
                }`}>
                  {isGenerating ? 'Running' : 'Idle'}
                </span>
              </div>
              
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Progress</span>
                  <span className="text-white">{progress.toFixed(1)}%</span>
                </div>
                <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>

              <div className="flex space-x-4">
                {!isGenerating ? (
                  <button
                    onClick={handleStartGeneration}
                    className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-medium hover:opacity-90 transition-opacity"
                  >
                    <Play className="w-5 h-5" />
                    <span>Start Generation</span>
                  </button>
                ) : (
                  <button
                    onClick={handleStopGeneration}
                    className="flex items-center space-x-2 px-6 py-3 bg-orange-500 rounded-lg font-medium hover:bg-orange-600 transition-colors"
                  >
                    <Pause className="w-5 h-5" />
                    <span>Pause Generation</span>
                  </button>
                )}
                
                <button
                  onClick={() => {
                    setStats({ generated: 0, rate: 0, elapsed: 0, quality: 99.2 });
                    setIsGenerating(false);
                  }}
                  className="flex items-center space-x-2 px-6 py-3 bg-slate-700 rounded-lg font-medium hover:bg-slate-600 transition-colors"
                >
                  <RefreshCw className="w-5 h-5" />
                  <span>Reset</span>
                </button>
                
                <button
                  disabled={stats.generated === 0}
                  className="flex items-center space-x-2 px-6 py-3 bg-slate-700 rounded-lg font-medium hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Download className="w-5 h-5" />
                  <span>Export</span>
                </button>
              </div>
            </div>

            {/* Recent Jobs */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Recent Jobs</h2>
              
              {jobs.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No generation jobs yet</p>
                  <p className="text-sm">Start your first generation to see jobs here</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {jobs.map((job) => (
                    <div key={job.id} className="flex items-center justify-between p-4 bg-slate-700/50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`w-2 h-2 rounded-full ${
                          job.status === 'running' ? 'bg-green-400 animate-pulse' :
                          job.status === 'completed' ? 'bg-blue-400' :
                          job.status === 'paused' ? 'bg-orange-400' : 'bg-gray-400'
                        }`} />
                        <div>
                          <div className="font-medium capitalize">{job.domain} Dataset</div>
                          <div className="text-sm text-gray-400">{job.created}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium">{job.target.toLocaleString()} items</div>
                        <div className="text-sm text-gray-400 capitalize">{job.status}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Configuration Panel */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Settings className="w-5 h-5 text-purple-400" />
                <h2 className="text-xl font-semibold">Configuration</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Domain</label>
                  <select
                    value={generationConfig.domain}
                    onChange={(e) => setGenerationConfig({...generationConfig, domain: e.target.value})}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="financial">Financial Education</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="legal">Legal</option>
                    <option value="technology">Technology</option>
                    <option value="science">Science</option>
                    <option value="custom">Custom Domain</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Target Count</label>
                  <input
                    type="number"
                    value={generationConfig.targetCount}
                    onChange={(e) => setGenerationConfig({...generationConfig, targetCount: parseInt(e.target.value) || 0})}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="100"
                    max="100000"
                    step="100"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
                  <input
                    type="number"
                    value={generationConfig.batchSize}
                    onChange={(e) => setGenerationConfig({...generationConfig, batchSize: parseInt(e.target.value) || 0})}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="5"
                    max="50"
                    step="5"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-2">Output Format</label>
                  <select
                    value={generationConfig.outputFormat}
                    onChange={(e) => setGenerationConfig({...generationConfig, outputFormat: e.target.value})}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="jsonl">JSONL</option>
                    <option value="csv">CSV</option>
                    <option value="parquet">Parquet</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Quick Tips */}
            <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 border border-purple-500/20 rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Zap className="w-5 h-5 text-yellow-400" />
                <h3 className="font-semibold">Quick Tips</h3>
              </div>
              <ul className="space-y-2 text-sm text-gray-300">
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Use batch size 25 for optimal speed on T4 GPU</span>
                </li>
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>Enable auto-save for long generation jobs</span>
                </li>
                <li className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  <span>JSONL format is recommended for ML training</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
