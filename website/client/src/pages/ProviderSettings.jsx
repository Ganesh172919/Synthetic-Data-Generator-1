import { useState, useEffect, useCallback } from 'react';
import {
  Cpu,
  CheckCircle,
  XCircle,
  AlertCircle,
  RefreshCw,
  Key,
  Zap,
  Globe,
  Server,
  Shield,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { AnimatedSection } from '../hooks/useIntersectionObserver';
import { useTheme } from '../hooks/useTheme';
import api from '../services/api';

const statusColors = {
  healthy: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  configured: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  available: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  unconfigured: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
  unavailable: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
};

const StatusIcon = ({ status }) => {
  switch (status) {
    case 'healthy':
    case 'available':
      return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    case 'configured':
      return <CheckCircle className="w-4 h-4 text-blue-400" />;
    case 'unconfigured':
      return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    case 'unavailable':
      return <XCircle className="w-4 h-4 text-red-400" />;
    default:
      return <AlertCircle className="w-4 h-4 text-gray-400" />;
  }
};

const providerIcons = {
  mock: <Cpu className="w-5 h-5" />,
  openai: <Zap className="w-5 h-5" />,
  huggingface: <Server className="w-5 h-5" />,
  anthropic: <Shield className="w-5 h-5" />,
  google: <Globe className="w-5 h-5" />,
  ollama: <Server className="w-5 h-5" />,
  azure_openai: <Cloud className="w-5 h-5" />,
  groq: <Zap className="w-5 h-5" />,
  together: <Cpu className="w-5 h-5" />,
  custom: <Server className="w-5 h-5" />,
};

function Cloud(props) {
  return (
    <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
    </svg>
  );
}

const ProviderCard = ({ provider, healthStatus, onRefreshHealth }) => {
  const { isDark } = useTheme();
  const [expanded, setExpanded] = useState(false);
  const status = healthStatus?.status || (provider.configured ? 'configured' : 'unconfigured');
  const colors = statusColors[status] || statusColors.unconfigured;

  return (
    <AnimatedSection animation="fade-up">
      <Card className={`transition-all duration-300 ${expanded ? 'ring-1 ring-purple-500/30' : ''}`}>
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${colors.bg} ${colors.text}`}>
              {providerIcons[provider.id] || <Cpu className="w-5 h-5" />}
            </div>
            <div>
              <h3 className="font-semibold text-white flex items-center gap-2">
                {provider.name}
                <Badge variant={status === 'healthy' || status === 'available' ? 'success' : status === 'unconfigured' ? 'warning' : status === 'configured' ? 'info' : 'error'}>
                  {status}
                </Badge>
              </h3>
              <p className="text-sm text-gray-400">{provider.description}</p>
            </div>
          </div>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-gray-400 hover:text-white transition-colors p-1"
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>

        {/* Quick info row */}
        <div className="mt-3 flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <StatusIcon status={status} />
            {provider.requiresKey ? 'Requires API Key' : 'No API Key Required'}
          </span>
          <span>{provider.models?.length || 0} models</span>
          {healthStatus?.latency_ms > 0 && (
            <span>{healthStatus.latency_ms}ms latency</span>
          )}
        </div>

        {/* Expanded details */}
        {expanded && (
          <div className="mt-4 pt-4 border-t border-slate-700 space-y-3 animate-fade-in">
            {/* Available models */}
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">Available Models</h4>
              <div className="flex flex-wrap gap-2">
                {provider.models?.map((model) => (
                  <Badge key={model} variant="default" className="text-xs">
                    {model}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Health status details */}
            {healthStatus && (
              <div>
                <h4 className="text-sm font-medium text-gray-300 mb-1">Health Check</h4>
                <p className="text-sm text-gray-400">{healthStatus.message}</p>
              </div>
            )}

            {/* API Key status */}
            {provider.requiresKey && (
              <div className="flex items-center gap-2 text-sm">
                <Key className="w-4 h-4 text-gray-500" />
                <span className={provider.configured ? 'text-emerald-400' : 'text-yellow-400'}>
                  {provider.configured ? 'API key configured' : 'API key not set — add to environment variables'}
                </span>
              </div>
            )}

            {/* Refresh health button */}
            <Button
              variant="secondary"
              size="sm"
              onClick={() => onRefreshHealth(provider.id)}
              leftIcon={<RefreshCw className="w-3 h-3" />}
            >
              Refresh Health
            </Button>
          </div>
        )}
      </Card>
    </AnimatedSection>
  );
};

const comparisonData = [
  { feature: 'Speed', mock: 'Instant', openai: 'Fast', anthropic: 'Fast', google: 'Fast', ollama: 'Varies', huggingface: 'Slow (local)', groq: 'Ultra-fast', together: 'Fast', azure_openai: 'Fast', custom: 'Varies' },
  { feature: 'Quality', mock: 'Test only', openai: 'Excellent', anthropic: 'Excellent', google: 'Excellent', ollama: 'Good', huggingface: 'Good', groq: 'Good', together: 'Good', azure_openai: 'Excellent', custom: 'Varies' },
  { feature: 'Cost', mock: 'Free', openai: 'Paid', anthropic: 'Paid', google: 'Paid', ollama: 'Free', huggingface: 'Free', groq: 'Free tier', together: 'Paid', azure_openai: 'Paid', custom: 'Varies' },
  { feature: 'Privacy', mock: 'Local', openai: 'Cloud', anthropic: 'Cloud', google: 'Cloud', ollama: 'Local', huggingface: 'Local', groq: 'Cloud', together: 'Cloud', azure_openai: 'Cloud', custom: 'Self-hosted' },
  { feature: 'Setup', mock: 'None', openai: 'API key', anthropic: 'API key', google: 'API key', ollama: 'Install', huggingface: 'GPU + libs', groq: 'API key', together: 'API key', azure_openai: 'Azure acct', custom: 'Endpoint URL' },
];

const ProviderSettings = () => {
  const { isDark } = useTheme();
  const [providers, setProviders] = useState([]);
  const [healthStatuses, setHealthStatuses] = useState({});
  const [loading, setLoading] = useState(true);
  const [showComparison, setShowComparison] = useState(false);

  const loadProviders = useCallback(async () => {
    try {
      const data = await api.getProviders();
      setProviders(data.providers || []);
    } catch (error) {
      console.error('Failed to load providers:', error);
    }
  }, []);

  const loadHealth = useCallback(async (providerId) => {
    try {
      const health = await api.getProviderHealth(providerId);
      setHealthStatuses((prev) => ({ ...prev, [providerId]: health }));
    } catch (error) {
      setHealthStatuses((prev) => ({
        ...prev,
        [providerId]: { status: 'unavailable', message: error.message },
      }));
    }
  }, []);

  const loadAllHealth = useCallback(async () => {
    for (const p of providers) {
      await loadHealth(p.id);
    }
  }, [providers, loadHealth]);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await loadProviders();
      setLoading(false);
    };
    init();
  }, [loadProviders]);

  useEffect(() => {
    if (providers.length > 0) {
      loadAllHealth();
    }
  }, [providers, loadAllHealth]);

  const configuredCount = providers.filter((p) => p.configured).length;
  const healthyCount = Object.values(healthStatuses).filter(
    (h) => h.status === 'healthy' || h.status === 'available'
  ).length;

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <AnimatedSection animation="fade-down" className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Provider Settings</h1>
          <p className="text-gray-400">
            Configure and monitor LLM providers for dataset generation.
          </p>
        </AnimatedSection>

        {/* Stats bar */}
        <AnimatedSection animation="fade-up" className="mb-8">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Total Providers</span>
                <span className="text-2xl font-bold text-white">{providers.length}</span>
              </div>
            </Card>
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Configured</span>
                <span className="text-2xl font-bold text-emerald-400">{configuredCount}</span>
              </div>
            </Card>
            <Card>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Healthy</span>
                <span className="text-2xl font-bold text-blue-400">{healthyCount}</span>
              </div>
            </Card>
          </div>
        </AnimatedSection>

        {/* Actions */}
        <div className="flex items-center gap-3 mb-6">
          <Button
            variant="secondary"
            onClick={loadAllHealth}
            leftIcon={<RefreshCw className="w-4 h-4" />}
          >
            Refresh All Health Checks
          </Button>
          <Button
            variant="secondary"
            onClick={() => setShowComparison(!showComparison)}
            leftIcon={<Cpu className="w-4 h-4" />}
          >
            {showComparison ? 'Hide' : 'Show'} Comparison Table
          </Button>
        </div>

        {/* Comparison table */}
        {showComparison && (
          <AnimatedSection animation="fade-down" className="mb-8">
            <Card>
              <h2 className="text-lg font-semibold mb-4 text-white">Provider Comparison</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2 px-3 text-gray-400 font-medium">Feature</th>
                      {['mock', 'openai', 'anthropic', 'google', 'ollama', 'huggingface', 'groq', 'together', 'azure_openai', 'custom'].map((p) => (
                        <th key={p} className="text-center py-2 px-2 text-gray-400 font-medium capitalize">
                          {p === 'azure_openai' ? 'Azure' : p === 'huggingface' ? 'HF' : p}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonData.map((row) => (
                      <tr key={row.feature} className="border-b border-slate-800">
                        <td className="py-2 px-3 text-gray-300 font-medium">{row.feature}</td>
                        {['mock', 'openai', 'anthropic', 'google', 'ollama', 'huggingface', 'groq', 'together', 'azure_openai', 'custom'].map((p) => (
                          <td key={p} className="text-center py-2 px-2 text-gray-400 text-xs">
                            {row[p]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </AnimatedSection>
        )}

        {/* Provider cards */}
        {loading ? (
          <div className="text-center py-12 text-gray-400">Loading providers...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {providers.map((provider) => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                healthStatus={healthStatuses[provider.id]}
                onRefreshHealth={loadHealth}
              />
            ))}
          </div>
        )}

        {/* Setup instructions */}
        <AnimatedSection animation="fade-up" className="mt-8">
          <Card variant="gradient">
            <h3 className="font-semibold mb-3 text-white flex items-center gap-2">
              <Key className="w-5 h-5 text-purple-400" />
              Environment Variables for API Keys
            </h3>
            <div className="space-y-2 text-sm text-gray-300 font-mono">
              <p><span className="text-purple-400">OPENAI_API_KEY</span> — OpenAI GPT models</p>
              <p><span className="text-purple-400">ANTHROPIC_API_KEY</span> — Anthropic Claude models</p>
              <p><span className="text-purple-400">GOOGLE_API_KEY</span> — Google Gemini models</p>
              <p><span className="text-purple-400">GROQ_API_KEY</span> — Groq fast inference</p>
              <p><span className="text-purple-400">TOGETHER_API_KEY</span> — Together.ai open models</p>
              <p><span className="text-purple-400">AZURE_OPENAI_API_KEY</span> + <span className="text-purple-400">AZURE_OPENAI_ENDPOINT</span> — Azure OpenAI</p>
              <p><span className="text-purple-400">CUSTOM_API_BASE</span> — Custom OpenAI-compatible endpoint</p>
              <p><span className="text-purple-400">OLLAMA_HOST</span> — Ollama server URL (default: http://localhost:11434)</p>
            </div>
          </Card>
        </AnimatedSection>
      </div>
    </div>
  );
};

export default ProviderSettings;
