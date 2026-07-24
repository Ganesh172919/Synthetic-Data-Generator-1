import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../hooks/useTheme';
import { api } from '../services/api';

const TIER_FEATURES = {
  free: {
    icon: '🆓',
    color: 'from-gray-500 to-gray-600',
    highlights: [
      '5 jobs per day',
      'Up to 1,000 rows per dataset',
      'Mock provider',
      'CSV & JSON export',
      '1 API key',
      '3-day retention',
    ],
  },
  pro: {
    icon: '⚡',
    color: 'from-purple-500 to-pink-500',
    highlights: [
      '50 jobs per day',
      'Up to 50,000 rows per dataset',
      'All providers (HuggingFace, OpenAI)',
      'Custom domains',
      'API access',
      '10 API keys',
      '30-day retention',
      'Advanced analytics',
      'Plugin marketplace',
    ],
  },
  enterprise: {
    icon: '🏢',
    color: 'from-amber-500 to-orange-500',
    highlights: [
      '500 jobs per day',
      'Up to 100,000 rows per dataset',
      'All providers + priority queue',
      'Team management',
      '100 API keys',
      '90-day retention',
      'SLA-backed support',
      'Dedicated infrastructure',
    ],
  },
};

export default function PricingPage() {
  const { isDark } = useTheme();
  const navigate = useNavigate();
  const [tiers, setTiers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getTiers()
      .then((data) => setTiers(data.tiers || []))
      .catch(() => setTiers([]))
      .finally(() => setLoading(false));
  }, []);

  const cardClass = (tierName) => {
    const isPopular = tierName === 'pro';
    return `relative rounded-2xl p-8 transition-all duration-300 ${
      isPopular ? 'scale-105 z-10' : ''
    } ${isDark
      ? `bg-white/5 border ${isPopular ? 'border-purple-500/50 shadow-lg shadow-purple-500/10' : 'border-white/10'}`
      : `bg-white border ${isPopular ? 'border-purple-500/50 shadow-xl shadow-purple-500/10' : 'border-gray-200'} shadow-lg`
    }`;
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center pt-20">
        <div className="animate-pulse text-lg">Loading pricing...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-16 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
              Simple, Transparent Pricing
            </span>
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Start free, scale as you grow. No hidden fees.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
          {tiers.map((tier) => {
            const meta = TIER_FEATURES[tier.id] || TIER_FEATURES.free;
            const isPopular = tier.id === 'pro';

            return (
              <div key={tier.id} className={cardClass(tier.id)}>
                {isPopular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <span className="bg-gradient-to-r from-purple-500 to-pink-500 text-white text-xs font-bold px-4 py-1 rounded-full">
                      MOST POPULAR
                    </span>
                  </div>
                )}

                <div className="text-center mb-6">
                  <div className="text-4xl mb-3">{meta.icon}</div>
                  <h3 className="text-xl font-bold capitalize">{tier.name}</h3>
                  <div className="mt-4">
                    <span className="text-4xl font-bold">${tier.price}</span>
                    {tier.price > 0 && (
                      <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>/month</span>
                    )}
                  </div>
                </div>

                <ul className="space-y-3 mb-8">
                  {meta.highlights.map((feature, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-green-500 mt-0.5 flex-shrink-0">✓</span>
                      <span className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{feature}</span>
                    </li>
                  ))}
                </ul>

                <button
                  onClick={() => navigate('/auth')}
                  className={`w-full py-3 rounded-xl font-semibold transition-all duration-200 ${
                    isPopular
                      ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-700 hover:to-pink-700'
                      : isDark
                        ? 'bg-white/10 text-white hover:bg-white/20'
                        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'
                  }`}
                >
                  {tier.price === 0 ? 'Get Started Free' : 'Start Free Trial'}
                </button>
              </div>
            );
          })}
        </div>

        <div className="mt-16 text-center">
          <h2 className="text-2xl font-bold mb-4">Enterprise Custom Plans</h2>
          <p className={`max-w-xl mx-auto mb-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Need higher limits, custom integrations, or on-premise deployment?
            Contact us for a tailored solution.
          </p>
          <button
            onClick={() => navigate('/documentation')}
            className={`px-8 py-3 rounded-xl font-semibold border transition-all ${
              isDark ? 'border-white/20 hover:bg-white/10' : 'border-gray-300 hover:bg-gray-100'
            }`}
          >
            Contact Sales
          </button>
        </div>
      </div>
    </div>
  );
}
