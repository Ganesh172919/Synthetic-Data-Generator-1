import { Link } from 'react-router-dom';
import { 
  Zap, Database, Shield, Clock, Sparkles, 
  ArrowRight, Check, Star, TrendingUp, Globe, Download
} from 'lucide-react';

const LandingPage = () => {
  const features = [
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Blazing Fast',
      description: 'Generate up to 167 Q&A pairs per minute with MEGA batch processing technology.',
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: 'ML-Ready Output',
      description: 'Industry-standard JSONL format ready for your training pipelines.',
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: 'Bulletproof Safety',
      description: 'Emergency save handlers and auto-download on disconnect.',
    },
    {
      icon: <Clock className="w-6 h-6" />,
      title: 'Resume Support',
      description: 'Checkpoint-based resume for interrupted sessions.',
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: 'Universal Templates',
      description: 'Generate datasets for ANY domain, not just finance.',
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      title: 'Quality Assured',
      description: 'Built-in pattern matching and content validation.',
    },
  ];

  const stats = [
    { value: '30,000+', label: 'Q&A Pairs in 3 Hours' },
    { value: '167/min', label: 'Generation Speed' },
    { value: '$0', label: 'Cost on Free Tier' },
    { value: '99.9%', label: 'Quality Rate' },
  ];

  const testimonials = [
    {
      quote: "SynthGen transformed how we create training data. What used to take weeks now takes hours.",
      author: "Sarah Chen",
      role: "ML Engineer at DataCorp",
      avatar: "SC"
    },
    {
      quote: "The quality of generated datasets is impressive. Our models improved by 15% after switching.",
      author: "Michael Park",
      role: "AI Researcher",
      avatar: "MP"
    },
    {
      quote: "Finally, a synthetic data tool that actually works. The speed is unmatched.",
      author: "Emily Rodriguez",
      role: "Data Scientist",
      avatar: "ER"
    }
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-900/20 to-pink-900/20" />
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500/30 rounded-full blur-[128px]" />
          <div className="absolute bottom-20 right-20 w-72 h-72 bg-pink-500/30 rounded-full blur-[128px]" />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
          <div className="text-center">
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-8">
              <Star className="w-4 h-4 text-yellow-400" />
              <span className="text-sm text-purple-300">Enterprise-Grade AI Dataset Generation</span>
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-white via-purple-200 to-white bg-clip-text text-transparent">
                Generate Synthetic Data
              </span>
              <br />
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
                At Unprecedented Speed
              </span>
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-300 max-w-3xl mx-auto mb-10">
              From 30,000 Q&A pairs in 3 hours on a FREE Google Colab T4 GPU—to unlimited possibilities. 
              Build high-quality datasets for machine learning and AI training.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                to="/dashboard"
                className="flex items-center space-x-2 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl font-semibold text-lg hover:opacity-90 transition-all shadow-lg shadow-purple-500/25"
              >
                <span>Start Generating Free</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                to="/documentation"
                className="flex items-center space-x-2 px-8 py-4 bg-slate-800 border border-slate-600 rounded-xl font-semibold text-lg hover:bg-slate-700 transition-all"
              >
                <span>View Documentation</span>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-y border-slate-700/50 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  {stat.value}
                </div>
                <div className="text-gray-400 mt-2">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Everything You Need for Dataset Generation
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Powerful features designed for researchers, data scientists, and ML engineers 
              who need high-quality synthetic data at scale.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="p-6 bg-slate-800/50 border border-slate-700/50 rounded-2xl hover:border-purple-500/50 transition-all duration-300 group"
              >
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-purple-400 mb-4 group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Get started with synthetic data generation in three simple steps
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { step: '01', title: 'Choose Your Domain', description: 'Select from pre-built templates or create a custom domain configuration.' },
              { step: '02', title: 'Configure Parameters', description: 'Set your target count, quality filters, and output format preferences.' },
              { step: '03', title: 'Generate & Export', description: 'Run the generator and download your ML-ready dataset in JSONL format.' },
            ].map((item, index) => (
              <div key={index} className="relative">
                <div className="text-6xl font-bold text-purple-500/20 absolute -top-4 -left-2">
                  {item.step}
                </div>
                <div className="relative pt-8 pl-4">
                  <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                  <p className="text-gray-400">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Loved by Data Scientists</h2>
            <p className="text-gray-400">See what our users are saying</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <div key={index} className="p-6 bg-slate-800/50 border border-slate-700/50 rounded-2xl">
                <div className="flex items-center mb-4">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                  ))}
                </div>
                <p className="text-gray-300 mb-6">&ldquo;{testimonial.quote}&rdquo;</p>
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-sm font-semibold">
                    {testimonial.avatar}
                  </div>
                  <div>
                    <div className="font-semibold">{testimonial.author}</div>
                    <div className="text-sm text-gray-400">{testimonial.role}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-purple-600 to-pink-600 p-12 text-center">
            <div className="absolute inset-0 opacity-40" style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg width='30' height='30' viewBox='0 0 30 30' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='1.5' cy='1.5' r='1.5' fill='rgba(255,255,255,0.07)'/%3E%3C/svg%3E")`
            }} />
            <div className="relative">
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                Ready to Generate Synthetic Data?
              </h2>
              <p className="text-lg text-purple-100 mb-8 max-w-2xl mx-auto">
                Start generating high-quality datasets today. No credit card required.
              </p>
              <Link
                to="/dashboard"
                className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-purple-600 rounded-xl font-semibold text-lg hover:bg-gray-100 transition-all"
              >
                <span>Get Started Free</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
