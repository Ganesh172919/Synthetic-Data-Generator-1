import { Link } from 'react-router-dom';
import { 
  Zap, Database, Shield, Clock, Sparkles, 
  ArrowRight, Check, Star, TrendingUp, Globe, ChevronRight,
  Play, Cpu, BarChart3
} from 'lucide-react';
import { AnimatedSection } from '../hooks/useIntersectionObserver';
import Button from '../components/ui/Button';
import Card from '../components/ui/Card';

/**
 * Landing Page Component
 * 
 * The main entry point for the application with hero section,
 * features, testimonials, and CTAs.
 * 
 * UX Improvements:
 * - Scroll-triggered animations for engagement
 * - F-pattern layout for optimal readability
 * - Clear visual hierarchy with typography
 * - Interactive hover states on cards
 * - Micro-interactions for delight
 *
 * Educational notes:
 * - This page is mostly static content + UI components; it does not call the API.
 * - CTAs route users to `/dashboard`, which is where job creation and downloads happen.
 * - Animations are driven by `AnimatedSection` (IntersectionObserver-based) and respect reduced motion.
 */
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
    { value: '30,000+', label: 'Q&A Pairs in 3 Hours', icon: <Database className="w-5 h-5" /> },
    { value: '167/min', label: 'Generation Speed', icon: <Zap className="w-5 h-5" /> },
    { value: '$0', label: 'Cost on Free Tier', icon: <TrendingUp className="w-5 h-5" /> },
    { value: '99.9%', label: 'Quality Rate', icon: <Star className="w-5 h-5" /> },
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

  const steps = [
    { 
      step: '01', 
      title: 'Choose Your Domain', 
      description: 'Select from pre-built templates or create a custom domain configuration.',
      icon: <Globe className="w-6 h-6" />
    },
    { 
      step: '02', 
      title: 'Configure Parameters', 
      description: 'Set your target count, quality filters, and output format preferences.',
      icon: <Cpu className="w-6 h-6" />
    },
    { 
      step: '03', 
      title: 'Generate & Export', 
      description: 'Run the generator and download your ML-ready dataset in JSONL format.',
      icon: <BarChart3 className="w-6 h-6" />
    },
  ];

  return (
    <div className="pt-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden min-h-[90vh] flex items-center">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute top-20 left-1/4 w-72 h-72 bg-purple-500/20 rounded-full blur-[120px] animate-pulse" />
          <div className="absolute bottom-20 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-[150px] animate-pulse" style={{ animationDelay: '1s' }} />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-purple-500/5 to-pink-500/5 rounded-full blur-3xl" />
        </div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-32">
          <div className="text-center">
            {/* Badge */}
            <AnimatedSection animation="fade-down" delay={0}>
              <div className="inline-flex items-center space-x-2 px-4 py-2 bg-purple-500/10 border border-purple-500/20 rounded-full mb-8 backdrop-blur-sm">
                <Star className="w-4 h-4 text-yellow-400" />
                <span className="text-sm text-purple-300 font-medium">Enterprise-Grade AI Dataset Generation</span>
              </div>
            </AnimatedSection>
            
            {/* Main Headline */}
            <AnimatedSection animation="fade-up" delay={100}>
              <h1 className="text-4xl sm:text-5xl lg:text-7xl font-extrabold mb-6 tracking-tight">
                <span className="bg-gradient-to-r from-white via-purple-100 to-white bg-clip-text text-transparent">
                  Generate Synthetic Data
                </span>
                <br />
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
                  At Unprecedented Speed
                </span>
              </h1>
            </AnimatedSection>
            
            {/* Subheadline */}
            <AnimatedSection animation="fade-up" delay={200}>
              <p className="text-lg sm:text-xl text-gray-300 max-w-3xl mx-auto mb-10 leading-relaxed">
                From 30,000 Q&A pairs in 3 hours on a FREE Google Colab T4 GPU—to unlimited possibilities. 
                Build high-quality datasets for machine learning and AI training.
              </p>
            </AnimatedSection>
            
            {/* CTA Buttons */}
            <AnimatedSection animation="fade-up" delay={300}>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Link to="/dashboard">
                  <Button 
                    size="xl" 
                    rightIcon={<ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />}
                    className="group"
                  >
                    Start Generating Free
                  </Button>
                </Link>
                <Link to="/documentation">
                  <Button variant="secondary" size="xl" leftIcon={<Play className="w-5 h-5" />}>
                    Watch Demo
                  </Button>
                </Link>
              </div>
            </AnimatedSection>
            
            {/* Trust indicators */}
            <AnimatedSection animation="fade-up" delay={400}>
              <div className="mt-12 flex flex-wrap items-center justify-center gap-6 text-sm text-gray-400">
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span>No credit card required</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span>Free tier available</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span>Open source</span>
                </div>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-y border-slate-700/30 bg-slate-900/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <AnimatedSection key={index} animation="scale" delay={index * 100}>
                <div className="text-center group">
                  <div className="inline-flex items-center justify-center w-12 h-12 mb-4 rounded-xl bg-purple-500/10 text-purple-400 group-hover:scale-110 transition-transform">
                    {stat.icon}
                  </div>
                  <div className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-1">
                    {stat.value}
                  </div>
                  <div className="text-gray-400 text-sm">{stat.label}</div>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection animation="fade-up" className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Everything You Need for Dataset Generation
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Powerful features designed for researchers, data scientists, and ML engineers 
              who need high-quality synthetic data at scale.
            </p>
          </AnimatedSection>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <AnimatedSection key={index} animation="fade-up" delay={index * 50}>
                <Card hover className="h-full group">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-purple-400 mb-4 group-hover:scale-110 transition-transform">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-semibold mb-2 group-hover:text-purple-400 transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400">{feature.description}</p>
                </Card>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-slate-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection animation="fade-up" className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Get started with synthetic data generation in three simple steps
            </p>
          </AnimatedSection>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {steps.map((item, index) => (
              <AnimatedSection key={index} animation="fade-up" delay={index * 100}>
                <div className="relative group">
                  {/* Connector line */}
                  {index < steps.length - 1 && (
                    <div className="hidden md:block absolute top-12 left-[60%] w-full h-px bg-gradient-to-r from-purple-500/50 to-transparent" />
                  )}
                  
                  <div className="relative">
                    {/* Step number background */}
                    <div className="text-8xl font-extrabold text-purple-500/5 absolute -top-4 -left-2 select-none">
                      {item.step}
                    </div>
                    
                    <div className="relative pt-8 pl-2">
                      <div className="w-12 h-12 mb-4 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform">
                        {item.icon}
                      </div>
                      <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                      <p className="text-gray-400">{item.description}</p>
                    </div>
                  </div>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection animation="fade-up" className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Loved by Data Scientists</h2>
            <p className="text-gray-400">See what our users are saying</p>
          </AnimatedSection>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {testimonials.map((testimonial, index) => (
              <AnimatedSection key={index} animation="fade-up" delay={index * 100}>
                <Card className="h-full">
                  {/* Stars */}
                  <div className="flex items-center mb-4">
                    {[...Array(5)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    ))}
                  </div>
                  
                  {/* Quote */}
                  <p className="text-gray-300 mb-6 leading-relaxed">
                    &ldquo;{testimonial.quote}&rdquo;
                  </p>
                  
                  {/* Author */}
                  <div className="flex items-center space-x-3 pt-4 border-t border-slate-700/50">
                    <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-sm font-semibold">
                      {testimonial.avatar}
                    </div>
                    <div>
                      <div className="font-semibold text-white">{testimonial.author}</div>
                      <div className="text-sm text-gray-400">{testimonial.role}</div>
                    </div>
                  </div>
                </Card>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <AnimatedSection animation="scale">
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-purple-600 to-pink-600 p-12 text-center">
              {/* Background pattern */}
              <div 
                className="absolute inset-0 opacity-30" 
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='30' height='30' viewBox='0 0 30 30' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='1.5' cy='1.5' r='1.5' fill='rgba(255,255,255,0.1)'/%3E%3C/svg%3E")`
                }} 
              />
              
              <div className="relative">
                <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                  Ready to Generate Synthetic Data?
                </h2>
                <p className="text-lg text-purple-100 mb-8 max-w-2xl mx-auto">
                  Start generating high-quality datasets today. No credit card required.
                </p>
                <Link to="/dashboard">
                  <Button 
                    variant="secondary" 
                    size="xl"
                    rightIcon={<ArrowRight className="w-5 h-5" />}
                    className="bg-white text-purple-600 hover:bg-gray-100 border-none shadow-xl"
                  >
                    Get Started Free
                  </Button>
                </Link>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
