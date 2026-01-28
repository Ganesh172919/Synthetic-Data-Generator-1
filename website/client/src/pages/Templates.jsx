import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Briefcase, Heart, Scale, Cpu, FlaskConical, GraduationCap,
  ArrowRight, Search, Star, Download, Eye
} from 'lucide-react';

const Templates = () => {
  const [templates, setTemplates] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(true);

  const categories = [
    { id: 'all', name: 'All Templates', icon: null },
    { id: 'financial', name: 'Finance', icon: <Briefcase className="w-4 h-4" /> },
    { id: 'healthcare', name: 'Healthcare', icon: <Heart className="w-4 h-4" /> },
    { id: 'legal', name: 'Legal', icon: <Scale className="w-4 h-4" /> },
    { id: 'technology', name: 'Technology', icon: <Cpu className="w-4 h-4" /> },
    { id: 'science', name: 'Science', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'education', name: 'Education', icon: <GraduationCap className="w-4 h-4" /> },
  ];

  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await fetch('/api/templates');
        const data = await response.json();
        setTemplates(data.templates);
      } catch (error) {
        console.error('Failed to fetch templates:', error);
        // Fallback to default templates
        setTemplates(defaultTemplates);
      } finally {
        setIsLoading(false);
      }
    };
    fetchTemplates();
  }, []);

  const defaultTemplates = [
    {
      id: 'fin-education',
      name: 'Financial Education Q&A',
      description: 'Personal finance, investing, budgeting, credit management, and retirement planning questions.',
      category: 'financial',
      rating: 4.9,
      downloads: 12500,
      topics: ['Personal Finance', 'Investing', 'Credit & Debt', 'Retirement'],
      featured: true
    },
    {
      id: 'healthcare-clinical',
      name: 'Clinical Knowledge Base',
      description: 'Medical terminology, symptoms, treatments, and healthcare procedures for training medical AI.',
      category: 'healthcare',
      rating: 4.8,
      downloads: 8700,
      topics: ['Medical Terms', 'Symptoms', 'Treatments', 'Procedures'],
      featured: true
    },
    {
      id: 'legal-contracts',
      name: 'Legal Document Analysis',
      description: 'Contract clauses, legal terminology, compliance requirements, and case law summaries.',
      category: 'legal',
      rating: 4.7,
      downloads: 6300,
      topics: ['Contracts', 'Compliance', 'Legal Terms', 'Case Law'],
      featured: false
    },
    {
      id: 'tech-programming',
      name: 'Programming Q&A',
      description: 'Code explanations, debugging help, best practices, and algorithm discussions.',
      category: 'technology',
      rating: 4.9,
      downloads: 15200,
      topics: ['Python', 'JavaScript', 'Algorithms', 'Best Practices'],
      featured: true
    },
    {
      id: 'science-research',
      name: 'Scientific Research Assistant',
      description: 'Research methodology, experiment design, data analysis, and academic writing.',
      category: 'science',
      rating: 4.6,
      downloads: 4500,
      topics: ['Methodology', 'Data Analysis', 'Papers', 'Citations'],
      featured: false
    },
    {
      id: 'edu-tutoring',
      name: 'Educational Tutoring',
      description: 'Math, science, language arts explanations suitable for K-12 and college students.',
      category: 'education',
      rating: 4.8,
      downloads: 9800,
      topics: ['Math', 'Science', 'English', 'History'],
      featured: false
    },
    {
      id: 'fin-trading',
      name: 'Trading Strategies',
      description: 'Stock market analysis, trading patterns, risk management, and portfolio optimization.',
      category: 'financial',
      rating: 4.5,
      downloads: 5600,
      topics: ['Stocks', 'Options', 'Risk Management', 'Technical Analysis'],
      featured: false
    },
    {
      id: 'healthcare-mental',
      name: 'Mental Health Support',
      description: 'Therapeutic conversations, coping strategies, and mental wellness guidance.',
      category: 'healthcare',
      rating: 4.7,
      downloads: 7200,
      topics: ['Therapy', 'Coping', 'Wellness', 'Self-Care'],
      featured: false
    },
  ];

  const filteredTemplates = (templates.length > 0 ? templates : defaultTemplates).filter(template => {
    const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
    const matchesSearch = template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const featuredTemplates = filteredTemplates.filter(t => t.featured);
  const regularTemplates = filteredTemplates.filter(t => !t.featured);

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Dataset Templates</h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Pre-built templates optimized for various domains. Start generating high-quality 
            synthetic data in seconds.
          </p>
        </div>

        {/* Search and Filter */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-slate-800 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                  selectedCategory === category.id
                    ? 'bg-purple-500 text-white'
                    : 'bg-slate-800 text-gray-300 hover:bg-slate-700'
                }`}
              >
                {category.icon}
                <span>{category.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Featured Templates */}
        {featuredTemplates.length > 0 && (
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-6 flex items-center space-x-2">
              <Star className="w-6 h-6 text-yellow-400" />
              <span>Featured Templates</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {featuredTemplates.map((template) => (
                <TemplateCard key={template.id} template={template} featured />
              ))}
            </div>
          </div>
        )}

        {/* All Templates */}
        <div>
          <h2 className="text-2xl font-bold mb-6">All Templates</h2>
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="bg-slate-800/50 rounded-xl p-6 animate-pulse">
                  <div className="h-6 bg-slate-700 rounded w-3/4 mb-4"></div>
                  <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
                  <div className="h-4 bg-slate-700 rounded w-2/3"></div>
                </div>
              ))}
            </div>
          ) : regularTemplates.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {regularTemplates.map((template) => (
                <TemplateCard key={template.id} template={template} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-400">
              <p>No templates found matching your criteria.</p>
            </div>
          )}
        </div>

        {/* CTA */}
        <div className="mt-16 text-center">
          <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 border border-purple-500/20 rounded-2xl p-8">
            <h3 className="text-2xl font-bold mb-4">Need a Custom Template?</h3>
            <p className="text-gray-400 mb-6 max-w-xl mx-auto">
              Use our Domain Builder to create custom templates tailored to your specific needs.
            </p>
            <Link
              to="/domain-builder"
              className="inline-flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-medium hover:opacity-90 transition-opacity"
            >
              <span>Build Custom Template</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

const TemplateCard = ({ template, featured = false }) => {
  const getCategoryIcon = (category) => {
    const icons = {
      financial: <Briefcase className="w-5 h-5" />,
      healthcare: <Heart className="w-5 h-5" />,
      legal: <Scale className="w-5 h-5" />,
      technology: <Cpu className="w-5 h-5" />,
      science: <FlaskConical className="w-5 h-5" />,
      education: <GraduationCap className="w-5 h-5" />,
    };
    return icons[category] || <Briefcase className="w-5 h-5" />;
  };

  return (
    <div className={`bg-slate-800/50 border rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300 ${
      featured ? 'border-purple-500/30' : 'border-slate-700/50'
    }`}>
      {featured && (
        <div className="flex items-center space-x-1 text-yellow-400 text-sm mb-3">
          <Star className="w-4 h-4 fill-yellow-400" />
          <span>Featured</span>
        </div>
      )}
      
      <div className="flex items-start justify-between mb-4">
        <div className="w-10 h-10 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg flex items-center justify-center text-purple-400">
          {getCategoryIcon(template.category)}
        </div>
        <div className="flex items-center space-x-1 text-sm">
          <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
          <span>{template.rating}</span>
        </div>
      </div>
      
      <h3 className="text-lg font-semibold mb-2">{template.name}</h3>
      <p className="text-gray-400 text-sm mb-4 line-clamp-2">{template.description}</p>
      
      <div className="flex flex-wrap gap-2 mb-4">
        {template.topics.slice(0, 3).map((topic, index) => (
          <span key={index} className="px-2 py-1 bg-slate-700/50 rounded text-xs text-gray-300">
            {topic}
          </span>
        ))}
        {template.topics.length > 3 && (
          <span className="px-2 py-1 bg-slate-700/50 rounded text-xs text-gray-300">
            +{template.topics.length - 3}
          </span>
        )}
      </div>
      
      <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
        <div className="flex items-center space-x-1 text-sm text-gray-400">
          <Download className="w-4 h-4" />
          <span>{template.downloads.toLocaleString()}</span>
        </div>
        <Link
          to={`/dashboard?template=${template.id}`}
          className="flex items-center space-x-1 text-purple-400 hover:text-purple-300 text-sm font-medium transition-colors"
        >
          <span>Use Template</span>
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  );
};

export default Templates;
