import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Briefcase, Heart, Scale, Cpu, FlaskConical, GraduationCap,
  ArrowRight, Search, Star, Download, Sparkles, Eye
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import Input from '../components/ui/Input';
import { SkeletonTemplateCard } from '../components/ui/Skeleton';
import { useToast } from '../components/ui/Toast';
import Modal from '../components/ui/Modal';
import { AnimatedSection } from '../hooks/useIntersectionObserver';

/**
 * Templates Page
 * 
 * Browse and select from pre-built dataset templates organized
 * by domain categories with search and filtering.
 * 
 * UX Improvements:
 * - Skeleton loading states
 * - Animated card reveals
 * - Interactive hover effects
 * - Clear visual hierarchy for featured templates
 * - Improved search experience
 */
const Templates = () => {
  const [templates, setTemplates] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [previewTemplate, setPreviewTemplate] = useState(null);

  const categories = [
    { id: 'all', name: 'All Templates', icon: <Sparkles className="w-4 h-4" /> },
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
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch templates:', error);
        // Fallback to default templates
        setTemplates(defaultTemplates);
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
        <AnimatedSection animation="fade-down" className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Dataset Templates</h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Pre-built templates optimized for various domains. Start generating high-quality 
            synthetic data in seconds.
          </p>
        </AnimatedSection>

        {/* Search and Filter */}
        <AnimatedSection animation="fade-up" delay={100} className="mb-8">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="relative flex-1">
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                leftIcon={<Search className="w-5 h-5" />}
                size="lg"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`
                    flex items-center space-x-2 px-4 py-2.5 rounded-xl
                    font-medium text-sm
                    transition-all duration-200
                    ${selectedCategory === category.id
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/25'
                      : 'bg-slate-800/50 text-gray-300 hover:bg-slate-700/50 border border-slate-700/50'
                    }
                  `}
                >
                  {category.icon}
                  <span className="hidden sm:inline">{category.name}</span>
                </button>
              ))}
            </div>
          </div>
        </AnimatedSection>

        {/* Featured Templates */}
        {featuredTemplates.length > 0 && !isLoading && (
          <AnimatedSection animation="fade-up" delay={200} className="mb-12">
            <h2 className="text-2xl font-bold mb-6 flex items-center space-x-2">
              <Star className="w-6 h-6 text-yellow-400" />
              <span>Featured Templates</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {featuredTemplates.map((template, index) => (
                <AnimatedSection key={template.id} animation="fade-up" delay={index * 50}>
                  <TemplateCard 
                    template={template} 
                    featured 
                    onPreview={() => setPreviewTemplate(template)}
                  />
                </AnimatedSection>
              ))}
            </div>
          </AnimatedSection>
        )}

        {/* All Templates */}
        <div>
          <h2 className="text-2xl font-bold mb-6">
            {selectedCategory === 'all' ? 'All Templates' : categories.find(c => c.id === selectedCategory)?.name}
          </h2>
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <SkeletonTemplateCard key={i} />
              ))}
            </div>
          ) : regularTemplates.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {regularTemplates.map((template, index) => (
                <AnimatedSection key={template.id} animation="fade-up" delay={index * 50}>
                  <TemplateCard 
                    template={template} 
                    onPreview={() => setPreviewTemplate(template)}
                  />
                </AnimatedSection>
              ))}
            </div>
          ) : filteredTemplates.length === 0 ? (
            <div className="text-center py-16">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-700/50 flex items-center justify-center">
                <Search className="w-8 h-8 text-gray-500" />
              </div>
              <h3 className="text-lg font-medium text-gray-300 mb-2">No templates found</h3>
              <p className="text-sm text-gray-500 mb-6">
                Try adjusting your search or filter criteria
              </p>
              <Button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedCategory('all');
                }}
                variant="secondary"
                size="sm"
              >
                Clear filters
              </Button>
            </div>
          ) : null}
        </div>

        {/* CTA */}
        <AnimatedSection animation="scale" delay={300} className="mt-16">
          <Card variant="gradient" padding="lg" className="text-center">
            <h3 className="text-2xl font-bold mb-4">Need a Custom Template?</h3>
            <p className="text-gray-300 mb-6 max-w-xl mx-auto">
              Use our Domain Builder to create custom templates tailored to your specific needs.
            </p>
            <Link to="/domain-builder">
              <Button rightIcon={<ArrowRight className="w-5 h-5" />}>
                Build Custom Template
              </Button>
            </Link>
          </Card>
        </AnimatedSection>
      </div>
      
      {/* Quick Preview Modal */}
      <Modal
        isOpen={!!previewTemplate}
        onClose={() => setPreviewTemplate(null)}
        title={previewTemplate?.name}
        maxWidth="lg"
      >
        <div className="space-y-6">
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-purple-500/10 rounded-xl text-purple-400">
               {previewTemplate && (
                  previewTemplate.options?.icon || <Briefcase className="w-8 h-8" />
               )}
            </div>
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <Badge variant="outline">{previewTemplate?.category}</Badge>
                {previewTemplate?.featured && (
                  <Badge variant="primary">Featured</Badge>
                )}
              </div>
              <p className="text-gray-300 text-lg leading-relaxed">
                {previewTemplate?.description}
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4 bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
            <div>
               <div className="text-sm text-gray-400 mb-1">Downloads</div>
               <div className="text-xl font-semibold flex items-center">
                 <Download className="w-4 h-4 mr-2 text-blue-400" />
                 {previewTemplate?.downloads?.toLocaleString()}
               </div>
            </div>
            <div>
               <div className="text-sm text-gray-400 mb-1">Rating</div>
               <div className="text-xl font-semibold flex items-center">
                 <Star className="w-4 h-4 mr-2 text-yellow-400" />
                 {previewTemplate?.rating}
               </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-300 mb-3">Included Topics</h4>
            <div className="flex flex-wrap gap-2">
              {previewTemplate?.topics?.map((topic, i) => (
                <span key={i} className="px-3 py-1 bg-slate-700/50 rounded-lg text-sm text-gray-300 border border-slate-700">
                  {topic}
                </span>
              ))}
            </div>
          </div>
          
          <div className="flex space-x-3 pt-4 border-t border-slate-700/50">
            <Button
              variant="primary"
              className="flex-1"
              onClick={() => {
                // Handle use template
                setPreviewTemplate(null);
                // would navigate to dashboard or generation
              }}
            >
              Use Template
            </Button>
            <Button
              variant="ghost"
              onClick={() => setPreviewTemplate(null)}
            >
              Close
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

const TemplateCard = ({ template, featured = false, onPreview }) => {
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
    <Card 
      hover 
      variant={featured ? 'featured' : 'default'}
      className="h-full flex flex-col group relative"
    >
      {featured && (
        <Badge variant="warning" size="sm" className="mb-3 w-fit">
          <Star className="w-3 h-3 mr-1" />
          Featured
        </Badge>
      )}
      
      <div className="flex items-start justify-between mb-4">
        <div className="w-11 h-11 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform">
          {getCategoryIcon(template.category)}
        </div>
        <div className="flex items-center space-x-1 text-sm">
          <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
          <span className="font-medium">{template.rating}</span>
        </div>
      </div>
      
      <h3 className="text-lg font-semibold mb-2 group-hover:text-purple-400 transition-colors">
        {template.name}
      </h3>
      <p className="text-gray-400 text-sm mb-4 line-clamp-2 flex-grow">
        {template.description}
      </p>
      
      <div className="flex flex-wrap gap-2 mb-4">
        {template.topics.slice(0, 3).map((topic, index) => (
          <Badge key={index} variant="default" size="sm">
            {topic}
          </Badge>
        ))}
        {template.topics.length > 3 && (
          <Badge variant="default" size="sm">
            +{template.topics.length - 3}
          </Badge>
        )}
      </div>
      
      <div className="flex items-center justify-between pt-4 border-t border-slate-700/50 mt-auto">
        <div className="flex items-center space-x-1.5 text-sm text-gray-400">
          <Download className="w-4 h-4" />
          <span>{template.downloads.toLocaleString()}</span>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={onPreview}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors border border-transparent hover:border-slate-600"
            title="Quick Preview"
          >
            <Eye className="w-4 h-4" />
          </button>
          <Link
            to={`/dashboard?template=${template.id}`}
            className="flex items-center space-x-1 text-purple-400 hover:text-purple-300 text-sm font-medium transition-colors group/link"
          >
            <span>Use Template</span>
            <ArrowRight className="w-4 h-4 group-hover/link:translate-x-1 transition-transform" />
          </Link>
        </div>
      </div>
    </Card>
  );
};

export default Templates;
