import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  Plus, Trash2, Save, ArrowRight, ArrowLeft, Lightbulb,
  FileText, Settings2, CheckCircle, AlertCircle, Layers, List
} from 'lucide-react';
import Button from '../components/ui/Button';
import Input from '../components/ui/Input';
import { useToast } from '../components/ui/Toast';
import api from '../services/api';

/**
 * Domain Builder Page
 *
 * Purpose:
 * - Provide a multi-step UI to define a “custom domain” (name, description, topics, question types, etc.)
 * - Save that configuration via the backend API so it can be reused for generation
 *
 * Data flow (reality-aligned):
 * - On save, calls `api.saveDomain(domainConfig)` → POST /api/domains
 * - The demo backend stores domains in memory (lost on server restart)
 *
 * Educational notes:
 * - The schema here is a "UI config schema" — it is not yet used by the demo backend to drive
 *   real dataset generation. It's a good starting point for later integration.
 * - Multi-step forms help users provide complex input without overwhelm, but add state complexity.
 */

const DomainBuilder = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(1);
  const totalSteps = 4;
  
  const [domainConfig, setDomainConfig] = useState({
    name: '',
    description: '',
    topics: [{ name: '', subtopics: [''] }],
    questionTypes: ['definition', 'explanation', 'comparison'],
    difficultyLevels: ['beginner', 'intermediate', 'advanced'],
    outputSettings: {
      minAnswerLength: 50,
      maxAnswerLength: 500,
      includeMetadata: true
    }
  });
  const [isSaving, setIsSaving] = useState(false);
  const [savedMessage, setSavedMessage] = useState('');

  const questionTypeOptions = [
    { value: 'definition', label: 'Definition', description: 'What is X?' },
    { value: 'explanation', label: 'Explanation', description: 'Explain how X works' },
    { value: 'comparison', label: 'Comparison', description: 'Compare X and Y' },
    { value: 'example', label: 'Example', description: 'Give an example of X' },
    { value: 'application', label: 'Application', description: 'How to apply X?' },
    { value: 'analysis', label: 'Analysis', description: 'Analyze the impact of X' },
  ];

  const addTopic = () => {
    setDomainConfig(prev => ({
      ...prev,
      topics: [...prev.topics, { name: '', subtopics: [''] }]
    }));
  };

  const removeTopic = (index) => {
    if (domainConfig.topics.length > 1) {
      setDomainConfig(prev => ({
        ...prev,
        topics: prev.topics.filter((_, i) => i !== index)
      }));
    }
  };

  const updateTopic = (index, field, value) => {
    setDomainConfig(prev => ({
      ...prev,
      topics: prev.topics.map((topic, i) => 
        i === index ? { ...topic, [field]: value } : topic
      )
    }));
  };

  const addSubtopic = (topicIndex) => {
    setDomainConfig(prev => ({
      ...prev,
      topics: prev.topics.map((topic, i) => 
        i === topicIndex ? { ...topic, subtopics: [...topic.subtopics, ''] } : topic
      )
    }));
  };

  const updateSubtopic = (topicIndex, subtopicIndex, value) => {
    setDomainConfig(prev => ({
      ...prev,
      topics: prev.topics.map((topic, i) => 
        i === topicIndex 
          ? { ...topic, subtopics: topic.subtopics.map((st, si) => si === subtopicIndex ? value : st) }
          : topic
      )
    }));
  };

  const removeSubtopic = (topicIndex, subtopicIndex) => {
    setDomainConfig(prev => ({
      ...prev,
      topics: prev.topics.map((topic, i) => 
        i === topicIndex && topic.subtopics.length > 1
          ? { ...topic, subtopics: topic.subtopics.filter((_, si) => si !== subtopicIndex) }
          : topic
      )
    }));
  };

  const toggleQuestionType = (type) => {
    setDomainConfig(prev => ({
      ...prev,
      questionTypes: prev.questionTypes.includes(type)
        ? prev.questionTypes.filter(t => t !== type)
        : [...prev.questionTypes, type]
    }));
  };

  const validateStep = (step) => {
    switch (step) {
      case 1:
        return domainConfig.name.length > 3 && domainConfig.description.length > 5;
      case 2:
        return domainConfig.topics.some(t => t.name.length > 0 && t.subtopics.some(st => st.length > 0));
      case 3:
        return domainConfig.questionTypes.length > 0;
      case 4:
        return true;
      default:
        return false;
    }
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, totalSteps));
      window.scrollTo(0, 0);
    } else {
      toast.error('Please complete all required fields');
    }
  };

  const handleBack = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1));
    window.scrollTo(0, 0);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await api.saveDomain(domainConfig);
      toast.success('Domain saved successfully!');
      
      // Navigate to dashboard or show success state
      setTimeout(() => navigate('/dashboard'), 1500);
    } catch (error) {
      console.error('Failed to save domain:', error);
      toast.error('Failed to save domain');
    } finally {
      setIsSaving(false);
    }
  };

    return (
    <div className="pt-24 pb-12 min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold mb-2">Custom Domain Builder</h1>
          <p className="text-gray-400 max-w-xl mx-auto">
            Design your dataset generation parameters in 4 simple steps
          </p>
        </div>

        {/* Stepper */}
        <div className="mb-10">
          <div className="flex items-center justify-between relative">
            {/* Progress Bar Background */}
            <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-slate-700 -z-10" />
            {/* Active Progress Bar */}
            <div 
              className="absolute left-0 top-1/2 h-0.5 bg-purple-500 -z-10 transition-all duration-500 ease-in-out" 
              style={{ width: `${((currentStep - 1) / (totalSteps - 1)) * 100}%` }}
            />
            
            {[
              { id: 1, label: 'Basic Info', icon: <FileText className="w-5 h-5" /> },
              { id: 2, label: 'Topics', icon: <Layers className="w-5 h-5" /> },
              { id: 3, label: 'Settings', icon: <Settings2 className="w-5 h-5" /> },
              { id: 4, label: 'Review', icon: <CheckCircle className="w-5 h-5" /> }
            ].map((step) => (
              <div key={step.id} className="flex flex-col items-center space-y-2 bg-slate-900 px-2 box-content">
                <div 
                  className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                    currentStep >= step.id 
                      ? 'bg-purple-600 border-purple-600 text-white shadow-lg shadow-purple-500/30' 
                      : 'bg-slate-800 border-slate-600 text-gray-500'
                  }`}
                >
                  {step.icon}
                </div>
                <span className={`text-sm font-medium ${
                  currentStep >= step.id ? 'text-purple-400' : 'text-gray-500'
                }`}>
                  {step.label}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 sm:p-8 shadow-xl min-h-[500px] flex flex-col">
          {/* Step 1: Basic Information */}
          {currentStep === 1 && (
            <div className="space-y-6 animate-fade-in flex-1">
              <h2 className="text-2xl font-semibold mb-6">Describe your Domain</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Domain Name *</label>
                  <Input
                    value={domainConfig.name}
                    onChange={(e) => setDomainConfig({...domainConfig, name: e.target.value})}
                    placeholder="e.g., Cryptocurrency Education"
                    autoFocus
                  />
                  <p className="text-xs text-gray-500 mt-1">Min 4 characters</p>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Description *</label>
                  <textarea
                    value={domainConfig.description}
                    onChange={(e) => setDomainConfig({...domainConfig, description: e.target.value})}
                    placeholder="Describe what kind of data this domain will generate..."
                    rows={6}
                    className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none transition-all placeholder:text-gray-600 text-white"
                  />
                  <p className="text-xs text-gray-500 mt-1">Min 6 characters</p>
                </div>
              </div>
            </div>
          )}

          {/* Step 2: Topics */}
          {currentStep === 2 && (
            <div className="space-y-6 animate-fade-in flex-1">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-semibold">Define Topics</h2>
                <Button 
                  onClick={addTopic}
                  variant="secondary"
                  size="sm"
                  leftIcon={<Plus className="w-4 h-4" />}
                >
                  Add Topic
                </Button>
              </div>
              
              <div className="space-y-6 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                {domainConfig.topics.map((topic, topicIndex) => (
                  <div key={topicIndex} className="p-5 bg-slate-800/40 rounded-xl border border-slate-700/50 hover:border-purple-500/20 transition-all">
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-400 font-bold shrink-0">
                        {topicIndex + 1}
                      </div>
                      <input
                        type="text"
                        value={topic.name}
                        onChange={(e) => updateTopic(topicIndex, 'name', e.target.value)}
                        placeholder="Topic name"
                        className="flex-1 px-4 py-2 bg-transparent border-b border-slate-600 focus:border-purple-500 focus:outline-none transition-colors text-lg font-medium"
                      />
                      {domainConfig.topics.length > 1 && (
                        <button
                          onClick={() => removeTopic(topicIndex)}
                          className="p-2 text-gray-500 hover:text-red-400 rounded-lg transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                    
                    <div className="ml-11 space-y-3">
                      {topic.subtopics.map((subtopic, subtopicIndex) => (
                        <div key={subtopicIndex} className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={subtopic}
                            onChange={(e) => updateSubtopic(topicIndex, subtopicIndex, e.target.value)}
                            placeholder={`Subtopic ${subtopicIndex + 1}`}
                            className="flex-1 px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-purple-500"
                          />
                          {topic.subtopics.length > 1 && (
                            <button
                              onClick={() => removeSubtopic(topicIndex, subtopicIndex)}
                              className="p-1.5 text-gray-500 hover:text-red-400 transition-colors"
                            >
                              <X className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                      ))}
                      <button
                        onClick={() => addSubtopic(topicIndex)}
                        className="text-xs font-medium text-purple-400 hover:text-purple-300 transition-colors flex items-center mt-2 pl-1"
                      >
                        <Plus className="w-3 h-3 mr-1" /> Add Subtopic
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Step 3: Settings */}
          {currentStep === 3 && (
            <div className="space-y-8 animate-fade-in flex-1">
              {/* Question Types */}
              <div>
                <h2 className="text-2xl font-semibold mb-6">Question Types</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {questionTypeOptions.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => toggleQuestionType(option.value)}
                      className={`p-4 rounded-xl border text-left transition-all ${
                        domainConfig.questionTypes.includes(option.value)
                          ? 'bg-purple-500/20 border-purple-500/50 text-white shadow-lg shadow-purple-900/20'
                          : 'bg-slate-800/30 border-slate-700 text-gray-400 hover:border-slate-500'
                      }`}
                    >
                      <div className="font-semibold mb-1">{option.label}</div>
                      <div className="text-xs opacity-70">{option.description}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Output Settings */}
              <div>
                <h3 className="text-lg font-medium text-gray-300 mb-4 flex items-center">
                  <Settings2 className="w-4 h-4 mr-2" /> Output Parameters
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 bg-slate-800/30 p-6 rounded-xl border border-slate-700/50">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Min Length (chars)</label>
                    <Input
                      type="number"
                      value={domainConfig.outputSettings.minAnswerLength}
                      onChange={(e) => setDomainConfig({
                        ...domainConfig,
                        outputSettings: { ...domainConfig.outputSettings, minAnswerLength: parseInt(e.target.value) || 0 }
                      })}
                      min="20"
                      max="200"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Max Length (chars)</label>
                    <Input
                      type="number"
                      value={domainConfig.outputSettings.maxAnswerLength}
                      onChange={(e) => setDomainConfig({
                        ...domainConfig,
                        outputSettings: { ...domainConfig.outputSettings, maxAnswerLength: parseInt(e.target.value) || 0 }
                      })}
                      min="100"
                      max="2000"
                    />
                  </div>
                  <div className="sm:col-span-2">
                    <label className="flex items-center space-x-3 cursor-pointer p-3 bg-slate-800/50 rounded-lg hover:bg-slate-800 transition-colors">
                      <input
                        type="checkbox"
                        checked={domainConfig.outputSettings.includeMetadata}
                        onChange={(e) => setDomainConfig({
                          ...domainConfig,
                          outputSettings: { ...domainConfig.outputSettings, includeMetadata: e.target.checked }
                        })}
                        className="w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-500 focus:ring-purple-500"
                      />
                      <span className="text-gray-300 font-medium">Include detailed metadata in dataset</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Review */}
          {currentStep === 4 && (
            <div className="space-y-6 animate-fade-in flex-1">
              <h2 className="text-2xl font-semibold mb-6">Review Configuration</h2>
              
              <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl overflow-hidden divide-y divide-slate-700/50">
                <div className="p-6">
                  <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">Basic Info</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <span className="text-gray-500 block text-xs mb-1">Domain Name</span>
                      <span className="text-lg font-medium text-white">{domainConfig.name}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs mb-1">Description</span>
                      <span className="text-gray-300">{domainConfig.description}</span>
                    </div>
                  </div>
                </div>
                
                <div className="p-6">
                  <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">Structure</h3>
                  <div className="space-y-4">
                    <div>
                      <span className="text-gray-500 block text-xs mb-2">Topics & Subtopics</span>
                      <div className="flex flex-wrap gap-2">
                        {domainConfig.topics.filter(t => t.name).map((t, i) => (
                          <div key={i} className="bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700 text-sm">
                            <span className="text-purple-400 font-medium">{t.name}</span>
                            <span className="text-gray-500 mx-2">|</span>
                            <span className="text-gray-400">{t.subtopics.filter(st => st).length} subtopics</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="p-6">
                  <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">Settings</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                     <div>
                       <span className="text-gray-500 block text-xs mb-2">Question Types</span>
                       <div className="flex flex-wrap gap-1">
                          {domainConfig.questionTypes.map(t => (
                            <span key={t} className="px-2 py-0.5 bg-purple-500/10 text-purple-300 rounded text-xs border border-purple-500/20">{t}</span>
                          ))}
                       </div>
                     </div>
                     <div>
                       <span className="text-gray-500 block text-xs mb-1">Output Limits</span>
                       <span className="text-gray-300">{domainConfig.outputSettings.minAnswerLength} - {domainConfig.outputSettings.maxAnswerLength} chars</span>
                     </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Navigation Actions */}
          <div className="flex items-center justify-between pt-8 mt-auto border-t border-slate-700/50">
            <Button
              variant="ghost"
              onClick={handleBack}
              disabled={currentStep === 1}
              className={`${currentStep === 1 ? 'invisible' : ''}`}
              leftIcon={<ArrowLeft className="w-5 h-5" />}
            >
              Back
            </Button>
            
            <div className="flex items-center space-x-3">
              {currentStep === 4 && (
                <span className="text-xs text-gray-500 mr-2">
                  Ready to save?
                </span>
              )}
              <Button
                variant={currentStep === 4 ? "primary" : "secondary"}
                onClick={currentStep === 4 ? handleSave : handleNext}
                isLoading={isSaving}
                rightIcon={currentStep === 4 ? <Save className="w-5 h-5" /> : <ArrowRight className="w-5 h-5" />}
                className={currentStep === 4 ? "bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 shadow-lg shadow-purple-500/25" : ""}
              >
                {currentStep === 4 ? 'Save Template' : 'Next Step'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DomainBuilder;
