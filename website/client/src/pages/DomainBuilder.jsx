import { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  Plus, Trash2, Save, ArrowRight, Lightbulb,
  FileText, Settings2, CheckCircle, AlertCircle
} from 'lucide-react';

const DomainBuilder = () => {
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

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const response = await fetch('/api/domains', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(domainConfig)
      });
      const data = await response.json();
      setSavedMessage('Domain configuration saved successfully!');
      setTimeout(() => setSavedMessage(''), 3000);
    } catch (error) {
      console.error('Failed to save domain:', error);
      setSavedMessage('Configuration ready! Save to your local browser.');
      setTimeout(() => setSavedMessage(''), 3000);
    } finally {
      setIsSaving(false);
    }
  };

  const isValid = domainConfig.name && 
                  domainConfig.topics.some(t => t.name && t.subtopics.some(st => st)) &&
                  domainConfig.questionTypes.length > 0;

  return (
    <div className="pt-20 pb-12 min-h-screen">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Custom Domain Builder</h1>
          <p className="text-gray-400">
            Create a custom domain configuration for generating specialized datasets
          </p>
        </div>

        {/* Success Message */}
        {savedMessage && (
          <div className="mb-6 p-4 bg-green-500/20 border border-green-500/30 rounded-xl flex items-center space-x-3">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-green-300">{savedMessage}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Form */}
          <div className="lg:col-span-2 space-y-6">
            {/* Basic Info */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <FileText className="w-5 h-5 text-purple-400" />
                <span>Basic Information</span>
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Domain Name *</label>
                  <input
                    type="text"
                    value={domainConfig.name}
                    onChange={(e) => setDomainConfig({...domainConfig, name: e.target.value})}
                    placeholder="e.g., Cryptocurrency Education"
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Description</label>
                  <textarea
                    value={domainConfig.description}
                    onChange={(e) => setDomainConfig({...domainConfig, description: e.target.value})}
                    placeholder="Describe what kind of data this domain will generate..."
                    rows={3}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                  />
                </div>
              </div>
            </div>

            {/* Topics */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center space-x-2">
                  <Settings2 className="w-5 h-5 text-purple-400" />
                  <span>Topics & Subtopics</span>
                </h2>
                <button
                  onClick={addTopic}
                  className="flex items-center space-x-1 px-3 py-1.5 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors text-sm"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add Topic</span>
                </button>
              </div>
              
              <div className="space-y-6">
                {domainConfig.topics.map((topic, topicIndex) => (
                  <div key={topicIndex} className="p-4 bg-slate-700/30 rounded-lg">
                    <div className="flex items-center space-x-3 mb-4">
                      <input
                        type="text"
                        value={topic.name}
                        onChange={(e) => updateTopic(topicIndex, 'name', e.target.value)}
                        placeholder={`Topic ${topicIndex + 1} name`}
                        className="flex-1 px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                      {domainConfig.topics.length > 1 && (
                        <button
                          onClick={() => removeTopic(topicIndex)}
                          className="p-2 text-red-400 hover:bg-red-500/20 rounded-lg transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                    
                    <div className="pl-4 border-l-2 border-slate-600 space-y-2">
                      <label className="block text-xs text-gray-500 mb-2">Subtopics</label>
                      {topic.subtopics.map((subtopic, subtopicIndex) => (
                        <div key={subtopicIndex} className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={subtopic}
                            onChange={(e) => updateSubtopic(topicIndex, subtopicIndex, e.target.value)}
                            placeholder={`Subtopic ${subtopicIndex + 1}`}
                            className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                          />
                          {topic.subtopics.length > 1 && (
                            <button
                              onClick={() => removeSubtopic(topicIndex, subtopicIndex)}
                              className="p-1.5 text-gray-400 hover:text-red-400 transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      ))}
                      <button
                        onClick={() => addSubtopic(topicIndex)}
                        className="text-sm text-purple-400 hover:text-purple-300 transition-colors"
                      >
                        + Add Subtopic
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Question Types */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Question Types</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {questionTypeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => toggleQuestionType(option.value)}
                    className={`p-4 rounded-lg border text-left transition-all ${
                      domainConfig.questionTypes.includes(option.value)
                        ? 'bg-purple-500/20 border-purple-500/50 text-white'
                        : 'bg-slate-700/30 border-slate-600 text-gray-300 hover:border-slate-500'
                    }`}
                  >
                    <div className="font-medium">{option.label}</div>
                    <div className="text-sm text-gray-400">{option.description}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Output Settings */}
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Output Settings</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Min Answer Length (chars)</label>
                  <input
                    type="number"
                    value={domainConfig.outputSettings.minAnswerLength}
                    onChange={(e) => setDomainConfig({
                      ...domainConfig,
                      outputSettings: { ...domainConfig.outputSettings, minAnswerLength: parseInt(e.target.value) || 0 }
                    })}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="20"
                    max="200"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Max Answer Length (chars)</label>
                  <input
                    type="number"
                    value={domainConfig.outputSettings.maxAnswerLength}
                    onChange={(e) => setDomainConfig({
                      ...domainConfig,
                      outputSettings: { ...domainConfig.outputSettings, maxAnswerLength: parseInt(e.target.value) || 0 }
                    })}
                    className="w-full px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                    min="100"
                    max="2000"
                  />
                </div>
              </div>
              
              <div className="mt-4">
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={domainConfig.outputSettings.includeMetadata}
                    onChange={(e) => setDomainConfig({
                      ...domainConfig,
                      outputSettings: { ...domainConfig.outputSettings, includeMetadata: e.target.checked }
                    })}
                    className="w-5 h-5 rounded border-slate-600 bg-slate-700 text-purple-500 focus:ring-purple-500"
                  />
                  <span className="text-gray-300">Include metadata (topic, difficulty, timestamp)</span>
                </label>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between pt-4">
              <div className="text-sm text-gray-400">
                {isValid ? (
                  <span className="flex items-center space-x-1 text-green-400">
                    <CheckCircle className="w-4 h-4" />
                    <span>Configuration is valid</span>
                  </span>
                ) : (
                  <span className="flex items-center space-x-1 text-yellow-400">
                    <AlertCircle className="w-4 h-4" />
                    <span>Fill in required fields</span>
                  </span>
                )}
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={handleSave}
                  disabled={!isValid || isSaving}
                  className="flex items-center space-x-2 px-6 py-3 bg-slate-700 rounded-lg font-medium hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Save className="w-5 h-5" />
                  <span>{isSaving ? 'Saving...' : 'Save Template'}</span>
                </button>
                <Link
                  to={isValid ? `/dashboard?domain=custom` : '#'}
                  className={`flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg font-medium transition-opacity ${
                    isValid ? 'hover:opacity-90' : 'opacity-50 cursor-not-allowed'
                  }`}
                  onClick={(e) => !isValid && e.preventDefault()}
                >
                  <span>Generate Dataset</span>
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </div>
            </div>
          </div>

          {/* Preview Panel */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 sticky top-24">
              <h2 className="text-xl font-semibold mb-4">Configuration Preview</h2>
              
              <div className="space-y-4 text-sm">
                <div>
                  <span className="text-gray-400">Domain:</span>
                  <p className="font-medium">{domainConfig.name || 'Not set'}</p>
                </div>
                
                <div>
                  <span className="text-gray-400">Topics ({domainConfig.topics.filter(t => t.name).length}):</span>
                  <ul className="mt-1 space-y-1">
                    {domainConfig.topics.filter(t => t.name).map((topic, i) => (
                      <li key={i} className="text-gray-300">
                        • {topic.name}
                        {topic.subtopics.filter(st => st).length > 0 && (
                          <span className="text-gray-500"> ({topic.subtopics.filter(st => st).length} subtopics)</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div>
                  <span className="text-gray-400">Question Types:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {domainConfig.questionTypes.map((type) => (
                      <span key={type} className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded text-xs">
                        {type}
                      </span>
                    ))}
                  </div>
                </div>
                
                <div>
                  <span className="text-gray-400">Answer Length:</span>
                  <p className="text-gray-300">
                    {domainConfig.outputSettings.minAnswerLength} - {domainConfig.outputSettings.maxAnswerLength} chars
                  </p>
                </div>
              </div>
            </div>

            {/* Tips */}
            <div className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 border border-purple-500/20 rounded-xl p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Lightbulb className="w-5 h-5 text-yellow-400" />
                <h3 className="font-semibold">Tips</h3>
              </div>
              <ul className="space-y-2 text-sm text-gray-300">
                <li>• Add 3-5 topics for diverse datasets</li>
                <li>• Include 2-4 subtopics per topic</li>
                <li>• Mix question types for better coverage</li>
                <li>• Longer answers work better for training</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DomainBuilder;
