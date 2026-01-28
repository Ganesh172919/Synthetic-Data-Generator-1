const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage (for demo purposes)
const jobs = new Map();
const domains = new Map();
const jobIntervals = new Map(); // Track intervals for cleanup

// Default templates
const templates = [
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
  }
];

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Get all templates
app.get('/api/templates', (req, res) => {
  res.json({ templates });
});

// Get template by ID
app.get('/api/templates/:id', (req, res) => {
  const template = templates.find(t => t.id === req.params.id);
  if (!template) {
    return res.status(404).json({ error: 'Template not found' });
  }
  res.json(template);
});

// Validation helpers
const validDomains = ['financial', 'healthcare', 'legal', 'technology', 'science', 'education', 'custom'];
const validOutputFormats = ['jsonl', 'csv', 'parquet'];

// Start generation job
app.post('/api/generate', (req, res) => {
  const { domain, targetCount, batchSize, outputFormat } = req.body;
  
  // Input validation
  if (!domain || typeof domain !== 'string') {
    return res.status(400).json({ error: 'Domain is required and must be a string' });
  }
  if (!validDomains.includes(domain)) {
    return res.status(400).json({ error: `Invalid domain. Must be one of: ${validDomains.join(', ')}` });
  }
  
  const parsedTargetCount = parseInt(targetCount) || 1000;
  const parsedBatchSize = parseInt(batchSize) || 25;
  
  if (parsedTargetCount < 100 || parsedTargetCount > 100000) {
    return res.status(400).json({ error: 'Target count must be between 100 and 100000' });
  }
  if (parsedBatchSize < 5 || parsedBatchSize > 50) {
    return res.status(400).json({ error: 'Batch size must be between 5 and 50' });
  }
  if (outputFormat && !validOutputFormats.includes(outputFormat)) {
    return res.status(400).json({ error: `Invalid output format. Must be one of: ${validOutputFormats.join(', ')}` });
  }
  
  const jobId = `gen_${uuidv4().substring(0, 8)}`;
  const job = {
    id: jobId,
    domain,
    targetCount: parsedTargetCount,
    batchSize: parsedBatchSize,
    outputFormat: outputFormat || 'jsonl',
    status: 'running',
    generated: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };
  
  jobs.set(jobId, job);
  
  // Simulate progress (in a real app, this would be handled by a job queue)
  simulateProgress(jobId);
  
  res.json({
    jobId,
    status: 'running',
    estimatedTime: `${Math.ceil(parsedTargetCount / 100)} minutes`
  });
});

// Get job status
app.get('/api/jobs/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json(job);
});

// List all jobs
app.get('/api/jobs', (req, res) => {
  const allJobs = Array.from(jobs.values()).sort((a, b) => 
    new Date(b.createdAt) - new Date(a.createdAt)
  );
  res.json({ jobs: allJobs });
});

// Save custom domain
app.post('/api/domains', (req, res) => {
  const domainConfig = req.body;
  
  // Input validation
  if (!domainConfig || typeof domainConfig !== 'object') {
    return res.status(400).json({ error: 'Domain configuration is required' });
  }
  if (!domainConfig.name || typeof domainConfig.name !== 'string' || domainConfig.name.trim().length === 0) {
    return res.status(400).json({ error: 'Domain name is required' });
  }
  if (domainConfig.name.length > 100) {
    return res.status(400).json({ error: 'Domain name must be less than 100 characters' });
  }
  if (!domainConfig.topics || !Array.isArray(domainConfig.topics) || domainConfig.topics.length === 0) {
    return res.status(400).json({ error: 'At least one topic is required' });
  }
  
  const domainId = `domain_${uuidv4().substring(0, 8)}`;
  
  const domain = {
    id: domainId,
    name: domainConfig.name.trim(),
    description: domainConfig.description || '',
    topics: domainConfig.topics,
    questionTypes: domainConfig.questionTypes || ['definition'],
    difficultyLevels: domainConfig.difficultyLevels || ['beginner'],
    outputSettings: domainConfig.outputSettings || {},
    createdAt: new Date().toISOString()
  };
  
  domains.set(domainId, domain);
  
  res.json({
    id: domainId,
    message: 'Domain configuration saved successfully'
  });
});

// Get domain by ID
app.get('/api/domains/:id', (req, res) => {
  const domain = domains.get(req.params.id);
  if (!domain) {
    return res.status(404).json({ error: 'Domain not found' });
  }
  res.json(domain);
});

// List all domains
app.get('/api/domains', (req, res) => {
  const allDomains = Array.from(domains.values());
  res.json({ domains: allDomains });
});

// Helper function to simulate job progress
function simulateProgress(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  // Clear any existing interval for this job
  if (jobIntervals.has(jobId)) {
    clearInterval(jobIntervals.get(jobId));
    jobIntervals.delete(jobId);
  }
  
  const interval = setInterval(() => {
    const currentJob = jobs.get(jobId);
    if (!currentJob || currentJob.status !== 'running') {
      clearInterval(interval);
      jobIntervals.delete(jobId);
      return;
    }
    
    const increment = Math.floor(Math.random() * 50) + 30;
    currentJob.generated = Math.min(currentJob.generated + increment, currentJob.targetCount);
    currentJob.updatedAt = new Date().toISOString();
    
    if (currentJob.generated >= currentJob.targetCount) {
      currentJob.status = 'completed';
      currentJob.downloadUrl = `/api/downloads/${jobId}.jsonl`;
      clearInterval(interval);
      jobIntervals.delete(jobId);
    }
    
    jobs.set(jobId, currentJob);
  }, 2000);
  
  // Store interval for cleanup
  jobIntervals.set(jobId, interval);
}

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 API endpoints available at http://localhost:${PORT}/api`);
});

module.exports = app;
