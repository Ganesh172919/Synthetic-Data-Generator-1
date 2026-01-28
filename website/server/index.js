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

// Start generation job
app.post('/api/generate', (req, res) => {
  const { domain, targetCount, batchSize, outputFormat } = req.body;
  
  const jobId = `gen_${uuidv4().substring(0, 8)}`;
  const job = {
    id: jobId,
    domain,
    targetCount: targetCount || 1000,
    batchSize: batchSize || 25,
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
    estimatedTime: `${Math.ceil(targetCount / 100)} minutes`
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
  const domainId = `domain_${uuidv4().substring(0, 8)}`;
  
  const domain = {
    id: domainId,
    ...domainConfig,
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
  
  const interval = setInterval(() => {
    const currentJob = jobs.get(jobId);
    if (!currentJob || currentJob.status !== 'running') {
      clearInterval(interval);
      return;
    }
    
    const increment = Math.floor(Math.random() * 50) + 30;
    currentJob.generated = Math.min(currentJob.generated + increment, currentJob.targetCount);
    currentJob.updatedAt = new Date().toISOString();
    
    if (currentJob.generated >= currentJob.targetCount) {
      currentJob.status = 'completed';
      currentJob.downloadUrl = `/api/downloads/${jobId}.jsonl`;
      clearInterval(interval);
    }
    
    jobs.set(jobId, currentJob);
  }, 2000);
}

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 API endpoints available at http://localhost:${PORT}/api`);
});

module.exports = app;
