/**
 * SynthGen — Demo API Server (Express)
 *
 * What this server is:
 * - A lightweight REST API that powers the web UI in `website/client/`
 * - A teaching-friendly example of a "job" based workflow (start → poll → download)
 *
 * What this server is NOT (yet):
 * - A production job runner
 * - A durable job database (jobs/domains are stored in memory)
 * - A real integration layer to the Python generators in `Pre-Work/`
 *
 * Reality-aligned behavior:
 * - `POST /api/generate` creates an in-memory job and simulates progress via `setInterval`.
 * - `GET /api/downloads/:jobId/:format` returns a *mock* dataset payload based on job config.
 *
 * Extension points (if you want "real generation"):
 * - Replace `simulateProgress(jobId)` with a queue/worker model
 * - Spawn `Pre-Work/universal_dataset_generator.py` (or similar) in a worker process
 * - Persist job state (SQLite/Postgres) and stream real output files for downloads
 *
 * See also:
 * - `docs/ARCHITECTURE.md`
 * - `docs/WEB_PLATFORM.md`
 * - `docs/SECURITY_AND_SAFETY.md`
 */

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
// GET /api/health
// - Purpose: basic liveness check for the UI and local scripts
// - Response: { status: "ok", timestamp: "..." }
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Get all templates
// GET /api/templates
// - Purpose: populate the Templates page in the UI
// - Response: { templates: [...] }
app.get('/api/templates', (req, res) => {
  res.json({ templates });
});

// Get template by ID
// GET /api/templates/:id
// - Purpose: fetch details for one template
// - Edge cases: 404 if the id doesn't exist
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
// POST /api/generate
// Request body (example):
//   {
//     "domain": "technology",
//     "targetCount": 1000,
//     "batchSize": 25,
//     "outputFormat": "jsonl"
//   }
//
// Response (demo):
//   { jobId, status: "running", estimatedTime: "10 minutes" }
//
// Reality note:
// - This endpoint does not start real generation; it starts a simulated progress timer.
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
// GET /api/jobs/:jobId
// - Purpose: polling endpoint for UI progress updates
// - Edge cases: 404 if job doesn't exist; job state is in memory (lost on restart)
app.get('/api/jobs/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  res.json(job);
});

// List all jobs
// GET /api/jobs
// - Purpose: show a job list/history in the UI
// - Reality note: this is limited to the current server process lifetime
app.get('/api/jobs', (req, res) => {
  const allJobs = Array.from(jobs.values()).sort((a, b) => 
    new Date(b.createdAt) - new Date(a.createdAt)
  );
  res.json({ jobs: allJobs });
});

// Stop a running job
// POST /api/jobs/:jobId/stop
// - Purpose: stop the simulated progress timer and mark job as "stopped"
// - Edge cases: 400 if job is not running
app.post('/api/jobs/:jobId/stop', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  if (job.status !== 'running') {
    return res.status(400).json({ error: 'Job is not running' });
  }
  
  // Clear the interval if running
  if (jobIntervals.has(req.params.jobId)) {
    clearInterval(jobIntervals.get(req.params.jobId));
    jobIntervals.delete(req.params.jobId);
  }
  
  job.status = 'stopped';
  job.updatedAt = new Date().toISOString();
  jobs.set(req.params.jobId, job);
  
  res.json({ message: 'Job stopped successfully', job });
});

// Delete a job
// DELETE /api/jobs/:jobId
// - Purpose: remove job metadata from memory and clear any timers
app.delete('/api/jobs/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  // Clear interval if exists
  if (jobIntervals.has(req.params.jobId)) {
    clearInterval(jobIntervals.get(req.params.jobId));
    jobIntervals.delete(req.params.jobId);
  }
  
  jobs.delete(req.params.jobId);
  res.json({ message: 'Job deleted successfully' });
});

// Download generated dataset (mock)
// GET /api/downloads/:jobId/:format
// - Purpose: download a dataset artifact when a job completes
// - Reality note: this endpoint does not read a real file; it synthesizes a payload in memory.
// - Formats:
//   - jsonl: application/x-ndjson (one JSON object per line)
//   - csv:  text/csv
//   - json: application/json
//
// Edge cases:
// - 400 if job is not completed yet
// - This demo endpoint does not implement path traversal protection because it does not accept filenames.
app.get('/api/downloads/:jobId/:format', (req, res) => {
  const { jobId, format } = req.params;
  const job = jobs.get(jobId);
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  if (job.status !== 'completed') {
    return res.status(400).json({ error: 'Job is not completed yet' });
  }
  
  // Generate mock data based on job configuration
  const mockData = generateMockDataset(job);
  
  res.setHeader('Content-Disposition', `attachment; filename="${jobId}.${format}"`);
  
  if (format === 'jsonl') {
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.send(mockData.map(item => JSON.stringify(item)).join('\n'));
  } else if (format === 'csv') {
    res.setHeader('Content-Type', 'text/csv');
    const headers = Object.keys(mockData[0]).join(',');
    const rows = mockData.map(item => Object.values(item).map(v => `"${v}"`).join(','));
    res.send([headers, ...rows].join('\n'));
  } else {
    res.setHeader('Content-Type', 'application/json');
    res.json(mockData);
  }
});

// Helper function to generate mock dataset
function generateMockDataset(job) {
  const sampleSize = Math.min(job.generated, 100); // Return max 100 samples
  const data = [];
  
  const topics = {
    financial: ['Investing', 'Budgeting', 'Credit', 'Retirement', 'Taxes'],
    healthcare: ['Symptoms', 'Treatments', 'Prevention', 'Wellness', 'Nutrition'],
    legal: ['Contracts', 'Rights', 'Compliance', 'Litigation', 'IP'],
    technology: ['Programming', 'AI/ML', 'Cloud', 'Security', 'DevOps'],
    science: ['Physics', 'Chemistry', 'Biology', 'Research', 'Data'],
    education: ['Math', 'Science', 'Language', 'History', 'Arts'],
    custom: ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']
  };
  
  const domainTopics = topics[job.domain] || topics.custom;
  
  for (let i = 0; i < sampleSize; i++) {
    const topic = domainTopics[i % domainTopics.length];
    data.push({
      id: `${job.domain}_${i + 1}`,
      topic: topic,
      question: `Sample question about ${topic} #${i + 1}?`,
      answer: `This is a comprehensive answer about ${topic}. It provides detailed information and insights relevant to the ${job.domain} domain.`,
      difficulty: ['beginner', 'intermediate', 'advanced'][i % 3],
      created_at: new Date().toISOString()
    });
  }
  
  return data;
}

// Save custom domain
// POST /api/domains
// - Purpose: store a "custom domain" configuration built in the UI
// - Reality note: stored in memory only; this is a demo persistence layer
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
// GET /api/domains/:id
// - Purpose: fetch one custom domain config
app.get('/api/domains/:id', (req, res) => {
  const domain = domains.get(req.params.id);
  if (!domain) {
    return res.status(404).json({ error: 'Domain not found' });
  }
  res.json(domain);
});

// List all domains
// GET /api/domains
// - Purpose: list all custom domains created since the server started
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
      // Educational note:
      // `downloadUrl` is illustrative metadata. The canonical download endpoint is:
      //   GET /api/downloads/:jobId/:format
      // Clients should construct URLs using (jobId, desiredFormat) rather than relying on this field.
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

// Production hardening notes (non-executable):
// - Add authentication/authorization before exposing the API beyond localhost.
// - Persist jobs/domains to a database instead of in-memory Maps.
// - Replace simulated progress with a durable queue + worker model.
// - Stream real dataset files from disk/object storage in the downloads endpoint.
// - Add rate limiting, request size limits, and structured logging for safety.

module.exports = app;
