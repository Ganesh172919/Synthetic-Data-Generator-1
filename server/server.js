const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Storage paths
const DATA_DIR = path.join(__dirname, '../data');
const OUTPUTS_DIR = path.join(DATA_DIR, 'outputs');
const CHECKPOINTS_DIR = path.join(DATA_DIR, 'checkpoints');
const CONFIGS_DIR = path.join(DATA_DIR, 'configs');

// Ensure directories exist
async function ensureDirectories() {
  for (const dir of [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, CONFIGS_DIR]) {
    await fs.mkdir(dir, { recursive: true });
  }
}

// Initialize on startup
ensureDirectories().catch(console.error);

// In-memory storage
const jobs = new Map();
const domains = new Map();
const activeProcesses = new Map(); // jobId -> Python process

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
    featured: true,
    generatorType: 'financial'
  },
  {
    id: 'healthcare-clinical',
    name: 'Clinical Knowledge Base',
    description: 'Medical terminology, symptoms, treatments, and healthcare procedures for training medical AI.',
    category: 'healthcare',
    rating: 4.8,
    downloads: 8700,
    topics: ['Medical Terms', 'Symptoms', 'Treatments', 'Procedures'],
    featured: true,
    generatorType: 'universal'
  },
  {
    id: 'legal-contracts',
    name: 'Legal Document Analysis',
    description: 'Contract clauses, legal terminology, compliance requirements, and case law summaries.',
    category: 'legal',
    rating: 4.7,
    downloads: 6300,
    topics: ['Contracts', 'Compliance', 'Legal Terms', 'Case Law'],
    featured: false,
    generatorType: 'universal'
  },
  {
    id: 'tech-programming',
    name: 'Programming Q&A',
    description: 'Code explanations, debugging help, best practices, and algorithm discussions.',
    category: 'technology',
    rating: 4.9,
    downloads: 15200,
    topics: ['Python', 'JavaScript', 'Algorithms', 'Best Practices'],
    featured: true,
    generatorType: 'universal'
  },
  {
    id: 'science-research',
    name: 'Scientific Research Assistant',
    description: 'Research methodology, experiment design, data analysis, and academic writing.',
    category: 'science',
    rating: 4.6,
    downloads: 4500,
    topics: ['Methodology', 'Data Analysis', 'Papers', 'Citations'],
    featured: false,
    generatorType: 'universal'
  },
  {
    id: 'edu-tutoring',
    name: 'Educational Tutoring',
    description: 'Math, science, language arts explanations suitable for K-12 and college students.',
    category: 'education',
    rating: 4.8,
    downloads: 9800,
    topics: ['Math', 'Science', 'English', 'History'],
    featured: false,
    generatorType: 'universal'
  }
];

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    pythonAvailable: checkPythonAvailable()
  });
});

// Check if Python is available
function checkPythonAvailable() {
  try {
    const pythonPath = process.env.PYTHON_PATH || 'python3';
    const result = require('child_process').spawnSync(pythonPath, ['--version'], { encoding: 'utf8' });
    return result.status === 0;
  } catch {
    return false;
  }
}

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
const validOutputFormats = ['jsonl', 'csv', 'json'];

// Start generation job
app.post('/api/generate', async (req, res) => {
  const { 
    domain, 
    targetCount, 
    batchSize, 
    outputFormat, 
    domainDescription,
    topics,
    templateId,
    modelName,
    temperature,
    useQuantization
  } = req.body;
  
  try {
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
    
    // Validate domainDescription
    if (domainDescription && typeof domainDescription === 'string' && domainDescription.length > 1000) {
      return res.status(400).json({ error: 'Domain description must be less than 1000 characters' });
    }
    
    // Validate topics array
    if (topics && Array.isArray(topics)) {
      for (const topic of topics) {
        if (typeof topic !== 'string' || topic.trim().length === 0) {
          return res.status(400).json({ error: 'All topics must be non-empty strings' });
        }
        if (topic.length > 200) {
          return res.status(400).json({ error: 'Topic must be less than 200 characters' });
        }
      }
    }
    
    // Validate temperature
    const parsedTemperature = typeof temperature === 'number' ? temperature : 0.8;
    if (parsedTemperature < 0 || parsedTemperature > 2.0) {
      return res.status(400).json({ error: 'Temperature must be between 0 and 2.0' });
    }
    
    // Validate modelName (basic format check)
    const allowedModels = [
      'mistralai/Mistral-7B-Instruct-v0.2',
      'microsoft/phi-2',
      'google/gemma-2b'
    ];
    const parsedModelName = modelName || 'mistralai/Mistral-7B-Instruct-v0.2';
    if (!allowedModels.includes(parsedModelName)) {
      return res.status(400).json({ 
        error: `Model must be one of: ${allowedModels.join(', ')}` 
      });
    }
    
    const jobId = `gen_${uuidv4().substring(0, 8)}`;
    const outputFileName = `dataset_${jobId}`;
    const configFile = path.join(CONFIGS_DIR, `${jobId}_config.json`);
    
    // Determine generator type
    let generatorType = 'universal';
    if (templateId) {
      const template = templates.find(t => t.id === templateId);
      if (template) {
        generatorType = template.generatorType || 'universal';
      }
    } else if (domain === 'financial') {
      generatorType = 'financial';
    }
    
    // Build configuration for Python generator
    const generatorConfig = {
      targetSize: parsedTargetCount,
      batchSize: parsedBatchSize,
      outputFile: path.join(OUTPUTS_DIR, outputFileName),
      outputFormat: outputFormat || 'jsonl',
      domainDescription: domainDescription || `${domain} dataset`,
      topics: topics || [],
      modelName: parsedModelName,
      temperature: parsedTemperature,
      useQuantization: useQuantization !== false,
      checkpointFile: path.join(CHECKPOINTS_DIR, `${jobId}_checkpoint.json`),
      saveInterval: 100
    };
    
    // Save config file
    try {
      await fs.writeFile(configFile, JSON.stringify(generatorConfig, null, 2));
    } catch (error) {
      console.error('Failed to save config file:', error);
      return res.status(500).json({ error: 'Failed to save configuration', details: error.message });
    }
    
    // Create job record
    const job = {
      id: jobId,
      domain,
      targetCount: parsedTargetCount,
      batchSize: parsedBatchSize,
      outputFormat: outputFormat || 'jsonl',
      outputFile: `${outputFileName}.${outputFormat || 'jsonl'}`,
      generatorType,
      status: 'initializing',
      generated: 0,
      rate: 0,
      progress: 0,
      estimatedTimeRemaining: null,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      logs: [],
      maxLogs: 100  // Limit log entries to prevent memory issues
    };
    
    jobs.set(jobId, job);
    
    // Start Python generator process
    startGeneratorProcess(jobId, configFile, generatorType);
    
    res.json({
      jobId,
      status: 'initializing',
      message: 'Generation job started'
    });
    
  } catch (error) {
    console.error('Error starting generation:', error);
    res.status(500).json({ error: 'Failed to start generation job', details: error.message });
  }
});

// Start Python generator subprocess
function startGeneratorProcess(jobId, configFile, generatorType) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  const pythonPath = process.env.PYTHON_PATH || 'python3';
  const runnerScript = path.join(__dirname, 'generator_runner.py');
  
  const args = [
    runnerScript,
    '--config', configFile,
    '--generator', generatorType
  ];
  
  console.log(`Starting generator: ${pythonPath} ${args.join(' ')}`);
  
  const pythonProcess = spawn(pythonPath, args, {
    cwd: __dirname,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });
  
  activeProcesses.set(jobId, pythonProcess);
  
  // Handle stdout - parse progress events
  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[${jobId}] stdout:`, output);
    
    // Parse progress events (format: PROGRESS:{json})
    const lines = output.split('\n');
    for (const line of lines) {
      if (line.startsWith('PROGRESS:')) {
        try {
          const event = JSON.parse(line.substring(9));
          handleProgressEvent(jobId, event);
        } catch (e) {
          console.error('Failed to parse progress event:', e);
        }
      }
    }
    
    // Add to job logs with size limit
    const currentJob = jobs.get(jobId);
    if (currentJob) {
      currentJob.logs.push({ timestamp: new Date().toISOString(), type: 'info', message: output.trim() });
      // Keep only the most recent maxLogs entries
      if (currentJob.logs.length > currentJob.maxLogs) {
        currentJob.logs = currentJob.logs.slice(-currentJob.maxLogs);
      }
    }
  });
  
  // Handle stderr
  pythonProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[${jobId}] stderr:`, output);
    
    // Parse error events (format: ERROR:{json})
    if (output.startsWith('ERROR:')) {
      try {
        const event = JSON.parse(output.substring(6));
        handleErrorEvent(jobId, event);
      } catch (e) {
        console.error('Failed to parse error event:', e);
      }
    }
    
    const currentJob = jobs.get(jobId);
    if (currentJob) {
      currentJob.logs.push({ timestamp: new Date().toISOString(), type: 'error', message: output.trim() });
      // Keep only the most recent maxLogs entries
      if (currentJob.logs.length > currentJob.maxLogs) {
        currentJob.logs = currentJob.logs.slice(-currentJob.maxLogs);
      }
    }
  });
  
  // Handle process exit
  pythonProcess.on('exit', (code, signal) => {
    console.log(`[${jobId}] Process exited with code ${code}, signal ${signal}`);
    
    const currentJob = jobs.get(jobId);
    if (currentJob && currentJob.status !== 'completed') {
      if (code === 0) {
        currentJob.status = 'completed';
        currentJob.progress = 100;
        currentJob.generated = currentJob.targetCount;
      } else {
        currentJob.status = 'failed';
        currentJob.error = `Process exited with code ${code}`;
      }
      currentJob.updatedAt = new Date().toISOString();
    }
    
    activeProcesses.delete(jobId);
  });
  
  pythonProcess.on('error', (error) => {
    console.error(`[${jobId}] Process error:`, error);
    
    const currentJob = jobs.get(jobId);
    if (currentJob) {
      currentJob.status = 'failed';
      currentJob.error = error.message;
      currentJob.updatedAt = new Date().toISOString();
    }
    
    activeProcesses.delete(jobId);
  });
  
  // Update job status
  job.status = 'running';
  job.updatedAt = new Date().toISOString();
}

// Handle progress events from Python
function handleProgressEvent(jobId, event) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  const { type, data } = event;
  
  switch (type) {
    case 'init':
      job.status = 'initializing';
      break;
    
    case 'start':
      job.status = 'running';
      break;
    
    case 'progress':
      job.generated = data.current || 0;
      job.rate = data.rate || 0;
      job.progress = data.percentage || 0;
      if (data.rate > 0 && job.targetCount > job.generated) {
        job.estimatedTimeRemaining = Math.ceil((job.targetCount - job.generated) / data.rate);
      }
      break;
    
    case 'complete':
      job.status = 'completed';
      job.generated = job.targetCount;
      job.progress = 100;
      job.downloadUrl = `/api/downloads/${job.id}/${job.outputFile}`;
      break;
    
    case 'shutdown':
      if (job.status === 'running') {
        job.status = 'paused';
      }
      break;
  }
  
  job.updatedAt = new Date().toISOString();
}

// Handle error events from Python
function handleErrorEvent(jobId, event) {
  const job = jobs.get(jobId);
  if (!job) return;
  
  job.status = 'failed';
  job.error = event.message;
  job.errorDetails = event.details;
  job.updatedAt = new Date().toISOString();
}

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

// Pause/stop a job
app.post('/api/jobs/:jobId/stop', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobs.get(jobId);
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  const process = activeProcesses.get(jobId);
  if (process) {
    process.kill('SIGTERM');
    job.status = 'stopped';
    job.updatedAt = new Date().toISOString();
    activeProcesses.delete(jobId);
    return res.json({ message: 'Job stopped successfully', job });
  }
  
  res.json({ message: 'Job already stopped', job });
});

// Download generated dataset
app.get('/api/downloads/:jobId/:filename', async (req, res) => {
  const { jobId, filename } = req.params;
  const job = jobs.get(jobId);
  
  if (!job) {
    return res.status(404).json({ error: 'Job not found' });
  }
  
  // Sanitize filename to prevent path traversal
  const sanitizedFilename = path.basename(filename);
  if (sanitizedFilename !== filename || filename.includes('..')) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  
  const filePath = path.join(OUTPUTS_DIR, sanitizedFilename);
  
  // Verify the resolved path is within OUTPUTS_DIR
  const resolvedPath = path.resolve(filePath);
  const resolvedOutputDir = path.resolve(OUTPUTS_DIR);
  if (!resolvedPath.startsWith(resolvedOutputDir)) {
    return res.status(400).json({ error: 'Invalid file path' });
  }
  
  try {
    await fs.access(filePath);
    res.download(filePath, sanitizedFilename);
  } catch (error) {
    res.status(404).json({ error: 'File not found', details: error.message });
  }
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

// Cleanup on shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, cleaning up...');
  for (const [jobId, process] of activeProcesses.entries()) {
    console.log(`Stopping job ${jobId}...`);
    process.kill('SIGTERM');
  }
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, cleaning up...');
  for (const [jobId, process] of activeProcesses.entries()) {
    console.log(`Stopping job ${jobId}...`);
    process.kill('SIGTERM');
  }
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 API endpoints available at http://localhost:${PORT}/api`);
  console.log(`🐍 Python available: ${checkPythonAvailable() ? 'Yes' : 'No'}`);
});

module.exports = app;
