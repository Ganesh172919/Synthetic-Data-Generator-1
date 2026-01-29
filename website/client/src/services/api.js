const API_BASE = '/api';

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const api = {
  // Health check
  checkHealth: async () => {
    const response = await fetch(`${API_BASE}/health`);
    return handleResponse(response);
  },

  // Generate dataset
  startGeneration: async (config) => {
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return handleResponse(response);
  },

  // Get job status
  getJobStatus: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    return handleResponse(response);
  },

  // List all jobs
  listJobs: async () => {
    const response = await fetch(`${API_BASE}/jobs`);
    return handleResponse(response);
  },

  // Stop a job
  stopJob: async (jobId) => {
    if (!jobId || typeof jobId !== 'string') {
      throw new Error('Invalid jobId');
    }
    const response = await fetch(`${API_BASE}/jobs/${jobId}/stop`, {
      method: 'POST'
    });
    return handleResponse(response);
  },

  // Download dataset
  getDownloadUrl: (jobId, filename) => {
    if (!jobId || typeof jobId !== 'string') {
      throw new Error('Invalid jobId');
    }
    if (!filename || typeof filename !== 'string' || filename.includes('..')) {
      throw new Error('Invalid filename');
    }
    return `${API_BASE}/downloads/${jobId}/${filename}`;
  },

  // List templates
  getTemplates: async () => {
    const response = await fetch(`${API_BASE}/templates`);
    return handleResponse(response);
  },

  // Get template by ID
  getTemplate: async (templateId) => {
    if (!templateId || typeof templateId !== 'string') {
      throw new Error('Invalid templateId');
    }
    const response = await fetch(`${API_BASE}/templates/${templateId}`);
    return handleResponse(response);
  },

  // Save custom domain
  saveDomain: async (domainConfig) => {
    const response = await fetch(`${API_BASE}/domains`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(domainConfig)
    });
    return handleResponse(response);
  },

  // Get domain by ID
  getDomain: async (domainId) => {
    if (!domainId || typeof domainId !== 'string') {
      throw new Error('Invalid domainId');
    }
    const response = await fetch(`${API_BASE}/domains/${domainId}`);
    return handleResponse(response);
  },

  // List all domains
  listDomains: async () => {
    const response = await fetch(`${API_BASE}/domains`);
    return handleResponse(response);
  }
};

export default api;
