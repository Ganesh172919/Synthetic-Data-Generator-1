const API_BASE = '/api';

export const api = {
  // Generate dataset
  startGeneration: async (config) => {
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  },

  // Get job status
  getJobStatus: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    return response.json();
  },

  // List templates
  getTemplates: async () => {
    const response = await fetch(`${API_BASE}/templates`);
    return response.json();
  },

  // Save custom domain
  saveDomain: async (domainConfig) => {
    const response = await fetch(`${API_BASE}/domains`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(domainConfig)
    });
    return response.json();
  },

  // Get domain by ID
  getDomain: async (domainId) => {
    const response = await fetch(`${API_BASE}/domains/${domainId}`);
    return response.json();
  }
};

export default api;
