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

  // List templates
  getTemplates: async () => {
    const response = await fetch(`${API_BASE}/templates`);
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
    const response = await fetch(`${API_BASE}/domains/${domainId}`);
    return handleResponse(response);
  }
};

export default api;
