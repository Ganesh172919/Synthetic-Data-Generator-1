const API_BASE = '/api';

const buildQuery = (params = {}) => {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      search.set(key, String(value));
    }
  });
  const query = search.toString();
  return query ? `?${query}` : '';
};

const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || `HTTP error ${response.status}`);
  }

  if (response.status === 204) {
    return null;
  }
  return response.json();
};

export const api = {
  checkHealth: async () => {
    const response = await fetch(`${API_BASE}/health`);
    return handleResponse(response);
  },

  startGeneration: async (config) => {
    const response = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse(response);
  },

  getJobStatus: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    return handleResponse(response);
  },

  listJobs: async (params = {}) => {
    const response = await fetch(`${API_BASE}/jobs${buildQuery(params)}`);
    return handleResponse(response);
  },

  stopJob: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/stop`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  retryJob: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/retry`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  deleteJob: async (jobId) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
      method: 'DELETE',
    });
    return handleResponse(response);
  },

  getJobPreview: async (jobId, limit = 20) => {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/preview${buildQuery({ limit })}`);
    return handleResponse(response);
  },

  getDownloadUrl: (jobId, format = 'jsonl') => {
    const validFormats = ['jsonl', 'csv', 'json'];
    if (!validFormats.includes(format)) {
      throw new Error('Invalid format. Use jsonl, csv, or json');
    }
    return `${API_BASE}/downloads/${jobId}/${format}`;
  },

  streamJobEvents: (jobId, sinceId = 0) => {
    return new EventSource(`${API_BASE}/jobs/${jobId}/events${buildQuery({ sinceId })}`);
  },

  getTemplates: async () => {
    const response = await fetch(`${API_BASE}/templates`);
    return handleResponse(response);
  },

  getTemplate: async (templateId) => {
    const response = await fetch(`${API_BASE}/templates/${templateId}`);
    return handleResponse(response);
  },

  saveDomain: async (domainConfig) => {
    const response = await fetch(`${API_BASE}/domains`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(domainConfig),
    });
    return handleResponse(response);
  },

  getDomain: async (domainId) => {
    const response = await fetch(`${API_BASE}/domains/${domainId}`);
    return handleResponse(response);
  },

  listDomains: async () => {
    const response = await fetch(`${API_BASE}/domains`);
    return handleResponse(response);
  },
};

export default api;
