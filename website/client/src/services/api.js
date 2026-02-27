const API_BASE = '/api';

const TOKEN_KEY = 'synthgen_token';

const getToken = () => {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
};

const setToken = (token) => {
  try { localStorage.setItem(TOKEN_KEY, token); } catch { /* localStorage unavailable */ }
};

const clearToken = () => {
  try { localStorage.removeItem(TOKEN_KEY); } catch { /* localStorage unavailable */ }
};

const authHeaders = () => {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
};

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

  // --- Auth ---

  register: async ({ email, username, password, displayName }) => {
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, username, password, displayName }),
    });
    return handleResponse(response);
  },

  login: async ({ email, password }) => {
    const response = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return handleResponse(response);
  },

  getProfile: async () => {
    const response = await fetch(`${API_BASE}/auth/profile`, {
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  updateProfile: async (data) => {
    const response = await fetch(`${API_BASE}/auth/profile`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(data),
    });
    return handleResponse(response);
  },

  changePassword: async ({ currentPassword, newPassword }) => {
    const response = await fetch(`${API_BASE}/auth/change-password`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ currentPassword, newPassword }),
    });
    return handleResponse(response);
  },

  // --- Billing & Subscriptions ---

  getTiers: async () => {
    const response = await fetch(`${API_BASE}/tiers`);
    return handleResponse(response);
  },

  getSubscription: async () => {
    const response = await fetch(`${API_BASE}/billing/subscription`, {
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  changeTier: async (tier) => {
    const response = await fetch(`${API_BASE}/billing/subscription/change`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ tier }),
    });
    return handleResponse(response);
  },

  getUsage: async () => {
    const response = await fetch(`${API_BASE}/billing/usage`, {
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  getQuotaCheck: async () => {
    const response = await fetch(`${API_BASE}/billing/quota-check`, {
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  // --- API Keys ---

  listApiKeys: async () => {
    const response = await fetch(`${API_BASE}/api-keys`, {
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  createApiKey: async (name) => {
    const response = await fetch(`${API_BASE}/api-keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ name }),
    });
    return handleResponse(response);
  },

  revokeApiKey: async (keyId) => {
    const response = await fetch(`${API_BASE}/api-keys/${keyId}/revoke`, {
      method: 'POST',
      headers: authHeaders(),
    });
    return handleResponse(response);
  },

  // --- Plugins ---

  listPlugins: async () => {
    const response = await fetch(`${API_BASE}/plugins`);
    return handleResponse(response);
  },
};

export { getToken, setToken, clearToken };
export default api;
