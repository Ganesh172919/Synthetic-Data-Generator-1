import { useState, useEffect, useCallback } from 'react';
import { useTheme } from '../hooks/useTheme';
import { useAuth } from '../hooks/useAuth';
import { api } from '../services/api';

function ProfileSection({ isDark }) {
  const { user, refreshProfile } = useAuth();
  const [form, setForm] = useState({ displayName: '', username: '' });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (user) {
      setForm({ displayName: user.displayName || '', username: user.username || '' });
    }
  }, [user]);

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');
    try {
      await api.updateProfile(form);
      await refreshProfile();
      setMessage('Profile updated');
    } catch (err) {
      setMessage(err.message);
    } finally {
      setSaving(false);
    }
  };

  const inputClass = `w-full px-4 py-2.5 rounded-xl border transition-all focus:outline-none focus:ring-2 focus:ring-purple-500 ${
    isDark ? 'bg-white/5 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-900'
  }`;

  return (
    <div className={`rounded-2xl p-6 ${isDark ? 'bg-white/5 border border-white/10' : 'bg-white border border-gray-200 shadow-sm'}`}>
      <h2 className="text-lg font-bold mb-4">Profile</h2>
      <form onSubmit={handleSave} className="space-y-4">
        <div>
          <label className={`block text-sm font-medium mb-1 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Email</label>
          <input type="email" disabled value={user?.email || ''} className={`${inputClass} opacity-60 cursor-not-allowed`} />
        </div>
        <div>
          <label className={`block text-sm font-medium mb-1 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Username</label>
          <input type="text" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} className={inputClass} />
        </div>
        <div>
          <label className={`block text-sm font-medium mb-1 ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>Display Name</label>
          <input type="text" value={form.displayName} onChange={(e) => setForm({ ...form, displayName: e.target.value })} className={inputClass} />
        </div>
        {message && <p className={`text-sm ${message.includes('updated') ? 'text-green-500' : 'text-red-400'}`}>{message}</p>}
        <button type="submit" disabled={saving} className="px-6 py-2 rounded-xl bg-purple-600 text-white font-medium hover:bg-purple-700 transition-colors disabled:opacity-50">
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </form>
    </div>
  );
}

function SubscriptionSection({ isDark }) {
  const { tier } = useAuth();
  const [usage, setUsage] = useState(null);

  useEffect(() => {
    api.getUsage().then(setUsage).catch(() => {});
  }, []);

  const tierColors = {
    free: 'text-gray-400',
    pro: 'text-purple-400',
    enterprise: 'text-amber-400',
  };

  return (
    <div className={`rounded-2xl p-6 ${isDark ? 'bg-white/5 border border-white/10' : 'bg-white border border-gray-200 shadow-sm'}`}>
      <h2 className="text-lg font-bold mb-4">Subscription</h2>
      <div className="flex items-center gap-3 mb-4">
        <span className={`text-2xl font-bold capitalize ${tierColors[tier] || ''}`}>{tier}</span>
        <span className={`text-xs px-2 py-1 rounded-full ${isDark ? 'bg-white/10' : 'bg-gray-100'}`}>Current Plan</span>
      </div>
      {usage && usage.limits && (
        <div className="grid grid-cols-2 gap-3 mt-4">
          <div className={`p-3 rounded-xl ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Jobs/Day</div>
            <div className="font-bold">{usage.limits.jobsPerDay}</div>
          </div>
          <div className={`p-3 rounded-xl ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Max Rows</div>
            <div className="font-bold">{usage.limits.maxTargetCount?.toLocaleString()}</div>
          </div>
          <div className={`p-3 rounded-xl ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>API Keys</div>
            <div className="font-bold">{usage.limits.apiKeysAllowed}</div>
          </div>
          <div className={`p-3 rounded-xl ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
            <div className={`text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>Retention</div>
            <div className="font-bold">{usage.limits.retentionDays} days</div>
          </div>
        </div>
      )}
    </div>
  );
}

function ApiKeysSection({ isDark }) {
  const [keys, setKeys] = useState([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [createdKey, setCreatedKey] = useState(null);
  const [creating, setCreating] = useState(false);

  const loadKeys = useCallback(async () => {
    try {
      const data = await api.listApiKeys();
      setKeys(data.keys || []);
    } catch { /* ignore load errors */ }
  }, []);

  useEffect(() => { loadKeys(); }, [loadKeys]);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    setCreating(true);
    try {
      const data = await api.createApiKey(newKeyName.trim());
      setCreatedKey(data.key.rawKey);
      setNewKeyName('');
      loadKeys();
    } catch (err) {
      alert(err.message);
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId) => {
    if (!confirm('Revoke this API key?')) return;
    try {
      await api.revokeApiKey(keyId);
      loadKeys();
    } catch (err) {
      alert(err.message);
    }
  };

  const inputClass = `flex-1 px-4 py-2.5 rounded-xl border transition-all focus:outline-none focus:ring-2 focus:ring-purple-500 ${
    isDark ? 'bg-white/5 border-white/10 text-white' : 'bg-white border-gray-200 text-gray-900'
  }`;

  return (
    <div className={`rounded-2xl p-6 ${isDark ? 'bg-white/5 border border-white/10' : 'bg-white border border-gray-200 shadow-sm'}`}>
      <h2 className="text-lg font-bold mb-4">API Keys</h2>

      {createdKey && (
        <div className="mb-4 p-3 rounded-xl bg-green-500/10 border border-green-500/20">
          <p className="text-sm text-green-400 mb-1 font-medium">New API key created! Save it now — it won&apos;t be shown again.</p>
          <code className={`block text-xs p-2 rounded-lg break-all ${isDark ? 'bg-black/30' : 'bg-gray-100'}`}>{createdKey}</code>
          <button onClick={() => setCreatedKey(null)} className="mt-2 text-xs text-green-400 hover:underline">Dismiss</button>
        </div>
      )}

      <form onSubmit={handleCreate} className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="Key name (e.g. production)"
          value={newKeyName}
          onChange={(e) => setNewKeyName(e.target.value)}
          className={inputClass}
        />
        <button type="submit" disabled={creating} className="px-4 py-2 rounded-xl bg-purple-600 text-white text-sm font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 whitespace-nowrap">
          {creating ? '...' : 'Create Key'}
        </button>
      </form>

      <div className="space-y-2">
        {keys.length === 0 && <p className={`text-sm ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>No API keys yet.</p>}
        {keys.map((key) => (
          <div key={key.id} className={`flex items-center justify-between p-3 rounded-xl ${isDark ? 'bg-white/5' : 'bg-gray-50'}`}>
            <div>
              <span className="font-medium text-sm">{key.name}</span>
              <span className={`ml-2 text-xs ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>{key.keyPrefix}</span>
              {!key.isActive && <span className="ml-2 text-xs text-red-400">Revoked</span>}
            </div>
            {key.isActive && (
              <button onClick={() => handleRevoke(key.id)} className="text-xs text-red-400 hover:text-red-300">Revoke</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const { isDark } = useTheme();
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center pt-20">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Sign in Required</h2>
          <p className={isDark ? 'text-gray-400' : 'text-gray-500'}>Please sign in to access settings.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-16 px-4">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">
          <span className="bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
            Settings
          </span>
        </h1>
        <div className="space-y-6">
          <ProfileSection isDark={isDark} />
          <SubscriptionSection isDark={isDark} />
          <ApiKeysSection isDark={isDark} />
        </div>
      </div>
    </div>
  );
}
