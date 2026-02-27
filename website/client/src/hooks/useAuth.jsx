import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { api, getToken, setToken, clearToken } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [subscription, setSubscription] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadProfile = useCallback(async () => {
    const token = getToken();
    if (!token) {
      setUser(null);
      setSubscription(null);
      setLoading(false);
      return;
    }
    try {
      const data = await api.getProfile();
      setUser(data.user);
      setSubscription(data.subscription);
    } catch {
      clearToken();
      setUser(null);
      setSubscription(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  const login = async (email, password) => {
    const data = await api.login({ email, password });
    setToken(data.token);
    setUser(data.user);
    await loadProfile();
    return data;
  };

  const register = async ({ email, username, password, displayName }) => {
    const data = await api.register({ email, username, password, displayName });
    setToken(data.token);
    setUser(data.user);
    await loadProfile();
    return data;
  };

  const logout = () => {
    clearToken();
    setUser(null);
    setSubscription(null);
  };

  const refreshProfile = () => loadProfile();

  return (
    <AuthContext.Provider value={{
      user,
      subscription,
      loading,
      isAuthenticated: !!user,
      isAdmin: user?.role === 'admin',
      tier: user?.tier || subscription?.tier || 'free',
      login,
      register,
      logout,
      refreshProfile,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}
