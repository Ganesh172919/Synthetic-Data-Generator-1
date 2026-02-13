import { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react';

/**
 * Theme Context
 * 
 * Provides dark/light mode support with system preference detection
 * and persistence via localStorage.
 * 
 * UX Decision: Respects user's system preference by default while
 * allowing manual override. Changes persist across sessions.
 *
 * Educational notes:
 * - We set a `data-theme` attribute on `<html>` so CSS variables (design tokens) can react to theme.
 * - `mounted` prevents a flash of incorrect theme during initial hydration.
 * - `matchMedia('(prefers-color-scheme: ...)')` lets us respond to OS-level theme changes.
 * - localStorage may be unavailable in some environments (privacy mode / blocked storage);
 *   this implementation assumes a typical browser environment.
 */

const ThemeContext = createContext(null);

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// Initialize theme from localStorage or system preference
function getInitialTheme() {
  if (typeof window === 'undefined') return 'dark';
  
  const storedTheme = localStorage.getItem('theme');
  if (storedTheme) return storedTheme;
  
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(getInitialTheme);
  const [mounted, setMounted] = useState(false);

  // Mark as mounted after hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Apply theme to document
  useEffect(() => {
    if (!mounted) return;
    
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update meta theme-color for mobile browsers
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (metaTheme) {
      metaTheme.setAttribute('content', theme === 'dark' ? '#0f172a' : '#ffffff');
    }
  }, [theme, mounted]);

  // Listen for system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e) => {
      const storedTheme = localStorage.getItem('theme');
      // Only auto-switch if user hasn't set a preference
      if (!storedTheme) {
        setThemeState(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const setTheme = useCallback((newTheme) => {
    setThemeState(newTheme);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const isDark = theme === 'dark';

  const value = useMemo(() => ({
    theme,
    setTheme,
    toggleTheme,
    isDark
  }), [theme, setTheme, toggleTheme, isDark]);

  // Prevent flash of wrong theme
  if (!mounted) {
    return null;
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export default ThemeProvider;
