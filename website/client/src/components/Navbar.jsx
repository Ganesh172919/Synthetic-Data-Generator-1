import { useState, useEffect, useCallback } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Menu, X, Zap, Database, Sun, Moon, User, LogOut, Settings } from 'lucide-react';
import { useTheme } from '../hooks/useTheme';
import { useAuth } from '../hooks/useAuth';

/**
 * Navbar Component
 * 
 * Fixed navigation with glass morphism effect, theme toggle,
 * and smooth mobile menu transitions.
 * 
 * UX Improvements:
 * - Backdrop blur for depth perception
 * - Active link indicators with smooth transitions
 * - Mobile-first responsive design
 * - Theme toggle with smooth icon transition
 *
 * Accessibility notes:
 * - Buttons include `aria-label` where the visual meaning is icon-only.
 * - `aria-expanded` is used for the mobile menu toggle.
 * - For production, consider adding focus trapping when the mobile menu is open.
 */
const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();
  const { isAuthenticated, logout } = useAuth();

  // Close mobile menu on route change
  const pathname = location.pathname;
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  // Add shadow on scroll
  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 10);
  }, []);
  
  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  const navLinks = [
    { path: '/', label: 'Home' },
    { path: '/dashboard', label: 'Dashboard' },
    { path: '/templates', label: 'Templates' },
    { path: '/domain-builder', label: 'Domain Builder' },
    { path: '/documentation', label: 'Docs' },
    { path: '/pricing', label: 'Pricing' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav 
      className={`
        fixed top-0 w-full z-50
        backdrop-blur-xl
        border-b transition-all duration-300
        ${isDark 
          ? 'bg-slate-900/80' 
          : 'bg-white/80'}
        ${scrolled 
          ? isDark 
            ? 'border-slate-700/50 shadow-lg shadow-black/10' 
            : 'border-slate-200 shadow-lg shadow-slate-200/50'
          : 'border-transparent'}
      `}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link 
            to="/" 
            className="flex items-center space-x-2 group"
          >
            <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25 transition-transform group-hover:scale-105">
              <Database className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              SynthGen
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`
                  relative px-4 py-2 rounded-lg
                  font-medium text-sm
                  transition-all duration-200
                  ${isActive(link.path)
                    ? isDark ? 'text-white' : 'text-slate-900'
                    : isDark 
                      ? 'text-gray-400 hover:text-white hover:bg-white/5' 
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }
                `}
              >
                {link.label}
                {/* Active indicator */}
                {isActive(link.path) && (
                  <span className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-purple-500 to-pink-500 shadow-[0_0_8px_rgba(168,85,247,0.5)]" />
                )}
              </Link>
            ))}
          </div>

          {/* Right side actions */}
          <div className="hidden md:flex items-center space-x-3">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className={`p-2.5 rounded-xl transition-all duration-200 ${
                isDark 
                  ? 'text-gray-400 hover:text-white hover:bg-white/5' 
                  : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
              }`}
              aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            >
              <div className="relative w-5 h-5">
                <Sun className={`absolute inset-0 transition-all duration-300 ${isDark ? 'opacity-100 rotate-0' : 'opacity-0 rotate-90'}`} />
                <Moon className={`absolute inset-0 transition-all duration-300 ${isDark ? 'opacity-0 -rotate-90' : 'opacity-100 rotate-0'}`} />
              </div>
            </button>
            
            {/* CTA / Auth Button */}
            {isAuthenticated ? (
              <div className="flex items-center space-x-2">
                <Link
                  to="/settings"
                  className={`p-2.5 rounded-xl transition-all duration-200 ${
                    isDark
                      ? 'text-gray-400 hover:text-white hover:bg-white/5'
                      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                  aria-label="Settings"
                >
                  <Settings className="w-5 h-5" />
                </Link>
                <button
                  onClick={() => { logout(); navigate('/'); }}
                  className={`p-2.5 rounded-xl transition-all duration-200 ${
                    isDark
                      ? 'text-gray-400 hover:text-white hover:bg-white/5'
                      : 'text-slate-500 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                  aria-label="Sign out"
                >
                  <LogOut className="w-5 h-5" />
                </button>
                <Link
                  to="/dashboard"
                  className="
                    flex items-center space-x-2 px-5 py-2.5
                    bg-gradient-to-r from-purple-500 to-pink-500
                    rounded-xl font-medium text-sm text-white
                    shadow-lg shadow-purple-500/25
                    hover:shadow-xl hover:shadow-purple-500/30
                    hover:-translate-y-0.5
                    active:translate-y-0
                    transition-all duration-200
                  "
                >
                  <Zap className="w-4 h-4" />
                  <span>Dashboard</span>
                </Link>
              </div>
            ) : (
              <div className="flex items-center space-x-2">
                <Link
                  to="/auth"
                  className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                    isDark
                      ? 'text-gray-300 hover:text-white hover:bg-white/5'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  Sign In
                </Link>
                <Link
                  to="/dashboard"
                  className="
                    flex items-center space-x-2 px-5 py-2.5
                    bg-gradient-to-r from-purple-500 to-pink-500
                    rounded-xl font-medium text-sm text-white
                    shadow-lg shadow-purple-500/25
                    hover:shadow-xl hover:shadow-purple-500/30
                    hover:-translate-y-0.5
                    active:translate-y-0
                    transition-all duration-200
                  "
                >
                  <Zap className="w-4 h-4" />
                  <span>Start Generating</span>
                </Link>
              </div>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="flex md:hidden items-center space-x-2">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5"
              aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
            >
              {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5"
              aria-label="Toggle menu"
              aria-expanded={isOpen}
            >
              <div className="relative w-6 h-6">
                <Menu className={`absolute inset-0 transition-all duration-200 ${isOpen ? 'opacity-0 rotate-90' : 'opacity-100 rotate-0'}`} />
                <X className={`absolute inset-0 transition-all duration-200 ${isOpen ? 'opacity-100 rotate-0' : 'opacity-0 -rotate-90'}`} />
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div 
        className={`
          md:hidden overflow-hidden
          transition-all duration-300 ease-out
          ${isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}
        `}
      >
        <div className="px-4 py-4 space-y-1 bg-slate-900/95 backdrop-blur-xl border-t border-slate-700/50">
          {navLinks.map((link, index) => (
            <Link
              key={link.path}
              to={link.path}
              className={`
                block px-4 py-3 rounded-xl
                font-medium transition-all duration-200
                ${isActive(link.path)
                  ? 'bg-purple-500/15 text-white border border-purple-500/20'
                  : 'text-gray-400 hover:text-white hover:bg-white/5'
                }
              `}
              style={{ 
                transitionDelay: isOpen ? `${index * 50}ms` : '0ms',
                transform: isOpen ? 'translateX(0)' : 'translateX(-10px)',
                opacity: isOpen ? 1 : 0
              }}
            >
              {link.label}
            </Link>
          ))}
          <Link
            to="/dashboard"
            className="
              flex items-center justify-center space-x-2
              w-full px-4 py-3 mt-3
              bg-gradient-to-r from-purple-500 to-pink-500
              rounded-xl font-medium
              shadow-lg shadow-purple-500/25
            "
          >
            <Zap className="w-4 h-4" />
            <span>Start Generating</span>
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
