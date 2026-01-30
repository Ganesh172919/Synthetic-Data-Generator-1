import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, useTheme } from './hooks/useTheme';
import { ToastProvider } from './components/ui/Toast';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import Templates from './pages/Templates';
import DomainBuilder from './pages/DomainBuilder';
import Documentation from './pages/Documentation';
import NotFound from './pages/NotFound';

/**
 * App Content Component
 * 
 * Separated to access theme context
 */
function AppContent() {
  const { isDark } = useTheme();
  
  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDark 
        ? 'bg-gradient-to-br from-slate-950 via-purple-950/50 to-slate-950 text-white' 
        : 'bg-gradient-to-br from-slate-50 via-purple-50/50 to-slate-100 text-slate-900'
    }`}>
      {/* Ambient glow effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-0 left-1/4 w-96 h-96 rounded-full blur-[128px] ${
          isDark ? 'bg-purple-500/10' : 'bg-purple-400/20'
        }`} />
        <div className={`absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-[128px] ${
          isDark ? 'bg-pink-500/10' : 'bg-pink-400/20'
        }`} />
      </div>
      
      <div className="relative z-10">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/templates" element={<Templates />} />
            <Route path="/domain-builder" element={<DomainBuilder />} />
            <Route path="/documentation" element={<Documentation />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </div>
  );
}

/**
 * App Component
 * 
 * Root component with providers for theme, toast notifications,
 * and routing. Uses a gradient background with glass morphism effects.
 */
function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <Router>
          <AppContent />
        </Router>
      </ToastProvider>
    </ThemeProvider>
  );
}

export default App;
