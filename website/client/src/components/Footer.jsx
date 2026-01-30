import { Database, Github, Twitter, Mail, Heart } from 'lucide-react';
import { Link } from 'react-router-dom';

/**
 * Footer Component
 * 
 * Site-wide footer with navigation links, social media,
 * and branding elements.
 * 
 * UX Improvements:
 * - Clear visual hierarchy
 * - Hover states for links
 * - Responsive layout
 * - Consistent spacing
 */
const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-slate-900/50 border-t border-slate-700/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 lg:gap-12">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center space-x-2 mb-4 group">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20 group-hover:shadow-purple-500/30 transition-shadow">
                <Database className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                SynthGen
              </span>
            </Link>
            <p className="text-gray-400 max-w-md mb-6 leading-relaxed">
              Enterprise-grade AI dataset generation platform. Generate high-quality 
              synthetic datasets at unprecedented speed using state-of-the-art LLMs.
            </p>
            <div className="flex space-x-4">
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-800/50 text-gray-400 hover:text-white hover:bg-slate-700/50 transition-all"
                aria-label="GitHub"
              >
                <Github className="w-5 h-5" />
              </a>
              <a 
                href="https://twitter.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-800/50 text-gray-400 hover:text-white hover:bg-slate-700/50 transition-all"
                aria-label="Twitter"
              >
                <Twitter className="w-5 h-5" />
              </a>
              <a 
                href="mailto:contact@synthgen.ai"
                className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-800/50 text-gray-400 hover:text-white hover:bg-slate-700/50 transition-all"
                aria-label="Email"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-white font-semibold mb-4">Product</h3>
            <ul className="space-y-3">
              {[
                { to: '/dashboard', label: 'Dashboard' },
                { to: '/templates', label: 'Templates' },
                { to: '/domain-builder', label: 'Domain Builder' },
                { to: '/documentation', label: 'Documentation' }
              ].map((link) => (
                <li key={link.to}>
                  <Link 
                    to={link.to} 
                    className="text-gray-400 hover:text-white transition-colors inline-block"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-white font-semibold mb-4">Resources</h3>
            <ul className="space-y-3">
              {[
                { href: '#', label: 'API Reference' },
                { href: 'https://github.com', label: 'GitHub' },
                { href: '#', label: 'Community' },
                { href: '#', label: 'Blog' }
              ].map((link) => (
                <li key={link.label}>
                  <a 
                    href={link.href}
                    target={link.href.startsWith('http') ? '_blank' : undefined}
                    rel={link.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                    className="text-gray-400 hover:text-white transition-colors inline-block"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-700/30 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-400 text-sm flex items-center gap-1">
            © {currentYear} Synthetic Data Generator. Built with 
            <Heart className="w-4 h-4 text-pink-500 inline" />
            MIT License.
          </p>
          <div className="flex space-x-6">
            <a 
              href="#" 
              className="text-gray-400 hover:text-white text-sm transition-colors"
            >
              Privacy Policy
            </a>
            <a 
              href="#" 
              className="text-gray-400 hover:text-white text-sm transition-colors"
            >
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
