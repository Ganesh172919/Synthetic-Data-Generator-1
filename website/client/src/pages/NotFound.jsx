import { Link } from 'react-router-dom';
import { Home, ArrowLeft, Search } from 'lucide-react';
import Button from '../components/ui/Button';

/**
 * NotFound Component
 * 
 * 404 Page with helpful navigation options.
 *
 * Educational notes:
 * - This page is rendered for the catch-all route (`path="*"`) in `src/App.jsx`.
 * - In SPAs, 404 handling is split between client routing and server routing.
 *   For production hosting, ensure your server rewrites unknown routes to `index.html`
 *   so React Router can handle them.
 */
const NotFound = () => {
  return (
    <div className="pt-20 pb-12 min-h-screen flex items-center justify-center">
      <div className="max-w-xl mx-auto px-4 text-center">
        {/* Large 404 */}
        <div className="relative mb-8">
          <h1 className="text-[12rem] font-extrabold leading-none text-transparent bg-clip-text bg-gradient-to-r from-purple-500/20 to-pink-500/20 select-none">
            404
          </h1>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-8xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              404
            </div>
          </div>
        </div>

        {/* Message */}
        <h2 className="text-2xl font-bold mb-4">Page Not Found</h2>
        <p className="text-gray-400 mb-8 max-w-md mx-auto">
          Oops! The page you're looking for doesn't exist or has been moved. 
          Let's get you back on track.
        </p>

        {/* Actions */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link to="/">
            <Button leftIcon={<Home className="w-5 h-5" />}>
              Back to Home
            </Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="secondary" leftIcon={<ArrowLeft className="w-5 h-5" />}>
              Go to Dashboard
            </Button>
          </Link>
        </div>

        {/* Quick Links */}
        <div className="mt-12 pt-8 border-t border-slate-700/50">
          <p className="text-sm text-gray-500 mb-4">Quick Links</p>
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <Link to="/templates" className="text-purple-400 hover:text-purple-300 transition-colors">
              Templates
            </Link>
            <Link to="/domain-builder" className="text-purple-400 hover:text-purple-300 transition-colors">
              Domain Builder
            </Link>
            <Link to="/documentation" className="text-purple-400 hover:text-purple-300 transition-colors">
              Documentation
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
