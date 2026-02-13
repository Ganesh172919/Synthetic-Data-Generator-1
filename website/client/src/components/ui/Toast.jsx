import { useState, useEffect, createContext, useContext, useCallback, useMemo } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';

/**
 * Toast Notification System
 * 
 * A lightweight toast notification system for displaying feedback messages.
 *
 * Usage (typical):
 * - Wrap app with <ToastProvider> (done in `src/App.jsx`)
 * - Call `const { toast } = useToast()` inside components
 * - Then `toast.success("Saved!")` / `toast.error("Failed")`, etc.
 *
 * API (informal):
 * - `toast.show(options | messageString)`
 * - `toast.success(message, options)`
 * - `toast.error(message, options)`
 * - `toast.warning(message, options)`
 * - `toast.info(message, options)`
 *
 * Accessibility notes:
 * - Toast items use `role="alert"` so screen readers announce them.
 * - For critical flows, don't rely only on toasts; also show inline errors near inputs.
 */

const ToastContext = createContext(null);

// Toast hook
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((toastData) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { ...toastData, id }]);
    return id;
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Create stable toast object with methods
  const toastObj = useMemo(() => ({
    show: (options) => {
      if (typeof options === 'string') {
        return addToast({ message: options, type: 'info' });
      }
      return addToast(options);
    },
    success: (message, options = {}) => addToast({ message, type: 'success', ...options }),
    error: (message, options = {}) => addToast({ message, type: 'error', ...options }),
    warning: (message, options = {}) => addToast({ message, type: 'warning', ...options }),
    info: (message, options = {}) => addToast({ message, type: 'info', ...options })
  }), [addToast]);

  const contextValue = useMemo(() => ({ toast: toastObj, removeToast }), [toastObj, removeToast]);

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}

function ToastContainer({ toasts, removeToast }) {
  return (
    <div className="fixed bottom-4 right-4 z-[1080] flex flex-col gap-3 max-w-sm w-full pointer-events-none">
      {toasts.map((toastItem, index) => (
        <ToastItem
          key={toastItem.id}
          toastData={toastItem}
          onClose={() => removeToast(toastItem.id)}
          index={index}
        />
      ))}
    </div>
  );
}

function ToastItem({ toastData, onClose, index }) {
  const { type = 'info', title, message, duration = 5000, action } = toastData;
  const [isVisible, setIsVisible] = useState(false);
  const [isLeaving, setIsLeaving] = useState(false);

  const handleClose = useCallback(() => {
    setIsLeaving(true);
    setTimeout(onClose, 200);
  }, [onClose]);

  useEffect(() => {
    const rafId = requestAnimationFrame(() => setIsVisible(true));
    let timer;
    if (duration > 0) {
      timer = setTimeout(handleClose, duration);
    }
    return () => {
      cancelAnimationFrame(rafId);
      if (timer) clearTimeout(timer);
    };
  }, [duration, handleClose]);

  const icons = {
    success: <CheckCircle className="w-5 h-5 text-emerald-400" />,
    error: <AlertCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-amber-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />
  };

  const backgrounds = {
    success: 'bg-emerald-500/10 border-emerald-500/30',
    error: 'bg-red-500/10 border-red-500/30',
    warning: 'bg-amber-500/10 border-amber-500/30',
    info: 'bg-blue-500/10 border-blue-500/30'
  };

  return (
    <div
      className={`
        pointer-events-auto
        bg-slate-800/95 backdrop-blur-lg
        border ${backgrounds[type]}
        rounded-xl p-4
        shadow-xl shadow-black/20
        transform transition-all duration-200 ease-out
        ${isVisible && !isLeaving ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
      `}
      role="alert"
      style={{ '--stagger': index }}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">{icons[type]}</div>
        <div className="flex-1 min-w-0">
          {title && <h4 className="text-sm font-semibold text-white mb-0.5">{title}</h4>}
          <p className="text-sm text-gray-300">{message}</p>
          {action && (
            <button
              onClick={() => { action.onClick(); handleClose(); }}
              className="mt-2 text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors"
            >
              {action.label}
            </button>
          )}
        </div>
        <button
          onClick={handleClose}
          className="flex-shrink-0 text-gray-400 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/5"
          aria-label="Close notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export default ToastItem;
