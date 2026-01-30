import { forwardRef } from 'react';

/**
 * Input Component
 * 
 * A flexible input component with support for various states,
 * icons, and validation feedback.
 * 
 * UX Decision: Uses subtle focus states with ring effects
 * for clear visual feedback without being distracting.
 */

const Input = forwardRef(({
  label,
  error,
  success,
  helperText,
  leftIcon,
  rightIcon,
  size = 'md',
  className = '',
  wrapperClassName = '',
  ...props
}, ref) => {
  const baseStyles = `
    w-full
    bg-slate-700/50 border border-slate-600
    text-white placeholder-slate-400
    rounded-xl
    transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500
    hover:border-slate-500
    disabled:opacity-50 disabled:cursor-not-allowed
  `;

  const sizes = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-3 text-base',
    lg: 'px-5 py-4 text-lg'
  };

  const stateStyles = error
    ? 'border-red-500 focus:ring-red-500/30 focus:border-red-500'
    : success
    ? 'border-emerald-500 focus:ring-emerald-500/30 focus:border-emerald-500'
    : '';

  const iconPadding = leftIcon ? 'pl-11' : '';
  const rightIconPadding = rightIcon ? 'pr-11' : '';

  return (
    <div className={`space-y-2 ${wrapperClassName}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-400">
          {label}
        </label>
      )}
      <div className="relative">
        {leftIcon && (
          <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">
            {leftIcon}
          </div>
        )}
        <input
          ref={ref}
          className={`
            ${baseStyles}
            ${sizes[size]}
            ${stateStyles}
            ${iconPadding}
            ${rightIconPadding}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400">
            {rightIcon}
          </div>
        )}
      </div>
      {(error || helperText || success) && (
        <p className={`text-sm ${
          error ? 'text-red-400' : 
          success ? 'text-emerald-400' : 
          'text-gray-500'
        }`}>
          {error || success || helperText}
        </p>
      )}
    </div>
  );
});

Input.displayName = 'Input';

/**
 * Textarea Component
 * 
 * A textarea variant of the Input component
 */
export const Textarea = forwardRef(({
  label,
  error,
  success,
  helperText,
  rows = 4,
  className = '',
  wrapperClassName = '',
  ...props
}, ref) => {
  const baseStyles = `
    w-full
    bg-slate-700/50 border border-slate-600
    text-white placeholder-slate-400
    rounded-xl
    transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500
    hover:border-slate-500
    disabled:opacity-50 disabled:cursor-not-allowed
    resize-y min-h-[100px]
    px-4 py-3
  `;

  const stateStyles = error
    ? 'border-red-500 focus:ring-red-500/30 focus:border-red-500'
    : success
    ? 'border-emerald-500 focus:ring-emerald-500/30 focus:border-emerald-500'
    : '';

  return (
    <div className={`space-y-2 ${wrapperClassName}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-400">
          {label}
        </label>
      )}
      <textarea
        ref={ref}
        rows={rows}
        className={`
          ${baseStyles}
          ${stateStyles}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      />
      {(error || helperText || success) && (
        <p className={`text-sm ${
          error ? 'text-red-400' : 
          success ? 'text-emerald-400' : 
          'text-gray-500'
        }`}>
          {error || success || helperText}
        </p>
      )}
    </div>
  );
});

Textarea.displayName = 'Textarea';

/**
 * Select Component
 * 
 * A styled select dropdown
 */
export const Select = forwardRef(({
  label,
  error,
  options = [],
  placeholder = 'Select an option',
  className = '',
  wrapperClassName = '',
  ...props
}, ref) => {
  const baseStyles = `
    w-full
    bg-slate-700/50 border border-slate-600
    text-white
    rounded-xl
    px-4 py-3
    transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500
    hover:border-slate-500
    disabled:opacity-50 disabled:cursor-not-allowed
    appearance-none
    cursor-pointer
    bg-no-repeat bg-right pr-10
  `;

  const stateStyles = error
    ? 'border-red-500 focus:ring-red-500/30 focus:border-red-500'
    : '';

  return (
    <div className={`space-y-2 ${wrapperClassName}`}>
      {label && (
        <label className="block text-sm font-medium text-gray-400">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          ref={ref}
          className={`
            ${baseStyles}
            ${stateStyles}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}
    </div>
  );
});

Select.displayName = 'Select';

export default Input;
