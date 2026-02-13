import { forwardRef } from 'react';

/**
 * Button Component
 * 
 * A versatile button component following the design system.
 * Supports multiple variants, sizes, and states.
 * 
 * UX Decision: Buttons use visual feedback with transforms and shadows
 * to provide instant feedback on hover/click, reducing perceived latency.
 *
 * Props (informal API):
 * - `variant`: primary | secondary | ghost | danger | success | outline
 * - `size`: xs | sm | md | lg | xl | icon | icon-sm | icon-lg
 * - `isLoading`: shows spinner and disables the button
 * - `disabled`: disables the button (also disabled when isLoading=true)
 * - `leftIcon` / `rightIcon`: optional icon elements
 *
 * Usage example:
 *   <Button variant="primary" isLoading={saving} leftIcon={<Save />}>Save</Button>
 *
 * Accessibility notes:
 * - Provide an accessible name (button text or `aria-label` for icon-only buttons).
 * - Loading states should still keep the purpose clear (we show "Loading..." for string children).
 */

const Button = forwardRef(({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  disabled = false,
  leftIcon,
  rightIcon,
  className = '',
  ...props
}, ref) => {
  const baseStyles = `
    inline-flex items-center justify-center gap-2
    font-medium rounded-xl
    transition-all duration-200 ease-out
    focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900
    disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
    active:scale-[0.98]
  `;

  const variants = {
    primary: `
      bg-gradient-to-r from-purple-500 to-pink-500
      text-white font-semibold
      shadow-lg shadow-purple-500/25
      hover:shadow-xl hover:shadow-purple-500/30 hover:-translate-y-0.5
    `,
    secondary: `
      bg-slate-800 border border-slate-600
      text-white
      hover:bg-slate-700 hover:border-purple-500/50
    `,
    ghost: `
      bg-transparent
      text-gray-300
      hover:bg-white/5 hover:text-white
    `,
    danger: `
      bg-red-500
      text-white
      hover:bg-red-600
    `,
    success: `
      bg-emerald-500
      text-white
      hover:bg-emerald-600
    `,
    outline: `
      bg-transparent border border-purple-500/50
      text-purple-400
      hover:bg-purple-500/10 hover:border-purple-500
    `
  };

  const sizes = {
    xs: 'px-3 py-1.5 text-xs',
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-sm',
    lg: 'px-8 py-4 text-base',
    xl: 'px-10 py-5 text-lg',
    icon: 'p-3',
    'icon-sm': 'p-2',
    'icon-lg': 'p-4'
  };

  const isDisabled = disabled || isLoading;

  return (
    <button
      ref={ref}
      disabled={isDisabled}
      className={`
        ${baseStyles}
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {isLoading ? (
        <>
          <Spinner size={size === 'xs' || size === 'sm' ? 'sm' : 'md'} />
          {typeof children === 'string' ? <span>Loading...</span> : children}
        </>
      ) : (
        <>
          {leftIcon && <span className="flex-shrink-0">{leftIcon}</span>}
          {children}
          {rightIcon && <span className="flex-shrink-0">{rightIcon}</span>}
        </>
      )}
    </button>
  );
});

Button.displayName = 'Button';

/**
 * Spinner Component
 * For loading states inside buttons
 */
const Spinner = ({ size = 'md' }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  return (
    <svg
      className={`animate-spin ${sizes[size]}`}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
};

export default Button;
