/**
 * Badge Component
 * 
 * Small labels for status indicators, counts, and tags.
 * 
 * UX Decision: Badges use subtle background colors with matching
 * text colors to maintain readability while being visually distinct.
 */

const Badge = ({
  children,
  variant = 'default',
  size = 'md',
  dot = false,
  pulsing = false,
  className = '',
  ...props
}) => {
  const variants = {
    default: 'bg-slate-700/50 text-gray-300',
    primary: 'bg-purple-500/15 text-purple-400',
    success: 'bg-emerald-500/15 text-emerald-400',
    warning: 'bg-amber-500/15 text-amber-400',
    error: 'bg-red-500/15 text-red-400',
    info: 'bg-blue-500/15 text-blue-400'
  };

  const dotColors = {
    default: 'bg-gray-400',
    primary: 'bg-purple-400',
    success: 'bg-emerald-400',
    warning: 'bg-amber-400',
    error: 'bg-red-400',
    info: 'bg-blue-400'
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm'
  };

  return (
    <span
      className={`
        inline-flex items-center gap-1.5
        font-medium rounded-full
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {dot && (
        <span
          className={`
            w-1.5 h-1.5 rounded-full
            ${dotColors[variant]}
            ${pulsing ? 'animate-pulse' : ''}
          `}
        />
      )}
      {children}
    </span>
  );
};

export default Badge;
