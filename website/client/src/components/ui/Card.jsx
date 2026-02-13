/**
 * Card Component
 * 
 * A flexible card component for displaying content in a contained format.
 * Supports various styles including glass morphism effects.
 * 
 * UX Decision: Cards use subtle hover states to indicate interactivity
 * and provide visual depth through shadows and border effects.
 *
 * Props (Card):
 * - `variant`: default | glass | elevated | outlined | gradient | featured
 * - `padding`: none | sm | md | lg | xl
 * - `hover`: enable hover affordance even if not clickable
 * - `onClick`: when provided, card becomes keyboard-focusable (tabIndex) and gets role="button"
 *
 * Usage example:
 *   <Card variant="glass" hover>
 *     <CardHeader>...</CardHeader>
 *     <CardContent>...</CardContent>
 *   </Card>
 *
 * Accessibility notes:
 * - If you use `onClick`, consider also adding keyboard handlers (Enter/Space) for full button parity.
 */

const Card = ({
  children,
  variant = 'default',
  hover = false,
  padding = 'md',
  className = '',
  onClick,
  ...props
}) => {
  const baseStyles = `
    rounded-2xl
    transition-all duration-300
  `;

  const variants = {
    default: `
      bg-slate-800/50 border border-slate-700/50
    `,
    glass: `
      bg-slate-800/30 backdrop-blur-xl
      border border-white/10
    `,
    elevated: `
      bg-slate-800 border border-slate-700
      shadow-xl shadow-black/20
    `,
    outlined: `
      bg-transparent border border-slate-700
    `,
    gradient: `
      bg-gradient-to-br from-purple-900/50 to-pink-900/50
      border border-purple-500/20
    `,
    featured: `
      bg-slate-800/50 border border-purple-500/30
      bg-gradient-to-br from-purple-500/5 to-pink-500/5
    `
  };

  const paddings = {
    none: 'p-0',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
    xl: 'p-10'
  };

  const hoverStyles = hover || onClick
    ? 'hover:border-purple-500/50 hover:-translate-y-1 hover:shadow-lg hover:shadow-purple-500/10 cursor-pointer'
    : '';

  return (
    <div
      className={`
        ${baseStyles}
        ${variants[variant]}
        ${paddings[padding]}
        ${hoverStyles}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      {...props}
    >
      {children}
    </div>
  );
};

/**
 * CardHeader Component
 */
export const CardHeader = ({
  children,
  className = '',
  ...props
}) => (
  <div
    className={`mb-4 ${className}`}
    {...props}
  >
    {children}
  </div>
);

/**
 * CardTitle Component
 */
export const CardTitle = ({
  children,
  className = '',
  ...props
}) => (
  <h3
    className={`text-xl font-semibold text-white ${className}`}
    {...props}
  >
    {children}
  </h3>
);

/**
 * CardDescription Component
 */
export const CardDescription = ({
  children,
  className = '',
  ...props
}) => (
  <p
    className={`text-sm text-gray-400 mt-1 ${className}`}
    {...props}
  >
    {children}
  </p>
);

/**
 * CardContent Component
 */
export const CardContent = ({
  children,
  className = '',
  ...props
}) => (
  <div className={className} {...props}>
    {children}
  </div>
);

/**
 * CardFooter Component
 */
export const CardFooter = ({
  children,
  className = '',
  divider = false,
  ...props
}) => (
  <div
    className={`
      mt-4 pt-4
      ${divider ? 'border-t border-slate-700/50' : ''}
      ${className}
    `.trim().replace(/\s+/g, ' ')}
    {...props}
  >
    {children}
  </div>
);

export default Card;
