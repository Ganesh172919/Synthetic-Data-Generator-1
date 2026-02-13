/**
 * Skeleton Loading Components
 * 
 * Skeleton loaders provide visual feedback during loading states,
 * improving perceived performance by showing the expected layout.
 * 
 * UX Decision: Skeletons use a subtle shimmer animation to indicate
 * loading without being distracting. They match the size and shape
 * of the content they're replacing.
 *
 * Accessibility notes:
 * - Skeletons are marked `aria-hidden="true"` because they are purely decorative.
 * - Pair skeletons with meaningful text updates (e.g., "Loading templates...") when appropriate.
 *
 * Edge cases:
 * - Avoid long-running skeleton states without progress or explanation; users may think the app is stuck.
 */

/**
 * Base Skeleton Component
 */
const Skeleton = ({
  className = '',
  variant = 'rectangular',
  width,
  height,
  ...props
}) => {
  const variants = {
    rectangular: 'rounded-lg',
    circular: 'rounded-full',
    text: 'rounded h-4',
    title: 'rounded h-6 w-3/4',
    avatar: 'rounded-full w-10 h-10',
    button: 'rounded-xl h-10 w-24',
    card: 'rounded-2xl h-48'
  };

  const style = {
    width: width,
    height: height
  };

  return (
    <div
      className={`
        bg-slate-700/50 
        animate-pulse
        ${variants[variant]}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      style={style}
      aria-hidden="true"
      {...props}
    />
  );
};

/**
 * Skeleton Text Lines
 */
export const SkeletonText = ({
  lines = 3,
  className = ''
}) => (
  <div className={`space-y-3 ${className}`}>
    {Array.from({ length: lines }).map((_, index) => (
      <Skeleton
        key={index}
        variant="text"
        className={index === lines - 1 ? 'w-4/5' : 'w-full'}
      />
    ))}
  </div>
);

/**
 * Skeleton Card
 */
export const SkeletonCard = ({ className = '' }) => (
  <div className={`bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 ${className}`}>
    <div className="flex items-start justify-between mb-4">
      <Skeleton variant="avatar" />
      <Skeleton className="w-12 h-4 rounded" />
    </div>
    <Skeleton variant="title" className="mb-4" />
    <SkeletonText lines={2} className="mb-4" />
    <div className="flex gap-2">
      <Skeleton className="h-6 w-16 rounded-full" />
      <Skeleton className="h-6 w-16 rounded-full" />
      <Skeleton className="h-6 w-16 rounded-full" />
    </div>
  </div>
);

/**
 * Skeleton Stats Card
 */
export const SkeletonStatsCard = ({ className = '' }) => (
  <div className={`bg-slate-800/50 rounded-xl p-6 border border-slate-700/50 ${className}`}>
    <div className="flex items-center justify-between mb-4">
      <Skeleton className="h-4 w-20 rounded" />
      <Skeleton variant="circular" className="w-5 h-5" />
    </div>
    <Skeleton className="h-8 w-24 rounded mb-1" />
    <Skeleton className="h-3 w-16 rounded" />
  </div>
);

/**
 * Skeleton Table Row
 */
export const SkeletonTableRow = ({
  columns = 4,
  className = ''
}) => (
  <div className={`flex items-center gap-4 p-4 ${className}`}>
    {Array.from({ length: columns }).map((_, index) => (
      <Skeleton
        key={index}
        className={`h-4 rounded ${
          index === 0 ? 'w-32' : 
          index === columns - 1 ? 'w-20' : 
          'flex-1'
        }`}
      />
    ))}
  </div>
);

/**
 * Skeleton List
 */
export const SkeletonList = ({
  items = 5,
  className = ''
}) => (
  <div className={`space-y-4 ${className}`}>
    {Array.from({ length: items }).map((_, index) => (
      <div key={index} className="flex items-center gap-4">
        <Skeleton variant="avatar" className="w-10 h-10" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-3/4 rounded" />
          <Skeleton className="h-3 w-1/2 rounded" />
        </div>
      </div>
    ))}
  </div>
);

/**
 * Skeleton Template Card (for Templates page)
 */
export const SkeletonTemplateCard = ({ className = '' }) => (
  <div className={`bg-slate-800/50 rounded-xl p-6 border border-slate-700/50 animate-pulse ${className}`}>
    <div className="flex items-start justify-between mb-4">
      <Skeleton className="w-10 h-10 rounded-lg" />
      <Skeleton className="w-12 h-4 rounded" />
    </div>
    <Skeleton className="h-5 w-3/4 rounded mb-2" />
    <Skeleton className="h-4 w-full rounded mb-1" />
    <Skeleton className="h-4 w-2/3 rounded mb-4" />
    <div className="flex gap-2 mb-4">
      <Skeleton className="h-6 w-16 rounded" />
      <Skeleton className="h-6 w-16 rounded" />
      <Skeleton className="h-6 w-16 rounded" />
    </div>
    <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
      <Skeleton className="h-4 w-16 rounded" />
      <Skeleton className="h-4 w-24 rounded" />
    </div>
  </div>
);

/**
 * Skeleton Progress Section
 */
export const SkeletonProgress = ({ className = '' }) => (
  <div className={`bg-slate-800/50 rounded-xl p-6 border border-slate-700/50 ${className}`}>
    <div className="flex items-center justify-between mb-4">
      <Skeleton className="h-6 w-40 rounded" />
      <Skeleton className="h-6 w-16 rounded-full" />
    </div>
    <div className="mb-4">
      <div className="flex justify-between mb-2">
        <Skeleton className="h-4 w-16 rounded" />
        <Skeleton className="h-4 w-12 rounded" />
      </div>
      <Skeleton className="h-4 w-full rounded-full" />
    </div>
    <div className="flex gap-4">
      <Skeleton className="h-10 w-36 rounded-lg" />
      <Skeleton className="h-10 w-20 rounded-lg" />
      <Skeleton className="h-10 w-20 rounded-lg" />
    </div>
  </div>
);

export default Skeleton;
