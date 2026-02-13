/**
 * UI Components Index
 * 
 * Central export for all reusable UI components.
 *
 * Educational note:
 * - This "barrel file" makes imports shorter (`import { Button } from .../ui`) at the cost of
 *   potentially hiding individual file boundaries.
 * - For small apps this is fine; for larger apps consider whether barrel exports affect tree-shaking.
 */

// Core Components
export { default as Button } from './Button';
export { default as Input, Textarea, Select } from './Input';
export { default as Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from './Card';
export { default as Modal, ModalActions } from './Modal';
export { ToastProvider, useToast } from './Toast';
export { default as Badge } from './Badge';
export { default as Progress, CircularProgress } from './Progress';

// Loading Components
export { 
  default as Skeleton,
  SkeletonText,
  SkeletonCard,
  SkeletonStatsCard,
  SkeletonTableRow,
  SkeletonList,
  SkeletonTemplateCard,
  SkeletonProgress
} from './Skeleton';
