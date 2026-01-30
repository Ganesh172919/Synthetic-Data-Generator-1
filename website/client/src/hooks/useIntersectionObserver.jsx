import { useEffect, useRef, useState } from 'react';

/**
 * useIntersectionObserver Hook
 * 
 * A custom hook for detecting when elements enter the viewport.
 * Used for scroll-triggered animations and lazy loading.
 * 
 * UX Decision: Animations are triggered only when elements become
 * visible, improving perceived performance and creating engagement.
 */

export function useIntersectionObserver({
  threshold = 0.1,
  rootMargin = '0px',
  triggerOnce = true,
  enabled = true
} = {}) {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasTriggered, setHasTriggered] = useState(false);
  const targetRef = useRef(null);

  useEffect(() => {
    if (!enabled) return;
    
    const target = targetRef.current;
    if (!target) return;

    // Skip if already triggered and triggerOnce is true
    if (triggerOnce && hasTriggered) return;

    // Check if IntersectionObserver is supported
    if (!('IntersectionObserver' in window)) {
      // Fallback: make visible immediately
      setIsIntersecting(true);
      setHasTriggered(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        const isVisible = entry.isIntersecting;
        setIsIntersecting(isVisible);
        
        if (isVisible && triggerOnce) {
          setHasTriggered(true);
          observer.unobserve(target);
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(target);

    return () => {
      observer.unobserve(target);
    };
  }, [threshold, rootMargin, triggerOnce, enabled, hasTriggered]);

  return { ref: targetRef, isIntersecting, hasTriggered };
}

/**
 * useAnimateOnScroll Hook
 * 
 * A simplified hook specifically for scroll-triggered animations.
 * Returns class names to apply based on visibility state.
 */
export function useAnimateOnScroll({
  animation = 'fade-up',
  delay = 0,
  threshold = 0.1,
  enabled = true
} = {}) {
  const { ref, isIntersecting } = useIntersectionObserver({
    threshold,
    triggerOnce: true,
    enabled
  });

  const animations = {
    'fade-up': {
      hidden: 'opacity-0 translate-y-8',
      visible: 'opacity-100 translate-y-0'
    },
    'fade-down': {
      hidden: 'opacity-0 -translate-y-8',
      visible: 'opacity-100 translate-y-0'
    },
    'fade-left': {
      hidden: 'opacity-0 translate-x-8',
      visible: 'opacity-100 translate-x-0'
    },
    'fade-right': {
      hidden: 'opacity-0 -translate-x-8',
      visible: 'opacity-100 translate-x-0'
    },
    'scale': {
      hidden: 'opacity-0 scale-95',
      visible: 'opacity-100 scale-100'
    },
    'fade': {
      hidden: 'opacity-0',
      visible: 'opacity-100'
    }
  };

  const selectedAnimation = animations[animation] || animations['fade-up'];
  
  const className = `
    transition-all duration-500 ease-out
    ${isIntersecting ? selectedAnimation.visible : selectedAnimation.hidden}
  `.trim();

  const style = delay > 0 ? { transitionDelay: `${delay}ms` } : {};

  return { ref, className, style, isVisible: isIntersecting };
}

/**
 * AnimatedSection Component
 * 
 * A wrapper component that applies scroll-triggered animations.
 */
export function AnimatedSection({
  children,
  animation = 'fade-up',
  delay = 0,
  threshold = 0.1,
  className = '',
  as = 'div',
  enabled = true,
  ...props
}) {
  const { ref, className: animationClass, style } = useAnimateOnScroll({
    animation,
    delay,
    threshold,
    enabled
  });

  // Check for reduced motion preference
  const prefersReducedMotion = typeof window !== 'undefined' 
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches 
    : false;
  
  const Component = as;

  if (prefersReducedMotion || !enabled) {
    return (
      <Component className={className} {...props}>
        {children}
      </Component>
    );
  }

  return (
    <Component
      ref={ref}
      className={`${animationClass} ${className}`}
      style={style}
      {...props}
    >
      {children}
    </Component>
  );
}

export default useIntersectionObserver;
