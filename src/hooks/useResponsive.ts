'use client';

/**
 * 📱 useResponsive Hook
 *
 * Premium responsive breakpoint detection with SSR safety
 * Returns current viewport size and device type
 */

import { useState, useEffect } from 'react';

export type DeviceType = 'mobile' | 'tablet' | 'desktop';
export type BreakpointName = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

export interface ResponsiveState {
  // Device type
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  deviceType: DeviceType;

  // Breakpoints
  isXS: boolean;  // < 480px
  isSM: boolean;  // 480-640px
  isMD: boolean;  // 641-768px
  isLG: boolean;  // 769-1024px
  isXL: boolean;  // > 1024px
  breakpoint: BreakpointName;

  // Viewport dimensions
  width: number;
  height: number;

  // Orientation
  isPortrait: boolean;
  isLandscape: boolean;

  // Touch capability
  isTouchDevice: boolean;
}

const BREAKPOINTS = {
  xs: 0,
  sm: 480,
  md: 641,
  lg: 769,
  xl: 1025,
} as const;

function getDeviceType(width: number): DeviceType {
  if (width < BREAKPOINTS.md) return 'mobile';
  if (width < BREAKPOINTS.xl) return 'tablet';
  return 'desktop';
}

function getBreakpoint(width: number): BreakpointName {
  if (width < BREAKPOINTS.sm) return 'xs';
  if (width < BREAKPOINTS.md) return 'sm';
  if (width < BREAKPOINTS.lg) return 'md';
  if (width < BREAKPOINTS.xl) return 'lg';
  return 'xl';
}

function isTouchDevice(): boolean {
  if (typeof window === 'undefined') return false;
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
}

export function useResponsive(): ResponsiveState {
  const [state, setState] = useState<ResponsiveState>({
    isMobile: false,
    isTablet: false,
    isDesktop: true,
    deviceType: 'desktop',
    isXS: false,
    isSM: false,
    isMD: false,
    isLG: false,
    isXL: true,
    breakpoint: 'xl',
    width: 1920,
    height: 1080,
    isPortrait: false,
    isLandscape: true,
    isTouchDevice: false,
  });

  useEffect(() => {
    const updateState = () => {
      if (typeof window === 'undefined') return;

      const width = window.innerWidth;
      const height = window.innerHeight;
      const deviceType = getDeviceType(width);
      const breakpoint = getBreakpoint(width);

      setState({
        // Device type
        isMobile: deviceType === 'mobile',
        isTablet: deviceType === 'tablet',
        isDesktop: deviceType === 'desktop',
        deviceType,

        // Breakpoints
        isXS: breakpoint === 'xs',
        isSM: breakpoint === 'sm',
        isMD: breakpoint === 'md',
        isLG: breakpoint === 'lg',
        isXL: breakpoint === 'xl',
        breakpoint,

        // Viewport dimensions
        width,
        height,

        // Orientation
        isPortrait: height > width,
        isLandscape: width >= height,

        // Touch capability
        isTouchDevice: isTouchDevice(),
      });
    };

    // Initial update
    updateState();

    // Listen for resize
    window.addEventListener('resize', updateState);
    window.addEventListener('orientationchange', updateState);

    return () => {
      window.removeEventListener('resize', updateState);
      window.removeEventListener('orientationchange', updateState);
    };
  }, []);

  return state;
}

/**
 * Helper hook for specific breakpoint queries
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const media = window.matchMedia(query);
    setMatches(media.matches);

    const listener = (e: MediaQueryListEvent) => setMatches(e.matches);
    media.addEventListener('change', listener);

    return () => media.removeEventListener('change', listener);
  }, [query]);

  return matches;
}

/**
 * Responsive value selector
 * Returns different values based on current breakpoint
 */
export function useResponsiveValue<T>(values: {
  xs?: T;
  sm?: T;
  md?: T;
  lg?: T;
  xl?: T;
  default: T;
}): T {
  const { breakpoint } = useResponsive();

  return values[breakpoint] ?? values.default;
}

/**
 * Export breakpoint constants for external use
 */
export { BREAKPOINTS };
