/**
 * Dynamic Import Utilities
 *
 * White-hat compliance: Code splitting improves performance and reduces
 * initial bundle size for better user experience
 */

import dynamic from 'next/dynamic';

/**
 * Lazy load heavy components with loading fallback
 */

// Chart components (heavy libraries)
export const DynamicApexChart = dynamic(
  () => import('react-apexcharts').then((mod) => mod.default),
  {
    loading: () => (
      <div className="w-full h-64 bg-gray-100 animate-pulse rounded flex items-center justify-center">
        <span className="text-gray-500">Loading chart...</span>
      </div>
    ),
    ssr: false, // Charts should only render on client
  }
);

// Heavy dashboard components
export const DynamicTradingView = dynamic(
  () => import('@/components/Chart/TradingViewChart'),
  {
    loading: () => (
      <div className="w-full h-96 bg-gray-100 animate-pulse rounded flex items-center justify-center">
        <span className="text-gray-500">Loading TradingView...</span>
      </div>
    ),
    ssr: false,
  }
);

// Settings panels (not critical for initial load)
export const DynamicSecuritySettings = dynamic(
  () => import('@/components/settings/SecuritySettings'),
  {
    loading: () => (
      <div className="p-4 bg-gray-100 animate-pulse rounded">
        <div className="h-8 bg-gray-200 rounded mb-4 w-1/3"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
      </div>
    ),
  }
);

export const DynamicNotificationSettings = dynamic(
  () => import('@/components/settings/NotificationChannels'),
  {
    loading: () => (
      <div className="p-4 bg-gray-100 animate-pulse rounded">
        <div className="h-8 bg-gray-200 rounded mb-4 w-1/3"></div>
        <div className="h-32 bg-gray-200 rounded"></div>
      </div>
    ),
  }
);

// Modal components (lazy load until needed)
// export const DynamicSignalDetailModal = dynamic(
//   () => import('@/components/modals/SignalDetailModal'),
//   {
//     loading: () => null, // Modals can have minimal loading state
//   }
// );

/**
 * Preload critical components
 *
 * Call this on user interaction (hover, focus) to preload
 * before actual navigation
 */
export function preloadComponent(componentName: keyof typeof ComponentPreloaders) {
  const preloader = ComponentPreloaders[componentName];
  if (preloader) {
    preloader();
  }
}

const ComponentPreloaders = {
  tradingView: () => import('@/components/Chart/TradingViewChart'),
  securitySettings: () => import('@/components/settings/SecuritySettings'),
  notificationSettings: () => import('@/components/settings/NotificationChannels'),
} as const;

/**
 * Create loading skeleton for consistent UX
 */
export function createLoadingSkeleton(height: string = 'h-64') {
  return (
    <div className={`w-full ${height} bg-gray-100 animate-pulse rounded`}>
      <div className="p-4 space-y-4">
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
      </div>
    </div>
  );
}

/**
 * Bundle size optimization helpers
 */

// Only import heavy libraries when needed
export async function loadHeavyLibrary(libraryName: 'tensorflow' | 'chart') {
  switch (libraryName) {
    case 'tensorflow':
      // Only load TensorFlow.js when AI features are used
      return import('@tensorflow/tfjs' as any);

    case 'chart':
      // Only load ApexCharts when chart is rendered
      return import('apexcharts');

    default:
      throw new Error(`Unknown library: ${libraryName}`);
  }
}

/**
 * Image optimization helpers
 */
export const IMAGE_SIZES = {
  thumbnail: { width: 64, height: 64 },
  small: { width: 128, height: 128 },
  medium: { width: 256, height: 256 },
  large: { width: 512, height: 512 },
  xlarge: { width: 1024, height: 1024 },
} as const;

export function getOptimizedImageProps(
  src: string,
  size: keyof typeof IMAGE_SIZES = 'medium'
) {
  const { width, height } = IMAGE_SIZES[size];

  return {
    src,
    width,
    height,
    quality: 85, // Good balance between quality and file size
    loading: 'lazy' as const, // Native lazy loading
    placeholder: 'blur' as const, // Show blur placeholder while loading
  };
}
