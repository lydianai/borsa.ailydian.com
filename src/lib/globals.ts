/**
 * GLOBAL VARIABLES & UTILITIES
 * Shared across the application
 */

// Environment check
export const isLocalhost = typeof window !== 'undefined' && 
  (window.location.hostname === 'localhost' || 
   window.location.hostname === '127.0.0.1' ||
   window.location.hostname === '');

// Export for global access
if (typeof window !== 'undefined') {
  (window as any).isLocalhost = isLocalhost;
}

export default isLocalhost;
