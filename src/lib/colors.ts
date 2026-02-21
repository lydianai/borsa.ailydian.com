/**
 * 🎨 UNIFIED COLOR SYSTEM
 * Modern premium color palette for consistent UI/UX
 *
 * Usage:
 * import { COLORS } from '@/lib/colors';
 * style={{ color: COLORS.success }}
 */

export const COLORS = {
  // Signal Colors
  success: '#10B981',    // Yeşil - BUY / Success / Positive
  warning: '#F59E0B',    // Amber - WAIT / Warning / Neutral
  danger: '#EF4444',     // Kırmızı - SELL / Error / Negative

  // Brand Colors
  info: '#3B82F6',       // Mavi - Info / Analysis
  premium: '#8B5CF6',    // Mor - Premium / Special
  cyan: '#00D4FF',       // Cyan - Nirvana / Highlight

  // Neutral Colors
  gray: {
    100: '#F3F4F6',
    200: '#E5E7EB',
    300: '#D1D5DB',
    400: '#9CA3AF',
    500: '#6B7280',
    600: '#4B5563',
    700: '#374151',
    800: '#1F2937',
    900: '#111827',
  },

  // Background Colors
  bg: {
    primary: '#000000',
    secondary: '#0a0a0a',
    card: '#111111',
    hover: '#1a1a1a',
  },

  // Text Colors
  text: {
    primary: '#FFFFFF',
    secondary: '#9CA3AF',
    muted: '#6B7280',
  },

  // Border Colors
  border: {
    default: '#222222',
    hover: '#333333',
    active: '#444444',
  },
} as const;

// Signal color helper
export const getSignalColor = (signal: 'BUY' | 'SELL' | 'WAIT' | 'HOLD' | 'NEUTRAL' | string): string => {
  const normalized = signal.toUpperCase();

  if (normalized === 'BUY') return COLORS.success;
  if (normalized === 'SELL') return COLORS.danger;
  if (normalized === 'WAIT' || normalized === 'HOLD') return COLORS.warning;
  if (normalized === 'NEUTRAL') return COLORS.gray[500];

  return COLORS.gray[500];
};

// Percentage color helper
export const getChangeColor = (change: number): string => {
  if (change > 0) return COLORS.success;
  if (change < -3) return COLORS.danger;
  return COLORS.warning;
};
