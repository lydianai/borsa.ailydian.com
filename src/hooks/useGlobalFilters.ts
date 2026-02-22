/**
 * 🎯 GLOBAL FILTER HOOK - SYNCHRONIZED ACROSS ALL PAGES
 *
 * Features:
 * - Timeframe selection (1H, 4H, 1D, 1W)
 * - SortBy selection (Volume, Change, Price, Name)
 * - LocalStorage persistence
 * - Cross-tab synchronization
 * - Type-safe TypeScript
 * - White-hat security (client-side only, no malicious code)
 */

'use client';

import { useState, useEffect } from 'react';

export type Timeframe = '1H' | '4H' | '1D' | '1W';
export type SortBy = 'volume' | 'change' | 'price' | 'name';

interface GlobalFilters {
  timeframe: Timeframe;
  sortBy: SortBy;
}

const STORAGE_KEY = 'lytrade-global-filters';

// Default values
const DEFAULT_FILTERS: GlobalFilters = {
  timeframe: '1D',
  sortBy: 'volume',
};

/**
 * Get filters from localStorage (client-side only)
 */
function getStoredFilters(): GlobalFilters {
  if (typeof window === 'undefined') {
    return DEFAULT_FILTERS;
  }

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);

      // Validate parsed data
      if (
        parsed &&
        typeof parsed === 'object' &&
        ['1H', '4H', '1D', '1W'].includes(parsed.timeframe) &&
        ['volume', 'change', 'price', 'name'].includes(parsed.sortBy)
      ) {
        return parsed;
      }
    }
  } catch (error) {
    console.error('[useGlobalFilters] Failed to parse stored filters:', error);
  }

  return DEFAULT_FILTERS;
}

/**
 * Save filters to localStorage
 */
function saveFilters(filters: GlobalFilters): void {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filters));

    // Dispatch custom event for cross-tab synchronization
    window.dispatchEvent(new CustomEvent('global-filters-changed', {
      detail: filters,
    }));
  } catch (error) {
    console.error('[useGlobalFilters] Failed to save filters:', error);
  }
}

/**
 * Hook for global filter state management
 *
 * Usage:
 * ```tsx
 * const { timeframe, sortBy, setTimeframe, setSortBy, resetFilters } = useGlobalFilters();
 * ```
 */
export function useGlobalFilters() {
  // Initialize from localStorage
  const [filters, setFilters] = useState<GlobalFilters>(getStoredFilters);

  // Sync with localStorage when filters change
  useEffect(() => {
    saveFilters(filters);
  }, [filters]);

  // Listen for cross-tab changes (when other tabs update filters)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        try {
          const newFilters = JSON.parse(e.newValue);

          // Validate before setting
          if (
            newFilters &&
            typeof newFilters === 'object' &&
            ['1H', '4H', '1D', '1W'].includes(newFilters.timeframe) &&
            ['volume', 'change', 'price', 'name'].includes(newFilters.sortBy)
          ) {
            setFilters(newFilters);
          }
        } catch (error) {
          console.error('[useGlobalFilters] Failed to sync from storage:', error);
        }
      }
    };

    // Listen for custom event (same-tab updates)
    const handleCustomEvent = (e: Event) => {
      const customEvent = e as CustomEvent<GlobalFilters>;
      if (customEvent.detail) {
        setFilters(customEvent.detail);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('global-filters-changed', handleCustomEvent);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('global-filters-changed', handleCustomEvent);
    };
  }, []);

  // Individual setters
  const setTimeframe = (timeframe: Timeframe) => {
    setFilters(prev => ({ ...prev, timeframe }));
  };

  const setSortBy = (sortBy: SortBy) => {
    setFilters(prev => ({ ...prev, sortBy }));
  };

  // Reset to defaults
  const resetFilters = () => {
    setFilters(DEFAULT_FILTERS);
  };

  return {
    timeframe: filters.timeframe,
    sortBy: filters.sortBy,
    setTimeframe,
    setSortBy,
    resetFilters,
  };
}
