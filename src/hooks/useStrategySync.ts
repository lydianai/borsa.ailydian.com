/**
 * STRATEGY SYNC HOOK v1.0
 *
 * Custom React hook for fetching unified strategy decisions from API
 *
 * Features:
 * - Real-time data fetching from /api/strategy-sync
 * - Auto-refresh with configurable interval
 * - TypeScript type safety
 * - Loading, error, and success states
 * - Automatic retry on error
 *
 * Usage:
 * const { data, loading, error, refresh } = useStrategySync('BTCUSDT', { autoRefresh: true, interval: 30000 });
 */

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { type UnifiedDecision } from '@/lib/unified-decision-engine';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface StrategyContributor {
  name: string;
  signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  confidence: number;
  impact: number;
}

export interface StrategySyncData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;

  unifiedDecision: {
    signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
    recommendation: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'WAIT' | 'SELL' | 'STRONG_SELL';
    confidence: number;
    consensus: number;
    marketCondition: 'volatile' | 'trending' | 'ranging' | 'uncertain';
  };

  weightedVotes: {
    BUY: number;
    SELL: number;
    WAIT: number;
    NEUTRAL: number;
  };

  topContributors: StrategyContributor[];

  strategies: {
    name: string;
    signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
    confidence: number;
    reason: string;
  }[];

  conflicts: string[];

  meta: {
    strategiesCount: number;
    timestamp: string;
    processingTimeMs: number;
  };
}

export interface UseStrategySyncOptions {
  autoRefresh?: boolean;
  interval?: number; // milliseconds (default: 30000 = 30s)
  retryOnError?: boolean;
  retryDelay?: number; // milliseconds (default: 5000 = 5s)
}

export interface UseStrategySyncResult {
  data: StrategySyncData | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  lastUpdate: Date | null;
}

// ============================================================================
// CUSTOM HOOK
// ============================================================================

export function useStrategySync(
  symbol: string,
  options: UseStrategySyncOptions = {}
): UseStrategySyncResult {
  const {
    autoRefresh = false,
    interval = 30000,
    retryOnError = true,
    retryDelay = 5000,
  } = options;

  // State
  const [data, setData] = useState<StrategySyncData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Refs for interval management
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Fetch data from API
   */
  const fetchData = useCallback(async () => {
    // Validate symbol
    if (!symbol || !/^[A-Z0-9]{6,12}$/.test(symbol)) {
      setError('Invalid symbol format. Expected uppercase alphanumeric, 6-12 characters.');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Cancel previous request if exists
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      // Fetch from API
      const response = await fetch(
        `/api/strategy-sync?symbol=${symbol}`,
        {
          signal: abortControllerRef.current.signal,
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const result: StrategySyncData = await response.json();

      // Update state
      setData(result);
      setLastUpdate(new Date());
      setError(null);

      console.log(`[useStrategySync] ${symbol}: ${result.unifiedDecision.recommendation} (${result.unifiedDecision.confidence}% confidence)`);
    } catch (err: any) {
      // Ignore abort errors (user-initiated cancellation)
      if (err.name === 'AbortError') {
        console.log('[useStrategySync] Request aborted');
        return;
      }

      const errorMessage = err.message || 'Unknown error occurred';
      setError(errorMessage);
      console.error('[useStrategySync] Error:', errorMessage);

      // Retry on error if enabled
      if (retryOnError && !retryTimeoutRef.current) {
        console.log(`[useStrategySync] Retrying in ${retryDelay}ms...`);
        retryTimeoutRef.current = setTimeout(() => {
          retryTimeoutRef.current = null;
          fetchData();
        }, retryDelay);
      }
    } finally {
      setLoading(false);
    }
  }, [symbol, retryOnError, retryDelay]);

  /**
   * Manual refresh function
   */
  const refresh = useCallback(async () => {
    await fetchData();
  }, [fetchData]);

  /**
   * Setup auto-refresh interval
   */
  useEffect(() => {
    // Initial fetch
    fetchData();

    // Setup auto-refresh if enabled
    if (autoRefresh && interval > 0) {
      console.log(`[useStrategySync] Auto-refresh enabled (${interval}ms interval)`);
      refreshIntervalRef.current = setInterval(() => {
        fetchData();
      }, interval);
    }

    // Cleanup function
    return () => {
      // Clear interval
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
        refreshIntervalRef.current = null;
      }

      // Clear retry timeout
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
        retryTimeoutRef.current = null;
      }

      // Abort pending request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, [symbol, autoRefresh, interval, fetchData]);

  return {
    data,
    loading,
    error,
    refresh,
    lastUpdate,
  };
}

// ============================================================================
// UTILITY HOOKS
// ============================================================================

/**
 * Hook for multiple symbols (watchlist)
 */
export function useMultipleStrategySync(
  symbols: string[],
  options: UseStrategySyncOptions = {}
): Record<string, UseStrategySyncResult> {
  const results: Record<string, UseStrategySyncResult> = {};

  symbols.forEach((symbol) => {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    results[symbol] = useStrategySync(symbol, options);
  });

  return results;
}

/**
 * Hook for single symbol with simplified return
 */
export function useQuickStrategySync(symbol: string) {
  const { data, loading, error } = useStrategySync(symbol, {
    autoRefresh: true,
    interval: 30000,
  });

  return {
    signal: data?.unifiedDecision.signal || null,
    recommendation: data?.unifiedDecision.recommendation || null,
    confidence: data?.unifiedDecision.confidence || 0,
    consensus: data?.unifiedDecision.consensus || 0,
    loading,
    error,
  };
}
