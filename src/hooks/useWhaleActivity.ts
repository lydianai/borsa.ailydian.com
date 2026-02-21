/**
 * WHALE ACTIVITY HOOK
 * React hook for fetching on-chain whale data
 *
 * Usage:
 * const { whaleData, loading, error, refresh } = useWhaleActivity('BTCUSDT');
 */

import { useEffect, useState, useCallback } from 'react';

interface WhaleActivityData {
  symbol: string;
  activity: 'accumulation' | 'distribution' | 'neutral';
  confidence: number;
  riskScore: number;
  exchangeNetflow: number;
  summary: string;
  recentTransactionsCount?: number;
  timestamp: Date;
}

interface UseWhaleActivityReturn {
  whaleData: WhaleActivityData | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/**
 * Fetch whale activity for a specific symbol
 */
export function useWhaleActivity(symbol: string): UseWhaleActivityReturn {
  const [whaleData, setWhaleData] = useState<WhaleActivityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const fetchWhaleActivity = useCallback(async () => {
    if (!symbol) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/onchain/whale-alerts?symbol=${symbol}`);
      const data = await response.json();

      if (data.success && data.data) {
        setWhaleData({
          ...data.data,
          timestamp: new Date(data.data.timestamp),
        });
      } else {
        setWhaleData(null);
        setError(data.error || 'No whale activity found');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch whale activity');
      setWhaleData(null);
    } finally {
      setLoading(false);
    }
  }, [symbol, refreshTrigger]);

  useEffect(() => {
    fetchWhaleActivity();
  }, [fetchWhaleActivity]);

  const refresh = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return {
    whaleData,
    loading,
    error,
    refresh,
  };
}

/**
 * Fetch whale activity for multiple symbols
 */
export function useWhaleActivityMultiple(symbols: string[]): {
  whaleDataMap: Map<string, WhaleActivityData>;
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const [whaleDataMap, setWhaleDataMap] = useState<Map<string, WhaleActivityData>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const fetchAllWhaleActivity = useCallback(async () => {
    if (!symbols || symbols.length === 0) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/onchain/whale-alerts');
      const data = await response.json();

      if (data.success && data.data && data.data.whaleActivity) {
        const map = new Map<string, WhaleActivityData>();
        data.data.whaleActivity.forEach((activity: WhaleActivityData) => {
          if (symbols.includes(activity.symbol)) {
            map.set(activity.symbol, {
              ...activity,
              timestamp: new Date(activity.timestamp || Date.now()),
            });
          }
        });
        setWhaleDataMap(map);
      } else {
        setError(data.error || 'Failed to fetch whale activity');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch whale activity');
    } finally {
      setLoading(false);
    }
  }, [symbols.join(','), refreshTrigger]);

  useEffect(() => {
    fetchAllWhaleActivity();
  }, [fetchAllWhaleActivity]);

  const refresh = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return {
    whaleDataMap,
    loading,
    error,
    refresh,
  };
}

/**
 * Fetch on-chain market overview
 */
export function useOnChainOverview(): {
  overview: {
    trending: Array<{
      symbol: string;
      activity: 'accumulation' | 'distribution' | 'neutral';
      confidence: number;
      summary: string;
    }>;
    mostAccumulated: { symbol: string; netflow: number; summary: string } | null;
    mostDistributed: { symbol: string; netflow: number; summary: string } | null;
    marketSentiment: 'bullish' | 'bearish' | 'neutral';
  } | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const [overview, setOverview] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const fetchOverview = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/onchain/overview');
      const data = await response.json();

      if (data.success && data.data) {
        setOverview(data.data);
      } else {
        setError(data.error || 'Failed to fetch overview');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch overview');
    } finally {
      setLoading(false);
    }
  }, [refreshTrigger]);

  useEffect(() => {
    fetchOverview();
  }, [fetchOverview]);

  const refresh = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return {
    overview,
    loading,
    error,
    refresh,
  };
}

/**
 * Fetch whale notifications
 */
export function useWhaleNotifications(config?: {
  symbols?: string[];
  minConfidence?: number;
  minNetflow?: number;
}): {
  notifications: Array<{
    id: string;
    type: string;
    title: string;
    message: string;
    timestamp: Date;
    icon: string;
    color: string;
  }>;
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const [notifications, setNotifications] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const fetchNotifications = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (config?.symbols) params.set('symbols', config.symbols.join(','));
      if (config?.minConfidence) params.set('minConfidence', config.minConfidence.toString());
      if (config?.minNetflow) params.set('minNetflow', config.minNetflow.toString());

      const url = `/api/onchain/notifications${params.toString() ? `?${params}` : ''}`;
      const response = await fetch(url);
      const data = await response.json();

      if (data.success && data.data) {
        setNotifications(
          data.data.notifications?.map((n: any) => ({
            ...n,
            timestamp: new Date(n.timestamp),
          })) || []
        );
      } else {
        setError(data.error || 'Failed to fetch notifications');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch notifications');
    } finally {
      setLoading(false);
    }
  }, [
    config?.symbols?.join(','),
    config?.minConfidence,
    config?.minNetflow,
    refreshTrigger,
  ]);

  useEffect(() => {
    fetchNotifications();
  }, [fetchNotifications]);

  const refresh = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return {
    notifications,
    loading,
    error,
    refresh,
  };
}
