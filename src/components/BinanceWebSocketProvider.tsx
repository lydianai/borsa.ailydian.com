/**
 * BINANCE WEBSOCKET PROVIDER
 * Client-side WebSocket connection manager for real-time market data
 * Beyaz Şapka: Educational purpose only
 */

'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { getBinanceWebSocketManager } from '@/lib/binance-websocket-manager';
import { MarketData } from '@/lib/binance-data-fetcher';

interface WebSocketContextType {
  marketData: Map<string, MarketData>;
  isConnected: boolean;
  cacheSize: number;
}

const WebSocketContext = createContext<WebSocketContextType>({
  marketData: new Map(),
  isConnected: false,
  cacheSize: 0,
});

export function BinanceWebSocketProvider({ children }: { children: ReactNode }) {
  const [marketData, setMarketData] = useState<Map<string, MarketData>>(new Map());
  const [isConnected, setIsConnected] = useState(false);
  const [cacheSize, setCacheSize] = useState(0);

  useEffect(() => {
    // Only run in browser
    if (typeof window === 'undefined') return;

    const wsManager = getBinanceWebSocketManager();

    if (!wsManager) {
      console.warn('[WebSocket Provider] WebSocket not available');
      return;
    }

    // Subscribe to market data updates
    const unsubscribe = wsManager.subscribe((data) => {
      setMarketData(new Map(data));
      setCacheSize(data.size);
    });

    // Check connection status periodically
    const statusInterval = setInterval(() => {
      const status = wsManager.getStatus();
      setIsConnected(status.connected);
      setCacheSize(status.cacheSize);
    }, 1000);

    return () => {
      unsubscribe();
      clearInterval(statusInterval);
    };
  }, []);

  return (
    <WebSocketContext.Provider value={{ marketData, isConnected, cacheSize }}>
      {children}
    </WebSocketContext.Provider>
  );
}

/**
 * Hook to access WebSocket market data
 */
export function useWebSocketMarketData() {
  return useContext(WebSocketContext);
}
