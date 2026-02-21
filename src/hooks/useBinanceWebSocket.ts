/**
 * BINANCE WEBSOCKET HOOK v1.0
 *
 * Real-time price updates from Binance WebSocket
 *
 * Features:
 * - Auto-reconnect on disconnect
 * - Cleanup on unmount
 * - TypeScript type safety
 * - Error handling
 *
 * Usage:
 * const { price, priceChange, priceChangePercent } = useBinanceWebSocket('BTCUSDT');
 */

'use client';

import { useState, useEffect, useRef } from 'react';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface BinanceTicker {
  symbol: string;
  price: number;
  priceChange: number;
  priceChangePercent: number;
  volume: number;
  high: number;
  low: number;
  timestamp: number;
}

export interface UseBinanceWebSocketResult {
  price: number | null;
  priceChange: number | null;
  priceChangePercent: number | null;
  volume: number | null;
  high: number | null;
  low: number | null;
  connected: boolean;
  error: string | null;
}

// ============================================================================
// CUSTOM HOOK
// ============================================================================

export function useBinanceWebSocket(symbol: string): UseBinanceWebSocketResult {
  const [price, setPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number | null>(null);
  const [priceChangePercent, setPriceChangePercent] = useState<number | null>(null);
  const [volume, setVolume] = useState<number | null>(null);
  const [high, setHigh] = useState<number | null>(null);
  const [low, setLow] = useState<number | null>(null);
  const [connected, setConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!symbol) return;

    const connect = () => {
      try {
        // Close existing connection
        if (wsRef.current) {
          wsRef.current.close();
        }

        // Binance WebSocket URL for 24hr ticker
        const wsUrl = `wss://fstream.binance.com/ws/${symbol.toLowerCase()}@ticker`;

        console.log(`[BinanceWS] Connecting to ${symbol}...`);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          console.log(`[BinanceWS] Connected to ${symbol}`);
          setConnected(true);
          setError(null);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            // Binance Futures ticker data structure
            setPrice(parseFloat(data.c)); // Last price
            setPriceChange(parseFloat(data.p)); // 24h price change
            setPriceChangePercent(parseFloat(data.P)); // 24h price change percent
            setVolume(parseFloat(data.v)); // 24h volume
            setHigh(parseFloat(data.h)); // 24h high
            setLow(parseFloat(data.l)); // 24h low
          } catch (err) {
            console.error('[BinanceWS] Parse error:', err);
          }
        };

        ws.onerror = (event) => {
          console.error('[BinanceWS] Error:', event);
          setError('WebSocket connection error');
          setConnected(false);
        };

        ws.onclose = () => {
          console.log(`[BinanceWS] Disconnected from ${symbol}`);
          setConnected(false);

          // Auto-reconnect after 5 seconds
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('[BinanceWS] Reconnecting...');
            connect();
          }, 5000);
        };

        wsRef.current = ws;
      } catch (err: any) {
        console.error('[BinanceWS] Connection error:', err);
        setError(err.message || 'Failed to connect');
        setConnected(false);
      }
    };

    connect();

    // Cleanup function
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [symbol]);

  return {
    price,
    priceChange,
    priceChangePercent,
    volume,
    high,
    low,
    connected,
    error,
  };
}
