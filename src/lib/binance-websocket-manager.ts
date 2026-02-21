/**
 * BINANCE WEBSOCKET MANAGER
 * Real-time market data via WebSocket to avoid API rate limits
 * Beyaz Şapka: Educational purpose only - real-time data streaming
 */

import { MarketData } from './binance-data-fetcher';

interface TickerData {
  e: string; // Event type
  E: number; // Event time
  s: string; // Symbol
  p: string; // Price change
  P: string; // Price change percent
  c: string; // Last price
  h: string; // High price
  l: string; // Low price
  v: string; // Total traded base asset volume
  q: string; // Total traded quote asset volume
}

export class BinanceWebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private isConnecting = false;
  private marketDataCache: Map<string, MarketData> = new Map();
  private subscribers: Set<(data: Map<string, MarketData>) => void> = new Set();
  private _symbols: string[] = [];

  constructor(symbols: string[] = []) {
    this.symbols = symbols;
  }

  /**
   * Connect to Binance WebSocket stream
   */
  public connect(): void {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      console.log('[Binance WS] Already connected or connecting');
      return;
    }

    this.isConnecting = true;
    console.log('[Binance WS] Connecting to ticker stream...');

    // Connect to all market tickers stream
    const wsUrl = 'wss://fstream.binance.com/ws/!ticker@arr';

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('✅ [Binance WS] Connected successfully');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onerror = (error) => {
        console.error('❌ [Binance WS] Error:', error);
      };

      this.ws.onclose = (event) => {
        console.log(`⚠️ [Binance WS] Connection closed (code: ${event.code})`);
        this.isConnecting = false;
        this.scheduleReconnect();
      };
    } catch (error) {
      console.error('❌ [Binance WS] Failed to create connection:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: string): void {
    try {
      const tickers: TickerData[] = JSON.parse(data);

      // Update cache with new data
      tickers.forEach((ticker) => {
        // Only process USDT perpetual contracts
        if (!ticker.s.endsWith('USDT')) return;

        const marketData: MarketData = {
          symbol: ticker.s,
          price: parseFloat(ticker.c),
          change24h: parseFloat(ticker.p),
          changePercent24h: parseFloat(ticker.P),
          volume24h: parseFloat(ticker.v),
          high24h: parseFloat(ticker.h),
          low24h: parseFloat(ticker.l),
          lastUpdate: new Date(ticker.E).toISOString(),
          closeTime: ticker.E,
        };

        this.marketDataCache.set(ticker.s, marketData);
      });

      // Notify subscribers
      this.notifySubscribers();
    } catch (error) {
      console.error('[Binance WS] Error parsing message:', error);
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error(`❌ [Binance WS] Max reconnect attempts (${this.maxReconnectAttempts}) reached`);
      return;
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);

    console.log(`🔄 [Binance WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Subscribe to market data updates
   */
  public subscribe(callback: (data: Map<string, MarketData>) => void): () => void {
    this.subscribers.add(callback);

    // Return unsubscribe function
    return () => {
      this.subscribers.delete(callback);
    };
  }

  /**
   * Notify all subscribers of data updates
   */
  private notifySubscribers(): void {
    this.subscribers.forEach((callback) => {
      try {
        callback(new Map(this.marketDataCache));
      } catch (error) {
        console.error('[Binance WS] Error notifying subscriber:', error);
      }
    });
  }

  /**
   * Get current market data from cache
   */
  public getMarketData(): MarketData[] {
    return Array.from(this.marketDataCache.values());
  }

  /**
   * Get specific symbol data
   */
  public getSymbolData(symbol: string): MarketData | undefined {
    return this.marketDataCache.get(symbol);
  }

  /**
   * Get cache size
   */
  public getCacheSize(): number {
    return this.marketDataCache.size;
  }

  /**
   * Disconnect WebSocket
   */
  public disconnect(): void {
    console.log('[Binance WS] Disconnecting...');

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnecting = false;
    this.reconnectAttempts = 0;
  }

  /**
   * Check if WebSocket is connected
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection status
   */
  public getStatus(): {
    connected: boolean;
    cacheSize: number;
    reconnectAttempts: number;
  } {
    return {
      connected: this.isConnected(),
      cacheSize: this.getCacheSize(),
      reconnectAttempts: this.reconnectAttempts,
    };
  }
}

// Singleton instance for server-side usage
let wsManagerInstance: BinanceWebSocketManager | null = null;

/**
 * Get or create WebSocket manager instance (Browser-only)
 * Note: This is a client-side only feature
 */
export function getBinanceWebSocketManager(): BinanceWebSocketManager | null {
  // WebSocket only works in browser environment
  if (typeof window === 'undefined' || typeof WebSocket === 'undefined') {
    console.warn('[Binance WS] WebSocket not available - browser-only feature');
    return null;
  }

  if (!wsManagerInstance) {
    wsManagerInstance = new BinanceWebSocketManager();
    wsManagerInstance.connect();
  }

  return wsManagerInstance;
}

/**
 * Cleanup function for graceful shutdown
 */
export function cleanupWebSocket(): void {
  if (wsManagerInstance) {
    wsManagerInstance.disconnect();
    wsManagerInstance = null;
  }
}
