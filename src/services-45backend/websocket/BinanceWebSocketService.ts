/**
 * BINANCE WEBSOCKET SERVICE
 * Real-time price streaming from Binance public WebSocket
 * WHITE-HAT COMPLIANT: Read-only, public data, no trading
 */

import WebSocket from 'ws';

export interface TickerData {
  symbol: string;
  price: number;
  change24h: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
}

export type TickerCallback = (data: TickerData) => void;

export class BinanceWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private subscribers: Map<string, Set<TickerCallback>> = new Map();
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  // Binance public WebSocket endpoint (no authentication required)
  private readonly WS_URL = 'wss://stream.binance.com:9443/ws';

  constructor() {
    console.log('🚀 Binance WebSocket Service initialized (READ-ONLY mode)');
  }

  /**
   * Connect to Binance WebSocket stream for specific symbols
   */
  public connect(symbols: string[]): void {
    if (this.isConnected) {
      console.log('⚠️  Already connected to Binance WebSocket');
      return;
    }

    // Convert symbols to Binance format (e.g., BTC/USDT -> btcusdt)
    const streams = symbols.map(s =>
      `${s.toLowerCase().replace('/', '')}@ticker`
    ).join('/');

    const wsUrl = `${this.WS_URL}/${streams}`;

    console.log(`📡 Connecting to Binance WebSocket: ${symbols.join(', ')}`);

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.on('open', () => {
        console.log('✅ Binance WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
      });

      this.ws.on('message', (data: WebSocket.Data) => {
        this.handleMessage(data);
      });

      this.ws.on('error', (error) => {
        console.error('❌ Binance WebSocket error:', error.message);
      });

      this.ws.on('close', () => {
        console.log('🔌 Binance WebSocket disconnected');
        this.isConnected = false;
        this.handleReconnect(symbols);
      });

    } catch (error: any) {
      console.error('❌ Failed to connect to Binance WebSocket:', error.message);
      this.handleReconnect(symbols);
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(data: WebSocket.Data): void {
    try {
      const message = JSON.parse(data.toString());

      // Handle ticker update
      if (message.e === '24hrTicker') {
        const tickerData: TickerData = {
          symbol: this.formatSymbol(message.s), // BTCUSDT -> BTC/USDT
          price: parseFloat(message.c),
          change24h: parseFloat(message.P),
          volume: parseFloat(message.v),
          high24h: parseFloat(message.h),
          low24h: parseFloat(message.l),
          timestamp: message.E,
        };

        // Notify all subscribers for this symbol
        this.notifySubscribers(tickerData.symbol, tickerData);
      }
    } catch (error: any) {
      console.error('❌ Error parsing WebSocket message:', error.message);
    }
  }

  /**
   * Format symbol from Binance format to standard format
   */
  private formatSymbol(binanceSymbol: string): string {
    // BTCUSDT -> BTC/USDT
    const match = binanceSymbol.match(/^([A-Z]+)(USDT|BTC|ETH|BNB)$/);
    if (match) {
      return `${match[1]}/${match[2]}`;
    }
    return binanceSymbol;
  }

  /**
   * Subscribe to ticker updates for a specific symbol
   */
  public subscribe(symbol: string, callback: TickerCallback): () => void {
    if (!this.subscribers.has(symbol)) {
      this.subscribers.set(symbol, new Set());
    }

    this.subscribers.get(symbol)!.add(callback);

    console.log(`📊 Subscribed to ${symbol} (${this.subscribers.get(symbol)!.size} subscribers)`);

    // Return unsubscribe function
    return () => {
      const callbacks = this.subscribers.get(symbol);
      if (callbacks) {
        callbacks.delete(callback);
        if (callbacks.size === 0) {
          this.subscribers.delete(symbol);
          console.log(`🔕 Unsubscribed from ${symbol}`);
        }
      }
    };
  }

  /**
   * Notify subscribers about ticker updates
   */
  private notifySubscribers(symbol: string, data: TickerData): void {
    const callbacks = this.subscribers.get(symbol);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error: any) {
          console.error(`❌ Error in subscriber callback for ${symbol}:`, error.message);
        }
      });
    }
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect(symbols: string[]): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached. Giving up.');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;

    console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.connect(symbols);
    }, delay);
  }

  /**
   * Disconnect from WebSocket
   */
  public disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnected = false;
    this.subscribers.clear();

    console.log('🔌 Binance WebSocket disconnected');
  }

  /**
   * Get connection status
   */
  public getStatus(): { connected: boolean; subscriberCount: number } {
    return {
      connected: this.isConnected,
      subscriberCount: Array.from(this.subscribers.values()).reduce(
        (sum, set) => sum + set.size,
        0
      ),
    };
  }
}

// Singleton instance
let binanceWSService: BinanceWebSocketService | null = null;

export function getBinanceWebSocketService(): BinanceWebSocketService {
  if (!binanceWSService) {
    binanceWSService = new BinanceWebSocketService();
  }
  return binanceWSService;
}
