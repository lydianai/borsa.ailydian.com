/**
 * BINANCE WEBSOCKET DATA SERVICE
 * Real-time price streaming with circuit breaker protection
 *
 * Features:
 * - Multi-symbol ticker streams
 * - Automatic reconnection with exponential backoff
 * - Circuit breaker protection
 * - Health monitoring
 * - Event-based price updates
 * - White-hat compliance: All connections logged
 */

import WebSocket from 'ws';
import { EventEmitter } from 'events';
import circuitBreakerManager from '../resilience/circuit-breaker';
import liveFeedManager from '../data/live-feed';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface BinanceTickerData {
  symbol: string;
  price: number;
  priceChange: number;
  priceChangePercent: number;
  volume: number;
  quoteVolume: number;
  timestamp: number;
}

export interface WebSocketConfig {
  baseUrl: string;
  reconnectDelay: number; // Initial reconnect delay in ms
  maxReconnectDelay: number; // Max reconnect delay in ms
  pingInterval: number; // Ping interval in ms
  pongTimeout: number; // Pong timeout in ms
}

export interface ConnectionStats {
  connected: boolean;
  connectedSince?: number;
  reconnectAttempts: number;
  messagesReceived: number;
  lastMessageTime?: number;
  subscriptions: string[];
  circuitBreakerState: string;
}

// ============================================================================
// BINANCE WEBSOCKET SERVICE
// ============================================================================

export class BinanceWebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private isConnecting: boolean = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private pingInterval: NodeJS.Timeout | null = null;
  private reconnectAttempts: number = 0;
  private currentReconnectDelay: number;
  private subscriptions: Set<string> = new Set();
  private connectedSince?: number;
  private messagesReceived: number = 0;
  private lastMessageTime?: number;
  private circuitBreaker = circuitBreakerManager.getBreaker('binance-websocket', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 30000, // 30 seconds
    monitoringPeriod: 60000,
  });

  private config: WebSocketConfig = {
    baseUrl: process.env.BINANCE_WS || 'wss://fstream.binance.com/ws',
    reconnectDelay: 1000, // 1 second
    maxReconnectDelay: 60000, // 1 minute
    pingInterval: 30000, // 30 seconds
    pongTimeout: 10000, // 10 seconds
  };

  constructor() {
    super();
    console.log('[BinanceWS] Initialized with config:', this.config);
    this.currentReconnectDelay = this.config.reconnectDelay;
  }

  /**
   * Connect to Binance WebSocket
   */
  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('[BinanceWS] Already connected');
      return;
    }

    if (this.isConnecting) {
      console.log('[BinanceWS] Connection already in progress');
      return;
    }

    this.isConnecting = true;

    try {
      await this.circuitBreaker.execute(
        async () => {
          await this.establishConnection();
        },
        async () => {
          console.warn('[BinanceWS] Circuit breaker fallback: Connection failed, will retry');
          throw new Error('Circuit breaker open, connection delayed');
        }
      );
    } catch (error: any) {
      console.error('[BinanceWS] Connection failed:', error.message);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Establish WebSocket connection
   */
  private async establishConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`[BinanceWS] Connecting to ${this.config.baseUrl}...`);

      this.ws = new WebSocket(this.config.baseUrl);

      // Connection timeout
      const timeout = setTimeout(() => {
        if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
          console.error('[BinanceWS] Connection timeout');
          this.ws.terminate();
          reject(new Error('Connection timeout'));
        }
      }, 10000);

      this.ws.on('open', () => {
        clearTimeout(timeout);
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.currentReconnectDelay = this.config.reconnectDelay;
        this.connectedSince = Date.now();

        console.log('[BinanceWS] ✅ Connected successfully');
        this.emit('connected');

        // Start ping/pong heartbeat
        this.startHeartbeat();

        // Resubscribe to symbols
        this.resubscribe();

        resolve();
      });

      this.ws.on('message', (data: Buffer) => {
        this.handleMessage(data);
      });

      this.ws.on('error', (error: Error) => {
        clearTimeout(timeout);
        console.error('[BinanceWS] WebSocket error:', error.message);
        this.emit('error', error);
        reject(error);
      });

      this.ws.on('close', (code: number, reason: Buffer) => {
        clearTimeout(timeout);
        console.warn(`[BinanceWS] Connection closed: ${code} - ${reason.toString()}`);
        this.cleanup();
        this.emit('disconnected', { code, reason: reason.toString() });
        this.scheduleReconnect();
      });

      this.ws.on('pong', () => {
        console.log('[BinanceWS] Pong received');
      });
    });
  }

  /**
   * Handle incoming WebSocket message
   */
  private handleMessage(data: Buffer): void {
    this.messagesReceived++;
    this.lastMessageTime = Date.now();

    try {
      const message = JSON.parse(data.toString());

      // Handle ticker update
      if (message.e === '24hrTicker') {
        const ticker: BinanceTickerData = {
          symbol: message.s,
          price: parseFloat(message.c),
          priceChange: parseFloat(message.p),
          priceChangePercent: parseFloat(message.P),
          volume: parseFloat(message.v),
          quoteVolume: parseFloat(message.q),
          timestamp: message.E,
        };

        this.emit('ticker', ticker);
        
        // Emit to live feed manager
        liveFeedManager.emitPriceUpdate({
          symbol: ticker.symbol,
          price: ticker.price,
          volume: ticker.volume,
          changePercent: ticker.priceChangePercent,
          timestamp: ticker.timestamp
        });
      }
    } catch (error: any) {
      console.error('[BinanceWS] Message parse error:', error.message);
    }
  }

  /**
   * Subscribe to symbol ticker streams
   */
  subscribe(symbols: string[]): void {
    if (!symbols || symbols.length === 0) {
      console.warn('[BinanceWS] No symbols to subscribe');
      return;
    }

    // Add symbols to subscription set
    symbols.forEach((symbol) => {
      const normalized = symbol.toUpperCase();
      this.subscriptions.add(normalized);
    });

    // If connected, send subscription message
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const streams = symbols.map((s) => `${s.toLowerCase()}@ticker`);

      const subscribeMessage = {
        method: 'SUBSCRIBE',
        params: streams,
        id: Date.now(),
      };

      console.log(`[BinanceWS] Subscribing to ${symbols.length} symbols:`, symbols.slice(0, 5));
      this.ws.send(JSON.stringify(subscribeMessage));
    } else {
      console.warn('[BinanceWS] Not connected, symbols will be subscribed on connection');
    }
  }

  /**
   * Unsubscribe from symbol ticker streams
   */
  unsubscribe(symbols: string[]): void {
    if (!symbols || symbols.length === 0) {
      return;
    }

    // Remove symbols from subscription set
    symbols.forEach((symbol) => {
      const normalized = symbol.toUpperCase();
      this.subscriptions.delete(normalized);
    });

    // If connected, send unsubscribe message
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const streams = symbols.map((s) => `${s.toLowerCase()}@ticker`);

      const unsubscribeMessage = {
        method: 'UNSUBSCRIBE',
        params: streams,
        id: Date.now(),
      };

      console.log(`[BinanceWS] Unsubscribing from ${symbols.length} symbols`);
      this.ws.send(JSON.stringify(unsubscribeMessage));
    }
  }

  /**
   * Resubscribe to all symbols after reconnection
   */
  private resubscribe(): void {
    if (this.subscriptions.size > 0) {
      const symbols = Array.from(this.subscriptions);
      console.log(`[BinanceWS] Resubscribing to ${symbols.length} symbols`);
      this.subscribe(symbols);
    }
  }

  /**
   * Start ping/pong heartbeat
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        console.log('[BinanceWS] Sending ping');
        this.ws.ping();
      }
    }, this.config.pingInterval);
  }

  /**
   * Stop ping/pong heartbeat
   */
  private stopHeartbeat(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      return; // Already scheduled
    }

    this.reconnectAttempts++;

    const delay = Math.min(
      this.currentReconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.config.maxReconnectDelay
    );

    console.log(`[BinanceWS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, delay);
  }

  /**
   * Cleanup resources
   */
  private cleanup(): void {
    this.stopHeartbeat();
    this.connectedSince = undefined;

    if (this.ws) {
      this.ws.removeAllListeners();
      this.ws = null;
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    console.log('[BinanceWS] Disconnecting...');

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    this.cleanup();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.subscriptions.clear();
    console.log('[BinanceWS] ✅ Disconnected');
  }

  /**
   * Get connection statistics
   */
  getStats(): ConnectionStats {
    return {
      connected: this.ws?.readyState === WebSocket.OPEN,
      connectedSince: this.connectedSince,
      reconnectAttempts: this.reconnectAttempts,
      messagesReceived: this.messagesReceived,
      lastMessageTime: this.lastMessageTime,
      subscriptions: Array.from(this.subscriptions),
      circuitBreakerState: this.circuitBreaker.getState(),
    };
  }

  /**
   * Health check
   */
  isHealthy(): boolean {
    const connected = this.ws?.readyState === WebSocket.OPEN;
    const circuitHealthy = this.circuitBreaker.isHealthy();
    const recentMessage =
      this.lastMessageTime !== undefined && Date.now() - this.lastMessageTime < 60000; // Last message within 1 minute

    return connected && circuitHealthy && (recentMessage || this.subscriptions.size === 0);
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const binanceWebSocketService = new BinanceWebSocketService();
export default binanceWebSocketService;
