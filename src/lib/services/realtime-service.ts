/**
 * REALTIME SERVICE
 * Initialize and manage real-time data processing
 */

import binanceWebSocketService from '../data-service/binance-websocket';
import realtimeProcessor from '../data/realtime-processor';
import { fetchBinanceFuturesData } from '../binance-data-fetcher';

// ============================================================================
// REALTIME SERVICE
// ============================================================================

class RealtimeService {
  private isInitialized: boolean = false;
  private symbols: string[] = [];

  constructor() {
    console.log('[RealtimeService] Initialized');
  }

  /**
   * Initialize real-time service
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('[RealtimeService] Already initialized');
      return;
    }

    try {
      console.log('[RealtimeService] Initializing...');
      
      // Start realtime processor
      realtimeProcessor.start();
      
      // Connect to Binance WebSocket
      await this.connectToBinance();
      
      this.isInitialized = true;
      console.log('[RealtimeService] ✅ Initialization complete');
    } catch (error) {
      console.error('[RealtimeService] Initialization error:', error);
    }
  }

  /**
   * Connect to Binance WebSocket and subscribe to symbols
   */
  private async connectToBinance(): Promise<void> {
    try {
      // Fetch symbols from Binance
      const symbols = await this.fetchSymbols();
      this.symbols = symbols;
      
      console.log(`[RealtimeService] Found ${symbols.length} symbols`);
      
      // Connect to WebSocket
      await binanceWebSocketService.connect();
      
      // Subscribe to symbols (limit to first 50 for testing)
      const symbolsToSubscribe = symbols.slice(0, 50);
      binanceWebSocketService.subscribe(symbolsToSubscribe);
      
      console.log(`[RealtimeService] Subscribed to ${symbolsToSubscribe.length} symbols`);
    } catch (error) {
      console.error('[RealtimeService] Binance connection error:', error);
    }
  }

  /**
   * Fetch symbols from Binance
   */
  private async fetchSymbols(): Promise<string[]> {
    try {
      const data = await fetchBinanceFuturesData();
      
      if (Array.isArray(data)) {
        // Extract symbols from array of ticker data
        return data.map((item: any) => item.symbol).filter(Boolean);
      } else if (data && typeof data === 'object') {
        // Handle object response
        return Object.keys(data).map(key => (data as any)[key].symbol || key);
      }
      
      // Fallback symbols
      return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT'
      ];
    } catch (error) {
      console.error('[RealtimeService] Symbol fetch error:', error);
      // Return fallback symbols
      return [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT'
      ];
    }
  }

  /**
   * Get service status
   */
  getStatus(): {
    initialized: boolean;
    symbolsCount: number;
    websocketConnected: boolean;
  } {
    return {
      initialized: this.isInitialized,
      symbolsCount: this.symbols.length,
      websocketConnected: binanceWebSocketService.getStats().connected
    };
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    console.log('[RealtimeService] Cleaning up...');
    realtimeProcessor.stop();
    binanceWebSocketService.disconnect();
    this.isInitialized = false;
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const realtimeService = new RealtimeService();
export default realtimeService;