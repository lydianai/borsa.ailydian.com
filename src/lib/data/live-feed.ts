/**
 * LIVE DATA FEED MANAGER
 * Real-time price update event emitter with LRU cache
 */

import { EventEmitter } from 'events';
import { LRUCache } from 'lru-cache';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface PriceUpdate {
  symbol: string;
  price: number;
  volume: number;
  changePercent: number;
  timestamp: number;
}

// ============================================================================
// LIVE FEED MANAGER
// ============================================================================

class LiveFeedManager extends EventEmitter {
  private cache: LRUCache<string, PriceUpdate>;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    
    // Initialize LRU cache with 1000 item limit
    this.cache = new LRUCache<string, PriceUpdate>({
      max: 1000,
      ttl: 300000, // 5 minutes TTL
    });

    // Start cleanup interval every 30 seconds
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 30000);
  }

  /**
   * Emit price update event
   */
  emitPriceUpdate(update: PriceUpdate): void {
    // Store in cache
    this.cache.set(update.symbol, update);
    
    // Emit event
    this.emit('priceUpdate', update);
  }

  /**
   * Get cached price data
   */
  getPrice(symbol: string): PriceUpdate | undefined {
    return this.cache.get(symbol);
  }

  /**
   * Get all cached prices
   */
  getAllPrices(): PriceUpdate[] {
    const prices: PriceUpdate[] = [];
    for (const [_symbol, price] of this.cache.entries()) {
      prices.push(price);
    }
    return prices;
  }

  /**
   * Get cache statistics
   */
  getStats(): { size: number; max: number } {
    return {
      size: this.cache.size,
      max: this.cache.max,
    };
  }

  /**
   * Cleanup old cache entries
   */
  private cleanup(): void {
    // LRU cache automatically handles cleanup, but we can log stats
    console.log(`[LiveFeed] Cache stats: ${this.cache.size}/${this.cache.max} items`);
  }

  /**
   * Stop cleanup interval
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const liveFeedManager = new LiveFeedManager();
export default liveFeedManager;