/**
 * REALTIME DATA PROCESSOR
 * Process live price updates and generate trading signals
 */

import liveFeedManager from './live-feed';
import { indicatorsAnalyzer } from '../indicators/analyzer';
import { realtimeTelegramNotifier } from '../telegram/realtime-notifier';

// ============================================================================
// REALTIME PROCESSOR
// ============================================================================

class RealtimeProcessor {
  private isProcessing: boolean = false;

  constructor() {
    console.log('[RealtimeProcessor] Initialized');
    
    // Listen for price updates
    liveFeedManager.on('priceUpdate', this.handlePriceUpdate.bind(this));
  }

  /**
   * Handle price update event
   */
  private async handlePriceUpdate(priceUpdate: any): Promise<void> {
    if (this.isProcessing) {
      return; // Avoid concurrent processing
    }

    try {
      this.isProcessing = true;
      
      const { symbol } = priceUpdate;
      
      // Perform technical analysis
      const analysis = await indicatorsAnalyzer.analyze(symbol, priceUpdate);
      
      // Send to Telegram notifier if conditions are met
      await realtimeTelegramNotifier.handleLiveSignal(symbol, analysis);
      
    } catch (error) {
      console.error('[RealtimeProcessor] Error processing price update:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Start processing (if needed for manual control)
   */
  start(): void {
    console.log('[RealtimeProcessor] Started listening for price updates');
  }

  /**
   * Stop processing
   */
  stop(): void {
    console.log('[RealtimeProcessor] Stopped listening for price updates');
    liveFeedManager.removeAllListeners('priceUpdate');
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const realtimeProcessor = new RealtimeProcessor();
export default realtimeProcessor;