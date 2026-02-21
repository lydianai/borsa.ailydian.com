/**
 * 🔄 CONTINUOUS SCANNER SERVICE
 * Automated market scanning scheduler for 522+ USDT perpetual contracts
 *
 * Features:
 * - Intelligent batching (50 symbols per batch)
 * - Priority-based scheduling (high volume coins first)
 * - Configurable scan intervals
 * - Circuit breaker protection
 * - Health monitoring
 * - White-hat compliance: All scans logged with audit trail
 */

import coinListService from './coin-list-service';
import { scanQueue } from '../queue/scan-queue';
import circuitBreakerManager from '../resilience/circuit-breaker';

// ============================================================================
// TYPES
// ============================================================================

export interface ScannerConfig {
  /**
   * Scan interval in milliseconds
   * Default: 5 minutes (300000ms)
   */
  scanIntervalMs: number;

  /**
   * Number of symbols per batch
   * Default: 50
   */
  batchSize: number;

  /**
   * Delay between batches in milliseconds
   * Default: 10 seconds (10000ms)
   */
  batchDelayMs: number;

  /**
   * Strategies to run on each scan
   */
  strategies: string[];

  /**
   * Enable priority-based scheduling
   * If true, high-volume coins are scanned first
   */
  priorityMode: boolean;

  /**
   * Auto-start on initialization
   */
  autoStart: boolean;
}

export interface ScannerStats {
  isRunning: boolean;
  totalScansTriggered: number;
  totalSymbolsScanned: number;
  totalBatchesProcessed: number;
  lastScanTime: string | null;
  nextScanTime: string | null;
  currentBatch: number;
  totalBatches: number;
  errors: number;
  circuitBreakerState: string;
}

// ============================================================================
// CONTINUOUS SCANNER SERVICE
// ============================================================================

export class ContinuousScannerService {
  private config: ScannerConfig;
  private isRunning: boolean = false;
  private scanInterval: NodeJS.Timeout | null = null;
  private batchTimeout: NodeJS.Timeout | null = null;

  // Statistics
  private stats = {
    totalScansTriggered: 0,
    totalSymbolsScanned: 0,
    totalBatchesProcessed: 0,
    lastScanTime: null as Date | null,
    nextScanTime: null as Date | null,
    currentBatch: 0,
    totalBatches: 0,
    errors: 0,
  };

  // Circuit breaker
  private circuitBreaker = circuitBreakerManager.getBreaker('continuous-scanner', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 60000, // 1 minute
    monitoringPeriod: 300000, // 5 minutes
  });

  constructor(config?: Partial<ScannerConfig>) {
    this.config = {
      scanIntervalMs: parseInt(process.env.SCAN_INTERVAL_MS || '300000', 10), // 5 min default
      batchSize: parseInt(process.env.SCAN_BATCH_SIZE || '50', 10),
      batchDelayMs: parseInt(process.env.SCAN_BATCH_DELAY_MS || '10000', 10), // 10 sec
      strategies: [
        'ma-pullback',
        'rsi-divergence',
        'bollinger-squeeze',
        'ema-ribbon',
        'volume-profile',
        'fibonacci',
        'ichimoku',
        'atr-volatility',
        'trend-reversal',
      ],
      priorityMode: process.env.SCAN_PRIORITY_MODE === 'true',
      autoStart: process.env.SCAN_AUTO_START === 'true',
      ...config,
    };

    console.log('[ContinuousScanner] Initialized with config:', this.config);

    if (this.config.autoStart) {
      this.start();
    }
  }

  /**
   * Start continuous scanning
   */
  start(): void {
    if (this.isRunning) {
      console.warn('[ContinuousScanner] Already running');
      return;
    }

    console.log('[ContinuousScanner] 🚀 Starting continuous scanner...');
    this.isRunning = true;

    // Run first scan immediately
    this.triggerScan();

    // Schedule recurring scans
    this.scanInterval = setInterval(() => {
      this.triggerScan();
    }, this.config.scanIntervalMs);

    console.log(
      `[ContinuousScanner] ✅ Started. Scanning every ${this.config.scanIntervalMs / 1000}s`
    );
  }

  /**
   * Stop continuous scanning
   */
  stop(): void {
    if (!this.isRunning) {
      console.warn('[ContinuousScanner] Not running');
      return;
    }

    console.log('[ContinuousScanner] 🛑 Stopping continuous scanner...');
    this.isRunning = false;

    if (this.scanInterval) {
      clearInterval(this.scanInterval);
      this.scanInterval = null;
    }

    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
      this.batchTimeout = null;
    }

    console.log('[ContinuousScanner] ✅ Stopped');
  }

  /**
   * Trigger a full market scan
   */
  private async triggerScan(): Promise<void> {
    const scanId = `scan-${Date.now()}`;
    const scanStartTime = Date.now();

    console.log(`[ContinuousScanner] 📊 Triggering scan ${scanId}...`);

    try {
      await this.circuitBreaker.execute(
        async () => {
          // 1. Fetch symbols
          const symbols = await this.fetchSymbols();

          if (symbols.length === 0) {
            throw new Error('No symbols fetched');
          }

          console.log(`[ContinuousScanner] Fetched ${symbols.length} symbols`);

          // 2. Create batches
          const batches = this.createBatches(symbols);
          this.stats.totalBatches = batches.length;
          this.stats.currentBatch = 0;

          console.log(`[ContinuousScanner] Created ${batches.length} batches`);

          // 3. Process batches sequentially
          for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            this.stats.currentBatch = i + 1;

            // Enqueue batch
            await this.enqueueBatch(scanId, batch, i + 1, batches.length);

            // Wait before next batch (rate limiting)
            if (i < batches.length - 1) {
              await this.sleep(this.config.batchDelayMs);
            }
          }

          // 4. Update statistics
          this.stats.totalScansTriggered++;
          this.stats.totalSymbolsScanned += symbols.length;
          this.stats.totalBatchesProcessed += batches.length;
          this.stats.lastScanTime = new Date();
          this.stats.nextScanTime = new Date(Date.now() + this.config.scanIntervalMs);

          const scanDuration = Date.now() - scanStartTime;
          console.log(
            `[ContinuousScanner] ✅ Scan ${scanId} completed in ${scanDuration}ms`
          );
        },
        async () => {
          // Fallback: Circuit breaker open
          this.stats.errors++;
          console.error('[ContinuousScanner] Circuit breaker open, skipping scan');
        }
      );
    } catch (error: any) {
      this.stats.errors++;
      console.error('[ContinuousScanner] Scan failed:', error.message);
    }
  }

  /**
   * Fetch symbols from CoinListService
   */
  private async fetchSymbols(): Promise<string[]> {
    if (this.config.priorityMode) {
      // Fetch top symbols by volume
      const topSymbols = await coinListService.getTopSymbolsByVolume(522);
      console.log(`[ContinuousScanner] Priority mode: Using top ${topSymbols.length} symbols`);
      return topSymbols;
    } else {
      // Fetch all symbols
      return await coinListService.getAllSymbols();
    }
  }

  /**
   * Create batches from symbols array
   */
  private createBatches(symbols: string[]): string[][] {
    const batches: string[][] = [];
    const batchSize = this.config.batchSize;

    for (let i = 0; i < symbols.length; i += batchSize) {
      batches.push(symbols.slice(i, i + batchSize));
    }

    return batches;
  }

  /**
   * Enqueue a batch to the scan queue
   */
  private async enqueueBatch(
    scanId: string,
    symbols: string[],
    batchNumber: number,
    totalBatches: number
  ): Promise<void> {
    const jobId = `${scanId}-batch-${batchNumber}`;

    try {
      const result = await scanQueue.enqueue({
        requestId: jobId,
        requestedBy: 'continuous-scanner',
        scopes: ['scan:enqueue'],
        symbols,
        strategies: this.config.strategies,
        priority: this.calculatePriority(batchNumber, totalBatches),
        timestamp: new Date().toISOString(),
      });

      console.log(
        `[ContinuousScanner] ✅ Enqueued batch ${batchNumber}/${totalBatches} (${symbols.length} symbols) - Job ID: ${result.jobId}`
      );
    } catch (error: any) {
      console.error(
        `[ContinuousScanner] Failed to enqueue batch ${batchNumber}:`,
        error.message
      );
      throw error;
    }
  }

  /**
   * Calculate job priority based on batch number
   * Earlier batches (high-volume coins) get higher priority
   */
  private calculatePriority(batchNumber: number, totalBatches: number): number {
    if (!this.config.priorityMode) {
      return 5; // Default priority
    }

    // Priority range: 10 (highest) to 1 (lowest)
    // First batch gets 10, last batch gets 1
    const priorityRange = 9; // 10 - 1
    const normalizedPosition = (batchNumber - 1) / (totalBatches - 1); // 0 to 1
    const priority = 10 - Math.floor(normalizedPosition * priorityRange);

    return Math.max(1, Math.min(10, priority));
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => {
      this.batchTimeout = setTimeout(resolve, ms);
    });
  }

  /**
   * Get scanner statistics
   */
  getStats(): ScannerStats {
    return {
      isRunning: this.isRunning,
      totalScansTriggered: this.stats.totalScansTriggered,
      totalSymbolsScanned: this.stats.totalSymbolsScanned,
      totalBatchesProcessed: this.stats.totalBatchesProcessed,
      lastScanTime: this.stats.lastScanTime ? this.stats.lastScanTime.toISOString() : null,
      nextScanTime: this.stats.nextScanTime ? this.stats.nextScanTime.toISOString() : null,
      currentBatch: this.stats.currentBatch,
      totalBatches: this.stats.totalBatches,
      errors: this.stats.errors,
      circuitBreakerState: this.circuitBreaker.getState(),
    };
  }

  /**
   * Get scanner configuration
   */
  getConfig(): ScannerConfig {
    return { ...this.config };
  }

  /**
   * Update scanner configuration (requires restart)
   */
  updateConfig(config: Partial<ScannerConfig>): void {
    const wasRunning = this.isRunning;

    if (wasRunning) {
      this.stop();
    }

    this.config = { ...this.config, ...config };
    console.log('[ContinuousScanner] Configuration updated:', this.config);

    if (wasRunning) {
      this.start();
    }
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      totalScansTriggered: 0,
      totalSymbolsScanned: 0,
      totalBatchesProcessed: 0,
      lastScanTime: null,
      nextScanTime: null,
      currentBatch: 0,
      totalBatches: 0,
      errors: 0,
    };
    console.log('[ContinuousScanner] Statistics reset');
  }

  /**
   * Health check
   */
  isHealthy(): boolean {
    const cbHealthy = this.circuitBreaker.isHealthy();

    // Healthy if:
    // 1. Circuit breaker is healthy
    // 2. Not too many errors (< 10)
    // 3. If running, last scan was recent (< 2x scan interval)
    if (!cbHealthy) return false;
    if (this.stats.errors >= 10) return false;

    if (this.isRunning && this.stats.lastScanTime) {
      const timeSinceLastScan = Date.now() - this.stats.lastScanTime.getTime();
      const maxAllowedGap = this.config.scanIntervalMs * 2;

      if (timeSinceLastScan > maxAllowedGap) {
        return false; // Scanner stuck
      }
    }

    return true;
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

const continuousScannerService = new ContinuousScannerService();
export default continuousScannerService;
