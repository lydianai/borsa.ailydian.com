/**
 * SIGNAL MONITOR SERVICE
 * Continuously processes data from APIs and generates real-time signals
 * Updates every 5 minutes for all top 100 coins
 */

import { quantumProEngine as _quantumProEngine } from './ai/QuantumProEngine';
import { riskManager } from './ai/RiskManagementModule';
import { realMarketData } from './api/RealMarketDataService';

interface SignalAlert {
  id: string;
  timestamp: number;
  symbol: string;
  action: 'BUY' | 'SELL';
  confidence: number;
  accuracyScore: number;
  detailedReasons: string[];
  fundamentals: any;
  riskScore: number;
  expiresAt: number;
}

export class SignalMonitorService {
  private isRunning: boolean = false;
  private updateInterval: number = 5 * 60 * 1000; // 5 minutes
  private alertHistory: SignalAlert[] = [];
  private activeAlerts: Map<string, SignalAlert> = new Map();

  /**
   * Start continuous monitoring
   */
  start() {
    if (this.isRunning) {
      console.log('⚠️ Signal monitor already running');
      return;
    }

    this.isRunning = true;
    console.log('🚀 Signal Monitor Service started');
    console.log(`   📡 Update interval: ${this.updateInterval / 1000}s`);
    console.log(`   🎯 Monitoring top 100 coins`);
    console.log(`   ✅ Real-time API integration active`);

    // Initial scan
    this.scanMarket();

    // Schedule recurring scans
    setInterval(() => {
      if (this.isRunning) {
        this.scanMarket();
      }
    }, this.updateInterval);
  }

  /**
   * Stop monitoring
   */
  stop() {
    this.isRunning = false;
    console.log('🛑 Signal Monitor Service stopped');
  }

  /**
   * Scan entire market for signals
   */
  private async scanMarket() {
    try {
      console.log('\n🔍 ======== MARKET SCAN STARTED ========');
      console.log(`⏰ ${new Date().toLocaleString()}`);

      // 1. Fetch top 100 coins with real data
      const coins = await realMarketData.getTop100Coins();
      console.log(`📊 Fetched ${coins.length} coins from APIs`);

      // 2. Generate signals for all coins
      // TODO: Re-enable after implementing generateAllSignals method
      const signals: any[] = []; // await quantumProEngine.generateAllSignals();
      console.log(`🎯 Generated ${signals.length} Quantum Pro signals (temporarily disabled)`);

      // 3. Process each signal
      let newAlerts = 0;
      let updatedAlerts = 0;

      for (const signal of signals) {
        // Skip if not strong enough
        if (signal.confidence < 0.70 || signal.action === 'NEUTRAL') {
          continue;
        }

        // Check risk management
        const riskCheck = riskManager.checkSignalRisk({
          symbol: signal.symbol,
          action: signal.action,
          confidence: signal.confidence,
          riskScore: signal.riskScore,
          price: 0 // Would get from market data
        });

        if (!riskCheck.approved) {
          console.log(`⛔ ${signal.symbol}: Signal rejected by risk manager - ${riskCheck.reason}`);
          continue;
        }

        // Create or update alert
        const alertId = `${signal.symbol}_${signal.action}`;
        const existingAlert = this.activeAlerts.get(alertId);

        const alert: SignalAlert = {
          id: alertId,
          timestamp: signal.timestamp,
          symbol: signal.symbol,
          action: signal.action,
          confidence: signal.confidence,
          accuracyScore: signal.accuracyScore,
          detailedReasons: signal.detailedReasons,
          fundamentals: signal.fundamentals,
          riskScore: signal.riskScore,
          expiresAt: Date.now() + (4 * 60 * 60 * 1000) // Expires in 4 hours
        };

        if (existingAlert) {
          // Update existing alert
          this.activeAlerts.set(alertId, alert);
          updatedAlerts++;
          console.log(`🔄 ${signal.symbol}: Alert updated (${(signal.confidence * 100).toFixed(1)}% confidence, ${(signal.accuracyScore * 100).toFixed(1)}% accuracy)`);
        } else {
          // New alert
          this.activeAlerts.set(alertId, alert);
          this.alertHistory.push(alert);
          newAlerts++;

          // Log detailed alert
          console.log(`\n🚨 ======== NEW ${signal.action} ALERT ========`);
          console.log(`💎 ${signal.symbol}`);
          console.log(`📊 Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
          console.log(`🎯 Accuracy Score: ${(signal.accuracyScore * 100).toFixed(1)}%`);
          console.log(`⚠️ Risk Score: ${(signal.riskScore * 100).toFixed(1)}%`);
          console.log(`✅ Confirmations: ${signal.confirmationCount}/${signal.timeframes.length} timeframes`);

          if (signal.fundamentals) {
            console.log(`💰 Market Cap: $${(signal.fundamentals.marketCap / 1000000000).toFixed(2)}B`);
            console.log(`📈 24h Change: ${signal.fundamentals.priceChange24h.toFixed(2)}%`);
            console.log(`📊 24h Volume: $${(signal.fundamentals.volume24h / 1000000).toFixed(2)}M`);
          }

          console.log(`\n📝 DETAILED REASONS:`);
          signal.detailedReasons.forEach(reason => {
            console.log(reason);
          });
          console.log(`=====================================\n`);

          // Trigger webhook/notification (if configured)
          this.sendNotification(alert);
        }
      }

      // 4. Clean up expired alerts
      const expiredCount = this.cleanupExpiredAlerts();

      // 5. Summary
      console.log(`\n📈 SCAN SUMMARY:`);
      console.log(`  🆕 New Alerts: ${newAlerts}`);
      console.log(`  🔄 Updated Alerts: ${updatedAlerts}`);
      console.log(`  ⏳ Expired Alerts: ${expiredCount}`);
      console.log(`  📊 Active Alerts: ${this.activeAlerts.size}`);
      console.log(`  📚 Total History: ${this.alertHistory.length}`);
      console.log(`======== MARKET SCAN COMPLETED ========\n`);

    } catch (error) {
      console.error('❌ Market scan error:', error);
    }
  }

  /**
   * Clean up expired alerts
   */
  private cleanupExpiredAlerts(): number {
    const now = Date.now();
    let expiredCount = 0;

    this.activeAlerts.forEach((alert, key) => {
      if (alert.expiresAt < now) {
        this.activeAlerts.delete(key);
        expiredCount++;
      }
    });

    return expiredCount;
  }

  /**
   * Send notification (webhook, email, etc.)
   */
  private sendNotification(alert: SignalAlert) {
    // TODO: Integrate with notification services
    // - Discord webhook
    // - Telegram bot
    // - Email service
    // - Mobile push notifications

    console.log(`🔔 Notification would be sent for ${alert.symbol} ${alert.action} signal`);
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): SignalAlert[] {
    return Array.from(this.activeAlerts.values());
  }

  /**
   * Get alert history
   */
  getAlertHistory(limit: number = 100): SignalAlert[] {
    return this.alertHistory.slice(-limit);
  }

  /**
   * Get stats
   */
  getStats() {
    const activeAlerts = this.getActiveAlerts();
    const buySignals = activeAlerts.filter(a => a.action === 'BUY').length;
    const sellSignals = activeAlerts.filter(a => a.action === 'SELL').length;

    const avgConfidence = activeAlerts.length > 0
      ? activeAlerts.reduce((sum, a) => sum + a.confidence, 0) / activeAlerts.length
      : 0;

    const avgAccuracy = activeAlerts.length > 0
      ? activeAlerts.reduce((sum, a) => sum + a.accuracyScore, 0) / activeAlerts.length
      : 0;

    return {
      isRunning: this.isRunning,
      updateInterval: this.updateInterval / 1000,
      activeAlerts: this.activeAlerts.size,
      totalHistory: this.alertHistory.length,
      buySignals,
      sellSignals,
      avgConfidence: (avgConfidence * 100).toFixed(1),
      avgAccuracy: (avgAccuracy * 100).toFixed(1),
      lastScanTime: activeAlerts.length > 0
        ? new Date(Math.max(...activeAlerts.map(a => a.timestamp))).toLocaleString()
        : 'Never'
    };
  }

  /**
   * Force immediate scan
   */
  async forceScan() {
    console.log('⚡ Force scanning market...');
    await this.scanMarket();
  }
}

// Singleton instance
export const signalMonitor = new SignalMonitorService();

// Auto-start in production
if (process.env.NODE_ENV === 'production' || process.env.AUTO_START_MONITOR === 'true') {
  signalMonitor.start();
}