/**
 * ⏰ OTONOM TARAMA ZAMANLAYICI - LyTrade LYDIAN 24/7
 *
 * Node-cron ile otomatik tarama işlerini başlatır.
 * Redis + BullMQ kuyruğuna job ekler.
 *
 * Tarama Aralıkları:
 * - Her 5 dakika: Ultra-aktif coinler ($10M+ hacim, 50 coin)
 * - Her 15 dakika: Orta-volatilite coinler ($5M+ hacim, 100 coin)
 * - Her 1 saat: Tüm 600+ coin, AI öğrenme aktif
 * - Her 4 saat: Strateji performans değerlendirmesi
 */

import cron from 'node-cron';
import { getAutonomousQueue } from '../queue/autonomous-queue';

// Cron job referansları
const cronJobs: cron.ScheduledTask[] = [];

/**
 * Tüm otonom tarama cron job'larını başlat
 */
export function startAutonomousScanner() {
  console.log('\n🤖 AUTONOMOUS SCANNER: Starting 24/7 automated scans...\n');

  // 1. Her 5 dakikada bir: Ultra-aktif coinler
  const ultraActiveCron = cron.schedule('*/5 * * * *', async () => {
    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] 5-Minute Ultra-Active Scan triggered`);

    try {
      const autonomousQueue = getAutonomousQueue();
      const job = await autonomousQueue.add('ultra-active-scan', {
        timeframe: '5m',
        minVolume: 10_000_000, // $10M+
        coinCount: 50,
        aiEnhanced: true,
      });

      console.log(`✅ Job enqueued: ${job.id}`);
    } catch (error: any) {
      console.error(`❌ Failed to enqueue ultra-active-scan:`, error.message);
    }
  });

  cronJobs.push(ultraActiveCron);
  console.log('✅ Cron: Ultra-Active Scan (*/5 * * * *) - Every 5 minutes');

  // 2. Her 15 dakikada bir: Orta-volatilite coinler
  const mediumVolatilityCron = cron.schedule('*/15 * * * *', async () => {
    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] 15-Minute Medium-Volatility Scan triggered`);

    try {
      const autonomousQueue = getAutonomousQueue();
      const job = await autonomousQueue.add('medium-volatility-scan', {
        timeframe: '15m',
        minVolume: 5_000_000, // $5M+
        coinCount: 100,
      });

      console.log(`✅ Job enqueued: ${job.id}`);
    } catch (error: any) {
      console.error(`❌ Failed to enqueue medium-volatility-scan:`, error.message);
    }
  });

  cronJobs.push(mediumVolatilityCron);
  console.log('✅ Cron: Medium-Volatility Scan (*/15 * * * *) - Every 15 minutes');

  // 3. Her 1 saatte bir: Tüm 600+ coin, AI öğrenme
  const fullMarketCron = cron.schedule('0 * * * *', async () => {
    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] Hourly Full-Market Scan triggered`);

    try {
      const autonomousQueue = getAutonomousQueue();
      const job = await autonomousQueue.add('full-market-scan', {
        timeframe: '1h',
        minVolume: 1_000_000, // $1M+
        coinCount: 600,
        aiEnhanced: true,
        strategyLearning: true, // AI öğrenme aktif
      });

      console.log(`✅ Job enqueued: ${job.id}`);
    } catch (error: any) {
      console.error(`❌ Failed to enqueue full-market-scan:`, error.message);
    }
  });

  cronJobs.push(fullMarketCron);
  console.log('✅ Cron: Full-Market Scan (0 * * * *) - Every hour');

  // 4. Her 4 saatte bir: Strateji performans değerlendirmesi
  const performanceReviewCron = cron.schedule('0 */4 * * *', async () => {
    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] Strategy Performance Review triggered`);

    try {
      const autonomousQueue = getAutonomousQueue();
      const job = await autonomousQueue.add('strategy-performance-review', {
        analyzeLastHours: 24,
        updateWeights: true, // Strateji ağırlıklarını güncelle
        generateNewParams: true, // Yeni parametreler öner
      });

      console.log(`✅ Job enqueued: ${job.id}`);
    } catch (error: any) {
      console.error(`❌ Failed to enqueue strategy-performance-review:`, error.message);
    }
  });

  cronJobs.push(performanceReviewCron);
  console.log('✅ Cron: Strategy Performance Review (0 */4 * * *) - Every 4 hours');

  console.log('\n✅ All autonomous cron jobs started successfully!\n');

  return cronJobs;
}

/**
 * Tüm cron job'ları durdur
 */
export function stopAutonomousScanner() {
  console.log('\n⚠️ AUTONOMOUS SCANNER: Stopping all cron jobs...\n');

  cronJobs.forEach((job, index) => {
    job.stop();
    console.log(`✅ Stopped cron job ${index + 1}`);
  });

  cronJobs.length = 0;

  console.log('\n✅ All cron jobs stopped successfully!\n');
}

/**
 * Cron job durumlarını al
 */
export function getAutonomousScannerStatus() {
  return {
    totalJobs: cronJobs.length,
    jobs: cronJobs.map((_job, index) => ({
      index: index + 1,
      running: true, // cronJobs array'inde sadece aktif job'lar tutuluyor
    })),
  };
}

/**
 * Manuel test job'u ekle (development için)
 */
export async function triggerManualScan(
  type: 'ultra-active' | 'medium-volatility' | 'full-market' | 'performance-review'
) {
  console.log(`\n🔧 Manual trigger: ${type}\n`);

  const autonomousQueue = getAutonomousQueue();

  switch (type) {
    case 'ultra-active':
      return await autonomousQueue.add('ultra-active-scan', {
        timeframe: '5m',
        minVolume: 10_000_000,
        coinCount: 50,
        aiEnhanced: true,
      });

    case 'medium-volatility':
      return await autonomousQueue.add('medium-volatility-scan', {
        timeframe: '15m',
        minVolume: 5_000_000,
        coinCount: 100,
      });

    case 'full-market':
      return await autonomousQueue.add('full-market-scan', {
        timeframe: '1h',
        minVolume: 1_000_000,
        coinCount: 600,
        aiEnhanced: true,
        strategyLearning: true,
      });

    case 'performance-review':
      return await autonomousQueue.add('strategy-performance-review', {
        analyzeLastHours: 24,
        updateWeights: true,
        generateNewParams: true,
      });

    default:
      throw new Error(`Unknown scan type: ${type}`);
  }
}

export default {
  startAutonomousScanner,
  stopAutonomousScanner,
  getAutonomousScannerStatus,
  triggerManualScan,
};
