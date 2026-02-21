#!/usr/bin/env node
/**
 * 🤖 TELEGRAM CRON SERVICE
 * PM2 ile 7/24 çalışan otomatik scheduler servisi
 *
 * Kullanım:
 * - Development: ts-node cron-service.ts
 * - Production: pm2 start ecosystem.config.js
 *
 * Özellikler:
 * - 1 saatlik: Market Correlation signals
 * - 4 saatlik: Omnipotent Futures + Crypto News
 * - Günlük: Nirvana + BTC-ETH + News özeti
 * - Haftalık: Nirvana haftalık özet
 */

import cron from 'node-cron';
import {
  runHourlyScheduler,
  run4HourlyScheduler,
  runDailyScheduler,
  runWeeklyScheduler,
} from './telegram-signal-scheduler';

// ============================================================================
// ENVIRONMENT CHECKS
// ============================================================================

function validateEnvironment() {
  const required = [
    'TELEGRAM_BOT_TOKEN',
    'TELEGRAM_ALLOWED_CHAT_IDS',
  ];

  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    console.error('❌ Eksik environment variables:', missing.join(', '));
    console.error('Lütfen .env.local dosyasını kontrol edin!');
    process.exit(1);
  }

  console.log('✅ Environment variables kontrol edildi');
}

// ============================================================================
// CRON JOBS
// ============================================================================

function startCronJobs() {
  console.log('🚀 Telegram Cron Service başlatılıyor...\n');

  validateEnvironment();

  // 1️⃣ SAATLİK SCHEDULER (Her saat başı)
  // Market Correlation yüksek confidence sinyalleri
  cron.schedule('0 * * * *', async () => {
    const now = new Date().toLocaleString('tr-TR');
    console.log(`\n⏰ [${now}] 1 Saatlik Scheduler Tetiklendi`);
    try {
      await runHourlyScheduler();
    } catch (error: any) {
      console.error('❌ 1 Saatlik Scheduler Hatası:', error.message);
    }
  });

  // 2️⃣ 4 SAATLİK SCHEDULER (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
  // Omnipotent Futures + Crypto News
  cron.schedule('0 */4 * * *', async () => {
    const now = new Date().toLocaleString('tr-TR');
    console.log(`\n⏰ [${now}] 4 Saatlik Scheduler Tetiklendi`);
    try {
      await run4HourlyScheduler();
    } catch (error: any) {
      console.error('❌ 4 Saatlik Scheduler Hatası:', error.message);
    }
  });

  // 3️⃣ GÜNLÜK SCHEDULER (UTC 00:00 = Türkiye 03:00)
  // Nirvana Dashboard + BTC-ETH Analysis + News Özeti
  cron.schedule('0 0 * * *', async () => {
    const now = new Date().toLocaleString('tr-TR');
    console.log(`\n⏰ [${now}] Günlük Scheduler Tetiklendi`);
    try {
      await runDailyScheduler();
    } catch (error: any) {
      console.error('❌ Günlük Scheduler Hatası:', error.message);
    }
  });

  // 4️⃣ HAFTALIK SCHEDULER (Pazartesi UTC 00:00 = Türkiye 03:00)
  // Nirvana Haftalık Özet
  cron.schedule('0 0 * * 1', async () => {
    const now = new Date().toLocaleString('tr-TR');
    console.log(`\n⏰ [${now}] Haftalık Scheduler Tetiklendi`);
    try {
      await runWeeklyScheduler();
    } catch (error: any) {
      console.error('❌ Haftalık Scheduler Hatası:', error.message);
    }
  });

  console.log('\n✅ Tüm Cron Job\'lar aktif edildi!\n');
  console.log('📅 Scheduler Takvimi:');
  console.log('  - 🕐 Saatlik: Market Correlation (Her saat başı)');
  console.log('  - 🕓 4 Saatlik: Omnipotent Futures + News (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)');
  console.log('  - 📅 Günlük: Nirvana + BTC-ETH + News Özeti (UTC 00:00 / TR 03:00)');
  console.log('  - 📆 Haftalık: Nirvana Özet (Pazartesi UTC 00:00 / TR 03:00)');
  console.log('\n⏳ Cron servisi çalışıyor... (Ctrl+C ile durdur)\n');
}

// ============================================================================
// GRACEFUL SHUTDOWN
// ============================================================================

process.on('SIGINT', () => {
  console.log('\n\n🛑 Cron servisi durduruluyor...');
  console.log('✅ Güvenli bir şekilde kapatıldı.');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n\n🛑 Cron servisi PM2 tarafından durduruldu.');
  process.exit(0);
});

// ============================================================================
// START SERVICE
// ============================================================================

startCronJobs();
