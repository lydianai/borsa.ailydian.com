/**
 * 📊 SIGNAL TRACKER SERVICE
 *
 * Her Telegram bildirimi gönderildiğinde sinyali kaydet:
 * - Unique ID ile signal kayıt
 * - Entry, TP levels, Stop Loss kayıt
 * - 24 saat sonra sonuç kontrolü için timestamp
 * - JSON dosyaya append
 *
 * Kullanım:
 * const { trackSignal } = require('./services/signal-tracker');
 * await trackSignal({ symbol, signal, entry, tp1, tp2, tp3, stopLoss, confidence, ... });
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ===== CONFIGURATION =====
const TRACKING_FILE = path.join(__dirname, '../tracking/signal-history.json');

// ===== UTILITIES =====
function generateSignalId() {
  return `SIG_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
}

function getCheckTime() {
  // 24 saat sonra kontrol edilecek
  const checkTime = new Date();
  checkTime.setHours(checkTime.getHours() + 24);
  return checkTime.toISOString();
}

// ===== TRACK SIGNAL =====
/**
 * Signal'i kaydet
 * @param {Object} signalData - Signal bilgileri
 * @param {string} signalData.symbol - Coin sembolü (örn: BTCUSDT)
 * @param {string} signalData.signalType - AL veya BEKLE
 * @param {number} signalData.entryPrice - Giriş fiyatı
 * @param {number} signalData.tp1 - Take Profit 1
 * @param {number} signalData.tp2 - Take Profit 2
 * @param {number} signalData.tp3 - Take Profit 3
 * @param {number} signalData.stopLoss - Stop Loss
 * @param {number} signalData.confidence - Güven skoru (0-100)
 * @param {string} signalData.source - Sinyal kaynağı (premium, market-briefing, 4h-top10, vb.)
 * @param {string[]} signalData.reasons - AL/BEKLE nedenleri
 * @param {number} signalData.leverage - Kaldıraç önerisi
 * @param {number} signalData.capitalAllocation - Sermaye tahsisi %
 * @returns {Promise<Object>} - Kaydedilen signal objesi
 */
async function trackSignal(signalData) {
  try {
    // 1. Tracking dosyasını oku
    let trackingDb = {
      signals: [],
      metadata: {
        totalSignals: 0,
        lastUpdate: null,
        version: '1.0.0'
      }
    };

    if (fs.existsSync(TRACKING_FILE)) {
      const fileContent = fs.readFileSync(TRACKING_FILE, 'utf8');
      trackingDb = JSON.parse(fileContent);
    }

    // 2. Yeni signal objesi oluştur
    const newSignal = {
      id: generateSignalId(),
      symbol: signalData.symbol,
      signalType: signalData.signalType, // AL veya BEKLE
      entryPrice: signalData.entryPrice,
      tp1: signalData.tp1,
      tp2: signalData.tp2,
      tp3: signalData.tp3,
      stopLoss: signalData.stopLoss,
      confidence: signalData.confidence,
      source: signalData.source, // premium-signals, market-briefing, 4h-top10
      reasons: signalData.reasons || [],
      leverage: signalData.leverage,
      capitalAllocation: signalData.capitalAllocation,
      timestamp: new Date().toISOString(),
      checkTime: getCheckTime(),
      status: 'PENDING', // PENDING, TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, EXPIRED
      outcome: null, // null, SUCCESS, FAILURE
      actualHighPrice: null, // En yüksek fiyat (24h içinde)
      actualLowPrice: null, // En düşük fiyat (24h içinde)
      pnl: null, // Profit/Loss %
      checkedAt: null // Kontrol edildiği zaman
    };

    // 3. Signal'i array'e ekle
    trackingDb.signals.push(newSignal);
    trackingDb.metadata.totalSignals = trackingDb.signals.length;
    trackingDb.metadata.lastUpdate = new Date().toISOString();

    // 4. Dosyaya kaydet
    fs.writeFileSync(TRACKING_FILE, JSON.stringify(trackingDb, null, 2), 'utf8');

    console.log(`✅ Signal kaydedildi: ${newSignal.id} - ${newSignal.symbol} (${newSignal.signalType})`);

    return newSignal;

  } catch (error) {
    console.error(`❌ Signal kaydetme hatası: ${error.message}`);
    throw error;
  }
}

// ===== GET PENDING SIGNALS =====
/**
 * Henüz kontrol edilmemiş sinyalleri getir
 * @returns {Array} - PENDING durumundaki sinyaller
 */
function getPendingSignals() {
  try {
    if (!fs.existsSync(TRACKING_FILE)) {
      return [];
    }

    const fileContent = fs.readFileSync(TRACKING_FILE, 'utf8');
    const trackingDb = JSON.parse(fileContent);

    // Kontrol zamanı gelmiş ve hala PENDING olan sinyaller
    const now = new Date();
    return trackingDb.signals.filter(signal => {
      const checkTime = new Date(signal.checkTime);
      return signal.status === 'PENDING' && checkTime <= now;
    });

  } catch (error) {
    console.error(`❌ Pending signals getirme hatası: ${error.message}`);
    return [];
  }
}

// ===== UPDATE SIGNAL OUTCOME =====
/**
 * Signal sonucunu güncelle
 * @param {string} signalId - Signal ID
 * @param {Object} outcomeData - Sonuç bilgileri
 * @param {string} outcomeData.status - TP1_HIT, TP2_HIT, TP3_HIT, SL_HIT, EXPIRED
 * @param {string} outcomeData.outcome - SUCCESS veya FAILURE
 * @param {number} outcomeData.actualHighPrice - 24h içindeki en yüksek fiyat
 * @param {number} outcomeData.actualLowPrice - 24h içindeki en düşük fiyat
 * @param {number} outcomeData.pnl - Profit/Loss %
 * @returns {boolean} - Başarı durumu
 */
function updateSignalOutcome(signalId, outcomeData) {
  try {
    if (!fs.existsSync(TRACKING_FILE)) {
      console.error(`❌ Tracking dosyası bulunamadı: ${TRACKING_FILE}`);
      return false;
    }

    const fileContent = fs.readFileSync(TRACKING_FILE, 'utf8');
    const trackingDb = JSON.parse(fileContent);

    // Signal'i bul
    const signalIndex = trackingDb.signals.findIndex(s => s.id === signalId);
    if (signalIndex === -1) {
      console.error(`❌ Signal bulunamadı: ${signalId}`);
      return false;
    }

    // Signal'i güncelle
    trackingDb.signals[signalIndex].status = outcomeData.status;
    trackingDb.signals[signalIndex].outcome = outcomeData.outcome;
    trackingDb.signals[signalIndex].actualHighPrice = outcomeData.actualHighPrice;
    trackingDb.signals[signalIndex].actualLowPrice = outcomeData.actualLowPrice;
    trackingDb.signals[signalIndex].pnl = outcomeData.pnl;
    trackingDb.signals[signalIndex].checkedAt = new Date().toISOString();

    trackingDb.metadata.lastUpdate = new Date().toISOString();

    // Dosyaya kaydet
    fs.writeFileSync(TRACKING_FILE, JSON.stringify(trackingDb, null, 2), 'utf8');

    console.log(`✅ Signal sonucu güncellendi: ${signalId} - ${outcomeData.status} (${outcomeData.outcome})`);

    return true;

  } catch (error) {
    console.error(`❌ Signal sonuç güncelleme hatası: ${error.message}`);
    return false;
  }
}

// ===== GET STATISTICS =====
/**
 * İstatistikleri hesapla
 * @param {Object} options - Filtreleme opsiyonları
 * @param {string} options.period - 'today', 'week', 'month', 'all'
 * @param {string} options.source - Signal kaynağı (opsiyonel)
 * @returns {Object} - İstatistik bilgileri
 */
function getStatistics(options = { period: 'today' }) {
  try {
    if (!fs.existsSync(TRACKING_FILE)) {
      return {
        totalSignals: 0,
        successfulSignals: 0,
        failedSignals: 0,
        pendingSignals: 0,
        accuracy: 0,
        avgPnl: 0
      };
    }

    const fileContent = fs.readFileSync(TRACKING_FILE, 'utf8');
    const trackingDb = JSON.parse(fileContent);

    // Tarih filtresi
    let filteredSignals = trackingDb.signals;

    if (options.period !== 'all') {
      const now = new Date();
      const startDate = new Date();

      if (options.period === 'today') {
        startDate.setHours(0, 0, 0, 0);
      } else if (options.period === 'week') {
        startDate.setDate(now.getDate() - 7);
      } else if (options.period === 'month') {
        startDate.setMonth(now.getMonth() - 1);
      }

      filteredSignals = filteredSignals.filter(signal => {
        const signalDate = new Date(signal.timestamp);
        return signalDate >= startDate;
      });
    }

    // Kaynak filtresi
    if (options.source) {
      filteredSignals = filteredSignals.filter(signal => signal.source === options.source);
    }

    // İstatistikleri hesapla
    const totalSignals = filteredSignals.length;
    const successfulSignals = filteredSignals.filter(s => s.outcome === 'SUCCESS').length;
    const failedSignals = filteredSignals.filter(s => s.outcome === 'FAILURE').length;
    const pendingSignals = filteredSignals.filter(s => s.status === 'PENDING').length;

    const completedSignals = successfulSignals + failedSignals;
    const accuracy = completedSignals > 0 ? (successfulSignals / completedSignals * 100).toFixed(2) : 0;

    // Ortalama PnL (sadece tamamlanmış sinyaller)
    const pnlSignals = filteredSignals.filter(s => s.pnl !== null);
    const avgPnl = pnlSignals.length > 0
      ? (pnlSignals.reduce((sum, s) => sum + s.pnl, 0) / pnlSignals.length).toFixed(2)
      : 0;

    return {
      totalSignals,
      successfulSignals,
      failedSignals,
      pendingSignals,
      accuracy: parseFloat(accuracy),
      avgPnl: parseFloat(avgPnl),
      period: options.period,
      source: options.source || 'all'
    };

  } catch (error) {
    console.error(`❌ İstatistik hesaplama hatası: ${error.message}`);
    return null;
  }
}

module.exports = {
  trackSignal,
  getPendingSignals,
  updateSignalOutcome,
  getStatistics
};
