/**
 * VWAP (Volume Weighted Average Price) Indicator
 * Kurumsal trader'ların #1 indikatörü
 *
 * Kullanım: Hedge fonlar, yatırım bankaları, prop firm'ler
 * Amaç: Günlük "adil fiyat" benchmark'ı, kurumsal alım/satım seviyesi
 */

export interface VWAPData {
  time: number;
  vwap: number;           // Volume Weighted Average Price
  upperBand1: number;     // +1 Standart Sapma
  upperBand2: number;     // +2 Standart Sapma
  lowerBand1: number;     // -1 Standart Sapma
  lowerBand2: number;     // -2 Standart Sapma
  cumulativeVolume: number;
  cumulativeTPV: number;  // Typical Price * Volume
}

export interface VWAPAnalysis {
  currentVWAP: number;
  pricePosition: 'above' | 'below' | 'at';  // Fiyatın VWAP'a göre konumu
  distanceFromVWAP: number;                  // VWAP'tan uzaklık (%)
  volumeStrength: 'strong' | 'moderate' | 'weak';
  signal: 'bullish' | 'bearish' | 'neutral';
  bandPosition: '2σ+' | '1σ+' | 'vwap' | '1σ-' | '2σ-';
}

/**
 * VWAP Hesaplama
 *
 * Formula:
 * VWAP = Σ(Typical Price × Volume) / Σ(Volume)
 * Typical Price = (High + Low + Close) / 3
 *
 * @param candles Mum verileri (high, low, close, volume)
 * @param anchor 'day' | 'week' | 'month' - VWAP başlangıç zamanı
 */
export function calculateVWAP(
  candles: any[],
  anchor: 'day' | 'week' | 'month' = 'day'
): VWAPData[] {
  if (candles.length === 0) return [];

  const vwapData: VWAPData[] = [];
  let cumulativeTPV = 0;  // Typical Price × Volume toplamı
  let cumulativeVolume = 0;
  let cumulativeTPSquared = 0;  // (Typical Price)² × Volume toplamı

  // Anchor point belirleme (günlük VWAP için her gün sıfırla)
  let lastAnchorTime = getAnchorTime(candles[0].time, anchor);

  for (let i = 0; i < candles.length; i++) {
    const candle = candles[i];
    const { high, low, close, volume, time } = candle;

    // Anchor zamanı geçtiyse sıfırla (örn: yeni gün başladı)
    const currentAnchorTime = getAnchorTime(time, anchor);
    if (currentAnchorTime !== lastAnchorTime) {
      cumulativeTPV = 0;
      cumulativeVolume = 0;
      cumulativeTPSquared = 0;
      lastAnchorTime = currentAnchorTime;
    }

    // Typical Price hesapla
    const typicalPrice = (high + low + close) / 3;

    // Kümülatif değerleri güncelle
    cumulativeTPV += typicalPrice * volume;
    cumulativeVolume += volume;
    cumulativeTPSquared += typicalPrice * typicalPrice * volume;

    // VWAP hesapla
    const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : close;

    // Standart Sapma hesapla
    const variance = cumulativeVolume > 0
      ? (cumulativeTPSquared / cumulativeVolume) - (vwap * vwap)
      : 0;
    const stdDev = Math.sqrt(Math.max(variance, 0));

    vwapData.push({
      time,
      vwap,
      upperBand1: vwap + stdDev,
      upperBand2: vwap + 2 * stdDev,
      lowerBand1: vwap - stdDev,
      lowerBand2: vwap - 2 * stdDev,
      cumulativeVolume,
      cumulativeTPV
    });
  }

  return vwapData;
}

/**
 * Anchor zamanı belirleme
 * Day: Günün başlangıcı (00:00)
 * Week: Haftanın başlangıcı (Pazartesi 00:00)
 * Month: Ayın başlangıcı (1. gün 00:00)
 */
function getAnchorTime(timestamp: number, anchor: 'day' | 'week' | 'month'): number {
  const date = new Date(timestamp * 1000);

  switch (anchor) {
    case 'day':
      date.setHours(0, 0, 0, 0);
      return Math.floor(date.getTime() / 1000);

    case 'week':
      const day = date.getDay();
      const diff = date.getDate() - day + (day === 0 ? -6 : 1); // Pazartesi'ye git
      date.setDate(diff);
      date.setHours(0, 0, 0, 0);
      return Math.floor(date.getTime() / 1000);

    case 'month':
      date.setDate(1);
      date.setHours(0, 0, 0, 0);
      return Math.floor(date.getTime() / 1000);

    default:
      return timestamp;
  }
}

/**
 * VWAP Analizi
 * Fiyatın VWAP'a göre konumunu ve sinyalleri analiz eder
 */
export function analyzeVWAP(
  currentPrice: number,
  vwapData: VWAPData[],
  recentVolume: number[]
): VWAPAnalysis {
  if (vwapData.length === 0) {
    return {
      currentVWAP: 0,
      pricePosition: 'at',
      distanceFromVWAP: 0,
      volumeStrength: 'weak',
      signal: 'neutral',
      bandPosition: 'vwap'
    };
  }

  const latest = vwapData[vwapData.length - 1];
  const vwap = latest.vwap;

  // Fiyat pozisyonu
  const distance = currentPrice - vwap;
  const distancePercent = (distance / vwap) * 100;
  const pricePosition: 'above' | 'below' | 'at' =
    Math.abs(distancePercent) < 0.1 ? 'at' :
    distancePercent > 0 ? 'above' : 'below';

  // Band pozisyonu
  let bandPosition: '2σ+' | '1σ+' | 'vwap' | '1σ-' | '2σ-' = 'vwap';
  if (currentPrice > latest.upperBand2) bandPosition = '2σ+';
  else if (currentPrice > latest.upperBand1) bandPosition = '1σ+';
  else if (currentPrice < latest.lowerBand2) bandPosition = '2σ-';
  else if (currentPrice < latest.lowerBand1) bandPosition = '1σ-';

  // Hacim gücü analizi (son 14 mum)
  const avgVolume = recentVolume.length > 0
    ? recentVolume.reduce((sum, v) => sum + v, 0) / recentVolume.length
    : 0;
  const latestVolume = recentVolume[recentVolume.length - 1] || 0;
  const volumeRatio = avgVolume > 0 ? latestVolume / avgVolume : 1;

  const volumeStrength: 'strong' | 'moderate' | 'weak' =
    volumeRatio > 1.5 ? 'strong' :
    volumeRatio > 0.8 ? 'moderate' : 'weak';

  // Sinyal analizi
  let signal: 'bullish' | 'bearish' | 'neutral' = 'neutral';

  // Bullish: Fiyat VWAP üzerinde + güçlü hacim
  if (pricePosition === 'above' && volumeStrength !== 'weak') {
    signal = 'bullish';
  }
  // Bearish: Fiyat VWAP altında + güçlü hacim
  else if (pricePosition === 'below' && volumeStrength !== 'weak') {
    signal = 'bearish';
  }
  // Aşırı alım (2σ üstü) = Potansiyel geri dönüş
  else if (bandPosition === '2σ+') {
    signal = 'bearish';
  }
  // Aşırı satım (2σ altı) = Potansiyel geri dönüş
  else if (bandPosition === '2σ-') {
    signal = 'bullish';
  }

  return {
    currentVWAP: vwap,
    pricePosition,
    distanceFromVWAP: Math.abs(distancePercent),
    volumeStrength,
    signal,
    bandPosition
  };
}

/**
 * VWAP Pullback Stratejisi
 * Fiyat VWAP'a yaklaştığında alım/satım fırsatı tespit eder
 */
export function detectVWAPPullback(
  candles: any[],
  vwapData: VWAPData[],
  lookback: number = 5
): {
  detected: boolean;
  type: 'bullish' | 'bearish' | null;
  distance: number;
  confidence: number;
} {
  if (candles.length < lookback || vwapData.length < lookback) {
    return { detected: false, type: null, distance: 0, confidence: 0 };
  }

  const recent = candles.slice(-lookback);
  const recentVWAP = vwapData.slice(-lookback);
  const currentPrice = recent[recent.length - 1].close;
  const currentVWAP = recentVWAP[recentVWAP.length - 1].vwap;

  // VWAP'a yakınlık (%)
  const distance = Math.abs((currentPrice - currentVWAP) / currentVWAP) * 100;

  // %0.5'ten yakınsa pullback potansiyeli var
  if (distance > 0.5) {
    return { detected: false, type: null, distance, confidence: 0 };
  }

  // Trend analizi (son 5 mum)
  const priceAboveVWAP = recent.filter((c, i) => c.close > recentVWAP[i].vwap).length;
  const trend = priceAboveVWAP > lookback / 2 ? 'bullish' : 'bearish';

  // Confidence hesaplama
  const proximityScore = (0.5 - distance) / 0.5; // Ne kadar yakınsa o kadar yüksek
  const trendStrength = Math.abs(priceAboveVWAP - lookback / 2) / (lookback / 2);
  const confidence = Math.min(100, (proximityScore * 60 + trendStrength * 40));

  return {
    detected: true,
    type: trend,
    distance,
    confidence
  };
}

/**
 * VWAP Anchored (Belirli bir noktadan başlayan VWAP)
 * Örnek: Önemli bir haberden sonra, yeni bir swing'den sonra
 */
export function calculateAnchoredVWAP(
  candles: any[],
  startIndex: number
): VWAPData[] {
  if (startIndex >= candles.length || startIndex < 0) {
    return [];
  }

  const anchoredCandles = candles.slice(startIndex);
  return calculateVWAP(anchoredCandles, 'day'); // Day anchor kullan ama starIndex'ten başla
}
