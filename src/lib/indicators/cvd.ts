/**
 * CVD (Cumulative Volume Delta) Indicator
 * Kurumsal trader'ların en çok kullandığı indikatör
 * Alıcı/Satıcı baskısını kümülatif olarak ölçer
 */

export interface CVDData {
  time: number;
  delta: number;           // Bu mumda alım - satım
  cumulativeDelta: number; // Kümülatif delta
  buyVolume: number;       // Alım hacmi
  sellVolume: number;      // Satım hacmi
  divergence?: 'bullish' | 'bearish' | null; // Fiyat-CVD uyuşmazlığı
}

export interface VolumeProfileLevel {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentage: number; // Toplam hacmin yüzdesi
}

export interface VolumeProfileData {
  levels: VolumeProfileLevel[];
  poc: number;           // Point of Control (en yüksek hacim fiyatı)
  valueAreaHigh: number; // Value Area High (%70 hacmin üst sınırı)
  valueAreaLow: number;  // Value Area Low (%70 hacmin alt sınırı)
  totalVolume: number;
}

/**
 * CVD Hesaplama
 * Her mumda alıcı baskısı - satıcı baskısı = Delta
 * Tüm deltaların toplamı = Cumulative Delta
 */
export function calculateCVD(candles: any[]): CVDData[] {
  if (candles.length === 0) return [];

  const cvdData: CVDData[] = [];
  let cumulativeDelta = 0;

  for (let i = 0; i < candles.length; i++) {
    const candle = candles[i];
    const { open, close, high, low, volume } = candle;

    // Mumun yönüne göre alım/satım tahmini
    // Yeşil mum (close > open) = Daha fazla alıcı baskısı
    // Kırmızı mum (close < open) = Daha fazla satıcı baskısı
    const isBullish = close > open;

    // Basit yaklaşım: Mumun body'sine göre oran
    const bodySize = Math.abs(close - open);
    const range = high - low;
    const bodyRatio = range > 0 ? bodySize / range : 0.5;

    let buyVolume: number;
    let sellVolume: number;

    if (isBullish) {
      // Yeşil mumda alıcılar daha güçlü
      buyVolume = volume * (0.5 + bodyRatio * 0.5);
      sellVolume = volume - buyVolume;
    } else {
      // Kırmızı mumda satıcılar daha güçlü
      sellVolume = volume * (0.5 + bodyRatio * 0.5);
      buyVolume = volume - sellVolume;
    }

    const delta = buyVolume - sellVolume;
    cumulativeDelta += delta;

    cvdData.push({
      time: candle.time,
      delta,
      cumulativeDelta,
      buyVolume,
      sellVolume,
      divergence: null // Divergence daha sonra tespit edilecek
    });
  }

  // Divergence tespiti
  detectCVDDivergence(candles, cvdData);

  return cvdData;
}

/**
 * CVD Divergence Tespiti
 * Fiyat yükseliyor ama CVD düşüyorsa = Bearish Divergence (Satış sinyali)
 * Fiyat düşüyor ama CVD yükseliyor = Bullish Divergence (Alım sinyali)
 */
function detectCVDDivergence(candles: any[], cvdData: CVDData[]): void {
  if (candles.length < 20) return;

  const lookback = 14; // 14 mum geriye bak

  for (let i = lookback; i < candles.length; i++) {
    const currentPrice = candles[i].close;
    const prevPrice = candles[i - lookback].close;
    const currentCVD = cvdData[i].cumulativeDelta;
    const prevCVD = cvdData[i - lookback].cumulativeDelta;

    // Bullish Divergence: Fiyat düşüyor ama CVD yükseliyor
    if (currentPrice < prevPrice && currentCVD > prevCVD) {
      cvdData[i].divergence = 'bullish';
    }
    // Bearish Divergence: Fiyat yükseliyor ama CVD düşüyor
    else if (currentPrice > prevPrice && currentCVD < prevCVD) {
      cvdData[i].divergence = 'bearish';
    }
  }
}

/**
 * Volume Profile Hesaplama
 * Fiyat seviyelerine göre hacim dağılımı
 */
export function calculateVolumeProfile(
  candles: any[],
  priceLevels: number = 50 // Kaç fiyat seviyesi oluşturulacak
): VolumeProfileData {
  if (candles.length === 0) {
    return {
      levels: [],
      poc: 0,
      valueAreaHigh: 0,
      valueAreaLow: 0,
      totalVolume: 0
    };
  }

  // Fiyat aralığını belirle
  const highPrice = Math.max(...candles.map(c => c.high));
  const lowPrice = Math.min(...candles.map(c => c.low));
  const priceRange = highPrice - lowPrice;
  const priceStep = priceRange / priceLevels;

  // Fiyat seviyelerini oluştur
  const volumeMap = new Map<number, { volume: number; buyVolume: number; sellVolume: number }>();

  for (let i = 0; i < priceLevels; i++) {
    const priceLevel = lowPrice + (i * priceStep);
    volumeMap.set(priceLevel, { volume: 0, buyVolume: 0, sellVolume: 0 });
  }

  // Her mumu fiyat seviyelerine dağıt
  let totalVolume = 0;

  for (const candle of candles) {
    const { high, low, open, close, volume } = candle;
    const isBullish = close > open;

    // Bu mumun hangi fiyat seviyelerini kapladığını bul
    const startLevel = Math.floor((low - lowPrice) / priceStep);
    const endLevel = Math.floor((high - lowPrice) / priceStep);

    const touchedLevels = endLevel - startLevel + 1;
    const volumePerLevel = volume / touchedLevels;

    for (let level = startLevel; level <= endLevel && level < priceLevels; level++) {
      const priceLevel = lowPrice + (level * priceStep);
      const existing = volumeMap.get(priceLevel);

      if (existing) {
        existing.volume += volumePerLevel;

        // Alım/Satım dağılımı
        if (isBullish) {
          existing.buyVolume += volumePerLevel * 0.6;
          existing.sellVolume += volumePerLevel * 0.4;
        } else {
          existing.buyVolume += volumePerLevel * 0.4;
          existing.sellVolume += volumePerLevel * 0.6;
        }
      }
    }

    totalVolume += volume;
  }

  // Volume Profile seviyelerini array'e çevir
  const levels: VolumeProfileLevel[] = [];
  volumeMap.forEach((data, price) => {
    levels.push({
      price,
      volume: data.volume,
      buyVolume: data.buyVolume,
      sellVolume: data.sellVolume,
      percentage: totalVolume > 0 ? (data.volume / totalVolume) * 100 : 0
    });
  });

  // En yüksek hacme sahip fiyat = POC (Point of Control)
  const poc = levels.reduce((max, level) =>
    level.volume > max.volume ? level : max
  ).price;

  // Value Area hesapla (%70 hacmin bulunduğu alan)
  const sortedByVolume = [...levels].sort((a, b) => b.volume - a.volume);
  let valueAreaVolume = 0;
  const valueAreaTarget = totalVolume * 0.7;
  const valueAreaPrices: number[] = [];

  for (const level of sortedByVolume) {
    if (valueAreaVolume >= valueAreaTarget) break;
    valueAreaVolume += level.volume;
    valueAreaPrices.push(level.price);
  }

  const valueAreaHigh = Math.max(...valueAreaPrices);
  const valueAreaLow = Math.min(...valueAreaPrices);

  return {
    levels: levels.sort((a, b) => b.price - a.price), // Fiyata göre sırala
    poc,
    valueAreaHigh,
    valueAreaLow,
    totalVolume
  };
}

/**
 * CVD Trend Analizi
 * Güçlü alıcı/satıcı baskısını tespit et
 */
export function analyzeCVDTrend(cvdData: CVDData[], period: number = 14): {
  trend: 'bullish' | 'bearish' | 'neutral';
  strength: number; // 0-100
  recentDelta: number;
} {
  if (cvdData.length < period) {
    return { trend: 'neutral', strength: 0, recentDelta: 0 };
  }

  const recent = cvdData.slice(-period);
  const startCVD = recent[0].cumulativeDelta;
  const endCVD = recent[recent.length - 1].cumulativeDelta;
  const recentDelta = endCVD - startCVD;

  // Trend belirleme
  let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (recentDelta > 0) trend = 'bullish';
  else if (recentDelta < 0) trend = 'bearish';

  // Güç hesaplama (normalize edilmiş)
  const maxAbsDelta = Math.max(...recent.map(d => Math.abs(d.delta)));
  const strength = maxAbsDelta > 0 ? Math.min(100, (Math.abs(recentDelta) / maxAbsDelta) * 100) : 0;

  return { trend, strength, recentDelta };
}
