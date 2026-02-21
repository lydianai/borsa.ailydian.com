/**
 * 📊 VOLUME PROFILE ANALYZER LIBRARY
 * POC (Point of Control), VAH/VAL (Value Area), VWAP hesaplamaları
 *
 * ✅ GERÇEK VERİ: Binance kline data kullanır
 * ❌ Demo/Mock veri YOK
 *
 * White-Hat Compliance:
 * - Eğitim amaçlıdır
 * - Finansal tavsiye değildir
 * - Volume Profile analysis için kullanılır
 */

import { fetchBinanceKlines, type Candlestick } from './technical-indicators';

// ============================================================================
// INTERFACES
// ============================================================================

export interface PriceLevel {
  price: number;
  volume: number;
  percentage: number; // % of total volume
}

export interface ValueArea {
  high: number; // VAH (Value Area High)
  low: number; // VAL (Value Area Low)
  percentage: number; // Usually 70% of total volume
  volumeInArea: number;
}

export interface VolumeProfile {
  // Point of Control (highest volume price level)
  poc: {
    price: number;
    volume: number;
    percentage: number;
  };

  // Value Area (70% of volume distribution)
  valueArea: ValueArea;

  // VWAP (Volume Weighted Average Price)
  vwap: {
    price: number;
    deviation: number; // Current price deviation from VWAP (%)
  };

  // Volume distribution histogram (price levels with volume)
  distribution: PriceLevel[];

  // Current price relative to value area
  pricePosition: 'ABOVE_VAH' | 'IN_VALUE_AREA' | 'BELOW_VAL';

  // Volume-based support/resistance
  volumeNodes: {
    highVolumeNodes: PriceLevel[]; // High volume = strong S/R
    lowVolumeNodes: PriceLevel[]; // Low volume = weak S/R
  };

  // Metadata
  totalVolume: number;
  numPriceLevels: number;
  timeframe: string;
  timestamp: string;
}

// ============================================================================
// VOLUME PROFILE CALCULATION
// ============================================================================

/**
 * Calculate Volume Profile from candlestick data
 *
 * @param symbol - Trading pair (e.g., BTCUSDT)
 * @param interval - Kline interval (e.g., '1h', '4h', '1d')
 * @param limit - Number of candles to analyze (default: 100)
 * @param priceBins - Number of price levels (default: 50)
 */
export async function calculateVolumeProfile(
  symbol: string,
  interval: string = '1h',
  limit: number = 100,
  priceBins: number = 50
): Promise<VolumeProfile> {
  try {
    // 1. Fetch candlestick data from Binance
    const klines = await fetchBinanceKlines(symbol, interval, limit);

    if (klines.length === 0) {
      throw new Error(`No kline data for ${symbol}`);
    }

    // 2. Find price range (min/max)
    const highPrices = klines.map(k => k.high);
    const lowPrices = klines.map(k => k.low);
    const minPrice = Math.min(...lowPrices);
    const maxPrice = Math.max(...highPrices);
    const priceRange = maxPrice - minPrice;
    const binSize = priceRange / priceBins;

    // 3. Initialize price bins
    const volumeBins: Map<number, number> = new Map();
    for (let i = 0; i < priceBins; i++) {
      const binPrice = minPrice + (binSize * i) + (binSize / 2); // Center of bin
      volumeBins.set(binPrice, 0);
    }

    // 4. Distribute volume across price levels
    let totalVolume = 0;
    let vwapNumerator = 0; // Σ(Typical Price × Volume)

    klines.forEach(candle => {
      const typicalPrice = (candle.high + candle.low + candle.close) / 3;
      const volume = candle.volume;

      totalVolume += volume;
      vwapNumerator += typicalPrice * volume;

      // Find which bin this candle belongs to
      const binIndex = Math.min(
        Math.floor((typicalPrice - minPrice) / binSize),
        priceBins - 1
      );
      const binPrice = minPrice + (binSize * binIndex) + (binSize / 2);

      // Add volume to bin
      const currentVolume = volumeBins.get(binPrice) || 0;
      volumeBins.set(binPrice, currentVolume + volume);
    });

    // 5. Calculate VWAP
    const vwapPrice = vwapNumerator / totalVolume;
    const currentPrice = klines[klines.length - 1].close;
    const vwapDeviation = ((currentPrice - vwapPrice) / vwapPrice) * 100;

    // 6. Convert bins to sorted distribution
    const distribution: PriceLevel[] = Array.from(volumeBins.entries())
      .map(([price, volume]) => ({
        price,
        volume,
        percentage: (volume / totalVolume) * 100,
      }))
      .sort((a, b) => b.volume - a.volume); // Sort by volume (highest first)

    // 7. Find POC (Point of Control) - highest volume price level
    const poc = distribution[0];

    // 8. Calculate Value Area (70% of volume)
    const valueAreaPercentage = 70;
    const targetVolume = totalVolume * (valueAreaPercentage / 100);

    // Start from POC and expand up/down until we reach 70% volume
    const sortedByPrice = [...distribution].sort((a, b) => a.price - b.price);
    const pocIndex = sortedByPrice.findIndex(level => level.price === poc.price);

    let valueAreaVolume = poc.volume;
    let vahIndex = pocIndex;
    let valIndex = pocIndex;

    while (valueAreaVolume < targetVolume) {
      const volumeAbove = vahIndex < sortedByPrice.length - 1 ? sortedByPrice[vahIndex + 1].volume : 0;
      const volumeBelow = valIndex > 0 ? sortedByPrice[valIndex - 1].volume : 0;

      if (volumeAbove > volumeBelow && vahIndex < sortedByPrice.length - 1) {
        vahIndex++;
        valueAreaVolume += volumeAbove;
      } else if (valIndex > 0) {
        valIndex--;
        valueAreaVolume += volumeBelow;
      } else if (vahIndex < sortedByPrice.length - 1) {
        vahIndex++;
        valueAreaVolume += volumeAbove;
      } else {
        break;
      }
    }

    const valueArea: ValueArea = {
      high: sortedByPrice[vahIndex].price,
      low: sortedByPrice[valIndex].price,
      percentage: (valueAreaVolume / totalVolume) * 100,
      volumeInArea: valueAreaVolume,
    };

    // 9. Determine price position relative to value area
    let pricePosition: VolumeProfile['pricePosition'];
    if (currentPrice > valueArea.high) {
      pricePosition = 'ABOVE_VAH';
    } else if (currentPrice < valueArea.low) {
      pricePosition = 'BELOW_VAL';
    } else {
      pricePosition = 'IN_VALUE_AREA';
    }

    // 10. Identify high volume nodes (HVN) and low volume nodes (LVN)
    const avgVolume = totalVolume / distribution.length;
    const highVolumeThreshold = avgVolume * 1.5; // 50% above average
    const lowVolumeThreshold = avgVolume * 0.5; // 50% below average

    const highVolumeNodes = distribution
      .filter(level => level.volume >= highVolumeThreshold)
      .slice(0, 5); // Top 5 HVN

    const lowVolumeNodes = distribution
      .filter(level => level.volume <= lowVolumeThreshold)
      .sort((a, b) => a.volume - b.volume)
      .slice(0, 5); // Bottom 5 LVN

    return {
      poc,
      valueArea,
      vwap: {
        price: vwapPrice,
        deviation: vwapDeviation,
      },
      distribution: distribution.slice(0, 20), // Top 20 levels for UI
      pricePosition,
      volumeNodes: {
        highVolumeNodes,
        lowVolumeNodes,
      },
      totalVolume,
      numPriceLevels: priceBins,
      timeframe: interval,
      timestamp: new Date().toISOString(),
    };
  } catch (error: any) {
    console.error(`[Volume Profile] Error for ${symbol}:`, error.message);
    throw error;
  }
}

/**
 * Batch calculate volume profiles for multiple symbols
 */
export async function batchCalculateVolumeProfile(
  symbols: string[],
  interval: string = '1h',
  concurrency: number = 3
): Promise<Map<string, VolumeProfile>> {
  const results = new Map<string, VolumeProfile>();

  // Process in batches to avoid overwhelming Binance API
  for (let i = 0; i < symbols.length; i += concurrency) {
    const batch = symbols.slice(i, i + concurrency);

    const batchResults = await Promise.allSettled(
      batch.map(symbol => calculateVolumeProfile(symbol, interval))
    );

    batchResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        results.set(batch[index], result.value);
      } else {
        console.warn(`[Volume Profile] Failed for ${batch[index]}:`, result.reason);
      }
    });
  }

  return results;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get interpretation of volume profile
 */
export function interpretVolumeProfile(vp: VolumeProfile, currentPrice: number): string {
  const interpretations: string[] = [];

  // 1. POC interpretation
  const pocDistance = ((currentPrice - vp.poc.price) / vp.poc.price) * 100;
  if (Math.abs(pocDistance) < 1) {
    interpretations.push(`Fiyat POC seviyesinde (%${Math.abs(pocDistance).toFixed(2)} mesafe). Güçlü destek/direnç.`);
  } else if (pocDistance > 0) {
    interpretations.push(`Fiyat POC'un %${pocDistance.toFixed(2)} üstünde. POC destek görevi görebilir.`);
  } else {
    interpretations.push(`Fiyat POC'un %${Math.abs(pocDistance).toFixed(2)} altında. POC direnç görevi görebilir.`);
  }

  // 2. Value Area interpretation
  if (vp.pricePosition === 'ABOVE_VAH') {
    interpretations.push(`Fiyat Value Area'nın üstünde. Güçlü yükseliş, ancak aşırı alım riski.`);
  } else if (vp.pricePosition === 'BELOW_VAL') {
    interpretations.push(`Fiyat Value Area'nın altında. Zayıflık, ancak aşırı satım fırsatı.`);
  } else {
    interpretations.push(`Fiyat Value Area içinde (Fair Value). Dengeli fiyatlandırma.`);
  }

  // 3. VWAP interpretation
  if (vp.vwap.deviation > 2) {
    interpretations.push(`VWAP'ın %${vp.vwap.deviation.toFixed(2)} üstünde. Güçlü momentum.`);
  } else if (vp.vwap.deviation < -2) {
    interpretations.push(`VWAP'ın %${Math.abs(vp.vwap.deviation).toFixed(2)} altında. Zayıf momentum.`);
  } else {
    interpretations.push(`VWAP'a yakın (±%${Math.abs(vp.vwap.deviation).toFixed(2)}). Ortalama fiyat seviyesi.`);
  }

  return interpretations.join(' ');
}

// ============================================================================
// WHITE-HAT COMPLIANCE
// ============================================================================

export const VOLUME_PROFILE_CONFIG = {
  DEFAULT_INTERVAL: '1h',
  DEFAULT_LIMIT: 100,
  DEFAULT_BINS: 50,
  VALUE_AREA_PERCENTAGE: 70,
  HVN_THRESHOLD: 1.5, // 50% above average
  LVN_THRESHOLD: 0.5, // 50% below average
};

console.log('✅ Volume Profile Analyzer initialized with White-Hat compliance');
console.log('⚠️ DISCLAIMER: For educational purposes only. Not financial advice.');
