/**
 * VOLUME PROFILE API
 * Calculates volume distribution across price levels
 * Shows high-volume nodes (HVN) and low-volume nodes (LVN)
 * Critical for identifying support/resistance levels
 */

import { NextRequest, NextResponse } from 'next/server';

interface VolumeProfileData {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentage: number;
}

/**
 * Calculate Volume Profile from kline data
 */
function calculateVolumeProfile(
  klines: any[],
  priceLevels: number = 50
): VolumeProfileData[] {
  if (klines.length === 0) return [];

  // Find price range
  const prices = klines.flatMap(k => [k.high, k.low]);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceStep = (maxPrice - minPrice) / priceLevels;

  // Initialize price buckets
  const volumeBuckets = new Map<number, { total: number; buy: number; sell: number }>();

  for (let i = 0; i < priceLevels; i++) {
    const priceLevel = minPrice + (i * priceStep);
    volumeBuckets.set(priceLevel, { total: 0, buy: 0, sell: 0 });
  }

  // Distribute volume across price levels
  for (const kline of klines) {
    const { open, close, high, low, volume } = kline;

    // Determine if candle is bullish or bearish
    const isBullish = close >= open;
    const candleRange = high - low;

    if (candleRange === 0) continue;

    // Distribute volume proportionally across the candle's price range
    const relevantBuckets = Array.from(volumeBuckets.keys()).filter(
      price => price >= low && price <= high
    );

    if (relevantBuckets.length === 0) continue;

    const volumePerBucket = volume / relevantBuckets.length;

    for (const priceLevel of relevantBuckets) {
      const bucket = volumeBuckets.get(priceLevel)!;
      bucket.total += volumePerBucket;

      // Distribute buy/sell based on candle color and position
      if (isBullish) {
        // Bullish candle: more buying pressure at higher prices
        const ratio = (priceLevel - low) / candleRange;
        bucket.buy += volumePerBucket * (0.5 + ratio * 0.5);
        bucket.sell += volumePerBucket * (0.5 - ratio * 0.5);
      } else {
        // Bearish candle: more selling pressure at higher prices
        const ratio = (priceLevel - low) / candleRange;
        bucket.sell += volumePerBucket * (0.5 + ratio * 0.5);
        bucket.buy += volumePerBucket * (0.5 - ratio * 0.5);
      }
    }
  }

  // Calculate total volume and convert to array
  const totalVolume = Array.from(volumeBuckets.values())
    .reduce((sum, bucket) => sum + bucket.total, 0);

  const volumeProfile: VolumeProfileData[] = Array.from(volumeBuckets.entries())
    .map(([price, bucket]) => ({
      price: Number(price.toFixed(2)),
      volume: Number(bucket.total.toFixed(2)),
      buyVolume: Number(bucket.buy.toFixed(2)),
      sellVolume: Number(bucket.sell.toFixed(2)),
      percentage: Number(((bucket.total / totalVolume) * 100).toFixed(2))
    }))
    .filter(item => item.volume > 0)
    .sort((a, b) => a.price - b.price);

  return volumeProfile;
}

/**
 * Identify Point of Control (POC), Value Area High (VAH), Value Area Low (VAL)
 */
function identifyKeyLevels(volumeProfile: VolumeProfileData[]): {
  poc: number;
  vah: number;
  val: number;
  valueAreaVolume: number;
} {
  if (volumeProfile.length === 0) {
    return { poc: 0, vah: 0, val: 0, valueAreaVolume: 0 };
  }

  // Find POC (Price level with highest volume)
  const poc = volumeProfile.reduce((max, current) =>
    current.volume > max.volume ? current : max
  ).price;

  // Calculate total volume
  const totalVolume = volumeProfile.reduce((sum, item) => sum + item.volume, 0);
  const valueAreaThreshold = totalVolume * 0.70; // 70% of volume

  // Find Value Area (70% of volume closest to POC)
  const sorted = [...volumeProfile].sort((a, b) => b.volume - a.volume);
  let valueAreaVolume = 0;
  const valueAreaPrices: number[] = [];

  for (const item of sorted) {
    if (valueAreaVolume >= valueAreaThreshold) break;
    valueAreaVolume += item.volume;
    valueAreaPrices.push(item.price);
  }

  const vah = Math.max(...valueAreaPrices);
  const val = Math.min(...valueAreaPrices);

  return {
    poc: Number(poc.toFixed(2)),
    vah: Number(vah.toFixed(2)),
    val: Number(val.toFixed(2)),
    valueAreaVolume: Number(valueAreaVolume.toFixed(2))
  };
}

/**
 * GET /api/charts/volume-profile?symbol=BTCUSDT&interval=1h&limit=500
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol')?.toUpperCase() || 'BTCUSDT';
    const interval = searchParams.get('interval') || '1h';
    const limit = Math.min(parseInt(searchParams.get('limit') || '500'), 1000);
    const priceLevels = Math.min(parseInt(searchParams.get('priceLevels') || '50'), 100);

    // Fetch kline data from Binance
    const binanceUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(binanceUrl);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    // Transform Binance kline format
    const klines = data.map((kline: any[]) => ({
      time: Math.floor(kline[0] / 1000),
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5])
    }));

    // Calculate volume profile
    const volumeProfile = calculateVolumeProfile(klines, priceLevels);
    const keyLevels = identifyKeyLevels(volumeProfile);

    // Find high-volume nodes (HVN) and low-volume nodes (LVN)
    const avgVolume = volumeProfile.reduce((sum, item) => sum + item.volume, 0) / volumeProfile.length;
    const hvnThreshold = avgVolume * 1.5;
    const lvnThreshold = avgVolume * 0.5;

    const hvnLevels = volumeProfile
      .filter(item => item.volume >= hvnThreshold)
      .map(item => item.price);

    const lvnLevels = volumeProfile
      .filter(item => item.volume <= lvnThreshold)
      .map(item => item.price);

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        interval,
        volumeProfile,
        keyLevels,
        hvnLevels, // High-volume nodes (strong support/resistance)
        lvnLevels, // Low-volume nodes (weak support/resistance)
        statistics: {
          totalVolume: volumeProfile.reduce((sum, item) => sum + item.volume, 0),
          avgVolume,
          priceRange: {
            high: Math.max(...volumeProfile.map(v => v.price)),
            low: Math.min(...volumeProfile.map(v => v.price))
          }
        }
      },
      processingTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error calculating volume profile:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to calculate volume profile',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}
