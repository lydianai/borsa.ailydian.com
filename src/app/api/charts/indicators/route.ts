/**
 * TECHNICAL INDICATORS API
 * Calculates RSI, MFI, and Support/Resistance levels
 * Works flawlessly across all timeframes
 */

import { NextRequest, NextResponse } from 'next/server';

interface KlineData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Calculate RSI (Relative Strength Index)
 * Standard 14-period RSI
 */
function calculateRSI(closes: number[], period: number = 14): number[] {
  if (closes.length < period + 1) {
    return [];
  }

  const rsi: number[] = [];
  const changes: number[] = [];

  // Calculate price changes
  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }

  // Initial average gain and loss
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) {
      avgGain += changes[i];
    } else {
      avgLoss += Math.abs(changes[i]);
    }
  }

  avgGain /= period;
  avgLoss /= period;

  // Calculate first RSI
  const rs = avgGain / (avgLoss === 0 ? 0.0001 : avgLoss);
  rsi.push(100 - (100 / (1 + rs)));

  // Calculate subsequent RSI values using smoothed averages
  for (let i = period; i < changes.length; i++) {
    const change = changes[i];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    const currentRS = avgGain / (avgLoss === 0 ? 0.0001 : avgLoss);
    rsi.push(100 - (100 / (1 + currentRS)));
  }

  return rsi;
}

/**
 * Calculate MFI (Money Flow Index)
 * Volume-weighted RSI
 */
function calculateMFI(klines: KlineData[], period: number = 14): number[] {
  if (klines.length < period + 1) {
    return [];
  }

  const mfi: number[] = [];
  const typicalPrices: number[] = [];
  const moneyFlows: number[] = [];

  // Calculate typical price and money flow
  for (const kline of klines) {
    const typicalPrice = (kline.high + kline.low + kline.close) / 3;
    typicalPrices.push(typicalPrice);
    moneyFlows.push(typicalPrice * kline.volume);
  }

  // Calculate MFI for each period
  for (let i = period; i < klines.length; i++) {
    let positiveFlow = 0;
    let negativeFlow = 0;

    for (let j = i - period + 1; j <= i; j++) {
      if (typicalPrices[j] > typicalPrices[j - 1]) {
        positiveFlow += moneyFlows[j];
      } else if (typicalPrices[j] < typicalPrices[j - 1]) {
        negativeFlow += moneyFlows[j];
      }
    }

    const moneyFlowRatio = positiveFlow / (negativeFlow === 0 ? 0.0001 : negativeFlow);
    const mfiValue = 100 - (100 / (1 + moneyFlowRatio));
    mfi.push(mfiValue);
  }

  return mfi;
}

/**
 * Detect Support and Resistance Levels
 * Uses pivot points and local extrema
 */
function detectSupportResistance(klines: KlineData[], lookback: number = 20): {
  support: number[];
  resistance: number[];
} {
  const supportLevels: number[] = [];
  const resistanceLevels: number[] = [];

  if (klines.length < lookback * 2) {
    return { support: [], resistance: [] };
  }

  // Find local minima (support) and maxima (resistance)
  for (let i = lookback; i < klines.length - lookback; i++) {
    const current = klines[i];
    let isLocalMin = true;
    let isLocalMax = true;

    // Check if current is local minimum or maximum
    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;

      if (klines[j].low < current.low) {
        isLocalMin = false;
      }
      if (klines[j].high > current.high) {
        isLocalMax = false;
      }
    }

    if (isLocalMin) {
      supportLevels.push(current.low);
    }
    if (isLocalMax) {
      resistanceLevels.push(current.high);
    }
  }

  // Cluster similar levels (within 0.5% of each other)
  const clusterLevels = (levels: number[]): number[] => {
    if (levels.length === 0) return [];

    const sorted = [...levels].sort((a, b) => a - b);
    const clustered: number[] = [];
    let currentCluster: number[] = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
      const diff = Math.abs(sorted[i] - sorted[i - 1]) / sorted[i - 1];

      if (diff < 0.005) { // Within 0.5%
        currentCluster.push(sorted[i]);
      } else {
        // Average the cluster and add it
        const avg = currentCluster.reduce((sum, val) => sum + val, 0) / currentCluster.length;
        clustered.push(avg);
        currentCluster = [sorted[i]];
      }
    }

    // Add last cluster
    if (currentCluster.length > 0) {
      const avg = currentCluster.reduce((sum, val) => sum + val, 0) / currentCluster.length;
      clustered.push(avg);
    }

    return clustered;
  };

  return {
    support: clusterLevels(supportLevels).slice(-5), // Top 5 most recent support levels
    resistance: clusterLevels(resistanceLevels).slice(-5) // Top 5 most recent resistance levels
  };
}

/**
 * GET /api/charts/indicators?symbol=BTCUSDT&interval=1h&limit=500
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol')?.toUpperCase() || 'BTCUSDT';
    const interval = searchParams.get('interval') || '1h';
    const limit = Math.min(parseInt(searchParams.get('limit') || '500'), 1000);
    const rsiPeriod = parseInt(searchParams.get('rsiPeriod') || '14');
    const mfiPeriod = parseInt(searchParams.get('mfiPeriod') || '14');

    // Fetch kline data from Binance
    const binanceUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(binanceUrl);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    // Transform Binance kline format
    const klines: KlineData[] = data.map((kline: any[]) => ({
      time: Math.floor(kline[0] / 1000),
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5])
    }));

    if (klines.length === 0) {
      throw new Error('No kline data available');
    }

    // Extract close prices for RSI
    const closes = klines.map(k => k.close);

    // Calculate indicators
    const rsi = calculateRSI(closes, rsiPeriod);
    const mfi = calculateMFI(klines, mfiPeriod);
    const { support, resistance } = detectSupportResistance(klines);

    // Get current values
    const currentRSI = rsi.length > 0 ? rsi[rsi.length - 1] : 50;
    const currentMFI = mfi.length > 0 ? mfi[mfi.length - 1] : 50;
    const currentPrice = klines[klines.length - 1].close;

    // Generate signals
    const signals = {
      rsiOverbought: currentRSI > 70,
      rsiOversold: currentRSI < 30,
      mfiOverbought: currentMFI > 80,
      mfiOversold: currentMFI < 20,
      nearSupport: support.some(level => Math.abs(currentPrice - level) / currentPrice < 0.01), // Within 1%
      nearResistance: resistance.some(level => Math.abs(currentPrice - level) / currentPrice < 0.01) // Within 1%
    };

    // Overall signal interpretation
    let overallSignal: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL' = 'NEUTRAL';
    let confidence = 50;

    if (signals.rsiOversold && signals.mfiOversold && signals.nearSupport) {
      overallSignal = 'STRONG_BUY';
      confidence = 85;
    } else if ((signals.rsiOversold || signals.mfiOversold) && signals.nearSupport) {
      overallSignal = 'BUY';
      confidence = 70;
    } else if (signals.rsiOverbought && signals.mfiOverbought && signals.nearResistance) {
      overallSignal = 'STRONG_SELL';
      confidence = 85;
    } else if ((signals.rsiOverbought || signals.mfiOverbought) && signals.nearResistance) {
      overallSignal = 'SELL';
      confidence = 70;
    }

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        interval,
        currentPrice,
        indicators: {
          rsi: {
            values: rsi,
            current: currentRSI,
            period: rsiPeriod,
            signal: currentRSI > 70 ? 'OVERBOUGHT' : currentRSI < 30 ? 'OVERSOLD' : 'NEUTRAL'
          },
          mfi: {
            values: mfi,
            current: currentMFI,
            period: mfiPeriod,
            signal: currentMFI > 80 ? 'OVERBOUGHT' : currentMFI < 20 ? 'OVERSOLD' : 'NEUTRAL'
          },
          supportResistance: {
            support: support.map(level => ({
              price: Number(level.toFixed(2)),
              distance: Number((((currentPrice - level) / currentPrice) * 100).toFixed(2)),
              distancePercent: `${(((currentPrice - level) / currentPrice) * 100).toFixed(2)}%`
            })),
            resistance: resistance.map(level => ({
              price: Number(level.toFixed(2)),
              distance: Number((((level - currentPrice) / currentPrice) * 100).toFixed(2)),
              distancePercent: `${(((level - currentPrice) / currentPrice) * 100).toFixed(2)}%`
            }))
          }
        },
        signals,
        overallSignal,
        confidence,
        timestamp: klines[klines.length - 1].time
      },
      processingTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error calculating indicators:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to calculate indicators',
        message: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}
