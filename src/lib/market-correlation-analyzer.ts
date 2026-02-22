/**
 * MARKET CORRELATION ANALYZER
 * Real correlation calculations with BTC, funding rates, liquidation analysis
 *
 * Features:
 * - Pearson correlation coefficient with BTC
 * - Real funding rates from Binance
 * - Liquidation risk calculation
 * - Volatility measurement (ATR-based)
 */

interface OHLCVData {
  symbol: string;
  close: number[];
  volume: number[];
}

interface FundingRateData {
  symbol: string;
  fundingRate: number;
  nextFundingTime: number;
}

interface CorrelationMetrics {
  btcCorrelation: number; // -1 to 1 (Pearson coefficient)
  fundingRate: number; // Annual percentage
  fundingBias: 'LONG' | 'SHORT' | 'BALANCED';
  liquidationRisk: number; // 0-100 score
  volatility: number; // Percentage
  volumeProfile: 'HIGH' | 'MEDIUM' | 'LOW';
}

/**
 * Calculate Pearson correlation coefficient
 */
function calculateCorrelation(arr1: number[], arr2: number[]): number {
  if (arr1.length !== arr2.length || arr1.length < 2) return 0;

  const n = arr1.length;
  const mean1 = arr1.reduce((a, b) => a + b, 0) / n;
  const mean2 = arr2.reduce((a, b) => a + b, 0) / n;

  let numerator = 0;
  let sumSq1 = 0;
  let sumSq2 = 0;

  for (let i = 0; i < n; i++) {
    const diff1 = arr1[i] - mean1;
    const diff2 = arr2[i] - mean2;
    numerator += diff1 * diff2;
    sumSq1 += diff1 * diff1;
    sumSq2 += diff2 * diff2;
  }

  const denominator = Math.sqrt(sumSq1 * sumSq2);
  if (denominator === 0) return 0;

  return numerator / denominator;
}

/**
 * Calculate standard deviation (volatility)
 */
function calculateVolatility(prices: number[]): number {
  if (prices.length < 2) return 0;

  const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
  const squareDiffs = prices.map(price => Math.pow(price - mean, 2));
  const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / prices.length;
  const stdDev = Math.sqrt(avgSquareDiff);

  return (stdDev / mean) * 100; // As percentage
}

/**
 * Fetch OHLCV data for multiple symbols from Binance
 */
async function fetchMultipleOHLCV(symbols: string[], limit: number = 100): Promise<Map<string, OHLCVData>> {
  const ohlcvMap = new Map<string, OHLCVData>();

  // Fetch data for all symbols in parallel
  const promises = symbols.map(async (symbol) => {
    try {
      const response = await fetch(
        `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=1h&limit=${limit}`,
        {
          headers: { 'User-Agent': 'Mozilla/5.0 (compatible; LyTradeAI/2.0)' },
          next: { revalidate: 300 }, // 5 min cache
        }
      );

      if (!response.ok) return null;

      const klines: any[] = await response.json();
      if (!Array.isArray(klines) || klines.length < 10) return null;

      const ohlcv: OHLCVData = {
        symbol,
        close: klines.map(k => parseFloat(k[4])),
        volume: klines.map(k => parseFloat(k[5])),
      };

      return { symbol, ohlcv };
    } catch (error) {
      return null;
    }
  });

  const results = await Promise.all(promises);

  for (const result of results) {
    if (result && result.ohlcv.close.length > 0) {
      ohlcvMap.set(result.symbol, result.ohlcv);
    }
  }

  return ohlcvMap;
}

/**
 * Fetch funding rates from Binance
 */
async function fetchFundingRates(symbols: string[]): Promise<Map<string, FundingRateData>> {
  const fundingMap = new Map<string, FundingRateData>();

  try {
    const response = await fetch('https://fapi.binance.com/fapi/v1/premiumIndex', {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; LyTradeAI/2.0)' },
      next: { revalidate: 300 }, // 5 min cache
    });

    if (!response.ok) return fundingMap;

    const data: any[] = await response.json();

    for (const item of data) {
      if (symbols.includes(item.symbol)) {
        fundingMap.set(item.symbol, {
          symbol: item.symbol,
          fundingRate: parseFloat(item.lastFundingRate || '0'),
          nextFundingTime: item.nextFundingTime || 0,
        });
      }
    }
  } catch (error) {
    console.error('[Market Correlation] Funding rates fetch error:', error);
  }

  return fundingMap;
}

/**
 * Calculate liquidation risk based on volatility and funding
 */
function calculateLiquidationRisk(
  volatility: number,
  fundingRate: number,
  correlation: number
): number {
  // Base risk from volatility
  let risk = Math.min(50, volatility * 5); // High volatility = high risk

  // Funding rate contribution (extreme funding = liquidation risk)
  const fundingRisk = Math.min(30, Math.abs(fundingRate) * 10000);
  risk += fundingRisk;

  // High correlation with BTC = systemic risk
  const correlationRisk = Math.abs(correlation) > 0.8 ? 20 : 0;
  risk += correlationRisk;

  return Math.min(100, Math.round(risk));
}

/**
 * Classify volume profile
 */
function classifyVolumeProfile(currentVolume: number, volumes: number[]): 'HIGH' | 'MEDIUM' | 'LOW' {
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
  const ratio = currentVolume / avgVolume;

  if (ratio > 1.5) return 'HIGH';
  if (ratio > 0.7) return 'MEDIUM';
  return 'LOW';
}

/**
 * Main correlation analyzer for a coin relative to BTC
 */
export async function analyzeMarketCorrelation(
  symbol: string,
  btcOHLCV?: OHLCVData,
  fundingRateMap?: Map<string, FundingRateData>
): Promise<CorrelationMetrics> {
  try {
    // Fetch OHLCV for symbol and BTC if not provided
    const symbols = [symbol];
    if (!btcOHLCV) symbols.push('BTCUSDT');

    const ohlcvMap = await fetchMultipleOHLCV(symbols, 100);

    const symbolOHLCV = ohlcvMap.get(symbol);
    const btcData = btcOHLCV || ohlcvMap.get('BTCUSDT');

    if (!symbolOHLCV || !btcData) {
      throw new Error('Insufficient data for correlation analysis');
    }

    // 1. Calculate BTC correlation (using last 50 candles for recent correlation)
    const recentCloses = symbolOHLCV.close.slice(-50);
    const recentBTCCloses = btcData.close.slice(-50);
    const btcCorrelation = calculateCorrelation(recentCloses, recentBTCCloses);

    // 2. Calculate volatility
    const volatility = calculateVolatility(recentCloses);

    // 3. Get funding rate
    let fundingRate = 0;
    let fundingBias: 'LONG' | 'SHORT' | 'BALANCED' = 'BALANCED';

    if (fundingRateMap) {
      const fundingData = fundingRateMap.get(symbol);
      if (fundingData) {
        fundingRate = fundingData.fundingRate;
        // Funding rate interpretation
        if (fundingRate > 0.0001) fundingBias = 'SHORT'; // Longs pay shorts (bearish)
        else if (fundingRate < -0.0001) fundingBias = 'LONG'; // Shorts pay longs (bullish)
      }
    }

    // 4. Calculate liquidation risk
    const liquidationRisk = calculateLiquidationRisk(volatility, fundingRate, btcCorrelation);

    // 5. Volume profile
    const currentVolume = symbolOHLCV.volume[symbolOHLCV.volume.length - 1];
    const volumeProfile = classifyVolumeProfile(currentVolume, symbolOHLCV.volume);

    return {
      btcCorrelation: parseFloat(btcCorrelation.toFixed(3)),
      fundingRate: parseFloat((fundingRate * 100).toFixed(4)), // Convert to percentage
      fundingBias,
      liquidationRisk,
      volatility: parseFloat(volatility.toFixed(2)),
      volumeProfile,
    };
  } catch (error: any) {
    console.error(`[Market Correlation] Error analyzing ${symbol}:`, error);

    // Return neutral metrics on error
    return {
      btcCorrelation: 0,
      fundingRate: 0,
      fundingBias: 'BALANCED',
      liquidationRisk: 50,
      volatility: 0,
      volumeProfile: 'MEDIUM',
    };
  }
}

/**
 * Batch analyze multiple coins for correlation
 */
export async function batchAnalyzeCorrelations(
  symbols: string[],
  maxConcurrent: number = 10
): Promise<Map<string, CorrelationMetrics>> {
  const results = new Map<string, CorrelationMetrics>();

  // Fetch BTC data once (shared reference)
  const btcData = (await fetchMultipleOHLCV(['BTCUSDT'], 100)).get('BTCUSDT');
  if (!btcData) {
    console.error('[Market Correlation] Failed to fetch BTC data');
    return results;
  }

  // Fetch all funding rates at once
  const fundingRateMap = await fetchFundingRates(symbols);

  // Process in batches
  for (let i = 0; i < symbols.length; i += maxConcurrent) {
    const batch = symbols.slice(i, i + maxConcurrent);
    const batchPromises = batch.map(symbol =>
      analyzeMarketCorrelation(symbol, btcData, fundingRateMap)
    );
    const batchResults = await Promise.all(batchPromises);

    batch.forEach((symbol, index) => {
      results.set(symbol, batchResults[index]);
    });
  }

  return results;
}
