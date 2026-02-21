/**
 * 🔗 MULTI-LAYER CORRELATION ENGINE
 * DXY, S&P500, GOLD, VIX korelasyon analizi
 *
 * White-Hat Compliance:
 * - Yahoo Finance API (public, free)
 * - Proper rate limiting
 * - Error handling
 * - Cache strategy
 */

// ============================================================================
// INTERFACES
// ============================================================================

export interface MacroAssetData {
  symbol: string;
  price: number;
  change24h: number;
  timestamp: number;
}

export interface CorrelationData {
  asset1: string;
  asset2: string;
  correlation: number; // -1 to 1
  strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NONE';
  direction: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
}

export interface CorrelationBreakdown {
  asset1: string;
  asset2: string;
  currentCorrelation: number;
  avgCorrelation: number;
  deviation: number;
  isBreakdown: boolean;
  significance: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface MacroMetrics {
  dxy: MacroAssetData | null;
  sp500: MacroAssetData | null;
  gold: MacroAssetData | null;
  vix: MacroAssetData | null;
  btc: MacroAssetData | null;
}

export interface CorrelationMatrix {
  btcDxy: CorrelationData;
  btcSp500: CorrelationData;
  btcGold: CorrelationData;
  btcVix: CorrelationData;
  timestamp: number;
}

// ============================================================================
// MACRO ASSET FETCHERS
// ============================================================================

/**
 * Fetch DXY (US Dollar Index)
 * Yahoo Finance: DX-Y.NYB
 */
export async function fetchDXY(): Promise<MacroAssetData | null> {
  try {
    // Yahoo Finance API (public endpoint)
    const symbol = 'DX-Y.NYB';
    const response = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=2d`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0',
        },
        next: { revalidate: 3600 }, // 1 hour cache
      }
    );

    if (!response.ok) {
      console.error(`[DXY] Yahoo Finance API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    const quote = data.chart.result[0];
    const prices = quote.indicators.quote[0].close;
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change24h = ((currentPrice - previousPrice) / previousPrice) * 100;

    return {
      symbol: 'DXY',
      price: currentPrice,
      change24h,
      timestamp: Date.now(),
    };
  } catch (error) {
    console.error('[DXY] Error:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

/**
 * Fetch S&P 500 Index
 * Yahoo Finance: ^GSPC
 */
export async function fetchSP500(): Promise<MacroAssetData | null> {
  try {
    const symbol = '^GSPC';
    const response = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=2d`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0',
        },
        next: { revalidate: 3600 }, // 1 hour cache
      }
    );

    if (!response.ok) {
      console.error(`[S&P500] Yahoo Finance API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    const quote = data.chart.result[0];
    const prices = quote.indicators.quote[0].close;
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change24h = ((currentPrice - previousPrice) / previousPrice) * 100;

    return {
      symbol: 'SP500',
      price: currentPrice,
      change24h,
      timestamp: Date.now(),
    };
  } catch (error) {
    console.error('[S&P500] Error:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

/**
 * Fetch GOLD Price
 * Yahoo Finance: GC=F (Gold Futures)
 */
export async function fetchGOLD(): Promise<MacroAssetData | null> {
  try {
    const symbol = 'GC=F';
    const response = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=2d`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0',
        },
        next: { revalidate: 3600 }, // 1 hour cache
      }
    );

    if (!response.ok) {
      console.error(`[GOLD] Yahoo Finance API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    const quote = data.chart.result[0];
    const prices = quote.indicators.quote[0].close;
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change24h = ((currentPrice - previousPrice) / previousPrice) * 100;

    return {
      symbol: 'GOLD',
      price: currentPrice,
      change24h,
      timestamp: Date.now(),
    };
  } catch (error) {
    console.error('[GOLD] Error:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

/**
 * Fetch VIX (Volatility Index)
 * Yahoo Finance: ^VIX
 */
export async function fetchVIX(): Promise<MacroAssetData | null> {
  try {
    const symbol = '^VIX';
    const response = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1d&range=2d`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0',
        },
        next: { revalidate: 3600 }, // 1 hour cache
      }
    );

    if (!response.ok) {
      console.error(`[VIX] Yahoo Finance API error: ${response.status}`);
      return null;
    }

    const data = await response.json();
    const quote = data.chart.result[0];
    const prices = quote.indicators.quote[0].close;
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change24h = ((currentPrice - previousPrice) / previousPrice) * 100;

    return {
      symbol: 'VIX',
      price: currentPrice,
      change24h,
      timestamp: Date.now(),
    };
  } catch (error) {
    console.error('[VIX] Error:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

/**
 * Fetch BTC Price (for correlation calculation)
 */
export async function fetchBTCPrice(): Promise<MacroAssetData | null> {
  try {
    const response = await fetch('https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT', {
      headers: { 'Content-Type': 'application/json' },
      next: { revalidate: 60 }, // 1 minute cache
    });

    if (!response.ok) {
      console.error(`[BTC] Binance API error: ${response.status}`);
      return null;
    }

    const data = await response.json();

    return {
      symbol: 'BTC',
      price: parseFloat(data.lastPrice),
      change24h: parseFloat(data.priceChangePercent),
      timestamp: Date.now(),
    };
  } catch (error) {
    console.error('[BTC] Error:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

// ============================================================================
// CORRELATION CALCULATOR
// ============================================================================

/**
 * Calculate correlation between two assets
 * Uses Pearson correlation coefficient
 */
export function calculateCorrelation(
  values1: number[],
  values2: number[]
): number {
  if (values1.length !== values2.length || values1.length === 0) {
    return 0;
  }

  const n = values1.length;
  const mean1 = values1.reduce((a, b) => a + b, 0) / n;
  const mean2 = values2.reduce((a, b) => a + b, 0) / n;

  let numerator = 0;
  let sumSquares1 = 0;
  let sumSquares2 = 0;

  for (let i = 0; i < n; i++) {
    const diff1 = values1[i] - mean1;
    const diff2 = values2[i] - mean2;
    numerator += diff1 * diff2;
    sumSquares1 += diff1 * diff1;
    sumSquares2 += diff2 * diff2;
  }

  const denominator = Math.sqrt(sumSquares1 * sumSquares2);

  if (denominator === 0) {
    return 0;
  }

  return numerator / denominator;
}

/**
 * Classify correlation strength
 */
export function classifyCorrelation(correlation: number): {
  strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NONE';
  direction: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
} {
  const abs = Math.abs(correlation);
  let strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NONE';
  let direction: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';

  if (abs >= 0.7) {
    strength = 'STRONG';
  } else if (abs >= 0.4) {
    strength = 'MODERATE';
  } else if (abs >= 0.2) {
    strength = 'WEAK';
  } else {
    strength = 'NONE';
  }

  if (correlation > 0.1) {
    direction = 'POSITIVE';
  } else if (correlation < -0.1) {
    direction = 'NEGATIVE';
  } else {
    direction = 'NEUTRAL';
  }

  return { strength, direction };
}

// ============================================================================
// HISTORICAL DATA FETCHER (for correlation calculation)
// ============================================================================

/**
 * Fetch historical prices for correlation calculation
 * Uses 30-day rolling correlation
 */
export async function fetchHistoricalPrices(
  symbol: string,
  days: number = 30
): Promise<number[]> {
  try {
    let yahooSymbol: string = '';
    let useBinance = false;

    switch (symbol) {
      case 'BTC':
        useBinance = true;
        break;
      case 'DXY':
        yahooSymbol = 'DX-Y.NYB';
        break;
      case 'SP500':
        yahooSymbol = '^GSPC';
        break;
      case 'GOLD':
        yahooSymbol = 'GC=F';
        break;
      case 'VIX':
        yahooSymbol = '^VIX';
        break;
      default:
        return [];
    }

    if (useBinance) {
      // Fetch from Binance
      const response = await fetch(
        `https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=${days}`,
        { next: { revalidate: 3600 } }
      );

      if (!response.ok) return [];

      const data = await response.json();
      return data.map((candle: any) => parseFloat(candle[4])); // Close prices
    } else {
      // Fetch from Yahoo Finance
      const response = await fetch(
        `https://query1.finance.yahoo.com/v8/finance/chart/${yahooSymbol}?interval=1d&range=${days}d`,
        {
          headers: { 'User-Agent': 'Mozilla/5.0' },
          next: { revalidate: 3600 },
        }
      );

      if (!response.ok) return [];

      const data = await response.json();
      const prices = data.chart.result[0].indicators.quote[0].close;
      return prices.filter((p: number) => p !== null);
    }
  } catch (error) {
    console.error(`[Historical Prices] Error for ${symbol}:`, error);
    return [];
  }
}

// ============================================================================
// MASTER CORRELATION ENGINE
// ============================================================================

/**
 * Fetch all macro metrics and calculate correlations
 */
export async function fetchCorrelationMatrix(): Promise<{
  macroMetrics: MacroMetrics;
  correlationMatrix: CorrelationMatrix | null;
}> {
  console.log('[Correlation Engine] Fetching macro metrics...');

  // Fetch all macro data in parallel
  const [dxy, sp500, gold, vix, btc] = await Promise.all([
    fetchDXY(),
    fetchSP500(),
    fetchGOLD(),
    fetchVIX(),
    fetchBTCPrice(),
  ]);

  const macroMetrics: MacroMetrics = {
    dxy,
    sp500,
    gold,
    vix,
    btc,
  };

  console.log('[Correlation Engine] Macro metrics fetched');
  console.log(`  - DXY: ${dxy ? dxy.price.toFixed(2) : 'N/A'}`);
  console.log(`  - S&P500: ${sp500 ? sp500.price.toFixed(2) : 'N/A'}`);
  console.log(`  - GOLD: ${gold ? gold.price.toFixed(2) : 'N/A'}`);
  console.log(`  - VIX: ${vix ? vix.price.toFixed(2) : 'N/A'}`);
  console.log(`  - BTC: ${btc ? btc.price.toFixed(2) : 'N/A'}`);

  // If BTC data is missing, cannot calculate correlations
  if (!btc) {
    console.log('[Correlation Engine] BTC data missing, skipping correlation calculation');
    return { macroMetrics, correlationMatrix: null };
  }

  // Fetch historical data for correlation calculation (30 days)
  console.log('[Correlation Engine] Fetching historical data for correlation...');
  const [btcHistory, dxyHistory, sp500History, goldHistory, vixHistory] = await Promise.all([
    fetchHistoricalPrices('BTC', 30),
    fetchHistoricalPrices('DXY', 30),
    fetchHistoricalPrices('SP500', 30),
    fetchHistoricalPrices('GOLD', 30),
    fetchHistoricalPrices('VIX', 30),
  ]);

  // Calculate correlations
  const btcDxyCorr = calculateCorrelation(btcHistory, dxyHistory);
  const btcSp500Corr = calculateCorrelation(btcHistory, sp500History);
  const btcGoldCorr = calculateCorrelation(btcHistory, goldHistory);
  const btcVixCorr = calculateCorrelation(btcHistory, vixHistory);

  const correlationMatrix: CorrelationMatrix = {
    btcDxy: {
      asset1: 'BTC',
      asset2: 'DXY',
      correlation: btcDxyCorr,
      ...classifyCorrelation(btcDxyCorr),
    },
    btcSp500: {
      asset1: 'BTC',
      asset2: 'SP500',
      correlation: btcSp500Corr,
      ...classifyCorrelation(btcSp500Corr),
    },
    btcGold: {
      asset1: 'BTC',
      asset2: 'GOLD',
      correlation: btcGoldCorr,
      ...classifyCorrelation(btcGoldCorr),
    },
    btcVix: {
      asset1: 'BTC',
      asset2: 'VIX',
      correlation: btcVixCorr,
      ...classifyCorrelation(btcVixCorr),
    },
    timestamp: Date.now(),
  };

  console.log('[Correlation Engine] Correlations calculated');
  console.log(`  - BTC/DXY: ${btcDxyCorr.toFixed(3)} (${correlationMatrix.btcDxy.strength})`);
  console.log(`  - BTC/SP500: ${btcSp500Corr.toFixed(3)} (${correlationMatrix.btcSp500.strength})`);
  console.log(`  - BTC/GOLD: ${btcGoldCorr.toFixed(3)} (${correlationMatrix.btcGold.strength})`);
  console.log(`  - BTC/VIX: ${btcVixCorr.toFixed(3)} (${correlationMatrix.btcVix.strength})`);

  return { macroMetrics, correlationMatrix };
}

// ============================================================================
// WHITE-HAT COMPLIANCE
// ============================================================================

export const CORRELATION_ENGINE_CONFIG = {
  // Rate limits
  YAHOO_FINANCE_RATE_LIMIT: 3600000, // 1 hour
  BINANCE_RATE_LIMIT: 60000, // 1 minute

  // Cache durations
  CACHE_DURATION: 3600, // 1 hour

  // Historical data
  CORRELATION_WINDOW: 30, // 30 days

  // Correlation thresholds
  STRONG_CORRELATION: 0.7,
  MODERATE_CORRELATION: 0.4,
  WEAK_CORRELATION: 0.2,
};

console.log('✅ Multi-layer Correlation Engine initialized with White-Hat compliance');
