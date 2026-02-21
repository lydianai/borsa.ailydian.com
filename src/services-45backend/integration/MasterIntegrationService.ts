/**
 * MASTER INTEGRATION SERVICE
 * Combines all data sources and services
 * - CoinMarketCap Top 100
 * - Binance OHLCV
 * - TA-Lib 500+ indicators
 * - Database
 * - Cache
 */

import { coinMarketCapService, CMCCoin } from '../market/CoinMarketCapService';
import { binanceOHLCVService, OHLCVCandle, Timeframe } from '../market/BinanceOHLCVService';
import axios from 'axios';

export interface IntegratedMarketData {
  coin: CMCCoin;
  ohlcv: {
    [key in Timeframe]?: OHLCVCandle[];
  };
  indicators: {
    rsi?: number[];
    macd?: { macd: number[]; signal: number[]; histogram: number[] };
    bbands?: { upper: number[]; middle: number[]; lower: number[] };
    atr?: number[];
    stoch?: { slowk: number[]; slowd: number[] };
    adx?: number[];
    obv?: number[];
    sma?: number[];
    ema?: number[];
  };
  technicalAnalysis: {
    trend: 'bullish' | 'bearish' | 'neutral';
    strength: number; // 0-100
    signals: string[];
    support: number[];
    resistance: number[];
  };
  lastUpdate: number;
}

export class MasterIntegrationService {
  private readonly talibUrl = process.env.TALIB_SERVICE_URL || 'http://localhost:5002';
  private cache: Map<string, { data: any; timestamp: number }> = new Map();
  private readonly cacheDuration = 300000; // 5 minutes

  /**
   * Get comprehensive market data for a symbol
   */
  async getComprehensiveData(
    symbol: string,
    timeframes: Timeframe[] = ['1h', '4h', '1d']
  ): Promise<IntegratedMarketData | null> {
    try {
      // 1. Get coin metadata from CoinMarketCap
      const allCoins = await coinMarketCapService.getTop100();
      const coin = allCoins.find(c => c.symbol === symbol);

      if (!coin) {
        console.error(`❌ Coin ${symbol} not found in Top 100`);
        return null;
      }

      // 2. Get OHLCV data from Binance
      const binanceSymbol = `${symbol}USDT`;
      const ohlcvData = await binanceOHLCVService.getMultiTimeframeOHLCV(
        binanceSymbol,
        timeframes,
        500
      );

      // 3. Calculate technical indicators (TA-Lib)
      const primaryTimeframe = timeframes[0];
      const primaryCandles = ohlcvData.timeframes[primaryTimeframe] || [];

      const indicators = await this.calculateAllIndicators(primaryCandles);

      // 4. Perform technical analysis
      const technicalAnalysis = this.analyzeTechnicals(primaryCandles, indicators);

      return {
        coin,
        ohlcv: ohlcvData.timeframes,
        indicators,
        technicalAnalysis,
        lastUpdate: Date.now(),
      };
    } catch (error: any) {
      console.error(`Error getting comprehensive data for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Get comprehensive data for Top 100 coins (batch)
   */
  async getTop100ComprehensiveData(
    timeframe: Timeframe = '1h',
    limit: number = 10
  ): Promise<IntegratedMarketData[]> {
    try {
      // Check cache
      const cacheKey = `top100_${timeframe}_${limit}`;
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.cacheDuration) {
        return cached.data;
      }

      // Get Top N coins
      const top100 = await coinMarketCapService.getTop100();
      const topN = top100.slice(0, limit);

      // Process in parallel (batches of 5)
      const results: IntegratedMarketData[] = [];
      const batchSize = 5;

      for (let i = 0; i < topN.length; i += batchSize) {
        const batch = topN.slice(i, i + batchSize);
        const batchPromises = batch.map(coin =>
          this.getComprehensiveData(coin.symbol, [timeframe])
        );

        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults.filter(r => r !== null) as IntegratedMarketData[]);

        // Small delay between batches
        if (i + batchSize < topN.length) {
          await this.delay(500);
        }
      }

      // Update cache
      this.cache.set(cacheKey, { data: results, timestamp: Date.now() });

      return results;
    } catch (error: any) {
      console.error('Error getting Top 100 comprehensive data:', error.message);
      return [];
    }
  }

  /**
   * Calculate all technical indicators via TA-Lib service
   */
  private async calculateAllIndicators(candles: OHLCVCandle[]): Promise<any> {
    if (candles.length === 0) {
      return {};
    }

    try {
      const ohlcv = {
        open: candles.map(c => c.open),
        high: candles.map(c => c.high),
        low: candles.map(c => c.low),
        close: candles.map(c => c.close),
        volume: candles.map(c => c.volume),
      };

      // Call TA-Lib service batch endpoint
      const response = await axios.post(
        `${this.talibUrl}/indicators/batch`,
        {
          ...ohlcv,
          indicators: ['RSI', 'MACD', 'BBANDS', 'ATR', 'STOCH', 'ADX', 'OBV', 'SMA', 'EMA'],
        },
        { timeout: 10000 }
      );

      if (response.data.success) {
        return response.data.indicators;
      } else {
        console.warn('⚠️  TA-Lib service returned error, using fallback');
        return this.calculateFallbackIndicators(candles);
      }
    } catch (error: any) {
      console.warn('⚠️  TA-Lib service unavailable, using fallback:', error.message);
      return this.calculateFallbackIndicators(candles);
    }
  }

  /**
   * Fallback indicator calculations (without TA-Lib)
   */
  private calculateFallbackIndicators(candles: OHLCVCandle[]): any {
    const closes = candles.map(c => c.close);

    // Simple RSI calculation
    const rsi = this.calculateSimpleRSI(closes, 14);

    // Simple SMA
    const sma = this.calculateSMA(closes, 20);

    // Simple EMA
    const ema = this.calculateEMA(closes, 20);

    return {
      RSI: rsi,
      SMA: sma,
      EMA: ema,
    };
  }

  /**
   * Analyze technical indicators and generate insights
   */
  private analyzeTechnicals(
    candles: OHLCVCandle[],
    indicators: any
  ): {
    trend: 'bullish' | 'bearish' | 'neutral';
    strength: number;
    signals: string[];
    support: number[];
    resistance: number[];
  } {
    const signals: string[] = [];
    let bullishSignals = 0;
    let bearishSignals = 0;

    if (candles.length === 0) {
      return { trend: 'neutral', strength: 0, signals: [], support: [], resistance: [] };
    }

    const currentPrice = candles[candles.length - 1].close;

    // RSI Analysis
    if (indicators.RSI && indicators.RSI.length > 0) {
      const rsi = indicators.RSI[indicators.RSI.length - 1];
      if (rsi < 30) {
        signals.push(`RSI oversold (${rsi.toFixed(2)}) - potential buy`);
        bullishSignals++;
      } else if (rsi > 70) {
        signals.push(`RSI overbought (${rsi.toFixed(2)}) - potential sell`);
        bearishSignals++;
      }
    }

    // MACD Analysis
    if (indicators.MACD) {
      const macdData = indicators.MACD;
      if (macdData.histogram && macdData.histogram.length > 1) {
        const lastHist = macdData.histogram[macdData.histogram.length - 1];
        const prevHist = macdData.histogram[macdData.histogram.length - 2];

        if (lastHist > 0 && prevHist <= 0) {
          signals.push('MACD bullish crossover');
          bullishSignals++;
        } else if (lastHist < 0 && prevHist >= 0) {
          signals.push('MACD bearish crossover');
          bearishSignals++;
        }
      }
    }

    // Moving Average Analysis
    if (indicators.SMA && indicators.SMA.length > 0 && indicators.EMA && indicators.EMA.length > 0) {
      const sma = indicators.SMA[indicators.SMA.length - 1];
      const ema = indicators.EMA[indicators.EMA.length - 1];

      if (currentPrice > sma && currentPrice > ema) {
        signals.push('Price above MA - bullish');
        bullishSignals++;
      } else if (currentPrice < sma && currentPrice < ema) {
        signals.push('Price below MA - bearish');
        bearishSignals++;
      }
    }

    // Stochastic Analysis
    if (indicators.STOCH) {
      const stoch = indicators.STOCH;
      if (stoch.slowk && stoch.slowk.length > 0) {
        const k = stoch.slowk[stoch.slowk.length - 1];
        if (k < 20) {
          signals.push(`Stochastic oversold (${k.toFixed(2)})`);
          bullishSignals++;
        } else if (k > 80) {
          signals.push(`Stochastic overbought (${k.toFixed(2)})`);
          bearishSignals++;
        }
      }
    }

    // Determine trend
    let trend: 'bullish' | 'bearish' | 'neutral';
    if (bullishSignals > bearishSignals + 1) {
      trend = 'bullish';
    } else if (bearishSignals > bullishSignals + 1) {
      trend = 'bearish';
    } else {
      trend = 'neutral';
    }

    // Calculate strength (0-100)
    const totalSignals = bullishSignals + bearishSignals;
    const strength = totalSignals > 0
      ? Math.abs(bullishSignals - bearishSignals) * 100 / (totalSignals + 2)
      : 0;

    // Detect support/resistance
    const { support, resistance } = binanceOHLCVService.detectSupportResistance(candles, 0.02);

    return {
      trend,
      strength,
      signals,
      support: support.slice(-3), // Last 3 support levels
      resistance: resistance.slice(-3), // Last 3 resistance levels
    };
  }

  /**
   * Generate trading signal based on comprehensive analysis
   */
  async generateTradingSignal(symbol: string): Promise<{
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    reasons: string[];
    entry: number;
    stopLoss: number;
    takeProfit: number;
  } | null> {
    try {
      const data = await this.getComprehensiveData(symbol, ['1h', '4h']);
      if (!data) return null;

      const { coin, technicalAnalysis, indicators } = data;
      const currentPrice = coin.price;

      let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
      let confidence = 0;
      const reasons: string[] = [];

      // Analyze signals
      if (technicalAnalysis.trend === 'bullish' && technicalAnalysis.strength > 60) {
        action = 'BUY';
        confidence = Math.min(0.95, (technicalAnalysis.strength / 100) * 0.9);
        reasons.push(`Strong bullish trend (${technicalAnalysis.strength.toFixed(0)}% strength)`);
        reasons.push(...technicalAnalysis.signals);
      } else if (technicalAnalysis.trend === 'bearish' && technicalAnalysis.strength > 60) {
        action = 'SELL';
        confidence = Math.min(0.95, (technicalAnalysis.strength / 100) * 0.9);
        reasons.push(`Strong bearish trend (${technicalAnalysis.strength.toFixed(0)}% strength)`);
        reasons.push(...technicalAnalysis.signals);
      } else {
        action = 'HOLD';
        confidence = 0.5;
        reasons.push('Market conditions unclear - waiting for better opportunity');
      }

      // Calculate stop-loss and take-profit
      const atr = indicators.atr && indicators.atr.length > 0
        ? indicators.atr[indicators.atr.length - 1]
        : currentPrice * 0.03; // 3% fallback

      const stopLoss = action === 'BUY'
        ? currentPrice - (atr * 2)
        : action === 'SELL'
        ? currentPrice + (atr * 2)
        : currentPrice;

      const takeProfit = action === 'BUY'
        ? currentPrice + (atr * 3)
        : action === 'SELL'
        ? currentPrice - (atr * 3)
        : currentPrice;

      return {
        action,
        confidence,
        reasons,
        entry: currentPrice,
        stopLoss,
        takeProfit,
      };
    } catch (error: any) {
      console.error(`Error generating signal for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Simple RSI calculation
   */
  private calculateSimpleRSI(closes: number[], period: number): number[] {
    const rsi: number[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = 1; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }

    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;

      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
  }

  /**
   * Simple Moving Average
   */
  private calculateSMA(values: number[], period: number): number[] {
    const sma: number[] = [];
    for (let i = period - 1; i < values.length; i++) {
      const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      sma.push(sum / period);
    }
    return sma;
  }

  /**
   * Exponential Moving Average
   */
  private calculateEMA(values: number[], period: number): number[] {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);

    // Start with SMA
    const sma = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
    ema.push(sma);

    for (let i = period; i < values.length; i++) {
      const currentEMA = (values[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
      ema.push(currentEMA);
    }

    return ema;
  }

  /**
   * Delay utility
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Clear cache
   */
  clearCache() {
    this.cache.clear();
    console.log('🗑️  Master integration cache cleared');
  }
}

// Singleton instance
export const masterIntegrationService = new MasterIntegrationService();
