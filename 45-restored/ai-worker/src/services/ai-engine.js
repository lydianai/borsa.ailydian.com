/**
 * Production-Grade AI Trading Engine
 *
 * CRITICAL SYSTEM - REAL MONEY TRADING
 *
 * Safety Features:
 * - Multi-indicator confirmation (RSI + MACD + Bollinger + Volume + Trend)
 * - Conservative confidence scoring (0.0 - 1.0)
 * - Risk management (stop-loss, take-profit calculations)
 * - Error handling and fallback mechanisms
 * - Extensive logging for audit trail
 * - NO SPECULATION - Only data-driven signals
 *
 * White-hat compliant: Ethical AI, transparent logic, no manipulation
 */

import { getMarketData, normalizeOHLCV, denormalizePrice } from './market-data.js';
import { calculateAllIndicators } from './technical-indicators.js';

/**
 * Generate trading signal with multi-indicator analysis
 *
 * @param {string} symbol - Trading pair (e.g., BTCUSDT)
 * @param {string} timeframe - Timeframe (1h, 4h, 1d)
 * @returns {Promise<Object>} - Trading signal with confidence score
 */
export async function generateTradingSignal(symbol, timeframe = '1h') {
  const startTime = Date.now();

  try {
    // Step 1: Fetch real market data from Binance
    console.info('Fetching market data:', { symbol, timeframe });
    const ohlcv = await getMarketData(symbol, timeframe, 100);

    if (!ohlcv || ohlcv.length < 50) {
      throw new Error('Insufficient market data for analysis');
    }

    const currentPrice = ohlcv[ohlcv.length - 1].close;

    // Step 2: Calculate all technical indicators
    console.info('Calculating technical indicators:', symbol);
    const indicators = calculateAllIndicators(ohlcv);

    // Step 3: Multi-indicator signal generation
    const signals = analyzeIndicators(indicators, currentPrice);

    // Step 4: Calculate confidence score (weighted average)
    const confidence = calculateConfidenceScore(signals);

    // Step 5: Determine action (BUY/SELL/HOLD)
    const action = determineAction(confidence, signals);

    // Step 6: Calculate entry, stop-loss, take-profit
    const { entryPrice, stopLoss, takeProfit } = calculateRiskLevels(
      currentPrice,
      action,
      indicators
    );

    // Step 7: Build signal response
    const signal = {
      symbol,
      timeframe,
      action,
      confidence: parseFloat(confidence.toFixed(4)),
      entryPrice: parseFloat(entryPrice.toFixed(2)),
      stopLoss: parseFloat(stopLoss.toFixed(2)),
      takeProfit: parseFloat(takeProfit.toFixed(2)),
      currentPrice: parseFloat(currentPrice.toFixed(2)),
      timestamp: Date.now(),
      source: 'ailydian-ai-engine',
      version: '2.0.0-production',
      indicators: {
        rsi: {
          value: parseFloat(indicators.rsi.current.toFixed(2)),
          signal: indicators.rsi.signal,
          weight: signals.rsi.weight
        },
        macd: {
          value: parseFloat(indicators.macd.current.toFixed(2)),
          signal: indicators.macd.crossover,
          histogram: parseFloat(indicators.macd.histogram.toFixed(2)),
          weight: signals.macd.weight
        },
        bollinger: {
          upper: parseFloat(indicators.bollinger.upper.toFixed(2)),
          middle: parseFloat(indicators.bollinger.middle.toFixed(2)),
          lower: parseFloat(indicators.bollinger.lower.toFixed(2)),
          position: indicators.bollinger.position,
          weight: signals.bollinger.weight
        },
        volume: {
          ratio: parseFloat(indicators.volume.volumeRatio.toFixed(2)),
          trend: indicators.volume.volumeTrend,
          isHighVolume: indicators.volume.isHighVolume,
          weight: signals.volume.weight
        },
        trend: {
          direction: indicators.trend,
          ema9: parseFloat(indicators.ema.ema9.toFixed(2)),
          ema21: parseFloat(indicators.ema.ema21.toFixed(2)),
          ema50: parseFloat(indicators.ema.ema50.toFixed(2)),
          weight: signals.trend.weight
        }
      },
      metadata: {
        requestId: crypto.randomUUID(),
        processingTime: Date.now() - startTime,
        dataPoints: ohlcv.length,
        algorithm: 'multi-indicator-consensus',
        riskLevel: calculateRiskLevel(confidence)
      }
    };

    // Step 8: Log signal for audit trail
    console.info('Signal generated:', {
      symbol,
      action,
      confidence,
      processingTime: signal.metadata.processingTime
    });

    return signal;

  } catch (error) {
    console.error('AI Engine error:', {
      symbol,
      timeframe,
      error: error.message,
      stack: error.stack
    });

    throw error;
  }
}

/**
 * Analyze indicators and generate individual signals
 * @private
 */
function analyzeIndicators(indicators, currentPrice) {
  const signals = {
    rsi: analyzeRSI(indicators.rsi),
    macd: analyzeMACD(indicators.macd),
    bollinger: analyzeBollinger(indicators.bollinger, currentPrice),
    volume: analyzeVolume(indicators.volume),
    trend: analyzeTrend(indicators.trend, indicators.ema)
  };

  return signals;
}

/**
 * Analyze RSI indicator
 * @private
 */
function analyzeRSI(rsi) {
  const value = rsi.current;

  let signal = 0; // -1 (bearish) to +1 (bullish)
  let weight = 0.2; // RSI weight in overall confidence

  if (value < 30) {
    // Oversold - bullish signal
    signal = Math.min((30 - value) / 10, 1);
  } else if (value > 70) {
    // Overbought - bearish signal
    signal = Math.max((70 - value) / 10, -1);
  } else {
    // Neutral zone
    signal = (value - 50) / 50; // -0.4 to +0.4
  }

  return { signal, weight, reason: rsi.signal };
}

/**
 * Analyze MACD indicator
 * @private
 */
function analyzeMACD(macd) {
  const histogram = macd.histogram;

  let signal = 0;
  let weight = 0.25;

  if (macd.crossover === 'bullish' && histogram > 0) {
    signal = Math.min(histogram / 100, 1);
  } else if (macd.crossover === 'bearish' && histogram < 0) {
    signal = Math.max(histogram / 100, -1);
  }

  return { signal, weight, reason: macd.crossover };
}

/**
 * Analyze Bollinger Bands
 * @private
 */
function analyzeBollinger(bollinger, currentPrice) {
  let signal = 0;
  let weight = 0.2;

  const middle = bollinger.middle;
  const upper = bollinger.upper;
  const lower = bollinger.lower;

  if (bollinger.position === 'below') {
    // Below lower band - oversold, bullish
    signal = 0.7;
  } else if (bollinger.position === 'above') {
    // Above upper band - overbought, bearish
    signal = -0.7;
  } else {
    // Inside bands - neutral to slightly directional
    const bandWidth = upper - lower;
    const positionInBand = (currentPrice - lower) / bandWidth;
    signal = (positionInBand - 0.5) * 0.5; // -0.25 to +0.25
  }

  return { signal, weight, reason: bollinger.position };
}

/**
 * Analyze Volume
 * @private
 */
function analyzeVolume(volume) {
  let signal = 0;
  let weight = 0.15;

  if (volume.isHighVolume) {
    // High volume confirms trend
    weight = 0.2;
    signal = volume.volumeTrend === 'increasing' ? 0.5 : -0.5;
  } else if (volume.isLowVolume) {
    // Low volume - less reliable
    weight = 0.1;
  }

  return { signal, weight, reason: volume.volumeTrend };
}

/**
 * Analyze Trend (EMA crossovers)
 * @private
 */
function analyzeTrend(trend, ema) {
  let signal = 0;
  let weight = 0.2;

  if (trend === 'bullish') {
    signal = 0.8;
  } else if (trend === 'bearish') {
    signal = -0.8;
  } else {
    signal = 0;
  }

  // Adjust based on EMA alignment
  if (ema.ema9 > ema.ema21 && ema.ema21 > ema.ema50) {
    // Strong bullish alignment
    signal = Math.min(signal + 0.2, 1);
  } else if (ema.ema9 < ema.ema21 && ema.ema21 < ema.ema50) {
    // Strong bearish alignment
    signal = Math.max(signal - 0.2, -1);
  }

  return { signal, weight, reason: trend };
}

/**
 * Calculate weighted confidence score
 * @private
 */
function calculateConfidenceScore(signals) {
  let totalWeight = 0;
  let weightedSum = 0;

  for (const indicator of Object.values(signals)) {
    weightedSum += indicator.signal * indicator.weight;
    totalWeight += indicator.weight;
  }

  // Normalize to 0-1 range
  const rawScore = (weightedSum / totalWeight + 1) / 2;

  // Apply conservative adjustment (reduce overconfidence)
  const confidence = Math.max(0.5, Math.min(0.95, rawScore));

  return confidence;
}

/**
 * Determine action based on confidence and signals
 * @private
 */
function determineAction(confidence, signals) {
  // Calculate net signal direction
  let netSignal = 0;
  let totalWeight = 0;

  for (const indicator of Object.values(signals)) {
    netSignal += indicator.signal * indicator.weight;
    totalWeight += indicator.weight;
  }

  const avgSignal = netSignal / totalWeight;

  // Conservative thresholds for real money trading
  if (avgSignal > 0.15 && confidence > 0.6) {
    return 'BUY';
  } else if (avgSignal < -0.15 && confidence > 0.6) {
    return 'SELL';
  } else {
    return 'HOLD';
  }
}

/**
 * Calculate entry, stop-loss, take-profit levels
 * @private
 */
function calculateRiskLevels(currentPrice, action, indicators) {
  const entryPrice = currentPrice;

  let stopLoss, takeProfit;

  if (action === 'BUY') {
    // Stop-loss: 2% below entry or below lower Bollinger Band
    const stopLossPercent = Math.min(
      entryPrice * 0.98,
      indicators.bollinger.lower * 0.995
    );
    stopLoss = stopLossPercent;

    // Take-profit: 4% above entry or near upper Bollinger Band
    const takeProfitPercent = entryPrice * 1.04;
    takeProfit = Math.min(takeProfitPercent, indicators.bollinger.upper * 0.995);

  } else if (action === 'SELL') {
    // Stop-loss: 2% above entry or above upper Bollinger Band
    const stopLossPercent = Math.max(
      entryPrice * 1.02,
      indicators.bollinger.upper * 1.005
    );
    stopLoss = stopLossPercent;

    // Take-profit: 4% below entry or near lower Bollinger Band
    const takeProfitPercent = entryPrice * 0.96;
    takeProfit = Math.max(takeProfitPercent, indicators.bollinger.lower * 1.005);

  } else {
    // HOLD - no risk levels
    stopLoss = entryPrice * 0.98;
    takeProfit = entryPrice * 1.02;
  }

  return { entryPrice, stopLoss, takeProfit };
}

/**
 * Calculate risk level based on confidence
 * @private
 */
function calculateRiskLevel(confidence) {
  if (confidence >= 0.8) {
    return 'LOW'; // High confidence = low risk
  } else if (confidence >= 0.65) {
    return 'MEDIUM';
  } else {
    return 'HIGH'; // Low confidence = high risk
  }
}

/**
 * Batch signal generation for multiple symbols
 * @param {Array} symbols - Array of trading pairs
 * @param {string} timeframe - Timeframe
 * @returns {Promise<Array>} - Array of trading signals
 */
export async function generateBatchSignals(symbols, timeframe = '1h') {
  const results = [];

  for (const symbol of symbols) {
    try {
      const signal = await generateTradingSignal(symbol, timeframe);
      results.push(signal);

      // Rate limiting: Wait 100ms between API calls
      await new Promise(resolve => setTimeout(resolve, 100));

    } catch (error) {
      console.error(`Batch signal error for ${symbol}:`, error.message);
      results.push({
        symbol,
        error: error.message,
        success: false
      });
    }
  }

  return results;
}
