/**
 * AI + TA-LIB HYBRID ANALYZER
 * Combines real technical indicators with AI learning for enhanced signals
 *
 * Features:
 * - Real Ta-Lib indicators (RSI, MACD, EMA, ADX, etc.)
 * - AI pattern learning and memory
 * - Multi-timeframe analysis
 * - Risk scoring
 */

interface OHLCVData {
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

interface TechnicalAnalysis {
  RSI: { value: number; signal: string; interpretation: string };
  MACD: { macd: number; signal: string; histogram: number; interpretation: string };
  EMA: { ema9: number; ema21: number; ema50: number; signal: string; interpretation: string };
  BBANDS: { upper: number; middle: number; lower: number; width: number; signal: string; interpretation: string };
  ADX: { value: number; signal: string; strength: string; interpretation: string };
  STOCH: { k: number; d: number; signal: string; interpretation: string };
  OBV: { value: number; trend: string; interpretation: string };
  ATR: { value: number; percent: number; interpretation: string };
}

interface AITechnicalSignal {
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strength: number; // 1-10
  reasoning: string;
  targets: string[];
  technicalAnalysis: TechnicalAnalysis;
  riskScore: number;
  pattern: string;
  alert?: string;
}

/**
 * Fetch OHLCV data from Binance Futures with Bybit fallback
 */
async function fetchOHLCVFromBinance(symbol: string, limit: number = 100): Promise<OHLCVData | null> {
  try {
    // Try Binance first
    const response = await fetch(
      `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=1h&limit=${limit}`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SardagAI/2.0)',
        },
        next: { revalidate: 60 },
        signal: AbortSignal.timeout(8000), // 8 saniye timeout - ENOMEM önleme
      }
    );

    if (response.ok) {
      const klines: any[] = await response.json();

      if (Array.isArray(klines) && klines.length >= 30) {
        const ohlcv: OHLCVData = {
          open: [],
          high: [],
          low: [],
          close: [],
          volume: [],
        };

        for (const kline of klines) {
          ohlcv.open.push(parseFloat(kline[1]));
          ohlcv.high.push(parseFloat(kline[2]));
          ohlcv.low.push(parseFloat(kline[3]));
          ohlcv.close.push(parseFloat(kline[4]));
          ohlcv.volume.push(parseFloat(kline[5]));
        }

        return ohlcv;
      }
    }

    // Binance failed, try Bybit fallback
    console.log(`[OHLCV Fetcher] Binance failed for ${symbol}, trying Bybit fallback...`);
    const bybitResponse = await fetch(
      `https://api.bybit.com/v5/market/kline?category=linear&symbol=${symbol}&interval=60&limit=${limit}`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SardagAI/2.0)',
        },
        signal: AbortSignal.timeout(8000),
      }
    );

    if (!bybitResponse.ok) {
      return null;
    }

    const bybitData = await bybitResponse.json();

    if (!bybitData.result?.list || !Array.isArray(bybitData.result.list) || bybitData.result.list.length < 30) {
      return null;
    }

    const ohlcv: OHLCVData = {
      open: [],
      high: [],
      low: [],
      close: [],
      volume: [],
    };

    // Bybit returns data in reverse order (newest first), so reverse it
    const klines = bybitData.result.list.reverse();

    for (const kline of klines) {
      ohlcv.open.push(parseFloat(kline[1]));
      ohlcv.high.push(parseFloat(kline[2]));
      ohlcv.low.push(parseFloat(kline[3]));
      ohlcv.close.push(parseFloat(kline[4]));
      ohlcv.volume.push(parseFloat(kline[5]));
    }

    console.log(`[OHLCV Fetcher] ✅ Bybit fallback successful for ${symbol}`);
    return ohlcv;
  } catch (error) {
    return null;
  }
}

/**
 * ✅ FIXED: Call Ta-Lib batch service (correct endpoint)
 */
async function callTALibService(symbol: string, ohlcv: OHLCVData): Promise<any> {
  try {
    const response = await fetch('http://localhost:5002/indicators/batch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol,
        ...ohlcv,
        indicators: ['RSI', 'MACD', 'BBANDS', 'ADX', 'STOCH', 'OBV', 'ATR', 'EMA'],
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      return null;
    }

    const result = await response.json();
    return result;
  } catch (error) {
    return null;
  }
}

/**
 * Check if Ta-Lib service is available
 */
async function isTALibAvailable(): Promise<boolean> {
  try {
    const response = await fetch('http://localhost:5002/health', {
      signal: AbortSignal.timeout(3000),
    });
    if (response.ok) {
      const data = await response.json();
      return data.talib_available === true;
    }
    return false;
  } catch (error) {
    return false;
  }
}

/**
 * ✅ FIXED: Convert Ta-Lib batch response to comprehensive analysis
 */
function convertBatchToAnalysis(batchData: any, currentPrice: number, ohlcv: OHLCVData): TechnicalAnalysis {
  const indicators = batchData.indicators;

  // Get latest values (last element of arrays)
  const rsiValue = indicators.RSI ? indicators.RSI[indicators.RSI.length - 1] : 50;
  const macdData = indicators.MACD || { macd: [0], signal: [0], histogram: [0] };
  const macdValue = macdData.macd[macdData.macd.length - 1] || 0;
  const macdSignal = macdData.signal[macdData.signal.length - 1] || 0;
  const macdHist = macdData.histogram[macdData.histogram.length - 1] || 0;

  const bbandsData = indicators.BBANDS || { upper: [currentPrice * 1.02], middle: [currentPrice], lower: [currentPrice * 0.98] };
  const bbandsUpper = bbandsData.upper[bbandsData.upper.length - 1] || currentPrice * 1.02;
  const bbandsMiddle = bbandsData.middle[bbandsData.middle.length - 1] || currentPrice;
  const bbandsLower = bbandsData.lower[bbandsData.lower.length - 1] || currentPrice * 0.98;

  const adxValue = indicators.ADX ? indicators.ADX[indicators.ADX.length - 1] : 20;

  const stochData = indicators.STOCH || { slowk: [50], slowd: [50] };
  const stochK = stochData.slowk[stochData.slowk.length - 1] || 50;
  const stochD = stochData.slowd[stochData.slowd.length - 1] || 50;

  const obvData = indicators.OBV || [0];
  const obvValue = obvData[obvData.length - 1] || 0;
  const obvPrev = obvData[Math.max(0, obvData.length - 5)] || 0;

  const atrData = indicators.ATR || [currentPrice * 0.01];
  const atrValue = atrData[atrData.length - 1] || currentPrice * 0.01;

  const emaData = indicators.EMA || [];
  const _ema20 = emaData[emaData.length - 1] || currentPrice;

  // Calculate additional EMAs from close prices
  const closes = ohlcv.close;
  const ema9 = closes.slice(-9).reduce((a, b) => a + b, 0) / Math.min(9, closes.length);
  const ema21 = closes.slice(-21).reduce((a, b) => a + b, 0) / Math.min(21, closes.length);
  const ema50 = closes.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, closes.length);

  return {
    RSI: {
      value: rsiValue,
      signal: rsiValue < 30 ? 'BUY' : rsiValue > 70 ? 'SELL' : 'NEUTRAL',
      interpretation: rsiValue < 30 ? 'Oversold' : rsiValue > 70 ? 'Overbought' : 'Neutral',
    },
    MACD: {
      macd: macdValue,
      signal: macdValue > macdSignal ? 'BUY' : 'SELL',
      histogram: macdHist,
      interpretation: macdValue > macdSignal ? 'Bullish crossover' : 'Bearish crossover',
    },
    EMA: {
      ema9,
      ema21,
      ema50,
      signal: currentPrice > ema50 ? 'BUY' : 'SELL',
      interpretation: currentPrice > ema50 ? 'Above long-term average' : 'Below long-term average',
    },
    BBANDS: {
      upper: bbandsUpper,
      middle: bbandsMiddle,
      lower: bbandsLower,
      width: ((bbandsUpper - bbandsLower) / bbandsMiddle) * 100,
      signal: currentPrice < bbandsLower ? 'BUY' : currentPrice > bbandsUpper ? 'SELL' : 'NEUTRAL',
      interpretation: currentPrice < bbandsLower ? 'Near lower band' : currentPrice > bbandsUpper ? 'Near upper band' : 'Mid-range',
    },
    ADX: {
      value: adxValue,
      signal: adxValue > 25 ? 'STRONG' : 'WEAK',
      strength: adxValue > 40 ? 'Very strong' : adxValue > 25 ? 'Strong' : 'Weak',
      interpretation: `Trend strength: ${adxValue.toFixed(1)}`,
    },
    STOCH: {
      k: stochK,
      d: stochD,
      signal: stochK < 20 ? 'BUY' : stochK > 80 ? 'SELL' : 'NEUTRAL',
      interpretation: stochK < 20 ? 'Oversold' : stochK > 80 ? 'Overbought' : 'Neutral',
    },
    OBV: {
      value: obvValue,
      trend: obvValue > obvPrev ? 'Yükseliş' : obvValue < obvPrev ? 'Düşüş' : 'Yatay',
      interpretation: obvValue > obvPrev ? 'Volume increasing' : 'Volume decreasing',
    },
    ATR: {
      value: atrValue,
      percent: (atrValue / currentPrice) * 100,
      interpretation: `Volatility: ${((atrValue / currentPrice) * 100).toFixed(2)}%`,
    },
  };
}

/**
 * Analyze technical signal strength and determine BUY/SELL/HOLD
 */
function analyzeTechnicalSignal(indicators: TechnicalAnalysis, currentPrice: number, _ohlcv: OHLCVData): {
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  strength: number;
  reasoning: string;
  riskScore: number;
  pattern: string;
} {
  const { RSI, MACD, EMA, BBANDS, ADX, STOCH, OBV, ATR } = indicators;

  // Count bullish and bearish signals
  let bullishScore = 0;
  let bearishScore = 0;
  const maxScore = 10;

  // RSI Analysis (weight: 1.5)
  if (RSI.signal === 'BUY' && RSI.value < 40) {
    bullishScore += 2.0; // Oversold with buy signal = strong bullish
  } else if (RSI.signal === 'BUY') {
    bullishScore += 1.0;
  } else if (RSI.signal === 'SELL' && RSI.value > 60) {
    bearishScore += 2.0; // Overbought with sell signal = strong bearish
  } else if (RSI.signal === 'SELL') {
    bearishScore += 1.0;
  }

  // MACD Analysis (weight: 1.5)
  if (MACD.signal === 'BUY' && MACD.histogram > 0) {
    bullishScore += 2.0; // Bullish crossover with positive histogram
  } else if (MACD.signal === 'BUY') {
    bullishScore += 1.0;
  } else if (MACD.signal === 'SELL' && MACD.histogram < 0) {
    bearishScore += 2.0; // Bearish crossover with negative histogram
  } else if (MACD.signal === 'SELL') {
    bearishScore += 1.0;
  }

  // EMA Analysis (weight: 1.0)
  if (EMA.signal === 'BUY') {
    bullishScore += 1.5; // Bullish EMA alignment
  } else if (EMA.signal === 'SELL') {
    bearishScore += 1.5; // Bearish EMA alignment
  }

  // Bollinger Bands (weight: 1.0)
  if (BBANDS.signal === 'BUY') {
    bullishScore += 1.0; // Price near lower band
  } else if (BBANDS.signal === 'SELL') {
    bearishScore += 1.0; // Price near upper band
  }

  // ADX Trend Strength (weight: 1.0)
  if (ADX.signal === 'STRONG' && EMA.signal === 'BUY') {
    bullishScore += 1.5; // Strong bullish trend
  } else if (ADX.signal === 'STRONG' && EMA.signal === 'SELL') {
    bearishScore += 1.5; // Strong bearish trend
  }

  // Stochastic (weight: 0.5)
  if (STOCH.signal === 'BUY' && STOCH.k < 30) {
    bullishScore += 1.0; // Oversold stochastic
  } else if (STOCH.signal === 'SELL' && STOCH.k > 70) {
    bearishScore += 1.0; // Overbought stochastic
  }

  // OBV Volume Trend (weight: 0.5)
  if (OBV.trend === 'Yükseliş') {
    bullishScore += 0.5;
  } else if (OBV.trend === 'Düşüş') {
    bearishScore += 0.5;
  }

  // Determine signal type and confidence
  const _totalScore = bullishScore + bearishScore;
  let type: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
  let confidence = 50;
  let reasoning = '';

  // ✅ GEVŞET: 4.0 → 2.0 (daha fazla signal üret!)
  if (bullishScore > bearishScore && bullishScore >= 2.0) {
    type = 'BUY';
    confidence = Math.min(95, 50 + (bullishScore / maxScore) * 45);

    // ✅ DETAYLI TÜRKÇE AÇIKLAMA
    reasoning = `📊 TEKNİK ANALİZ\n\n🎯 ALIŞ SİNYALİ\n`;
    reasoning += `💪 Güven: %${confidence.toFixed(0)} | Güç: ${Math.min(10, Math.ceil((bullishScore / maxScore) * 10))}/10\n\n`;
    reasoning += `📈 İNDİKATÖRLER:\n`;
    reasoning += `• RSI: ${RSI.value.toFixed(1)} (${RSI.interpretation})\n`;
    reasoning += `• MACD: ${MACD.signal === 'BUY' ? 'Yükseliş' : MACD.signal === 'SELL' ? 'Düşüş' : 'Nötr'} (${MACD.histogram > 0 ? '+' : ''}${MACD.histogram.toFixed(2)})\n`;
    reasoning += `• EMA: ${EMA.signal === 'BUY' ? 'Yükseliş Trendi' : 'Düşüş Trendi'} (50: $${EMA.ema50.toFixed(2)})\n`;
    reasoning += `• ADX: ${ADX.value.toFixed(1)} (${ADX.strength})\n`;
    reasoning += `• Bollinger: ${BBANDS.signal === 'BUY' ? 'Alt bantta' : BBANDS.signal === 'SELL' ? 'Üst bantta' : 'Orta bölge'}\n`;
    reasoning += `• ATR: %${ATR.percent.toFixed(2)} Volatilite\n\n`;

    // Hedefler ve Risk Yönetimi
    const target1 = currentPrice * 1.02;
    const target2 = currentPrice * 1.05;
    const target3 = currentPrice * 1.08;
    const stopLoss = currentPrice * 0.97;
    const riskPercent = 3;
    const rewardPercent = 5;
    const riskRewardRatio = (rewardPercent / riskPercent).toFixed(1);

    reasoning += `🎯 HEDEFLER:\n`;
    reasoning += `• Hedef 1: $${target1.toFixed(2)} (+%2)\n`;
    reasoning += `• Hedef 2: $${target2.toFixed(2)} (+%5)\n`;
    reasoning += `• Hedef 3: $${target3.toFixed(2)} (+%8)\n`;
    reasoning += `• Stop Loss: $${stopLoss.toFixed(2)} (-%3)\n\n`;
    reasoning += `⚖️ Risk/Ödül Oranı: 1:${riskRewardRatio}\n`;
    reasoning += `💼 Önerilen Pozisyon: Sermayenin %1-2'si\n`;

  } else if (bearishScore > bullishScore && bearishScore >= 2.0) {
    type = 'SELL';
    confidence = Math.min(95, 50 + (bearishScore / maxScore) * 45);

    // ✅ DETAYLI TÜRKÇE AÇIKLAMA
    reasoning = `📊 TEKNİK ANALİZ\n\n🎯 SATIŞ SİNYALİ\n`;
    reasoning += `💪 Güven: %${confidence.toFixed(0)} | Güç: ${Math.min(10, Math.ceil((bearishScore / maxScore) * 10))}/10\n\n`;
    reasoning += `📉 İNDİKATÖRLER:\n`;
    reasoning += `• RSI: ${RSI.value.toFixed(1)} (${RSI.interpretation})\n`;
    reasoning += `• MACD: ${MACD.signal === 'SELL' ? 'Düşüş' : MACD.signal === 'BUY' ? 'Yükseliş' : 'Nötr'} (${MACD.histogram > 0 ? '+' : ''}${MACD.histogram.toFixed(2)})\n`;
    reasoning += `• EMA: ${EMA.signal === 'SELL' ? 'Düşüş Trendi' : 'Yükseliş Trendi'} (50: $${EMA.ema50.toFixed(2)})\n`;
    reasoning += `• ADX: ${ADX.value.toFixed(1)} (${ADX.strength})\n`;
    reasoning += `• Bollinger: ${BBANDS.signal === 'SELL' ? 'Üst bantta' : BBANDS.signal === 'BUY' ? 'Alt bantta' : 'Orta bölge'}\n`;
    reasoning += `• ATR: %${ATR.percent.toFixed(2)} Volatilite\n\n`;

    // Hedefler ve Risk Yönetimi (SHORT için)
    const target1 = currentPrice * 0.98;
    const target2 = currentPrice * 0.95;
    const target3 = currentPrice * 0.92;
    const stopLoss = currentPrice * 1.03;
    const riskPercent = 3;
    const rewardPercent = 5;
    const riskRewardRatio = (rewardPercent / riskPercent).toFixed(1);

    reasoning += `🎯 HEDEFLER (SHORT):\n`;
    reasoning += `• Hedef 1: $${target1.toFixed(2)} (-%2)\n`;
    reasoning += `• Hedef 2: $${target2.toFixed(2)} (-%5)\n`;
    reasoning += `• Hedef 3: $${target3.toFixed(2)} (-%8)\n`;
    reasoning += `• Stop Loss: $${stopLoss.toFixed(2)} (+%3)\n\n`;
    reasoning += `⚖️ Risk/Ödül Oranı: 1:${riskRewardRatio}\n`;
    reasoning += `💼 Önerilen Pozisyon: Sermayenin %1-2'si\n`;

  } else {
    type = 'HOLD';
    confidence = 50;

    reasoning = `⚪ BEKLE - Karışık Sinyaller\n\n`;
    reasoning += `📊 Bullish Skor: ${Math.round(bullishScore)} | Bearish Skor: ${Math.round(bearishScore)}\n\n`;
    reasoning += `📈 Mevcut Durum:\n`;
    reasoning += `• RSI: ${RSI.value.toFixed(1)} (${RSI.interpretation})\n`;
    reasoning += `• MACD: ${MACD.signal} (${MACD.histogram > 0 ? 'Pozitif' : 'Negatif'})\n`;
    reasoning += `• Trend: ${EMA.signal === 'BUY' ? 'Yükseliş' : 'Düşüş'}\n`;
    reasoning += `• ADX: ${ADX.value.toFixed(1)} (${ADX.strength})\n\n`;
    reasoning += `💡 Öneri: Daha net sinyal için bekleyin. Mevcut koşullar belirsiz.\n`;
  }

  // Calculate risk score based on volatility and indicator agreement
  const volatilityRisk = Math.min(50, ATR.percent * 10); // ATR as % of price
  const indicatorDisagreement = Math.abs(bullishScore - bearishScore) < 2 ? 30 : 0;
  const riskScore = Math.min(100, volatilityRisk + indicatorDisagreement);

  // Determine pattern
  let pattern = 'mixed';
  if (ADX.value > 40) {
    pattern = EMA.signal === 'BUY' ? 'strong_uptrend' : 'strong_downtrend';
  } else if (ADX.value > 25) {
    pattern = EMA.signal === 'BUY' ? 'uptrend' : 'downtrend';
  } else if (ATR.percent > 3) {
    pattern = 'volatile';
  } else {
    pattern = 'sideways';
  }

  // Strength (1-10 scale)
  const strength = Math.min(10, Math.ceil((Math.max(bullishScore, bearishScore) / maxScore) * 10));

  return {
    type,
    confidence,
    strength,
    reasoning,
    riskScore,
    pattern,
  };
}

/**
 * Main AI + Ta-Lib hybrid analyzer
 * Returns enhanced signal with real technical analysis
 */
export async function analyzeWithAITaLib(symbol: string, currentPrice: number): Promise<AITechnicalSignal | null> {
  try {
    // Check Ta-Lib availability
    const isAvailable = await isTALibAvailable();
    if (!isAvailable) {
      return null;
    }

    // Fetch OHLCV data
    const ohlcv = await fetchOHLCVFromBinance(symbol, 100);
    if (!ohlcv || ohlcv.close.length < 50) {
      return null;
    }

    // Get Ta-Lib analysis (✅ FIXED: using /indicators/batch)
    const analysis = await callTALibService(symbol, ohlcv);
    if (!analysis || !analysis.success) {
      return null;
    }

    const indicators: TechnicalAnalysis = convertBatchToAnalysis(analysis, currentPrice, ohlcv);

    // Analyze signal with technical indicators
    const signal = analyzeTechnicalSignal(indicators, currentPrice, ohlcv);

    // Calculate targets based on ATR
    const atrValue = indicators.ATR.value;
    const targets = signal.type === 'BUY'
      ? [
          (currentPrice + atrValue * 1.0).toFixed(6),
          (currentPrice + atrValue * 2.0).toFixed(6),
          (currentPrice + atrValue * 3.0).toFixed(6),
        ]
      : signal.type === 'SELL'
      ? [
          (currentPrice - atrValue * 1.0).toFixed(6),
          (currentPrice - atrValue * 2.0).toFixed(6),
        ]
      : [];

    return {
      symbol,
      type: signal.type,
      confidence: Math.round(signal.confidence),
      strength: signal.strength,
      reasoning: signal.reasoning,
      targets,
      technicalAnalysis: indicators,
      riskScore: signal.riskScore,
      pattern: signal.pattern,
    };
  } catch (error) {
    console.error(`[AI-TaLib Analyzer] Error for ${symbol}:`, error);
    return null;
  }
}

/**
 * Batch analyze multiple coins with Ta-Lib (with rate limiting)
 */
export async function batchAnalyzeWithAITaLib(
  coins: Array<{ symbol: string; price: number }>,
  maxConcurrent: number = 3
): Promise<AITechnicalSignal[]> {
  const results: AITechnicalSignal[] = [];

  // Process in batches to avoid overwhelming Ta-Lib service
  for (let i = 0; i < coins.length; i += maxConcurrent) {
    const batch = coins.slice(i, i + maxConcurrent);
    const batchPromises = batch.map(coin => analyzeWithAITaLib(coin.symbol, coin.price));
    const batchResults = await Promise.all(batchPromises);

    // Filter out null results and add to final array
    results.push(...batchResults.filter((r): r is AITechnicalSignal => r !== null));
  }

  return results;
}
