import { NextRequest, NextResponse } from 'next/server';
import { getAIModelsEndpoint, getTALibEndpoint } from '@/lib/api-config';

interface QuantumSignalRequest {
  symbol: string;
  config: {
    multiTimeframe: boolean;
    adaptivePositionSizing: boolean;
    aiModelWeights: {
      lstm: number;
      gru: number;
      transformer: number;
      gradientBoosting: number;
    };
  };
  apiKey: string;
  apiSecret: string;
}

// Quantum AI sinyali oluştur - tüm AI modelleri + 158 TA-Lib indikatörü
async function generateQuantumSignal(request: QuantumSignalRequest): Promise<any> {
  const { symbol, config } = request;

  try {
    // Paralel olarak tüm kaynaklardan veri al
    const [aiPredictions, taLibData, binanceData] = await Promise.all([
      getAIPredictions(symbol, config.aiModelWeights),
      getTALibIndicators(symbol),
      getBinanceMarketData(symbol),
    ]);

    // Quantum özelliklerini tespit et
    const quantumFeatures = detectQuantumFeatures(binanceData, taLibData);

    // Risk değerlendirmesi yap
    const riskAssessment = assessRisk(binanceData, taLibData, quantumFeatures);

    // Tüm sinyalleri birleştir ve ensemble tahmin oluştur
    const ensembleSignal = combineSignals(aiPredictions, taLibData, quantumFeatures, riskAssessment);

    // Adaptive position sizing hesapla
    const positionSize = config.adaptivePositionSizing
      ? calculateAdaptivePositionSize(ensembleSignal, riskAssessment)
      : null;

    return {
      action: ensembleSignal.action,
      confidence: ensembleSignal.confidence,
      reason: ensembleSignal.reason,
      aiPredictions: aiPredictions,
      taLibIndicators: {
        rsi: taLibData.rsi?.value || 50,
        macd: taLibData.macd?.signal || 'NEUTRAL',
        bbands: taLibData.bbands?.position || 'MIDDLE',
      },
      quantumFeatures,
      riskAssessment,
      recommendedPositionSize: positionSize,
    };
  } catch (error: any) {
    console.error('Quantum signal generation error:', error.message);
    throw error;
  }
}

// AI modellerinden tahmin al (14 model)
async function getAIPredictions(symbol: string, weights: any): Promise<any> {
  try {
    const response = await fetch(getAIModelsEndpoint('/predict/single'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, timeframe: '1h' }),
    });

    if (!response.ok) throw new Error('AI service unavailable');

    const data = await response.json();

    // Her model tipinin tahminini al ve ağırlıklandır
    return {
      lstm: (data.predictions?.lstm?.change_prediction || 0) * weights.lstm,
      gru: (data.predictions?.gru?.change_prediction || 0) * weights.gru,
      transformer: (data.predictions?.transformer?.change_prediction || 0) * weights.transformer,
      gradientBoosting: (data.predictions?.gradient_boosting?.change_prediction || 0) * weights.gradientBoosting,
    };
  } catch (error) {
    // Fallback: tahmini değerler
    return {
      lstm: 0,
      gru: 0,
      transformer: 0,
      gradientBoosting: 0,
    };
  }
}

// TA-Lib indikatörleri al (158 indikatör)
async function getTALibIndicators(symbol: string): Promise<any> {
  try {
    const response = await fetch(getTALibEndpoint('/indicators/batch'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol,
        timeframe: '1h',
        indicators: ['rsi', 'macd', 'bbands', 'ema', 'sma', 'stoch', 'adx', 'obv', 'atr'],
      }),
    });

    if (!response.ok) throw new Error('TA-Lib service unavailable');

    const data = await response.json();
    return data.indicators || {};
  } catch (error) {
    return {
      rsi: { value: 50 },
      macd: { signal: 'NEUTRAL' },
      bbands: { position: 'MIDDLE' },
    };
  }
}

// Binance piyasa verisi al
async function getBinanceMarketData(symbol: string): Promise<any> {
  try {
    const response = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}`);
    if (!response.ok) throw new Error('Binance API error');
    return await response.json();
  } catch (error) {
    return {
      priceChangePercent: '0',
      volume: '0',
      highPrice: '0',
      lowPrice: '0',
    };
  }
}

// Quantum özelliklerini tespit et
function detectQuantumFeatures(marketData: any, taLibData: any): any {
  const priceChange = parseFloat(marketData.priceChangePercent || '0');
  const volume = parseFloat(marketData.volume || '0');

  // Piyasa rejimi belirle
  let marketRegime = 'RANGING';
  if (Math.abs(priceChange) > 2 && taLibData.adx?.value > 25) {
    marketRegime = 'TRENDING';
  } else if (Math.abs(priceChange) > 5) {
    marketRegime = 'VOLATILE';
  } else if (Math.abs(priceChange) < 0.5) {
    marketRegime = 'CONSOLIDATING';
  }

  // Gürültü seviyesi
  const noiseLevel = Math.abs(priceChange) > 3 ? 'HIGH' : Math.abs(priceChange) > 1 ? 'MEDIUM' : 'LOW';

  // Sinyal netliği
  const rsi = taLibData.rsi?.value || 50;
  const signalClarity = (rsi < 30 || rsi > 70) ? 'CLEAR' : (rsi < 40 || rsi > 60) ? 'MODERATE' : 'WEAK';

  // Piyasa gücü
  const marketStrength = volume > 10000 ? 'STRONG' : volume > 5000 ? 'MODERATE' : 'WEAK';

  return {
    marketRegime,
    noiseLevel,
    signalClarity,
    marketStrength,
  };
}

// Risk değerlendirmesi yap
function assessRisk(marketData: any, taLibData: any, quantumFeatures: any): any {
  const priceChange = Math.abs(parseFloat(marketData.priceChangePercent || '0'));
  const atr = taLibData.atr?.value || 0;

  // Volatilite riski
  const volatilityRisk = priceChange > 5 || atr > 1000 ? 'HIGH' : priceChange > 2 || atr > 500 ? 'MEDIUM' : 'LOW';

  // Trend gücü
  const adx = taLibData.adx?.value || 0;
  const trendStrength = adx > 40 ? 'STRONG' : adx > 25 ? 'MODERATE' : 'WEAK';

  // Genel risk
  let overallRisk = 'MEDIUM';
  if (volatilityRisk === 'HIGH' || quantumFeatures.noiseLevel === 'HIGH') {
    overallRisk = 'HIGH';
  } else if (volatilityRisk === 'LOW' && quantumFeatures.signalClarity === 'CLEAR') {
    overallRisk = 'LOW';
  }

  return {
    overallRisk,
    volatilityRisk,
    trendStrength,
  };
}

// Sinyalleri birleştir ve ensemble tahmin oluştur
function combineSignals(aiPredictions: any, taLibData: any, quantumFeatures: any, riskAssessment: any): any {
  // AI tahminlerinin ortalaması
  const aiAverage = (
    aiPredictions.lstm +
    aiPredictions.gru +
    aiPredictions.transformer +
    aiPredictions.gradientBoosting
  ) / 4;

  // TA-Lib sinyal skoru
  const rsi = taLibData.rsi?.value || 50;
  let taLibScore = 0;

  if (rsi < 30) taLibScore = 1; // Oversold - AL
  else if (rsi > 70) taLibScore = -1; // Overbought - SAT
  else if (rsi < 40) taLibScore = 0.5;
  else if (rsi > 60) taLibScore = -0.5;

  // MACD sinyali
  if (taLibData.macd?.signal === 'BUY') taLibScore += 0.5;
  else if (taLibData.macd?.signal === 'SELL') taLibScore -= 0.5;

  // Ensemble tahmin (AI 70%, TA-Lib 30%)
  const ensembleScore = (aiAverage * 0.7) + (taLibScore * 0.3);

  // Güven skoru hesapla
  let confidence = Math.abs(ensembleScore) * 0.5 + 0.3; // Base confidence

  // Quantum özelliklere göre güven ayarla
  if (quantumFeatures.signalClarity === 'CLEAR') confidence += 0.15;
  if (quantumFeatures.marketRegime === 'TRENDING') confidence += 0.1;
  if (riskAssessment.overallRisk === 'LOW') confidence += 0.05;

  confidence = Math.min(confidence, 0.95); // Max %95

  // Aksiyon belirle
  let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
  if (ensembleScore > 0.5 && confidence > 0.6) action = 'BUY';
  else if (ensembleScore < -0.5 && confidence > 0.6) action = 'SELL';

  // Sebep oluştur
  const reason = `Quantum AI: ${action} (AI: ${(aiAverage * 100).toFixed(1)}%, TA-Lib: ${(taLibScore * 100).toFixed(0)}%, Rejim: ${quantumFeatures.marketRegime})`;

  return {
    action,
    confidence,
    reason,
    ensembleScore,
  };
}

// Adaptive position sizing hesapla
function calculateAdaptivePositionSize(signal: any, riskAssessment: any): number {
  let baseSize = 100; // USDT

  // Güvene göre çarp
  const confidenceMultiplier = signal.confidence;

  // Risk seviyesine göre azalt
  const riskMultiplier = riskAssessment.overallRisk === 'HIGH' ? 0.5 : riskAssessment.overallRisk === 'MEDIUM' ? 0.75 : 1.0;

  return baseSize * confidenceMultiplier * riskMultiplier;
}

export async function POST(request: NextRequest) {
  try {
    const body: QuantumSignalRequest = await request.json();

    const signal = await generateQuantumSignal(body);

    return NextResponse.json({
      success: true,
      signal,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to generate quantum signal',
      },
      { status: 500 }
    );
  }
}
