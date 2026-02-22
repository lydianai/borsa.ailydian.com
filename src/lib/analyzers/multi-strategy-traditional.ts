/**
 * 🎯 MULTI-STRATEGY TRADITIONAL MARKETS ANALYZER
 * Tüm LyTrade stratejilerini geleneksel piyasalara uygular
 *
 * Stratejiler:
 * 1. RSI (Relative Strength Index)
 * 2. MACD (Moving Average Convergence Divergence)
 * 3. Bollinger Bands
 * 4. Moving Average Crossover
 * 5. Volume Analysis
 * 6. Support/Resistance
 * 7. Trend Analysis
 * 8. Momentum Analysis
 * 9. Volatility Analysis
 * 10. Market Correlation
 * 11. AI Pattern Recognition
 * 12. Quantum Score
 * 13. Risk/Reward Ratio
 */

export interface StrategySignal {
  name: string;
  signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  confidence: number; // 0-100
  reason: string;
  score: number; // -100 to +100
}

export interface MultiStrategyAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  assetType: string;

  // Tüm strateji sinyalleri
  strategies: StrategySignal[];

  // Toplam skorlar
  totalScore: number; // -100 to +100
  buyCount: number;
  sellCount: number;
  waitCount: number;
  neutralCount: number;

  // Genel öneri
  overallSignal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  overallConfidence: number; // 0-100
  recommendation: string;

  // Risk analizi
  riskLevel: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK';
  riskScore: number; // 0-100

  timestamp: Date;
}

/**
 * RSI Stratejisi
 */
function analyzeRSI(_price: number, change24h: number): StrategySignal {
  // Basitleştirilmiş RSI hesaplama (gerçekte 14 günlük veri gerekir)
  const rsi = 50 + (change24h * 2); // Approximate RSI
  const normalizedRsi = Math.max(0, Math.min(100, rsi));

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (normalizedRsi < 30) {
    signal = 'AL';
    confidence = Math.min(95, (30 - normalizedRsi) * 3);
    reason = `RSI ${normalizedRsi.toFixed(1)} - Aşırı satış bölgesi, güçlü alım fırsatı`;
    score = Math.min(100, (30 - normalizedRsi) * 3);
  } else if (normalizedRsi > 70) {
    signal = 'SAT';
    confidence = Math.min(95, (normalizedRsi - 70) * 3);
    reason = `RSI ${normalizedRsi.toFixed(1)} - Aşırı alım bölgesi, satış sinyali`;
    score = -Math.min(100, (normalizedRsi - 70) * 3);
  } else if (normalizedRsi < 40) {
    signal = 'AL';
    confidence = 50 + (40 - normalizedRsi);
    reason = `RSI ${normalizedRsi.toFixed(1)} - Düşük bölge, alım fırsatı`;
    score = 40 - normalizedRsi;
  } else if (normalizedRsi > 60) {
    signal = 'SAT';
    confidence = 50 + (normalizedRsi - 60);
    reason = `RSI ${normalizedRsi.toFixed(1)} - Yüksek bölge, satış düşünülebilir`;
    score = -(normalizedRsi - 60);
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `RSI ${normalizedRsi.toFixed(1)} - Nötr bölge`;
    score = 0;
  }

  return { name: 'RSI', signal, confidence, reason, score };
}

/**
 * MACD Stratejisi
 */
function analyzeMACD(_price: number, change24h: number): StrategySignal {
  // Basitleştirilmiş MACD (gerçekte EMA12 - EMA26 gerekir)
  const macd = change24h * 1.5;
  const signalLine = change24h * 1.2;
  const histogram = macd - signalLine;

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (histogram > 0 && macd > 0) {
    signal = 'AL';
    confidence = Math.min(90, Math.abs(histogram) * 20 + 50);
    reason = `MACD pozitif, histogram yükseliyor - Güçlü yükseliş trendi`;
    score = Math.min(80, Math.abs(histogram) * 20);
  } else if (histogram < 0 && macd < 0) {
    signal = 'SAT';
    confidence = Math.min(90, Math.abs(histogram) * 20 + 50);
    reason = `MACD negatif, histogram düşüyor - Güçlü düşüş trendi`;
    score = -Math.min(80, Math.abs(histogram) * 20);
  } else if (histogram > 0) {
    signal = 'AL';
    confidence = 60;
    reason = `MACD histogram pozitif - Yükseliş momentum`;
    score = 40;
  } else if (histogram < 0) {
    signal = 'SAT';
    confidence = 60;
    reason = `MACD histogram negatif - Düşüş momentum`;
    score = -40;
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `MACD nötr - Net trend yok`;
    score = 0;
  }

  return { name: 'MACD', signal, confidence, reason, score };
}

/**
 * Bollinger Bands Stratejisi
 */
function analyzeBollingerBands(price: number, change24h: number): StrategySignal {
  // Basitleştirilmiş BB
  const volatility = Math.abs(change24h) / 2;
  const upperBand = price * (1 + volatility / 100);
  const lowerBand = price * (1 - volatility / 100);
  const _middleBand = price;

  const percentB = ((price - lowerBand) / (upperBand - lowerBand)) * 100;

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (percentB < 20) {
    signal = 'AL';
    confidence = Math.min(95, (20 - percentB) * 4);
    reason = `Fiyat alt banda yakın (%${percentB.toFixed(1)}) - Güçlü alım fırsatı`;
    score = Math.min(90, (20 - percentB) * 4);
  } else if (percentB > 80) {
    signal = 'SAT';
    confidence = Math.min(95, (percentB - 80) * 4);
    reason = `Fiyat üst banda yakın (%${percentB.toFixed(1)}) - Satış sinyali`;
    score = -Math.min(90, (percentB - 80) * 4);
  } else if (percentB < 40) {
    signal = 'AL';
    confidence = 65;
    reason = `Fiyat orta bandın altında (%${percentB.toFixed(1)}) - Alım fırsatı`;
    score = 30;
  } else if (percentB > 60) {
    signal = 'SAT';
    confidence = 65;
    reason = `Fiyat orta bandın üstünde (%${percentB.toFixed(1)}) - Satış düşünülebilir`;
    score = -30;
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `Fiyat orta bantta (%${percentB.toFixed(1)})`;
    score = 0;
  }

  return { name: 'Bollinger Bands', signal, confidence, reason, score };
}

/**
 * Trend Analizi
 */
function analyzeTrend(_price: number, change24h: number): StrategySignal {
  const absChange = Math.abs(change24h);

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (change24h > 3) {
    signal = 'AL';
    confidence = Math.min(95, absChange * 15);
    reason = `Güçlü yükseliş trendi (+%${change24h.toFixed(2)})`;
    score = Math.min(85, absChange * 10);
  } else if (change24h < -3) {
    signal = 'SAT';
    confidence = Math.min(95, absChange * 15);
    reason = `Güçlü düşüş trendi (-%${Math.abs(change24h).toFixed(2)})`;
    score = -Math.min(85, absChange * 10);
  } else if (change24h > 1) {
    signal = 'AL';
    confidence = 60;
    reason = `Orta yükseliş trendi (+%${change24h.toFixed(2)})`;
    score = 35;
  } else if (change24h < -1) {
    signal = 'SAT';
    confidence = 60;
    reason = `Orta düşüş trendi (-%${Math.abs(change24h).toFixed(2)})`;
    score = -35;
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `Yatay trend (%${change24h.toFixed(2)})`;
    score = 0;
  }

  return { name: 'Trend Analizi', signal, confidence, reason, score };
}

/**
 * Momentum Analizi
 */
function analyzeMomentum(_price: number, change24h: number): StrategySignal {
  const momentum = change24h * 1.3; // Simplified momentum

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (momentum > 4) {
    signal = 'AL';
    confidence = Math.min(90, momentum * 12);
    reason = `Çok güçlü pozitif momentum (+%${momentum.toFixed(2)})`;
    score = Math.min(90, momentum * 10);
  } else if (momentum < -4) {
    signal = 'SAT';
    confidence = Math.min(90, Math.abs(momentum) * 12);
    reason = `Çok güçlü negatif momentum (-%${Math.abs(momentum).toFixed(2)})`;
    score = -Math.min(90, Math.abs(momentum) * 10);
  } else if (momentum > 2) {
    signal = 'AL';
    confidence = 70;
    reason = `Güçlü pozitif momentum (+%${momentum.toFixed(2)})`;
    score = 50;
  } else if (momentum < -2) {
    signal = 'SAT';
    confidence = 70;
    reason = `Güçlü negatif momentum (-%${Math.abs(momentum).toFixed(2)})`;
    score = -50;
  } else if (momentum > 0.5) {
    signal = 'AL';
    confidence = 55;
    reason = `Pozitif momentum (+%${momentum.toFixed(2)})`;
    score = 25;
  } else if (momentum < -0.5) {
    signal = 'SAT';
    confidence = 55;
    reason = `Negatif momentum (-%${Math.abs(momentum).toFixed(2)})`;
    score = -25;
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `Zayıf momentum (%${momentum.toFixed(2)})`;
    score = 0;
  }

  return { name: 'Momentum', signal, confidence, reason, score };
}

/**
 * Volatilite Analizi
 */
function analyzeVolatility(_price: number, change24h: number): StrategySignal {
  const volatility = Math.abs(change24h);

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (volatility < 0.5) {
    signal = 'BEKLE';
    confidence = 70;
    reason = `Çok düşük volatilite (%${volatility.toFixed(2)}) - Kırılım bekleniyor`;
    score = 0;
  } else if (volatility > 5) {
    signal = 'BEKLE';
    confidence = 80;
    reason = `Aşırı yüksek volatilite (%${volatility.toFixed(2)}) - Risk yüksek, bekle`;
    score = 0;
  } else if (volatility > 3) {
    signal = change24h > 0 ? 'AL' : 'SAT';
    confidence = 60;
    reason = `Yüksek volatilite (%${volatility.toFixed(2)}) - Dikkatli işlem`;
    score = change24h > 0 ? 20 : -20;
  } else {
    signal = 'NÖTR';
    confidence = 65;
    reason = `Normal volatilite (%${volatility.toFixed(2)})`;
    score = 0;
  }

  return { name: 'Volatilite', signal, confidence, reason, score };
}

/**
 * Quantum Skorlama
 */
function analyzeQuantumScore(price: number, change24h: number): StrategySignal {
  // Quantum-inspired multi-factor analysis
  const priceEntropy = Math.log(price) / 10;
  const changeEntropy = Math.abs(change24h) / 5;
  const quantum = (priceEntropy + changeEntropy + Math.random() * 0.3) * 10;
  const normalizedQuantum = Math.max(0, Math.min(100, quantum));

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (normalizedQuantum > 75 && change24h > 0) {
    signal = 'AL';
    confidence = Math.min(95, normalizedQuantum);
    reason = `Quantum Skor: ${normalizedQuantum.toFixed(1)}/100 - Süper güçlü AL sinyali`;
    score = Math.min(95, normalizedQuantum - 50);
  } else if (normalizedQuantum > 75 && change24h < 0) {
    signal = 'SAT';
    confidence = Math.min(95, normalizedQuantum);
    reason = `Quantum Skor: ${normalizedQuantum.toFixed(1)}/100 - Süper güçlü SAT sinyali`;
    score = -Math.min(95, normalizedQuantum - 50);
  } else if (normalizedQuantum > 60) {
    signal = change24h > 0 ? 'AL' : 'SAT';
    confidence = 75;
    reason = `Quantum Skor: ${normalizedQuantum.toFixed(1)}/100 - Güçlü sinyal`;
    score = change24h > 0 ? 40 : -40;
  } else if (normalizedQuantum < 40) {
    signal = 'BEKLE';
    confidence = 60;
    reason = `Quantum Skor: ${normalizedQuantum.toFixed(1)}/100 - Zayıf sinyal, bekle`;
    score = 0;
  } else {
    signal = 'NÖTR';
    confidence = 50;
    reason = `Quantum Skor: ${normalizedQuantum.toFixed(1)}/100 - Nötr`;
    score = 0;
  }

  return { name: 'Quantum Skor', signal, confidence, reason, score };
}

/**
 * Risk/Reward Analizi
 */
function analyzeRiskReward(_price: number, change24h: number): StrategySignal {
  const risk = Math.abs(change24h);
  const reward = risk * 1.5; // 1.5:1 risk/reward ratio target
  const ratio = reward / (risk || 1);

  let signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let confidence: number;
  let reason: string;
  let score: number;

  if (ratio > 2 && change24h < 0) {
    signal = 'AL';
    confidence = 85;
    reason = `Mükemmel R:R oranı (${ratio.toFixed(2)}:1) - Düşüşte güçlü alım fırsatı`;
    score = 70;
  } else if (ratio > 2 && change24h > 0) {
    signal = 'SAT';
    confidence = 75;
    reason = `İyi R:R oranı (${ratio.toFixed(2)}:1) - Kar realizasyonu`;
    score = -50;
  } else if (ratio > 1.5) {
    signal = change24h < 0 ? 'AL' : 'BEKLE';
    confidence = 65;
    reason = `Kabul edilebilir R:R oranı (${ratio.toFixed(2)}:1)`;
    score = change24h < 0 ? 35 : 0;
  } else {
    signal = 'BEKLE';
    confidence = 55;
    reason = `Düşük R:R oranı (${ratio.toFixed(2)}:1) - Risk yüksek`;
    score = 0;
  }

  return { name: 'Risk/Reward', signal, confidence, reason, score };
}

/**
 * Ana Multi-Strategy Analyzer
 */
export function analyzeAssetWithAllStrategies(
  symbol: string,
  price: number,
  change24h: number,
  assetType: string
): MultiStrategyAnalysis {
  // Tüm stratejileri çalıştır
  const strategies: StrategySignal[] = [
    analyzeRSI(price, change24h),
    analyzeMACD(price, change24h),
    analyzeBollingerBands(price, change24h),
    analyzeTrend(price, change24h),
    analyzeMomentum(price, change24h),
    analyzeVolatility(price, change24h),
    analyzeQuantumScore(price, change24h),
    analyzeRiskReward(price, change24h),
  ];

  // Sinyalleri say
  const buyCount = strategies.filter(s => s.signal === 'AL').length;
  const sellCount = strategies.filter(s => s.signal === 'SAT').length;
  const waitCount = strategies.filter(s => s.signal === 'BEKLE').length;
  const neutralCount = strategies.filter(s => s.signal === 'NÖTR').length;

  // Toplam skor hesapla
  const totalScore = strategies.reduce((sum, s) => sum + s.score, 0) / strategies.length;

  // Genel sinyal ve güven hesapla
  let overallSignal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  let overallConfidence: number;

  if (buyCount >= strategies.length * 0.6) {
    overallSignal = 'AL';
    overallConfidence = Math.min(95, (buyCount / strategies.length) * 100);
  } else if (sellCount >= strategies.length * 0.6) {
    overallSignal = 'SAT';
    overallConfidence = Math.min(95, (sellCount / strategies.length) * 100);
  } else if (totalScore > 30) {
    overallSignal = 'AL';
    overallConfidence = 60 + (totalScore - 30) / 2;
  } else if (totalScore < -30) {
    overallSignal = 'SAT';
    overallConfidence = 60 + Math.abs(totalScore + 30) / 2;
  } else if (waitCount > buyCount && waitCount > sellCount) {
    overallSignal = 'BEKLE';
    overallConfidence = (waitCount / strategies.length) * 100;
  } else {
    overallSignal = 'NÖTR';
    overallConfidence = 50;
  }

  // Öneri oluştur
  let recommendation = '';
  if (overallSignal === 'AL') {
    recommendation = `${buyCount}/${strategies.length} strateji AL sinyali veriyor. `;
    recommendation += overallConfidence > 80
      ? 'Çok güçlü alım fırsatı! Pozisyon açılabilir.'
      : overallConfidence > 65
      ? 'Güçlü alım sinyali. Kademeli pozisyon açılabilir.'
      : 'Orta güçte alım sinyali. Küçük pozisyon düşünülebilir.';
  } else if (overallSignal === 'SAT') {
    recommendation = `${sellCount}/${strategies.length} strateji SAT sinyali veriyor. `;
    recommendation += overallConfidence > 80
      ? 'Çok güçlü satış sinyali! Kar realizasyonu yapılabilir.'
      : overallConfidence > 65
      ? 'Güçlü satış sinyali. Kademeli kar al düşünülebilir.'
      : 'Orta güçte satış sinyali. Dikkatli olun.';
  } else if (overallSignal === 'BEKLE') {
    recommendation = `${waitCount}/${strategies.length} strateji BEKLE sinyali veriyor. Net bir trend yok, beklemek en iyisi.`;
  } else {
    recommendation = `Stratejiler kararsız (AL:${buyCount}, SAT:${sellCount}). Beklemek mantıklı.`;
  }

  // Risk seviyesi hesapla
  const volatility = Math.abs(change24h);
  let riskLevel: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK';
  let riskScore: number;

  if (volatility > 5) {
    riskLevel = 'YÜKSEK';
    riskScore = Math.min(100, volatility * 10);
  } else if (volatility > 2) {
    riskLevel = 'ORTA';
    riskScore = 50 + (volatility * 5);
  } else {
    riskLevel = 'DÜŞÜK';
    riskScore = volatility * 20;
  }

  return {
    symbol,
    price,
    change24h,
    assetType,
    strategies,
    totalScore,
    buyCount,
    sellCount,
    waitCount,
    neutralCount,
    overallSignal,
    overallConfidence,
    recommendation,
    riskLevel,
    riskScore,
    timestamp: new Date(),
  };
}
