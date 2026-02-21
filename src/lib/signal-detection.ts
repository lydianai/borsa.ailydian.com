/**
 * Advanced Signal Detection for LONG positions
 * Based on candle patterns + indicator confirmations
 */

export interface LongSignal {
  detected: boolean;
  confidence: number; // 0-100
  candlePattern: string;
  indicators: {
    rsi: { value: number; signal: string };
    mfi: { value: number; signal: string };
    ma: { signal: string };
    bollinger: { signal: string };
  };
  explanation: string;
  timestamp: number;
  price: number;
}

export function detectLongSignal(
  candles: any[],
  rsiData: any[],
  mfiData: any[],
  ma7: any[],
  ma25: any[],
  bollingerBands: any[]
): LongSignal | null {
  if (candles.length < 10) return null;

  const lastCandle = candles[candles.length - 1];
  const prevCandle = candles[candles.length - 2];
  const lastRSI = rsiData[rsiData.length - 1]?.y || 50;
  const lastMFI = mfiData[mfiData.length - 1]?.y || 50;

  let confidence = 0;
  let candlePattern = '';
  const indicators: any = {
    rsi: { value: lastRSI, signal: '' },
    mfi: { value: lastMFI, signal: '' },
    ma: { signal: '' },
    bollinger: { signal: '' }
  };
  const reasons: string[] = [];

  // 1. CANDLE PATTERN DETECTION
  const isBullishEngulfing =
    prevCandle.close < prevCandle.open && // Prev bearish
    lastCandle.close > lastCandle.open && // Current bullish
    lastCandle.open < prevCandle.close &&
    lastCandle.close > prevCandle.open;

  const isHammer =
    lastCandle.close > lastCandle.open &&
    (lastCandle.low - lastCandle.open) > 2 * (lastCandle.close - lastCandle.open) &&
    (lastCandle.high - lastCandle.close) < 0.1 * (lastCandle.close - lastCandle.open);

  const isMorningStar =
    candles.length >= 3 &&
    candles[candles.length - 3].close < candles[candles.length - 3].open &&
    Math.abs(candles[candles.length - 2].close - candles[candles.length - 2].open) <
      (candles[candles.length - 3].high - candles[candles.length - 3].low) * 0.3 &&
    lastCandle.close > lastCandle.open;

  if (isBullishEngulfing) {
    candlePattern = 'Bullish Engulfing';
    confidence += 25;
    reasons.push('🟢 Yükseliş Yutan Mum (Bullish Engulfing) tespit edildi');
  } else if (isHammer) {
    candlePattern = 'Hammer';
    confidence += 20;
    reasons.push('🔨 Çekiç (Hammer) formasyonu oluştu');
  } else if (isMorningStar) {
    candlePattern = 'Morning Star';
    confidence += 30;
    reasons.push('⭐ Sabah Yıldızı (Morning Star) formasyonu');
  }

  // 2. RSI CONFIRMATION
  if (lastRSI < 30) {
    confidence += 20;
    indicators.rsi.signal = 'Aşırı Satım (Oversold)';
    reasons.push('📊 RSI: ' + lastRSI.toFixed(0) + ' - Aşırı satım bölgesinde, toparlanma bekleniyor');
  } else if (lastRSI >= 30 && lastRSI < 50) {
    confidence += 10;
    indicators.rsi.signal = 'Toparlanma';
    reasons.push('📊 RSI: ' + lastRSI.toFixed(0) + ' - Toparlanma aşamasında');
  }

  // 3. MFI CONFIRMATION
  if (lastMFI < 20) {
    confidence += 15;
    indicators.mfi.signal = 'Güçlü Alım Bölgesi';
    reasons.push('💰 MFI: ' + lastMFI.toFixed(0) + ' - Para akışı çok düşük, alım fırsatı');
  } else if (lastMFI < 40) {
    confidence += 10;
    indicators.mfi.signal = 'Alım Bölgesi';
    reasons.push('💰 MFI: ' + lastMFI.toFixed(0) + ' - Para akışı alım seviyesinde');
  }

  // 4. MOVING AVERAGE CONFIRMATION
  if (ma7.length > 0 && ma25.length > 0) {
    const lastMA7 = ma7[ma7.length - 1]?.y;
    const lastMA25 = ma25[ma25.length - 1]?.y;
    const price = lastCandle.close;

    if (lastMA7 > lastMA25 && price > lastMA7) {
      confidence += 15;
      indicators.ma.signal = 'Golden Cross Etkisi';
      reasons.push('📈 Fiyat MA7 ve MA25 üzerinde, trend yukarı yönlü');
    } else if (price > lastMA7 && lastMA7 < lastMA25) {
      confidence += 10;
      indicators.ma.signal = 'MA7 Kırılımı';
      reasons.push('📈 Fiyat MA7 üzerine çıktı, momentum artıyor');
    }
  }

  // 5. BOLLINGER BANDS CONFIRMATION
  if (bollingerBands.length > 0) {
    const lastBB = bollingerBands[bollingerBands.length - 1];
    const price = lastCandle.close;

    if (price < lastBB.lower) {
      confidence += 15;
      indicators.bollinger.signal = 'Alt Banttan Geri Dönüş';
      reasons.push('📉 Fiyat Bollinger alt bandının altında, geri dönüş olasılığı yüksek');
    } else if (price > lastBB.lower && price < lastBB.middle) {
      confidence += 5;
      indicators.bollinger.signal = 'Orta Banda Doğru Hareket';
      reasons.push('📊 Fiyat Bollinger alt bandı ile orta bant arasında');
    }
  }

  // SIGNAL DECISION
  const detected = confidence >= 40; // Minimum %40 güven

  if (!detected) return null;

  const explanation =
    '🎯 LONG SİNYALİ ALGILANDI (Güven: %' + confidence + ')\n\n' +
    (candlePattern ? '📊 Mum Formasyonu: ' + candlePattern + '\n\n' : '') +
    reasons.join('\n') + '\n\n' +
    '⏰ Zaman: ' + new Date(lastCandle.time * 1000).toLocaleString('tr-TR') + '\n' +
    '💵 Fiyat: $' + lastCandle.close.toFixed(4) + '\n\n' +
    '⚠️ Risk Yönetimi:\n' +
    '• Stop-Loss: $' + (lastCandle.close * 0.98).toFixed(4) + ' (-%2)\n' +
    '• Take-Profit: $' + (lastCandle.close * 1.05).toFixed(4) + ' (+%5)';

  return {
    detected: true,
    confidence,
    candlePattern,
    indicators,
    explanation,
    timestamp: lastCandle.time,
    price: lastCandle.close
  };
}
