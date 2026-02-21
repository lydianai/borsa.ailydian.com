import { NextRequest, NextResponse } from 'next/server';

const BINANCE_API = 'https://fapi.binance.com/fapi/v1';
const WHALE_THRESHOLD = 100000; // $100K+ trades

interface Trade {
  id: number;
  price: string;
  qty: string;
  quoteQty: string;
  time: number;
  isBuyerMaker: boolean;
}

/**
 * Whale Activity Detection ve Market Pressure Analysis
 * Gerçek Binance Futures verilerinden whale hareketlerini ve piyasa baskısını hesaplar
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'BTCUSDT';

  try {
    console.log(`[Whale Activity] ${symbol} analiz ediliyor...`);

    // 1. Son 1000 trade'i al
    const tradesResponse = await fetch(`${BINANCE_API}/aggTrades?symbol=${symbol}&limit=1000`);
    if (!tradesResponse.ok) throw new Error('Trade verisi alınamadı');
    const trades: Trade[] = await tradesResponse.json();

    // 2. Order book depth al
    const depthResponse = await fetch(`${BINANCE_API}/depth?symbol=${symbol}&limit=1000`);
    if (!depthResponse.ok) throw new Error('Order book verisi alınamadı');
    const depth = await depthResponse.json();

    // 3. Güncel fiyat al
    const tickerResponse = await fetch(`${BINANCE_API}/ticker/24hr?symbol=${symbol}`);
    if (!tickerResponse.ok) throw new Error('Fiyat verisi alınamadı');
    const ticker = await tickerResponse.json();
    const currentPrice = parseFloat(ticker.lastPrice);

    // ========== WHALE TRADE ANALİZİ ==========
    const whaleTrades = trades.filter((trade: Trade) => {
      const tradeValue = parseFloat(trade.quoteQty);
      return tradeValue >= WHALE_THRESHOLD;
    });

    let whaleCount = whaleTrades.length;
    let buyVolume = 0;
    let sellVolume = 0;
    let totalVolume = 0;

    whaleTrades.forEach((trade: Trade) => {
      const volume = parseFloat(trade.quoteQty);
      totalVolume += volume;

      if (!trade.isBuyerMaker) {
        buyVolume += volume; // BUY order
      } else {
        sellVolume += volume; // SELL order
      }
    });

    const avgTradeSize = whaleCount > 0 ? totalVolume / whaleCount : 0;
    const whaleDetected = whaleCount >= 3 && totalVolume >= 300000;

    // ========== MARKET PRESSURE ==========
    const bids = depth.bids as [string, string][];
    const asks = depth.asks as [string, string][];

    let bidVolume = 0;
    let askVolume = 0;

    bids.slice(0, 100).forEach(([price, qty]) => {
      bidVolume += parseFloat(qty) * parseFloat(price);
    });

    asks.slice(0, 100).forEach(([price, qty]) => {
      askVolume += parseFloat(qty) * parseFloat(price);
    });

    const bidAskRatio = askVolume > 0 ? bidVolume / askVolume : 1;
    const buyPressure = buyVolume;
    const sellPressure = sellVolume;
    const netPressure = buyPressure - sellPressure;
    const pressureRatio = sellPressure > 0 ? buyPressure / sellPressure : 1;

    let pressureSignal = 'NÖTR';
    let pressureConfidence = 50;

    if (bidAskRatio > 1.3 && pressureRatio > 1.3 && netPressure > 100000) {
      pressureSignal = 'ALIM';
      pressureConfidence = Math.min(95, 50 + Math.round((bidAskRatio - 1) * 20) + Math.round((pressureRatio - 1) * 15));
    } else if (bidAskRatio < 0.7 && pressureRatio < 0.7 && netPressure < -100000) {
      pressureSignal = 'SATIM';
      pressureConfidence = Math.min(95, 50 + Math.round((1 - bidAskRatio) * 20) + Math.round((1 - pressureRatio) * 15));
    } else if (bidAskRatio > 1.15 && pressureRatio > 1.1) {
      pressureSignal = 'ALIM';
      pressureConfidence = Math.min(85, 50 + Math.round((bidAskRatio - 1) * 15));
    } else if (bidAskRatio < 0.85 && pressureRatio < 0.9) {
      pressureSignal = 'SATIM';
      pressureConfidence = Math.min(85, 50 + Math.round((1 - bidAskRatio) * 15));
    }

    // ========== BİRİKİM PATTERN ==========
    const now = Date.now();
    const oneHourAgo = now - (60 * 60 * 1000);
    const recentWhaleTrades = whaleTrades.filter((trade: Trade) => trade.time >= oneHourAgo);
    
    let recentBuyVolume = 0;
    let recentSellVolume = 0;

    recentWhaleTrades.forEach((trade: Trade) => {
      const volume = parseFloat(trade.quoteQty);
      if (!trade.isBuyerMaker) {
        recentBuyVolume += volume;
      } else {
        recentSellVolume += volume;
      }
    });

    let accumulationSignal = 'Belirsiz';
    let accumulationConfidence = 0;

    if (recentBuyVolume > recentSellVolume * 1.5 && recentBuyVolume > 200000) {
      accumulationSignal = 'Güçlü birikim (whale alımları)';
      accumulationConfidence = Math.min(95, Math.round((recentBuyVolume / recentSellVolume) * 30));
    } else if (recentSellVolume > recentBuyVolume * 1.5 && recentSellVolume > 200000) {
      accumulationSignal = 'Dağıtım (whale satışları)';
      accumulationConfidence = Math.min(95, Math.round((recentSellVolume / recentBuyVolume) * 30));
    } else if (recentBuyVolume > recentSellVolume * 1.2) {
      accumulationSignal = 'Hafif birikim';
      accumulationConfidence = 60;
    } else if (recentSellVolume > recentBuyVolume * 1.2) {
      accumulationSignal = 'Hafif dağıtım';
      accumulationConfidence = 60;
    }

    return NextResponse.json({
      success: true,
      data: {
        whale_activity: whaleDetected ? {
          detected: true,
          whale_count: whaleCount,
          buy_volume: buyVolume,
          sell_volume: sellVolume,
          total_volume: totalVolume,
          avg_trade_size: avgTradeSize,
          timeframe: '1000 trades',
          description: `${whaleCount} büyük whale işlemi tespit edildi (toplam $${(totalVolume / 1e6).toFixed(2)}M)`
        } : null,
        pressure: {
          signal: pressureSignal,
          confidence: pressureConfidence,
          bid_ask_ratio: bidAskRatio,
          net_pressure: netPressure,
          buy_pressure: buyPressure,
          sell_pressure: sellPressure,
          description: pressureSignal === 'ALIM'
            ? `Güçlü alım baskısı (${pressureConfidence}% güven)`
            : pressureSignal === 'SATIM'
            ? `Güçlü satım baskısı (${pressureConfidence}% güven)`
            : 'Dengeli piyasa'
        },
        accumulation: {
          signal: accumulationSignal,
          confidence: accumulationConfidence,
          recent_buy_volume: recentBuyVolume,
          recent_sell_volume: recentSellVolume,
          period: '1 saat',
          description: accumulationSignal === 'Belirsiz'
            ? 'Belirgin bir birikim/dağıtım paterni yok'
            : accumulationSignal
        },
        current_price: currentPrice,
        symbol: symbol,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error: any) {
    console.error('[Whale Activity] Hata:', error);
    return NextResponse.json({
      success: false,
      error: error.message,
      data: {
        whale_activity: null,
        pressure: null,
        accumulation: null,
        current_price: null
      }
    }, { status: 500 });
  }
}
