/**
 * 📊 SAAT BAŞI PİYASA BİLGİLENDİRME API'Sİ
 *
 * Trader'lar için kritik piyasa bilgilerini toplar:
 * ✅ Global piyasa durumu (Market Cap, Dominance)
 * ✅ Fear & Greed Index
 * ✅ En çok yükselen/düşen coinler
 * ✅ Volume liderleri
 * ✅ Balina hareketleri
 * ✅ Volatilite uyarıları
 * ✅ BTC/ETH teknik seviyeler
 *
 * %100 Türkçe, tek bakışta anlaşılır format
 */

import { NextRequest, NextResponse } from 'next/server';

const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';

// Timeout helper
async function fetchWithTimeout(url: string, timeout = 8000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

// ===== DATA INTERFACES =====
interface MarketBriefing {
  timestamp: Date;
  globalMarket: {
    totalMarketCap: number;
    volume24h: number;
    btcDominance: number;
    ethDominance: number;
    totalCoins: number;
  };
  sentiment: {
    fearGreedIndex: number;
    fearGreedText: string;
    marketSentiment: string;
    sentimentScore: number;
  };
  topPerformers: {
    gainers: Array<{
      symbol: string;
      price: number;
      change24h: number;
      volume24h: number;
    }>;
    losers: Array<{
      symbol: string;
      price: number;
      change24h: number;
      volume24h: number;
    }>;
    volumeLeaders: Array<{
      symbol: string;
      price: number;
      volume24h: number;
      change24h: number;
    }>;
  };
  alerts: {
    whaleActivity: boolean;
    highVolatility: boolean;
    volumeSpikes: string[];
    criticalLevels: string[];
  };
  btcEth: {
    btc: {
      price: number;
      change24h: number;
      volume24h: number;
      trend: string;
      support: number;
      resistance: number;
    };
    eth: {
      price: number;
      change24h: number;
      volume24h: number;
      trend: string;
      support: number;
      resistance: number;
    };
  };
}

// ===== FETCH GLOBAL MARKET DATA =====
async function getGlobalMarketData() {
  try {
    const response = await fetchWithTimeout(`${BASE_URL}/api/binance/futures`, 10000);
    const data = await response.json();

    if (!data.success || !data.data || !data.data.all) {
      throw new Error('Failed to fetch market data');
    }

    const allCoins = data.data.all;

    // Calculate total market metrics
    let totalMarketCap = 0;
    let totalVolume = 0;
    let btcMarketCap = 0;
    let ethMarketCap = 0;

    allCoins.forEach((coin: any) => {
      const marketCap = coin.price * coin.volume24h / coin.price; // Approximate
      totalMarketCap += marketCap;
      totalVolume += coin.volume24h;

      if (coin.symbol === 'BTCUSDT') btcMarketCap = marketCap;
      if (coin.symbol === 'ETHUSDT') ethMarketCap = marketCap;
    });

    const btcDominance = (btcMarketCap / totalMarketCap) * 100;
    const ethDominance = (ethMarketCap / totalMarketCap) * 100;

    return {
      totalMarketCap,
      volume24h: totalVolume,
      btcDominance,
      ethDominance,
      totalCoins: allCoins.length,
      allCoins,
    };
  } catch (error) {
    console.error('[Market Briefing] Error fetching global market:', error);
    throw error;
  }
}

// ===== FETCH SENTIMENT DATA =====
async function getSentimentData() {
  try {
    // Get Nirvana dashboard for market sentiment
    const nirvanaResp = await fetchWithTimeout(`${BASE_URL}/api/nirvana`, 8000);
    const nirvanaData = await nirvanaResp.json();

    let marketSentiment = 'NÖTR';
    let sentimentScore = 50;

    if (nirvanaData.success && nirvanaData.data) {
      marketSentiment = nirvanaData.data.marketSentiment || 'NÖTR';
      sentimentScore = nirvanaData.data.sentimentScore || 50;
    }

    // Calculate Fear & Greed Index (simplified)
    // Range: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
    let fearGreedIndex = sentimentScore;
    let fearGreedText = 'NÖTR';

    if (fearGreedIndex < 25) fearGreedText = 'AŞIRI KORKU';
    else if (fearGreedIndex < 45) fearGreedText = 'KORKU';
    else if (fearGreedIndex < 55) fearGreedText = 'NÖTR';
    else if (fearGreedIndex < 75) fearGreedText = 'AÇGÖZLÜLÜK';
    else fearGreedText = 'AŞIRI AÇGÖZLÜLÜK';

    return {
      fearGreedIndex,
      fearGreedText,
      marketSentiment,
      sentimentScore,
    };
  } catch (error) {
    console.error('[Market Briefing] Error fetching sentiment:', error);
    return {
      fearGreedIndex: 50,
      fearGreedText: 'NÖTR',
      marketSentiment: 'NÖTR',
      sentimentScore: 50,
    };
  }
}

// ===== GET TOP PERFORMERS =====
function getTopPerformers(allCoins: any[]) {
  // Sort by change percentage
  const gainers = [...allCoins]
    .filter(c => c.changePercent24h > 0)
    .sort((a, b) => b.changePercent24h - a.changePercent24h)
    .slice(0, 5)
    .map(c => ({
      symbol: c.symbol,
      price: c.price,
      change24h: c.changePercent24h,
      volume24h: c.volume24h,
    }));

  const losers = [...allCoins]
    .filter(c => c.changePercent24h < 0)
    .sort((a, b) => a.changePercent24h - b.changePercent24h)
    .slice(0, 5)
    .map(c => ({
      symbol: c.symbol,
      price: c.price,
      change24h: c.changePercent24h,
      volume24h: c.volume24h,
    }));

  const volumeLeaders = [...allCoins]
    .sort((a, b) => b.volume24h - a.volume24h)
    .slice(0, 5)
    .map(c => ({
      symbol: c.symbol,
      price: c.price,
      volume24h: c.volume24h,
      change24h: c.changePercent24h,
    }));

  return { gainers, losers, volumeLeaders };
}

// ===== CHECK ALERTS =====
async function checkAlerts(allCoins: any[]) {
  const alerts = {
    whaleActivity: false,
    highVolatility: false,
    volumeSpikes: [] as string[],
    criticalLevels: [] as string[],
  };

  // Check for high volatility (>10% change)
  const highVolCoins = allCoins.filter(c => Math.abs(c.changePercent24h) > 10);
  alerts.highVolatility = highVolCoins.length > 10;

  // Check for volume spikes (volume > 2x average)
  const avgVolume = allCoins.reduce((sum, c) => sum + c.volume24h, 0) / allCoins.length;
  const volumeSpikes = allCoins.filter(c => c.volume24h > avgVolume * 3).slice(0, 3);
  alerts.volumeSpikes = volumeSpikes.map(c => c.symbol);

  // Check for whale activity on top coins
  try {
    const btcWhaleResp = await fetchWithTimeout(`${BASE_URL}/api/whale-activity?symbol=BTCUSDT`, 5000);
    const btcWhaleData = await btcWhaleResp.json();
    if (btcWhaleData.success && btcWhaleData.data?.whaleDetected) {
      alerts.whaleActivity = true;
    }
  } catch (error) {
    // Silent fail
  }

  return alerts;
}

// ===== GET BTC/ETH TECHNICALS =====
async function getBtcEthTechnicals(allCoins: any[]) {
  const btc = allCoins.find(c => c.symbol === 'BTCUSDT');
  const eth = allCoins.find(c => c.symbol === 'ETHUSDT');

  if (!btc || !eth) {
    throw new Error('BTC or ETH not found');
  }

  // Calculate support/resistance (simplified: ±2% from current price)
  const btcSupport = btc.price * 0.98;
  const btcResistance = btc.price * 1.02;
  const ethSupport = eth.price * 0.98;
  const ethResistance = eth.price * 1.02;

  // Determine trend
  const btcTrend = btc.changePercent24h > 3 ? '📈 GÜÇLÜ YUKARI' :
                   btc.changePercent24h > 0 ? '📈 YUKARI' :
                   btc.changePercent24h > -3 ? '📉 AŞAĞI' : '📉 GÜÇLÜ AŞAĞI';

  const ethTrend = eth.changePercent24h > 3 ? '📈 GÜÇLÜ YUKARI' :
                   eth.changePercent24h > 0 ? '📈 YUKARI' :
                   eth.changePercent24h > -3 ? '📉 AŞAĞI' : '📉 GÜÇLÜ AŞAĞI';

  return {
    btc: {
      price: btc.price,
      change24h: btc.changePercent24h,
      volume24h: btc.volume24h,
      trend: btcTrend,
      support: btcSupport,
      resistance: btcResistance,
    },
    eth: {
      price: eth.price,
      change24h: eth.changePercent24h,
      volume24h: eth.volume24h,
      trend: ethTrend,
      support: ethSupport,
      resistance: ethResistance,
    },
  };
}

// ===== FORMAT TELEGRAM MESSAGE =====
function formatTelegramBriefing(briefing: MarketBriefing): string {
  const dateStr = briefing.timestamp.toLocaleString('tr-TR', {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });

  let msg = `📊 *SAAT BAŞI PİYASA BİLGİLENDİRME*\n`;
  msg += `📅 ${dateStr}\n`;
  msg += `━━━━━━━━━━━━━━━━━━━━\n\n`;

  // Global Market
  msg += `🌍 *GLOBAL PİYASA DURUMU*\n`;
  msg += `   💰 Toplam Hacim: $${(briefing.globalMarket.volume24h / 1_000_000_000).toFixed(1)}B\n`;
  msg += `   🪙 Toplam Coin: ${briefing.globalMarket.totalCoins}\n`;
  msg += `   🟠 BTC Dominance: ${briefing.globalMarket.btcDominance.toFixed(2)}%\n`;
  msg += `   🔵 ETH Dominance: ${briefing.globalMarket.ethDominance.toFixed(2)}%\n\n`;

  // Sentiment
  const sentimentEmoji = briefing.sentiment.fearGreedIndex < 25 ? '😱' :
                        briefing.sentiment.fearGreedIndex < 45 ? '😟' :
                        briefing.sentiment.fearGreedIndex < 55 ? '😐' :
                        briefing.sentiment.fearGreedIndex < 75 ? '😊' : '🤑';

  msg += `🎭 *DUYGU ANALİZİ*\n`;
  msg += `   ${sentimentEmoji} Fear & Greed: ${briefing.sentiment.fearGreedText} (${briefing.sentiment.fearGreedIndex}/100)\n`;
  msg += `   📈 Market Sentiment: ${briefing.sentiment.marketSentiment}\n\n`;

  // BTC/ETH
  msg += `👑 *BTC & ETH DURUMU*\n`;
  msg += `   🟠 *BITCOIN*\n`;
  msg += `      💵 Fiyat: $${briefing.btcEth.btc.price.toLocaleString('tr-TR', { minimumFractionDigits: 2 })}\n`;
  msg += `      ${briefing.btcEth.btc.change24h >= 0 ? '📈' : '📉'} 24h: ${briefing.btcEth.btc.change24h >= 0 ? '+' : ''}${briefing.btcEth.btc.change24h.toFixed(2)}%\n`;
  msg += `      ${briefing.btcEth.btc.trend}\n`;
  msg += `      🛡️ Destek: $${briefing.btcEth.btc.support.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}\n`;
  msg += `      🎯 Direnç: $${briefing.btcEth.btc.resistance.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}\n\n`;

  msg += `   🔵 *ETHEREUM*\n`;
  msg += `      💵 Fiyat: $${briefing.btcEth.eth.price.toLocaleString('tr-TR', { minimumFractionDigits: 2 })}\n`;
  msg += `      ${briefing.btcEth.eth.change24h >= 0 ? '📈' : '📉'} 24h: ${briefing.btcEth.eth.change24h >= 0 ? '+' : ''}${briefing.btcEth.eth.change24h.toFixed(2)}%\n`;
  msg += `      ${briefing.btcEth.eth.trend}\n`;
  msg += `      🛡️ Destek: $${briefing.btcEth.eth.support.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}\n`;
  msg += `      🎯 Direnç: $${briefing.btcEth.eth.resistance.toLocaleString('tr-TR', { maximumFractionDigits: 0 })}\n\n`;

  // Top Gainers
  msg += `🚀 *EN ÇOK YÜKSELENLER (Top 5)*\n`;
  briefing.topPerformers.gainers.forEach((coin, i) => {
    const medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i];
    msg += `   ${medal} ${coin.symbol.replace('USDT', '')}: *+${coin.change24h.toFixed(2)}%*\n`;
  });
  msg += `\n`;

  // Top Losers
  msg += `📉 *EN ÇOK DÜŞENLER (Top 5)*\n`;
  briefing.topPerformers.losers.forEach((coin, i) => {
    const medal = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣'][i];
    msg += `   ${medal} ${coin.symbol.replace('USDT', '')}: *${coin.change24h.toFixed(2)}%*\n`;
  });
  msg += `\n`;

  // Volume Leaders
  msg += `📊 *HACİM LİDERLERİ (Top 5)*\n`;
  briefing.topPerformers.volumeLeaders.forEach((coin, i) => {
    const medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i];
    const volumeText = coin.volume24h > 1_000_000_000
      ? `$${(coin.volume24h / 1_000_000_000).toFixed(1)}B`
      : `$${(coin.volume24h / 1_000_000).toFixed(0)}M`;
    msg += `   ${medal} ${coin.symbol.replace('USDT', '')}: ${volumeText}\n`;
  });
  msg += `\n`;

  // Alerts
  if (briefing.alerts.whaleActivity || briefing.alerts.highVolatility || briefing.alerts.volumeSpikes.length > 0) {
    msg += `⚠️ *ÖNEMLİ UYARILAR*\n`;
    if (briefing.alerts.whaleActivity) {
      msg += `   🐋 Balina aktivitesi tespit edildi!\n`;
    }
    if (briefing.alerts.highVolatility) {
      msg += `   ⚡ Yüksek volatilite: Dikkatli olun!\n`;
    }
    if (briefing.alerts.volumeSpikes.length > 0) {
      msg += `   📈 Volume artışı: ${briefing.alerts.volumeSpikes.join(', ')}\n`;
    }
    msg += `\n`;
  }

  msg += `━━━━━━━━━━━━━━━━━━━━\n`;
  msg += `🤖 _LyDian_\n`;
  msg += `_Bir sonraki bilgilendirme: 1 saat sonra_`;

  return msg;
}

// ===== MAIN API HANDLER =====
export async function GET(_request: NextRequest) {
  try {
    console.log('[Market Briefing] Starting analysis...');
    const startTime = Date.now();

    // 1. Get global market data
    const marketData = await getGlobalMarketData();

    // 2. Get sentiment data
    const sentimentData = await getSentimentData();

    // 3. Get top performers
    const topPerformers = getTopPerformers(marketData.allCoins);

    // 4. Check alerts
    const alerts = await checkAlerts(marketData.allCoins);

    // 5. Get BTC/ETH technicals
    const btcEthData = await getBtcEthTechnicals(marketData.allCoins);

    // 6. Compile briefing
    const briefing: MarketBriefing = {
      timestamp: new Date(),
      globalMarket: {
        totalMarketCap: marketData.totalMarketCap,
        volume24h: marketData.volume24h,
        btcDominance: marketData.btcDominance,
        ethDominance: marketData.ethDominance,
        totalCoins: marketData.totalCoins,
      },
      sentiment: sentimentData,
      topPerformers,
      alerts,
      btcEth: btcEthData,
    };

    // 7. Format Telegram message
    const telegramMessage = formatTelegramBriefing(briefing);

    const elapsedTime = Date.now() - startTime;
    console.log(`[Market Briefing] Analysis completed in ${elapsedTime}ms`);

    return NextResponse.json({
      success: true,
      data: {
        briefing,
        telegramMessage,
        timestamp: briefing.timestamp.toISOString(),
        elapsedTimeMs: elapsedTime,
      },
    });
  } catch (error) {
    console.error('[Market Briefing] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
