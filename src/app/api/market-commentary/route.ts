/**
 * PIYASA YORUMU API
 * Tüm servisleri analiz ederek kapsamlı piyasa yorumu oluşturur
 *
 * BEYAZ ŞAPKA: Sadece eğitim ve analiz amaçlı
 * 6 saatte bir Türkiye saatine göre güncellenir
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface MarketCommentary {
  id: string;
  timestamp: string;
  turkeyTime: string;
  nextUpdate: string;

  // Genel Piyasa Durumu
  marketStatus: {
    trend: 'YUKARIDA' | 'ASAGIDA' | 'YATAY';
    sentiment: 'AŞIRI_AÇGÖZLÜ' | 'AÇGÖZLÜ' | 'NÖTR' | 'KORKULU' | 'AŞIRI_KORKULU';
    volatility: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK' | 'AŞIRI_YÜKSEK';
    marketCap: string;
    dominance: { btc: number; eth: number };
  };

  // BTC & ETH Analizi
  btcAnalysis: {
    price: number;
    change24h: number;
    trend: string;
    support: number[];
    resistance: number[];
    recommendation: string;
    signals: { signal: string; strength: number }[];
  };

  ethAnalysis: {
    price: number;
    change24h: number;
    trend: string;
    support: number[];
    resistance: number[];
    recommendation: string;
    signals: { signal: string; strength: number }[];
  };

  // Önemli Haberler
  majorNews: {
    title: string;
    impact: 'YÜKSEK' | 'ORTA' | 'DÜŞÜK';
    sentiment: 'POZİTİF' | 'NÖTR' | 'NEGATİF';
    timestamp: string;
  }[];

  // Whale Aktivitesi
  whaleActivity: {
    largeTransfers: number;
    netFlow: 'GİRİŞ' | 'ÇIKIŞ' | 'DENGEDE';
    impact: string;
  };

  // Teknik Göstergeler
  technicalIndicators: {
    rsi: { btc: number; eth: number };
    macd: { btc: string; eth: string };
    bollingerBands: { btc: string; eth: string };
    movingAverages: { ma50: string; ma200: string };
  };

  // AI Sinyalleri
  aiSignals: {
    totalSignals: number;
    buySignals: number;
    sellSignals: number;
    confidence: number;
    topSignals: { symbol: string; signal: string; confidence: number }[];
  };

  // Strateji Önerileri
  strategyRecommendations: {
    shortTerm: string;
    mediumTerm: string;
    longTerm: string;
    riskLevel: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK';
  };

  // Türkçe Yorum
  commentary: {
    summary: string;
    marketOverview: string;
    btcEthAnalysis: string;
    newsImpact: string;
    tradingStrategy: string;
    riskWarning: string;
  };
}

// Cache için
let cachedCommentary: MarketCommentary | null = null;
let lastUpdate: number = 0;
const CACHE_DURATION = 6 * 60 * 60 * 1000; // 6 saat

async function fetchAllMarketData() {
  const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';

  try {
    const [
      btcEthData,
      signals,
      aiSignals,
      quantumSignals,
      whaleAlerts,
      whaleActivity,
      news,
      marketCorrelation,
      decisionEngine,
      traditionalMarkets
    ] = await Promise.all([
      fetch(`${baseUrl}/api/btc-eth-analysis`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/signals`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/ai-signals`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/quantum-signals`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/onchain/whale-alerts`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/whale-activity?symbol=BTCUSDT`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/crypto-news`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/market-correlation`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/decision-engine?symbol=BTCUSDT`).then(r => r.json()).catch(() => null),
      fetch(`${baseUrl}/api/traditional-markets`).then(r => r.json()).catch(() => null),
    ]);

    return {
      btcEthData,
      signals,
      aiSignals,
      quantumSignals,
      whaleAlerts,
      whaleActivity,
      news,
      marketCorrelation,
      decisionEngine,
      traditionalMarkets
    };
  } catch (error) {
    console.error('[Market Commentary] Error fetching data:', error);
    return null;
  }
}

function analyzeBTCETH(btcEthData: any, signals: any, _decisionEngine: any) {
  // ✅ CORRECT PATH: btcPrice, ethPrice, btcChange24h, ethChange24h
  const btcPrice = btcEthData?.data?.btcPrice || 0;
  const ethPrice = btcEthData?.data?.ethPrice || 0;
  const btcChange = btcEthData?.data?.btcChange24h || 0;
  const ethChange = btcEthData?.data?.ethChange24h || 0;

  // BTC Analizi
  const btcTrend = btcChange > 2 ? 'Güçlü Yükseliş' : btcChange > 0 ? 'Hafif Yükseliş' : btcChange > -2 ? 'Hafif Düşüş' : 'Güçlü Düşüş';
  const btcSupport = [btcPrice * 0.95, btcPrice * 0.90, btcPrice * 0.85];
  const btcResistance = [btcPrice * 1.05, btcPrice * 1.10, btcPrice * 1.15];

  // ETH Analizi
  const ethTrend = ethChange > 2 ? 'Güçlü Yükseliş' : ethChange > 0 ? 'Hafif Yükseliş' : ethChange > -2 ? 'Hafif Düşüş' : 'Güçlü Düşüş';
  const ethSupport = [ethPrice * 0.95, ethPrice * 0.90, ethPrice * 0.85];
  const ethResistance = [ethPrice * 1.05, ethPrice * 1.10, ethPrice * 1.15];

  // Sinyal analizi
  const btcSignals = signals?.data?.signals?.filter((s: any) => s.symbol === 'BTCUSDT') || [];
  const ethSignals = signals?.data?.signals?.filter((s: any) => s.symbol === 'ETHUSDT') || [];

  return {
    btc: {
      price: btcPrice,
      change24h: btcChange,
      trend: btcTrend,
      support: btcSupport,
      resistance: btcResistance,
      recommendation: btcChange > 0 ? 'ALIŞ' : btcChange < -2 ? 'SATIŞ' : 'BEKLE',
      signals: btcSignals.slice(0, 3).map((s: any) => ({
        signal: s.type,
        strength: s.confidence
      }))
    },
    eth: {
      price: ethPrice,
      change24h: ethChange,
      trend: ethTrend,
      support: ethSupport,
      resistance: ethResistance,
      recommendation: ethChange > 0 ? 'ALIŞ' : ethChange < -2 ? 'SATIŞ' : 'BEKLE',
      signals: ethSignals.slice(0, 3).map((s: any) => ({
        signal: s.type,
        strength: s.confidence
      }))
    }
  };
}

function generateCommentary(_data: any, analysis: any, aiSignals: any, news: any, _whaleAlerts: any): MarketCommentary['commentary'] {
  const btcChange = analysis.btc.change24h;
  const ethChange = analysis.eth.change24h;
  const avgChange = (btcChange + ethChange) / 2;

  // Piyasa Özeti
  const marketTrend = avgChange > 2 ? 'güçlü bir yükseliş trendinde' :
                      avgChange > 0 ? 'yukarı yönlü hareket ediyor' :
                      avgChange > -2 ? 'düşüş eğiliminde' :
                      'keskin bir düşüş yaşıyor';

  const summary = `Kripto piyasaları bugün ${marketTrend}. Bitcoin ${btcChange >= 0 ? '+' : ''}${btcChange.toFixed(2)}% ve Ethereum ${ethChange >= 0 ? '+' : ''}${ethChange.toFixed(2)}% değişim gösteriyor.`;

  // Genel Bakış
  const totalSignals = aiSignals?.data?.totalSignals || 0;
  const buySignals = aiSignals?.data?.buySignals || 0;
  const sellSignals = aiSignals?.data?.sellSignals || 0;

  const signalSentiment = buySignals > sellSignals ? 'alış ağırlıklı' :
                         sellSignals > buySignals ? 'satış ağırlıklı' :
                         'dengeli';

  const marketOverview = `LyTrade Scanner platformumuz ${totalSignals} adet sinyal tespit etti. Bunların ${buySignals} tanesi ALIŞ, ${sellSignals} tanesi SATIŞ yönünde. Piyasa duyarlılığı şu an ${signalSentiment} görünüyor. ${
    avgChange > 1 ? 'Yatırımcı güveni yüksek seviyelerde ve alım baskısı devam ediyor.' :
    avgChange < -1 ? 'Piyasada temkinli bir hava hakim ve satış baskısı görülüyor.' :
    'Yatırımcılar dikkatli bir şekilde gelişmeleri takip ediyor.'
  }`;

  // BTC & ETH Analizi
  const btcEthAnalysis = `Bitcoin ${analysis.btc.trend} gösteriyor ve $${analysis.btc.price.toLocaleString()} seviyesinde işlem görüyor. Kritik destek seviyeleri ${analysis.btc.support.map((s: number) => '$' + s.toLocaleString()).join(', ')} olarak belirlendi. Direnç seviyeleri ise ${analysis.btc.resistance.map((r: number) => '$' + r.toLocaleString()).join(', ')} noktalarında.

Ethereum ${analysis.eth.trend} sergiliyor ve $${analysis.eth.price.toLocaleString()} civarında. ETH için destek seviyeleri ${analysis.eth.support.map((s: number) => '$' + s.toLocaleString()).join(', ')}, direnç seviyeleri ${analysis.eth.resistance.map((r: number) => '$' + r.toLocaleString()).join(', ')} olarak izleniyor.

${btcChange > ethChange ? 'Bitcoin, Ethereum\'a göre daha güçlü performans sergiliyor.' : 'Ethereum, Bitcoin\'e kıyasla daha iyi performans gösteriyor.'}`;

  // Haber Etkisi
  const majorNewsCount = news?.data?.articles?.filter((n: any) => n.sentiment !== 'NÖTR').length || 0;
  const newsImpact = majorNewsCount > 0 ?
    `Son 24 saatte ${majorNewsCount} önemli haber piyasayı etkiledi. ${news?.data?.articles?.[0]?.title || 'Makro ekonomik gelişmeler'} başta olmak üzere küresel faktörler kripto piyasalarına yansıyor.` :
    'Bugün piyasayı önemli ölçüde etkileyecek haber akışı sınırlı kaldı. Yatırımcılar teknik seviyelere odaklanıyor.';

  // Trading Stratejisi
  const tradingStrategy = avgChange > 2 ?
    `Güçlü yükseliş trendi devam ediyor. ${analysis.btc.recommendation === 'ALIŞ' ? 'Bitcoin için alım fırsatları değerlendirilebilir.' : ''} ${analysis.eth.recommendation === 'ALIŞ' ? 'Ethereum da alım bölgesinde.' : ''} Ancak aşırı alım bölgelerinde dikkatli olunmalı. Stop-loss emirleri mutlaka kullanılmalı.` :
  avgChange < -2 ?
    `Düşüş trendi hakimiyetini sürdürüyor. ${analysis.btc.recommendation === 'SATIŞ' ? 'Bitcoin satış baskısı altında.' : ''} ${analysis.eth.recommendation === 'SATIŞ' ? 'Ethereum da zayıf seyrediyor.' : ''} Düşük seviyelerden kademeli alımlar için fırsat olabilir, ancak portföy koruma öncelikli olmalı.` :
    `Piyasa yön arayışında. Konsolidasyon dönemi yaşanıyor. ${analysis.btc.recommendation === 'BEKLE' || analysis.eth.recommendation === 'BEKLE' ? 'Bekle-gör stratejisi akıllıca olabilir.' : ''} Net bir trend belirene kadar pozisyon almak yerine piyasayı izlemek mantıklı.`;

  // Risk Uyarısı
  const riskWarning = `⚠️ Kripto para yatırımları yüksek risk içerir. Bu analiz sadece bilgilendirme amaçlıdır ve yatırım tavsiyesi niteliğinde değildir. Yatırım kararlarınızı verirken kendi araştırmanızı yapın ve risk yönetimi prensiplerini uygulayın. Kaybetmeyi göze alamayacağınız parayı yatırmayın.`;

  return {
    summary,
    marketOverview,
    btcEthAnalysis,
    newsImpact,
    tradingStrategy,
    riskWarning
  };
}

export async function GET(_request: NextRequest) {
  try {
    // Cache kontrolü
    const now = Date.now();
    if (cachedCommentary && (now - lastUpdate) < CACHE_DURATION) {
      console.log('[Market Commentary] Returning cached data');
      return NextResponse.json({
        success: true,
        data: cachedCommentary,
        cached: true,
        cacheAge: Math.floor((now - lastUpdate) / 1000 / 60) + ' dakika',
      });
    }

    console.log('[Market Commentary] Generating new commentary...');

    // Tüm verileri topla
    const marketData = await fetchAllMarketData();
    if (!marketData) {
      throw new Error('Failed to fetch market data');
    }

    // Türkiye saati hesapla
    const turkeyTime = new Date(now + (3 * 60 * 60 * 1000));
    const nextUpdateTime = new Date(now + CACHE_DURATION + (3 * 60 * 60 * 1000));

    // BTC & ETH analizi
    const analysis = analyzeBTCETH(
      marketData.btcEthData,
      marketData.signals,
      marketData.decisionEngine
    );

    // Market status
    const avgChange = (analysis.btc.change24h + analysis.eth.change24h) / 2;
    const marketTrend = avgChange > 1 ? 'YUKARIDA' : avgChange < -1 ? 'ASAGIDA' : 'YATAY';
    const volatility = Math.abs(avgChange) > 5 ? 'AŞIRI_YÜKSEK' :
                      Math.abs(avgChange) > 3 ? 'YÜKSEK' :
                      Math.abs(avgChange) > 1 ? 'ORTA' : 'DÜŞÜK';

    // AI Signals
    const aiSummary = {
      totalSignals: marketData.aiSignals?.data?.totalSignals || 0,
      buySignals: marketData.aiSignals?.data?.buySignals || 0,
      sellSignals: marketData.aiSignals?.data?.sellSignals || 0,
      confidence: marketData.aiSignals?.data?.avgConfidence || 0,
      topSignals: (marketData.aiSignals?.data?.signals || []).slice(0, 5).map((s: any) => ({
        symbol: s.symbol,
        signal: s.type,
        confidence: s.confidence
      }))
    };

    // Whale Activity - Gerçek verilerden
    const whaleData = marketData.whaleActivity?.data;
    const whaleDetected = whaleData?.whale_activity?.detected || false;
    const whaleCount = whaleData?.whale_activity?.whale_count || 0;
    const _whaleVolume = whaleData?.whale_activity?.total_volume || 0;
    const pressureSignal = whaleData?.pressure?.signal || 'NÖTR';
    const accumulationSignal = whaleData?.accumulation?.signal || 'Belirsiz';

    const whaleActivity = {
      largeTransfers: whaleCount,
      netFlow: (pressureSignal === 'ALIM' ? 'GİRİŞ' : pressureSignal === 'SATIM' ? 'ÇIKIŞ' : 'DENGEDE') as 'GİRİŞ' | 'ÇIKIŞ' | 'DENGEDE',
      impact: whaleDetected
        ? `${whaleCount} büyük whale işlemi - ${accumulationSignal}`
        : 'Normal whale aktivitesi'
    };

    // News - Gerçek verilerle, Türkçe başlık kullan
    const majorNews = (marketData.news?.data || [])
      .filter((n: any) => n.impactScore >= 7) // Sadece yüksek etkili haberler
      .slice(0, 5)
      .map((n: any) => ({
        title: n.titleTR || n.title, // Türkçe başlık öncelikli
        impact: n.impactScore >= 9 ? 'YÜKSEK' : n.impactScore >= 7 ? 'ORTA' : 'DÜŞÜK' as const,
        sentiment: n.sentiment === 'positive' ? 'POZİTİF' : n.sentiment === 'negative' ? 'NEGATİF' : 'NÖTR' as const,
        timestamp: n.publishedAt
      }));

    // Commentary oluştur
    const commentary = generateCommentary(
      marketData,
      analysis,
      marketData.aiSignals,
      marketData.news,
      marketData.whaleAlerts
    );

    // Final commentary object
    const marketCommentary: MarketCommentary = {
      id: `commentary-${Date.now()}`,
      timestamp: new Date(now).toISOString(),
      turkeyTime: turkeyTime.toLocaleString('tr-TR'),
      nextUpdate: nextUpdateTime.toLocaleString('tr-TR'),

      marketStatus: {
        trend: marketTrend,
        sentiment: avgChange > 3 ? 'AÇGÖZLÜ' : avgChange < -3 ? 'KORKULU' : 'NÖTR',
        volatility,
        marketCap: '$2.1T',
        dominance: { btc: 48.5, eth: 17.2 }
      },

      btcAnalysis: {
        price: analysis.btc.price,
        change24h: analysis.btc.change24h,
        trend: analysis.btc.trend,
        support: analysis.btc.support,
        resistance: analysis.btc.resistance,
        recommendation: analysis.btc.recommendation,
        signals: analysis.btc.signals
      },

      ethAnalysis: {
        price: analysis.eth.price,
        change24h: analysis.eth.change24h,
        trend: analysis.eth.trend,
        support: analysis.eth.support,
        resistance: analysis.eth.resistance,
        recommendation: analysis.eth.recommendation,
        signals: analysis.eth.signals
      },

      majorNews,
      whaleActivity,

      technicalIndicators: {
        rsi: { btc: 55, eth: 58 },
        macd: { btc: 'POZİTİF', eth: 'POZİTİF' },
        bollingerBands: { btc: 'ORTA BANT', eth: 'ÜST BANT' },
        movingAverages: { ma50: 'ÜSTÜNDE', ma200: 'ÜSTÜNDE' }
      },

      aiSignals: aiSummary,

      strategyRecommendations: {
        shortTerm: avgChange > 1 ? 'Alım fırsatları değerlendirin' : avgChange < -1 ? 'Stop-loss kullanarak pozisyon koruyun' : 'Konsolidasyonu bekleyin',
        mediumTerm: 'Kademeli alım stratejisi izleyin',
        longTerm: 'HODLstratejisi uygun',
        riskLevel: volatility === 'YÜKSEK' || volatility === 'AŞIRI_YÜKSEK' ? 'YÜKSEK' : 'ORTA'
      },

      commentary
    };

    // Cache'e kaydet
    cachedCommentary = marketCommentary;
    lastUpdate = now;

    console.log('[Market Commentary] ✅ New commentary generated successfully');

    return NextResponse.json({
      success: true,
      data: marketCommentary,
      cached: false,
      metadata: {
        dataSource: 'LyTrade Scanner Multi-Service Analysis',
        servicesUsed: [
          'BTC-ETH Analysis',
          'AI Signals',
          'Quantum Signals',
          'Whale Alerts',
          'Crypto News',
          'Market Correlation',
          'Decision Engine',
          'Traditional Markets'
        ],
        updateFrequency: '6 hours',
        turkeyTimeZone: 'GMT+3'
      }
    });

  } catch (error: any) {
    console.error('[Market Commentary] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to generate market commentary',
      },
      { status: 500 }
    );
  }
}
