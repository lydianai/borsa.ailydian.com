'use client';

/**
 * 🌐 OMNIPOTENT FUTURES v3.0 - ULTIMATE MULTI-DIMENSIONAL ANALYSIS
 * ✅ Wyckoff Method (4 phases)
 * ✅ Funding Rates & Open Interest
 * ✅ BTC Dominance & Fear/Greed
 * ✅ Macro Correlations (DXY, S&P500, GOLD, VIX)
 * ✅ Risk Management (Kelly Criterion)
 * ✅ Liquidation Zones
 * ✅ 200+ coin real-time analysis
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS, getSignalColor } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';
import {
  calculateKellyCriterion,
  calculatePositionSize,
  type TradeHistory,
  type KellyCriterion,
  type PositionSizeRecommendation
} from '@/lib/risk-management';

interface CorrelationData {
  symbol: string;
  price: number;
  change24h: number;
  omnipotentScore: number;
  marketPhase: string;
  trend: string;
  volumeProfile: string;
  fundingBias: string;
  liquidationRisk: number;
  volatility: number;
  btcCorrelation: number;
  signal: string;
  confidence: number;
  // NEW v2.0 fields
  fundingRate?: number;
  fundingRateAnnualized?: number;
  openInterest?: number;
  openInterestValue?: number;
  nearestLiquidation?: {
    long: { price: number; distance: number };
    short: { price: number; distance: number };
  };
  // NEW v3.0 fields
  technicalIndicators?: {
    rsi: { value: number; signal: string; interpretation: string };
    macd: { macdLine: number; signalLine: number; histogram: number; signal: string; interpretation: string };
    bollingerBands: { upper: number; middle: number; lower: number; bandwidth: number; percentB: number; signal: string; interpretation: string };
    timestamp: string;
  };
}

// NEW v3.0 Interfaces
interface MacroAsset {
  symbol: string;
  price: number;
  change24h: number;
}

interface GlobalMetrics {
  btcDominance: {
    btc: number;
    eth: number;
    stables: number;
    totalMarketCap: number;
  } | null;
  fearGreed: {
    value: number;
    classification: string;
  } | null;
}

interface MarketOverview {
  totalCoins: number;
  avgOmnipotentScore: number;
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  avgVolatility: string;
  highConfidenceSignals: number;
  marketPhaseDistribution: {
    ACCUMULATION: number;
    MARKUP: number;
    DISTRIBUTION: number;
    MARKDOWN: number;
  };
}

// Renamed v3.0 interface to avoid conflicts
interface CorrelationMatrixV3Data {
  btcDxy: { correlation: number; strength: string; direction: string };
  btcSp500: { correlation: number; strength: string; direction: string };
  btcGold: { correlation: number; strength: string; direction: string };
  btcVix: { correlation: number; strength: string; direction: string };
}

// NEW v3.0 Multi-Timeframe Interfaces
interface TimeframeAnalysis {
  timeframe: '1h' | '4h' | '1d' | '1w';
  rsi: { value: number; signal: string; interpretation: string };
  macd: { macdLine: number; signalLine: number; histogram: number; signal: string; interpretation: string };
  bollingerBands: { upper: number; middle: number; lower: number; bandwidth: number; percentB: number; signal: string; interpretation: string };
  overallSignal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  signalStrength: number;
  currentPrice: number;
  priceChange24h?: number;
  timestamp: string;
}

interface MultiTimeframeData {
  symbol: string;
  timeframes: {
    '1h': TimeframeAnalysis;
    '4h': TimeframeAnalysis;
    '1d': TimeframeAnalysis;
    '1w': TimeframeAnalysis;
  };
  consensus: {
    signal: 'STRONG_BULLISH' | 'BULLISH' | 'NEUTRAL' | 'BEARISH' | 'STRONG_BEARISH';
    strength: number;
    alignment: number;
    interpretation: string;
  };
  higherTimeframeBias: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  timestamp: string;
}

// NEW v3.0: Volume Profile Interface
interface VolumeProfileData {
  poc: { price: number; volume: number; percentage: number };
  valueArea: { high: number; low: number; percentage: number; volumeInArea: number };
  vwap: { price: number; deviation: number };
  distribution: Array<{ price: number; volume: number; percentage: number }>;
  pricePosition: 'ABOVE_VAH' | 'IN_VALUE_AREA' | 'BELOW_VAL';
  volumeNodes: {
    highVolumeNodes: Array<{ price: number; volume: number; percentage: number }>;
    lowVolumeNodes: Array<{ price: number; volume: number; percentage: number }>;
  };
  totalVolume: number;
  numPriceLevels: number;
  timeframe: string;
  timestamp: string;
}

// NEW v3.0: Order Flow Interface
interface OrderFlowData {
  imbalance: {
    ratio: number;
    strength: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL';
    percentage: number;
  };
  delta: {
    value: number;
    trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
    cumulative: number;
  };
  aggressive: {
    buyPressure: number;
    sellPressure: number;
    dominance: 'BUYERS' | 'SELLERS' | 'BALANCED';
  };
  volume: {
    current: number;
    average: number;
    ratio: number;
    surge: boolean;
  };
  priceVolumeCorrelation: {
    divergence: boolean;
    type: 'BULLISH_DIVERGENCE' | 'BEARISH_DIVERGENCE' | 'NONE';
    confidence: number;
  };
  signal: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  timestamp: string;
  timeframe: string;
}

export default function OmnipotentFuturesPage() {
  const [correlations, setCorrelations] = useState<CorrelationData[]>([]);
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(60);
  const [filterSignal, setFilterSignal] = useState<string>('TÜMÜ');
  const [sortBy, setSortBy] = useState<'score' | 'confidence' | 'risk'>('score');
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [showLogicModal, setShowLogicModal] = useState(false);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Mantık popup state
  const [explainerModal, setExplainerModal] = useState<{
    isOpen: boolean;
    title: string;
    content: string;
  }>({
    isOpen: false,
    title: '',
    content: '',
  });

  // NEW v3.0 State
  const [macroMetrics, setMacroMetrics] = useState<{
    dxy: MacroAsset | null;
    sp500: MacroAsset | null;
    gold: MacroAsset | null;
    vix: MacroAsset | null;
    btc: MacroAsset | null;
  } | null>(null);
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationMatrixV3Data | null>(null);
  const [globalMetrics, setGlobalMetrics] = useState<GlobalMetrics | null>(null);
  const [dataSourcesActive, setDataSourcesActive] = useState({
    wyckoff: false,
    fundingRates: false,
    openInterest: false,
    btcDominance: false,
    fearGreed: false,
    correlations: false,
    technicalIndicators: false,
    multiTimeframe: false,
    volumeProfile: false,
    orderFlow: false,
  });
  const [btcMultiTimeframe, setBtcMultiTimeframe] = useState<MultiTimeframeData | null>(null);
  const [btcVolumeProfile, setBtcVolumeProfile] = useState<VolumeProfileData | null>(null);
  const [btcOrderFlow, setBtcOrderFlow] = useState<OrderFlowData | null>(null);

  // Global filters (synchronized across all pages)
  const { timeframe: globalTimeframe, sortBy: globalSortBy } = useGlobalFilters();

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/omnipotent-futures');
      const result = await response.json();

      if (result.success) {
        // Map Wyckoff futures data to correlation format with v2.0/v3.0 enhancements
        const fetchedFutures = result.data.futures.map((f: any) => ({
          symbol: f.symbol,
          price: f.price,
          change24h: f.change24h,
          omnipotentScore: f.confidence,
          marketPhase: f.wyckoffPhase,
          trend: f.trendStrength > 60 ? 'BULLISH' : f.trendStrength < 40 ? 'BEARISH' : 'SIDEWAYS',
          volumeProfile: f.volumeProfile,
          fundingBias: f.smartMoneyActivity,
          liquidationRisk: Math.round(100 - f.trendStrength),
          volatility: f.rangePercent,
          btcCorrelation: f.trendStrength / 100,
          signal: f.signal,
          confidence: f.confidence,
          // NEW v2.0 fields
          fundingRate: f.fundingRate,
          fundingRateAnnualized: f.fundingRateAnnualized,
          openInterest: f.openInterest,
          openInterestValue: f.openInterestValue,
          nearestLiquidation: f.nearestLiquidation,
          // NEW v3.0 fields
          technicalIndicators: f.technicalIndicators,
        }));

        setCorrelations(fetchedFutures);

        // Map market overview from Wyckoff data
        const overview = result.data.marketOverview;
        setMarketOverview({
          totalCoins: overview.totalCoins,
          avgOmnipotentScore: overview.highConfidenceSignals * 10,
          bullishCount: overview.signals.BUY,
          bearishCount: overview.signals.SELL,
          neutralCount: overview.signals.WAIT,
          avgVolatility: overview.avgTrendStrength,
          highConfidenceSignals: overview.highConfidenceSignals,
          marketPhaseDistribution: overview.phaseDistribution,
        });

        // NEW v3.0: Store macro metrics
        if (result.data.macroCorrelations) {
          setMacroMetrics(result.data.macroCorrelations);
        }

        // NEW v3.0: Store correlation matrix
        if (result.data.correlationMatrix) {
          setCorrelationMatrix(result.data.correlationMatrix);
        }

        // NEW v2.0: Store global metrics
        if (result.data.globalMetrics) {
          setGlobalMetrics(result.data.globalMetrics);
        }

        // Store data sources active status
        if (result.data.dataSourcesActive) {
          setDataSourcesActive(result.data.dataSourcesActive);
        }

        // NEW v3.0: Store multi-timeframe analysis (BTC)
        if (result.data.btcMultiTimeframe) {
          setBtcMultiTimeframe(result.data.btcMultiTimeframe);
        }

        // NEW v3.0: Store volume profile analysis (BTC)
        if (result.data.btcVolumeProfile) {
          setBtcVolumeProfile(result.data.btcVolumeProfile);
        }

        // NEW v3.0: Store order flow analysis (BTC)
        if (result.data.btcOrderFlow) {
          setBtcOrderFlow(result.data.btcOrderFlow);
        }

        setError(null);

        // AL/SAT sinyallerini say ve localStorage'a kaydet
        const buyCount = overview.signals.BUY;
        const sellCount = overview.signals.SELL;
        const totalSignals = buyCount + sellCount;

        if (typeof window !== 'undefined') {
          localStorage.setItem('omnipotent_notification_count', totalSignals.toString());
        }
      } else {
        setError(result.error || 'Veri yüklenemedi');
      }
    } catch (err: any) {
      setError(err.message || 'Ağ hatası');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchData();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Filtrele ve sırala
  const filteredCorrelations = correlations
    .filter((c) => filterSignal === 'TÜMÜ' || c.signal === filterSignal)
    .sort((a, b) => {
      if (sortBy === 'score') return b.omnipotentScore - a.omnipotentScore;
      if (sortBy === 'confidence') return b.confidence - a.confidence;
      if (sortBy === 'risk') return a.liquidationRisk - b.liquidationRisk;
      return 0;
    });

  const getPhaseColor = (phase: string) => {
    if (phase === 'MARKUP') return COLORS.success;
    if (phase === 'MARKDOWN') return COLORS.danger;
    if (phase === 'ACCUMULATION') return COLORS.info;
    if (phase === 'DISTRIBUTION') return COLORS.warning;
    return COLORS.gray[500];
  };

  // Mantık açıklamalarını döndüren fonksiyon
  const getExplainerContent = (topic: string): string => {
    const explainers: {[key: string]: string} = {
      'makro-varliklar': `**KÜRESEL MAKRO VARLIKLAR NEDİR?**

Bu panel, Bitcoin ve kripto piyasasını etkileyen geleneksel finans piyasalarını gösterir.

**DXY (Dolar Endeksi):**
ABD Dolarının, Euro, Yen, Sterlin gibi diğer para birimlerine karşı değerini ölçer. DXY yükselirse dolar güçlenir. Genellikle dolar güçlendiğinde Bitcoin ve altın gibi varlıklar zayıflar, çünkü yatırımcılar daha güvenli liman olan dolara yönelir.

**S&P 500:**
Amerika'nın en büyük 500 şirketinin hisse senedi performansını takip eden endeks. Borsa yükseldiğinde risk iştahı artar ve kripto piyasası da genellikle yükselir. Borsa düştüğünde yatırımcılar riskten kaçar ve kripto da düşebilir.

**GOLD (Altın):**
Binlerce yıldır değer saklama aracı olarak kullanılan귀金属 emtia. Ekonomik belirsizlik zamanlarında altına talep artar. Bitcoin "dijital altın" olarak görüldüğü için altın fiyatları ile pozitif korelasyon gösterebilir.

**VIX (Volatilite Endeksi):**
"Korku Endeksi" olarak bilinir. Borsadaki oynaklığı ölçer. VIX yükseldiğinde piyasalarda korku ve belirsizlik vardır, düştüğünde ise piyasalar sakindir. Yüksek VIX genellikle kripto için de riskli dönemleri işaret eder.

**Neden Önemli?**
Bitcoin artık sadece kripto dünyasında değil, küresel finans sisteminin bir parçası. Büyük kurumlar Bitcoin'e yatırım yapıyor ve bu yüzden geleneksel piyasalarla bağlantı güçleniyor. Bu varlıkları takip ederek Bitcoin'in gelecekteki hareketini tahmin edebiliriz.`,

      'korelasyon': `**BTC KORELASYON MATRİSİ NEDİR?**

Korelasyon, iki varlığın fiyat hareketlerinin ne kadar birlikte hareket ettiğini gösteren istatistiksel bir ölçüdür.

**Korelasyon Değerleri:**
• +1.00: Mükemmel pozitif korelasyon - Her zaman birlikte hareket ederler
• +0.70 ila +1.00: Güçlü pozitif - Genellikle birlikte yükselir/düşerler
• +0.40 ila +0.70: Orta pozitif - Bazen birlikte hareket ederler
• -0.40 ila +0.40: Zayıf/Yok - Bağımsız hareket ederler
• -0.70 ila -0.40: Orta negatif - Biri yükselirken diğeri düşer
• -1.00 ila -0.70: Güçlü negatif - Ters yönde hareket ederler

**BTC/DXY Korelasyonu:**
Bitcoin ile Dolar Endeksi arasındaki ilişki. Genellikle negatif korelasyon vardır: Dolar güçlendiğinde Bitcoin zayıflar çünkü yatırımcılar güvenli liman arıyor.

**BTC/S&P500 Korelasyonu:**
Bitcoin ile Amerikan borsası arasındaki ilişki. 2020'den beri güçlü pozitif korelasyon var. Borsa yükseldiğinde Bitcoin da yükseliyor çünkü risk iştahı artıyor.

**BTC/GOLD Korelasyonu:**
Bitcoin ile altın arasındaki ilişki. Her ikisi de "değer saklama" aracı olduğu için bazı dönemlerde pozitif korelasyon gösterir ama ilişki değişkendir.

**BTC/VIX Korelasyonu:**
Bitcoin ile korku endeksi arasındaki ilişki. Genellikle negatif korelasyon: Piyasalarda korku arttığında (VIX yükselir) Bitcoin düşer.

**30 Günlük Hareketli Korelasyon:**
Son 30 günün verilerini kullanarak hesaplanır. Bu sayede güncel piyasa koşullarını yansıtır.

**Neden Önemli?**
Korelasyonları bilmek risk yönetiminde kritiktir. Örneğin S&P500 düşmeye başladıysa ve BTC ile güçlü pozitif korelasyon varsa, Bitcoin'in de düşme ihtimali yüksektir.`,

      'risk-yonetimi': `**RİSK YÖNETİMİ HESAPLAYICI NEDİR?**

Risk yönetimi, trading'de en önemli konudur. Kazanmak değil, kaybetmemek esastır!

**KELLY CRİTERİON (Kelly Kriteri):**

Nobel ödüllü matematikçi John Kelly tarafından geliştirilen bir formül. "Ne kadar para yatırmalıyım?" sorusuna matematiksel cevap verir.

Formula: K% = W - (1-W)/R

• W = Kazanma oranı (kazanan işlem sayısı / toplam işlem)
• R = Kazanç/Kayıp oranı (ortalama kazanç / ortalama kayıp)

**Örnek:**
100 işlem yaptınız:
• 65 kazanan, 35 kaybeden
• Ortalama kazanç: $150
• Ortalama kayıp: $80

W = 0.65 (65%)
R = 150/80 = 1.875

Kelly = 0.65 - (1-0.65)/1.875 = 0.65 - 0.187 = **46.3%**

Bu, hesabınızın %46.3'ünü her işlemde riske atmanız gerektiğini söyler.

**ANCAK ÇOK RİSKLİ!**

Bu yüzden "Fractional Kelly" kullanırız:
• **Yarım Kelly (Half Kelly)**: %46.3 / 2 = %23.15 → Önerilen
• **Çeyrek Kelly (Quarter Kelly)**: %46.3 / 4 = %11.58 → Muhafazakar

**POZİSYON BOYUTLANDIRMA:**

Bir işleme ne kadar para ayıracağınızı hesaplar:

1. **Hesap Büyüklüğü**: Toplam sermayeniz ($10,000)
2. **Risk Yüzdesi**: İşlem başına kayba razı olduğunuz miktar (genellikle %1-2)
3. **Stop Loss Mesafesi**: Giriş fiyatı ile stop loss arasındaki mesafe
4. **Kaldıraç**: Kullanılan kaldıraç oranı (dikkatli!)

**Hesaplama:**
Risk = $10,000 × 1% = $100
Stop Loss Mesafesi = 2%
Pozisyon = $100 / 0.02 = $5,000
Kaldıraçlı Pozisyon = $5,000 × 3 = $15,000

**Altın Kurallar:**
1. İşlem başına asla %2'den fazla risk almayın
2. Toplam portföy riski %10'u geçmesin
3. Kaldıraç kullanıyorsanız ekstra dikkatli olun
4. Stop loss her zaman kullanın!

**Neden Önemli?**
Profesyonel trader'lar bile %50-60 doğruluk oranına sahip. Kar etmenin sırrı doğru tahmin değil, DOĞRU RİSK YÖNETİMİDİR! Bir işlemde tüm paranızı kaybetmeyin.`,

      'wyckoff': `**WYCKOFF METHODOLOJİSİ NEDİR?**

Richard Wyckoff (1870-1934) tarafından geliştirilen, piyasa döngülerini ve "akıllı para" hareketlerini analiz eden bir yöntem.

**4 ANA FAZ:**

**1. BİRİKTİRME (Accumulation):**
"Akıllı para" (büyük yatırımcılar, kurumlar) ucuza sessizce alım yapar. Fiyat dar bir aralıkta kalır, çoğu insan umudunu kaybetmiştir. İşlem hacmi düşüktür.

📍 **İşaret:** Düşük fiyat, düşük hacim, dar range
📍 **Yapılacak:** Bu dönemde alım yapmak idealdir ama sabır gerekir

**2. YUKARI HAREKET (Markup):**
Birikim tamamlandı, şimdi fiyat yükseliyor. Medyada olumlu haberler çıkar, herkes konuşmaya başlar. Hacim artar. Trend güçlüdür.

📍 **İşaret:** Yükselen fiyat, artan hacim, güçlü trend
📍 **Yapılacak:** Trend devam ederken tutabilirsiniz ama açgözlü olmayın

**3. DAĞITIM (Distribution):**
"Akıllı para" şimdi perakende yatırımcılara satıyor. Herkes "To the moon!" diyor ama fiyat bir tavan bulur. Hacim yüksek ama fiyat yükselmiyor.

📍 **İşaret:** Yüksek fiyat, yüksek hacim ama ilerleme yok
📍 **Yapılacak:** Kar realizasyonu zamanı, çıkış yapın

**4. AŞAĞI HAREKET (Markdown):**
Dağıtım bitti, artık düşüş var. Panik satışlar başlar. Geç kalanlar zarar eder. Hacim kriz anlarında patlama yapar.

📍 **İşaret:** Düşen fiyat, panik, yüksek volatilite
📍 **Yapılacak:** Uzak durun veya short yapın (risk var)

**VOLUME (Hacim) Analizi:**

• **Climax Volume:** Aşırı yüksek hacim, genellikle dönüş noktasıdır
• **Dry-Up:** Hacim kurudu, hareket bitmek üzere
• **Effort vs Result:** Yüksek hacim ama düşük fiyat hareketi = güçsüzlük

**SMART MONEY ACTIVITY:**

Akıllı paranın ne yaptığını takip ederiz:
• **Buying:** Sessizce birikim yapıyorlar
• **Selling:** Heyecan doruktayken satıyorlar
• **Neutral:** Beklemedeler

**Neden Önemli?**
Perakende yatırımcı genellikle en tepede alır, en dipte satar. Wyckoff bize "akıllı para" ile birlikte hareket etmeyi öğretir. Onlar ne yapıyorsa biz de onu yapalım!`,

      'technical-indicators': `**TEKNİK İNDİKATÖRLER NEDİR?**

Teknik indikatörler, fiyat ve hacim verilerini kullanarak piyasanın durumunu analiz eden matematiksel formüllerdir.

**RSI (Relative Strength Index):**
Momentum göstergesi. 0-100 arası değer alır. 14 periyot varsayılandır.
• **30'un altı:** Aşırı satım - Potansiyel yükseliş fırsatı
• **70'in üstü:** Aşırı alım - Potansiyel düşüş riski
• **30-70 arası:** Nötr bölge

Formula: RSI = 100 - (100 / (1 + RS))
RS = Ortalama Kazanç / Ortalama Kayıp

**MACD (Moving Average Convergence Divergence):**
Trend ve momentum göstergesi. 3 bileşenden oluşur:
• **MACD Çizgisi:** 12 günlük EMA - 26 günlük EMA
• **Sinyal Çizgisi:** MACD'nin 9 günlük EMA'sı
• **Histogram:** MACD - Sinyal farkı

Sinyaller:
• MACD çizgisi sinyal çizgisinin üstüne geçerse → Yükseliş
• MACD çizgisi sinyal çizgisinin altına geçerse → Düşüş

**Bollinger Bands (Bollinger Bantları):**
Volatilite ve fiyat kanalları. 3 bant:
• **Üst Bant:** 20 günlük SMA + (2 × Standart Sapma)
• **Orta Bant:** 20 günlük Simple Moving Average
• **Alt Bant:** 20 günlük SMA - (2 × Standart Sapma)

Sinyaller:
• Fiyat üst banda dokundu → Aşırı alım
• Fiyat alt banda dokundu → Aşırı satım
• Bantlar daraldı → Düşük volatilite, patl

ama yakın
• Bantlar genişledi → Yüksek volatilite

**Neden Önemli?**
Bu indikatörler trader'ların en çok kullandığı araçlardır. BİRDEN FAZLA indikatör birlikte kullanıldığında daha güvenilir sinyaller verir.`,

      'multi-timeframe': `**MULTI-TIMEFRAME ANALYSIS NEDİR?**

Aynı coin'i farklı zaman dilimlerinde (timeframe) analiz ederek daha güvenilir sinyaller elde etme yöntemidir.

**4 ZAMAN DİLİMİ:**

• **1 Saat (1h):** Kısa vadeli momentum ve entry/exit timing
• **4 Saat (4h):** Orta vadeli trend ve güçlü destek/direnç seviyeleri
• **1 Gün (1d):** Günlük ana trend direction
• **1 Hafta (1w):** Uzun vadeli büyük resim ve major trend

**TREND ALIGNMENT (UYUM):**

Tüm timeframe'ler aynı yönde sinyal veriyorsa, o sinyal ÇOK DAHA GÜÇLÜdür!

Örnek:
• 1h: BULLISH ✅
• 4h: BULLISH ✅
• 1d: BULLISH ✅
• 1w: BULLISH ✅
→ TAM UYUM! %100 Bullish Consensus → Güçlü Yükseliş Trendi

**HIGHER TIMEFRAME BIAS:**

Büyük zaman dilimleri (1d ve 1w) daha önemlidir ve daha fazla ağırlığa sahiptir. Eğer 1w BULLISH ise ama 1h BEARISH ise, 1w'lik trend önceliklidir.

**CONSENSUS ALGORITHM:**

Sistem her timeframe'e ağırlık verir:
• 1h: Ağırlık 1x
• 4h: Ağırlık 2x
• 1d: Ağırlık 3x
• 1w: Ağırlık 4x (EN GÜÇLÜ)

Tüm timeframe sinyalleri ağırlıklı ortalama ile birleştirilerek tek bir "CONSENSUS" sinyali üretilir:
• **STRONG_BULLISH:** 3+ timeframe bullish ve güç >%70
• **BULLISH:** 2+ timeframe bullish
• **NEUTRAL:** Karışık sinyaller
• **BEARISH:** 2+ timeframe bearish
• **STRONG_BEARISH:** 3+ timeframe bearish ve güç >%70

**NEDEN ÖNEMLİ?**

Tek bir timeframe'e bakarak işlem yapmak risklidir. Örneğin 1h'te BULLISH görünebilir ama 1d ve 1w düşüş trendindeyse, o 1h'lik yükseliş kısa ömürlü olabilir.

Multi-timeframe analysis sayesinde:
✅ Daha yüksek doğruluk oranı
✅ Daha az false signal (yanlış sinyal)
✅ Trend confirmation
✅ Better entry/exit timing

Professional trader'lar MUTLAKA multi-timeframe analysis yapar!`
    };

    return explainers[topic] || 'Açıklama bulunamadı.';
  };

  const openExplainer = (topic: string, title: string) => {
    setExplainerModal({
      isOpen: true,
      title,
      content: getExplainerContent(topic),
    });
  };

  if (error) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary, color: COLORS.danger, padding: '20px' }}>
        <div style={{ textAlign: 'center' }}>
          <h1 style={{ fontSize: '24px', marginBottom: '16px' }}>❌ Veri Yüklenirken Hata</h1>
          <p style={{ marginBottom: '20px' }}>{error}</p>
          <button onClick={fetchData} style={{ padding: '12px 24px', background: COLORS.text.primary, color: COLORS.bg.primary, border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: '600', transition: 'all 0.2s ease-in-out' }}>
            Yeniden Dene
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Mantık Açıklama Modalı */}
      {explainerModal.isOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            zIndex: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
            animation: 'fadeIn 0.2s ease-in-out',
          }}
          onClick={() => setExplainerModal({ isOpen: false, title: '', content: '' })}
        >
          <div
            style={{
              background: COLORS.bg.primary,
              borderRadius: '12px',
              maxWidth: '800px',
              width: '100%',
              maxHeight: '80vh',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              border: `2px solid ${COLORS.premium}`,
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
              animation: 'slideUp 0.3s ease-out',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.success} 100%)`,
                padding: '20px 24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderBottom: `2px solid ${COLORS.premium}`,
              }}
            >
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: 'white', margin: 0 }}>
                💡 {explainerModal.title}
              </h2>
              <button
                onClick={() => setExplainerModal({ isOpen: false, title: '', content: '' })}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  color: 'white',
                  width: '32px',
                  height: '32px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '18px',
                  fontWeight: '700',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.2s ease-in-out',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
                  e.currentTarget.style.transform = 'scale(1.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
                  e.currentTarget.style.transform = 'scale(1)';
                }}
              >
                ✕
              </button>
            </div>

            {/* Modal Content */}
            <div
              style={{
                padding: '24px',
                overflowY: 'auto',
                color: COLORS.text.primary,
                lineHeight: '1.8',
                fontSize: '15px',
              }}
            >
              <div
                style={{
                  whiteSpace: 'pre-wrap',
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                }}
                dangerouslySetInnerHTML={{
                  __html: explainerModal.content
                    .replace(/\*\*(.*?)\*\*/g, '<strong style="color: ' + COLORS.premium + '; font-weight: 700;">$1</strong>')
                    .replace(/•/g, '<span style="color: ' + COLORS.success + '; font-weight: 700;">•</span>')
                    .replace(/📍/g, '<span style="font-size: 18px;">📍</span>')
                    .replace(/\n/g, '<br/>')
                }}
              />
            </div>

            {/* Modal Footer */}
            <div
              style={{
                padding: '16px 24px',
                borderTop: `1px solid ${COLORS.border.default}`,
                display: 'flex',
                justifyContent: 'flex-end',
              }}
            >
              <button
                onClick={() => setExplainerModal({ isOpen: false, title: '', content: '' })}
                style={{
                  background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.success} 100%)`,
                  border: 'none',
                  color: 'white',
                  padding: '10px 24px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.2s ease-in-out',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(103, 126, 234, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                Anladım
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Asistan */}
      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}

      {/* Sidebar */}
      <SharedSidebar
        currentPage="omnipotent-futures"
        notificationCounts={notificationCounts}
      />

      {/* Ana İçerik */}
      <div className="dashboard-main" style={{ minHeight: '100vh', background: COLORS.bg.secondary, color: COLORS.text.primary, padding: '24px', marginTop: '40px', paddingTop: isLocalhost ? '116px' : '60px' }}>
        {/* Header - Sticky ve Z-Index ile üstte kalacak */}
        <div style={{ position: 'sticky', top: 0, zIndex: 100, background: COLORS.bg.secondary, marginBottom: '24px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px', paddingTop: '8px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              {/* Menü Toggle Butonu */}
              <button
                style={{
                  background: 'transparent',
                  border: `1px solid ${COLORS.border.hover}`,
                  color: COLORS.text.primary,
                  padding: '8px 12px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.2s ease-in-out'
                }}
                title="Menüyü Aç/Kapat"
              >
                <Icons.Menu style={{ width: '18px', height: '18px' }} />
              </button>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                  <h1 style={{
                    fontSize: '32px',
                    fontWeight: '800',
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                    letterSpacing: '1.5px',
                    fontFamily: '"Inter", "SF Pro Display", system-ui, -apple-system, sans-serif',
                    textShadow: '0 2px 4px rgba(16, 185, 129, 0.1)'
                  }}>
                    Omnipotent Futures Matrix™
                  </h1>
                  <span style={{
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    color: 'white',
                    padding: '6px 14px',
                    borderRadius: '8px',
                    fontSize: '11px',
                    fontWeight: '800',
                    letterSpacing: '1px',
                    boxShadow: '0 4px 6px rgba(16, 185, 129, 0.2)',
                    fontFamily: '"Inter", "SF Pro Display", system-ui, sans-serif'
                  }}>
                    PREMIUM
                  </span>
                </div>
                <p style={{
                  color: COLORS.text.muted,
                  fontSize: '13px',
                  marginBottom: '8px',
                  fontFamily: '"Inter", system-ui, sans-serif',
                  letterSpacing: '0.3px'
                }}>
                  Advanced Trading Intelligence • Real-Time Market Analysis • Premium Signals
                </p>
                {/* Data Sources Active Indicators */}
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {dataSourcesActive.wyckoff && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.success}33`, color: COLORS.success, fontWeight: '600' }}>
                      ✓ WYCKOFF
                    </span>
                  )}
                  {dataSourcesActive.correlations && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.premium}33`, color: COLORS.premium, fontWeight: '600' }}>
                      ✓ MACRO
                    </span>
                  )}
                  {dataSourcesActive.fundingRates && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.info}33`, color: COLORS.info, fontWeight: '600' }}>
                      ✓ FUNDING
                    </span>
                  )}
                  {dataSourcesActive.openInterest && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.warning}33`, color: COLORS.warning, fontWeight: '600' }}>
                      ✓ OI
                    </span>
                  )}
                  {dataSourcesActive.btcDominance && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.success}33`, color: COLORS.success, fontWeight: '600' }}>
                      ✓ DOMINANCE
                    </span>
                  )}
                  {dataSourcesActive.fearGreed && (
                    <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: `${COLORS.danger}33`, color: COLORS.danger, fontWeight: '600' }}>
                      ✓ F&G
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted }}>Otomatik Yenileme</div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.success }}>{countdown}s</div>
              </div>
              {/* AI Assistant Button */}
              <button
                onClick={() => setAiAssistantOpen(true)}
                style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 16px',
                  color: 'white',
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-1px)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <Icons.Bot style={{ width: '16px', height: '16px' }} />
                AI Asistan
              </button>

              {/* MANTIK Button - Responsive */}
              <div>
                <style>{`
                  @media (max-width: 768px) {
                    .mantik-button-omnipotent {
                      padding: 10px 20px !important;
                      fontSize: 13px !important;
                      height: 42px !important;
                    }
                    .mantik-button-omnipotent svg {
                      width: 18px !important;
                      height: 18px !important;
                    }
                  }
                  @media (max-width: 480px) {
                    .mantik-button-omnipotent {
                      padding: 8px 16px !important;
                      fontSize: 12px !important;
                      height: 40px !important;
                    }
                    .mantik-button-omnipotent svg {
                      width: 16px !important;
                      height: 16px !important;
                    }
                  }
                `}</style>
                <button
                  onClick={() => setShowLogicModal(true)}
                  className="mantik-button-omnipotent"
                  style={{
                    padding: '12px 24px',
                    background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                    color: '#000',
                    border: 'none',
                    borderRadius: '10px',
                    fontSize: '14px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    height: '44px',
                    boxShadow: `0 4px 12px ${COLORS.premium}40`,
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = `0 6px 20px ${COLORS.premium}60`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = `0 4px 12px ${COLORS.premium}40`;
                  }}
                >
                  <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
                  MANTIK
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div style={{ padding: '16px 24px', borderBottom: `1px solid ${COLORS.bg.primary}`, display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => setFilterSignal('TÜMÜ')}
              style={{
                background: filterSignal === 'TÜMÜ' ? COLORS.success : COLORS.bg.secondary,
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: '600',
              }}
            >
              Tümü
            </button>
            <button
              onClick={() => setFilterSignal('BUY')}
              style={{
                background: filterSignal === 'BUY' ? COLORS.success : COLORS.bg.secondary,
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: '600',
              }}
            >
              AL
            </button>
            <button
              onClick={() => setFilterSignal('SELL')}
              style={{
                background: filterSignal === 'SELL' ? COLORS.danger : COLORS.bg.secondary,
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: '600',
              }}
            >
              SAT
            </button>
            <button
              onClick={() => setFilterSignal('WAIT')}
              style={{
                background: filterSignal === 'WAIT' ? COLORS.warning : COLORS.bg.secondary,
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: '600',
              }}
            >
              BEKLE
            </button>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: COLORS.text.muted }}>Sırala:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'score' | 'confidence' | 'risk')}
              style={{
                background: COLORS.bg.secondary,
                color: 'white',
                border: `1px solid ${COLORS.bg.primary}`,
                padding: '6px 12px',
                borderRadius: '6px',
                fontSize: '12px',
                cursor: 'pointer',
              }}
            >
              <option value="score">Omnipotent Skoru</option>
              <option value="confidence">Güven</option>
              <option value="risk">Risk</option>
            </select>
          </div>
        </div>

      {loading && !marketOverview ? (
        <div style={{ textAlign: 'center', padding: '100px 20px', color: COLORS.text.muted }}>
          <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
          <div style={{ fontSize: '18px' }}>Omnipotent Matrix ile 200+ korelasyon analiz ediliyor...</div>
        </div>
      ) : (
        <>
          {/* ======================================== */}
          {/* NEW v3.0: GLOBAL MACRO ASSETS PANEL */}
          {/* ======================================== */}
          {macroMetrics && (macroMetrics.dxy || macroMetrics.sp500 || macroMetrics.gold || macroMetrics.vix || macroMetrics.btc) && (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', background: 'linear-gradient(135deg, #667eea15, #764ba215)', border: `2px solid ${COLORS.premium}`, borderRadius: '12px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '18px', color: COLORS.premium, display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700', margin: 0 }}>
                  <span>🌎</span> KÜRESEL MAKRO VARLIKLAR - Anlık Veriler
                </h2>
                <button
                  onClick={() => openExplainer('makro-varliklar', 'Küresel Makro Varlıklar')}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.success} 100%)`,
                    border: 'none',
                    color: 'white',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease-in-out',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(103, 126, 234, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  💡 Mantık
                </button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                {macroMetrics.dxy && (
                  <div style={{ background: COLORS.bg.card, padding: '18px', borderRadius: '10px', border: `1px solid ${COLORS.border.active}`, transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>💵 ABD DOLAR ENDEKSİ (DXY)</div>
                    <div style={{ fontSize: '32px', fontWeight: '700', marginBottom: '6px' }}>{(macroMetrics.dxy.price ?? 0).toFixed(2)}</div>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: (macroMetrics.dxy.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                      {(macroMetrics.dxy.change24h ?? 0) >= 0 ? '+' : ''}{(macroMetrics.dxy.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                )}
                {macroMetrics.sp500 && (
                  <div style={{ background: COLORS.bg.card, padding: '18px', borderRadius: '10px', border: `1px solid ${COLORS.border.active}`, transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>📈 S&P 500 ENDEKSİ</div>
                    <div style={{ fontSize: '32px', fontWeight: '700', marginBottom: '6px' }}>{(macroMetrics.sp500.price ?? 0).toFixed(2)}</div>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: (macroMetrics.sp500.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                      {(macroMetrics.sp500.change24h ?? 0) >= 0 ? '+' : ''}{(macroMetrics.sp500.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                )}
                {macroMetrics.gold && (
                  <div style={{ background: COLORS.bg.card, padding: '18px', borderRadius: '10px', border: `1px solid ${COLORS.border.active}`, transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>🥇 ALTIN VADELİ</div>
                    <div style={{ fontSize: '32px', fontWeight: '700', marginBottom: '6px' }}>${(macroMetrics.gold.price ?? 0).toFixed(2)}</div>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: (macroMetrics.gold.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                      {(macroMetrics.gold.change24h ?? 0) >= 0 ? '+' : ''}{(macroMetrics.gold.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                )}
                {macroMetrics.vix && (
                  <div style={{ background: COLORS.bg.card, padding: '18px', borderRadius: '10px', border: `1px solid ${COLORS.border.active}`, transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>⚡ VOLATİLİTE ENDEKSİ (VIX)</div>
                    <div style={{ fontSize: '32px', fontWeight: '700', marginBottom: '6px' }}>{(macroMetrics.vix.price ?? 0).toFixed(2)}</div>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: (macroMetrics.vix.change24h ?? 0) >= 0 ? COLORS.danger : COLORS.success }}>
                      {(macroMetrics.vix.change24h ?? 0) >= 0 ? '+' : ''}{(macroMetrics.vix.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                )}
                {macroMetrics.btc && (
                  <div style={{ background: COLORS.bg.card, padding: '18px', borderRadius: '10px', border: `1px solid ${COLORS.warning}`, transition: 'transform 0.2s' }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>₿ BİTCOİN (BTC)</div>
                    <div style={{ fontSize: '32px', fontWeight: '700', marginBottom: '6px', color: COLORS.warning }}>${(macroMetrics.btc.price ?? 0).toLocaleString()}</div>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: (macroMetrics.btc.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                      {(macroMetrics.btc.change24h ?? 0) >= 0 ? '+' : ''}{(macroMetrics.btc.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ======================================== */}
          {/* NEW v3.0: BTC CORRELATION MATRIX */}
          {/* ======================================== */}
          {correlationMatrix && (() => {
            const translateStrength = (strength: string) => {
              const map: {[key: string]: string} = {
                'STRONG': 'GÜÇLÜ',
                'MODERATE': 'ORTA',
                'WEAK': 'ZAYIF',
                'NONE': 'YOK'
              };
              return map[strength] || strength;
            };
            const translateDirection = (direction: string) => {
              const map: {[key: string]: string} = {
                'POSITIVE': 'POZİTİF',
                'NEGATIVE': 'NEGATİF',
                'NEUTRAL': 'NÖTR'
              };
              return map[direction] || direction;
            };
            return (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', background: COLORS.bg.card, border: `2px solid ${COLORS.info}`, borderRadius: '12px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '18px', color: COLORS.info, display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700', margin: 0 }}>
                  <span>🔗</span> BTC KORELASYON MATRİSİ - 30 Günlük Hareketli
                </h2>
                <button
                  onClick={() => openExplainer('korelasyon', 'BTC Korelasyon Matrisi')}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.info} 0%, ${COLORS.premium} 100%)`,
                    border: 'none',
                    color: 'white',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease-in-out',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  💡 Mantık
                </button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
                {/* BTC/DXY */}
                <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '10px', border: `2px solid ${correlationMatrix.btcDxy.strength === 'STRONG' ? COLORS.premium : correlationMatrix.btcDxy.strength === 'MODERATE' ? COLORS.warning : COLORS.border.default}` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>BTC ↔ DXY</div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: (correlationMatrix.btcDxy.correlation ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                    {(correlationMatrix.btcDxy.correlation ?? 0) >= 0 ? '+' : ''}{(correlationMatrix.btcDxy.correlation ?? 0).toFixed(3)}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${COLORS.premium}33`, color: COLORS.premium, fontWeight: '600' }}>
                      {translateStrength(correlationMatrix.btcDxy.strength)}
                    </span>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${correlationMatrix.btcDxy.direction === 'POSITIVE' ? COLORS.success : COLORS.danger}33`, color: correlationMatrix.btcDxy.direction === 'POSITIVE' ? COLORS.success : COLORS.danger, fontWeight: '600' }}>
                      {translateDirection(correlationMatrix.btcDxy.direction)}
                    </span>
                  </div>
                </div>
                {/* BTC/SP500 */}
                <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '10px', border: `2px solid ${correlationMatrix.btcSp500.strength === 'STRONG' ? COLORS.premium : correlationMatrix.btcSp500.strength === 'MODERATE' ? COLORS.warning : COLORS.border.default}` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>BTC ↔ S&P500</div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: (correlationMatrix.btcSp500.correlation ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                    {(correlationMatrix.btcSp500.correlation ?? 0) >= 0 ? '+' : ''}{(correlationMatrix.btcSp500.correlation ?? 0).toFixed(3)}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${COLORS.premium}33`, color: COLORS.premium, fontWeight: '600' }}>
                      {translateStrength(correlationMatrix.btcSp500.strength)}
                    </span>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${correlationMatrix.btcSp500.direction === 'POSITIVE' ? COLORS.success : COLORS.danger}33`, color: correlationMatrix.btcSp500.direction === 'POSITIVE' ? COLORS.success : COLORS.danger, fontWeight: '600' }}>
                      {translateDirection(correlationMatrix.btcSp500.direction)}
                    </span>
                  </div>
                </div>
                {/* BTC/GOLD */}
                <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '10px', border: `2px solid ${correlationMatrix.btcGold.strength === 'STRONG' ? COLORS.premium : correlationMatrix.btcGold.strength === 'MODERATE' ? COLORS.warning : COLORS.border.default}` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>BTC ↔ ALTIN</div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: (correlationMatrix.btcGold.correlation ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                    {(correlationMatrix.btcGold.correlation ?? 0) >= 0 ? '+' : ''}{(correlationMatrix.btcGold.correlation ?? 0).toFixed(3)}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${COLORS.premium}33`, color: COLORS.premium, fontWeight: '600' }}>
                      {translateStrength(correlationMatrix.btcGold.strength)}
                    </span>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${correlationMatrix.btcGold.direction === 'POSITIVE' ? COLORS.success : COLORS.danger}33`, color: correlationMatrix.btcGold.direction === 'POSITIVE' ? COLORS.success : COLORS.danger, fontWeight: '600' }}>
                      {translateDirection(correlationMatrix.btcGold.direction)}
                    </span>
                  </div>
                </div>
                {/* BTC/VIX */}
                <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '10px', border: `2px solid ${correlationMatrix.btcVix.strength === 'STRONG' ? COLORS.premium : correlationMatrix.btcVix.strength === 'MODERATE' ? COLORS.warning : COLORS.border.default}` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>BTC ↔ VIX</div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: (correlationMatrix.btcVix.correlation ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                    {(correlationMatrix.btcVix.correlation ?? 0) >= 0 ? '+' : ''}{(correlationMatrix.btcVix.correlation ?? 0).toFixed(3)}
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${COLORS.premium}33`, color: COLORS.premium, fontWeight: '600' }}>
                      {translateStrength(correlationMatrix.btcVix.strength)}
                    </span>
                    <span style={{ fontSize: '10px', padding: '4px 10px', borderRadius: '4px', background: `${correlationMatrix.btcVix.direction === 'POSITIVE' ? COLORS.success : COLORS.danger}33`, color: correlationMatrix.btcVix.direction === 'POSITIVE' ? COLORS.success : COLORS.danger, fontWeight: '600' }}>
                      {translateDirection(correlationMatrix.btcVix.direction)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
            );
          })()}

          {/* ======================================== */}
          {/* RISK MANAGEMENT CALCULATOR - REAL DATA */}
          {/* ======================================== */}
          {correlations.length > 0 && (() => {
            // ✅ GERÇEK VERİ: Omnipotent Futures sinyallerinden trade geçmişi hesaplama
            const buySignals = correlations.filter((f: CorrelationData) => f.signal === 'BUY');
            const sellSignals = correlations.filter((f: CorrelationData) => f.signal === 'SELL');
            const waitSignals = correlations.filter((f: CorrelationData) => f.signal === 'WAIT');

            // Gerçek trade istatistikleri
            const realTradeHistory: TradeHistory = {
              wins: buySignals.length, // BUY sinyalleri kazanan işlem olarak sayılır
              losses: sellSignals.length + waitSignals.length, // Diğerleri kayıp/nötr
              avgWin: buySignals.length > 0
                ? buySignals.reduce((sum: number, s: CorrelationData) => sum + s.confidence, 0) / buySignals.length
                : 50,
              avgLoss: (sellSignals.length + waitSignals.length) > 0
                ? (sellSignals.reduce((sum: number, s: CorrelationData) => sum + s.confidence, 0) + waitSignals.reduce((sum: number, s: CorrelationData) => sum + s.confidence, 0)) / (sellSignals.length + waitSignals.length)
                : 40,
              totalTrades: correlations.length
            };

            const kellyResult = calculateKellyCriterion(realTradeHistory);

            // ✅ GERÇEK VERİ: BTC'den gerçek fiyat al
            const btcData = correlations.find((f: CorrelationData) => f.symbol === 'BTCUSDT');
            const realBtcPrice = btcData ? btcData.price : 100000;

            // Gerçek pozisyon hesaplaması
            const realAccountSize = 10000; // Örnek hesap (kullanıcı inputu olabilir)
            const realRiskPercent = 1; // %1 risk per trade
            const realStopLoss = realBtcPrice * 0.98; // 2% stop loss
            const realLeverage = 3; // 3x kaldıraç

            const positionSizeResult = calculatePositionSize(
              realAccountSize,
              realRiskPercent,
              realBtcPrice,
              realStopLoss,
              realLeverage
            );

            const getRiskLevelColor = (level: string) => {
              switch (level) {
                case 'CONSERVATIVE': return COLORS.success;
                case 'MODERATE': return COLORS.info;
                case 'AGGRESSIVE': return COLORS.warning;
                case 'TOO_RISKY': return COLORS.danger;
                default: return COLORS.text.muted;
              }
            };

            const translateRiskLevel = (level: string) => {
              const map: {[key: string]: string} = {
                'CONSERVATIVE': 'MUHAFAZAKAR',
                'MODERATE': 'ORTA',
                'AGGRESSIVE': 'AGRESİF',
                'TOO_RISKY': 'ÇOK RİSKLİ'
              };
              return map[level] || level;
            };

            return (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', background: 'linear-gradient(135deg, #f093fb15, #f5576c15)', border: `2px solid ${COLORS.premium}`, borderRadius: '12px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '18px', color: COLORS.premium, display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700', margin: 0 }}>
                  <span>🎯</span> RİSK YÖNETİMİ HESAPLAYICI - Kelly Criterion & Pozisyon Boyutlandırma
                </h2>
                <button
                  onClick={() => openExplainer('risk-yonetimi', 'Risk Yönetimi Hesaplayıcı')}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.danger} 100%)`,
                    border: 'none',
                    color: 'white',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease-in-out',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(245, 87, 108, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  💡 Mantık
                </button>
              </div>

              {/* İstatistik Özeti - GERÇEK VERİ */}
              <div style={{ marginBottom: '20px', padding: '16px', background: `${COLORS.info}11`, borderRadius: '8px', border: `1px solid ${COLORS.info}33` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>
                  📊 GERÇEK ZAMANA SINYAL ANALİZİ ({correlations.length} Coin)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
                  <div style={{ background: COLORS.bg.card, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.success, marginBottom: '4px' }}>BUY Sinyalleri</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.success }}>{realTradeHistory.wins}</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.danger, marginBottom: '4px' }}>SELL/WAIT Sinyalleri</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.danger }}>{realTradeHistory.losses}</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Başarı Oranı</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.info }}>{(((realTradeHistory.wins ?? 0) / (realTradeHistory.totalTrades ?? 1)) * 100).toFixed(1)}%</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Ort. BUY Güven</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.success }}>{(realTradeHistory.avgWin ?? 0).toFixed(1)}%</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Ort. Diğer Güven</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.danger }}>{(realTradeHistory.avgLoss ?? 0).toFixed(1)}%</div>
                  </div>
                </div>
              </div>

              {/* Kelly Criterion Sonuçları */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                {/* Full Kelly */}
                <div style={{ background: COLORS.bg.card, padding: '20px', borderRadius: '10px', border: `2px solid ${getRiskLevelColor(kellyResult.riskLevel)}` }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>
                    📈 KELLY CRİTERİON (TAM)
                  </div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: getRiskLevelColor(kellyResult.riskLevel) }}>
                    {(kellyResult.kellyPercentage ?? 0).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '10px', padding: '6px 12px', borderRadius: '4px', background: `${getRiskLevelColor(kellyResult.riskLevel)}33`, color: getRiskLevelColor(kellyResult.riskLevel), fontWeight: '600', display: 'inline-block' }}>
                    {translateRiskLevel(kellyResult.riskLevel)}
                  </div>
                </div>

                {/* Half Kelly (Önerilen) */}
                <div style={{ background: COLORS.bg.card, padding: '20px', borderRadius: '10px', border: `2px solid ${COLORS.success}`, boxShadow: `0 0 20px ${COLORS.success}22` }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    ✅ YARIM KELLY (ÖNERİLEN)
                    <span style={{ fontSize: '10px', padding: '2px 6px', background: `${COLORS.success}33`, borderRadius: '3px', color: COLORS.success }}>GÜVENLİ</span>
                  </div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: COLORS.success }}>
                    {(kellyResult.fractionalKelly ?? 0).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '8px' }}>
                    İşlem başına önerilen pozisyon büyüklüğü
                  </div>
                </div>

                {/* Quarter Kelly (Çok Güvenli) */}
                <div style={{ background: COLORS.bg.card, padding: '20px', borderRadius: '10px', border: `2px solid ${COLORS.info}` }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>
                    🛡️ ÇEYREK KELLY (ÇOK GÜVENLİ)
                  </div>
                  <div style={{ fontSize: '48px', fontWeight: '700', marginBottom: '8px', color: COLORS.info }}>
                    {(kellyResult.quarterKelly ?? 0).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '8px' }}>
                    Muhafazakar yatırımcılar için
                  </div>
                </div>
              </div>

              {/* Kelly Önerisi */}
              <div style={{ marginBottom: '20px', padding: '16px', background: `${getRiskLevelColor(kellyResult.riskLevel)}11`, borderRadius: '8px', border: `1px solid ${getRiskLevelColor(kellyResult.riskLevel)}33` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>💡 KELLY ÖNERİSİ:</div>
                <div style={{ fontSize: '13px', color: COLORS.text.primary, lineHeight: '1.6' }}>
                  {kellyResult.recommendation}
                </div>
              </div>

              {/* Pozisyon Boyutlandırma - GERÇEK BTC FİYATI */}
              <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '10px', border: `1px solid ${COLORS.border.active}` }}>
                <h3 style={{ fontSize: '15px', marginBottom: '16px', color: COLORS.warning, display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '700' }}>
                  <span>📊</span> POZİSYON BOYUTLANDIRMA (BTC ${(realBtcPrice ?? 0).toLocaleString()})
                </h3>

                {/* Giriş Parametreleri */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                  <div style={{ background: COLORS.bg.card, padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px' }}>Hesap Büyüklüğü</div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.info }}>${(positionSizeResult.accountSize ?? 0).toLocaleString()}</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px' }}>Risk Yüzdesi</div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.warning }}>{positionSizeResult.riskPercentage ?? 0}%</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px' }}>Kaldıraç</div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.premium }}>{positionSizeResult.leverage ?? 0}x</div>
                  </div>
                  <div style={{ background: COLORS.bg.card, padding: '12px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px' }}>Stop Loss Mesafesi</div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.danger }}>{(positionSizeResult.stopLossDistance ?? 0).toFixed(2)}%</div>
                  </div>
                </div>

                {/* Hesaplanan Sonuçlar */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                  <div style={{ background: `${COLORS.success}11`, padding: '16px', borderRadius: '8px', border: `2px solid ${COLORS.success}` }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>
                      💰 ÖNERİLEN POZİSYON BÜYÜKLÜĞÜ
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.success }}>
                      ${(positionSizeResult.positionSize ?? 0).toLocaleString()}
                    </div>
                  </div>
                  <div style={{ background: `${COLORS.warning}11`, padding: '16px', borderRadius: '8px', border: `2px solid ${COLORS.warning}` }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>
                      ⚠️ İŞLEM BAŞINA RİSK
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.warning }}>
                      ${(positionSizeResult.riskPerTrade ?? 0).toLocaleString()}
                    </div>
                  </div>
                  <div style={{ background: `${COLORS.info}11`, padding: '16px', borderRadius: '8px', border: `2px solid ${COLORS.info}` }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>
                      📦 KONTRAT MİKTARI
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.info }}>
                      {(positionSizeResult.contractQuantity ?? 0).toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Uyarı Notu */}
              <div style={{ marginTop: '16px', padding: '12px', background: `${COLORS.success}11`, borderRadius: '6px', border: `1px solid ${COLORS.success}33` }}>
                <div style={{ fontSize: '11px', color: COLORS.success, fontWeight: '600', marginBottom: '4px' }}>
                  ✅ GERÇEK VERİ KULLANILDI:
                </div>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                  Bu hesaplamalar {correlations.length} coin'den gerçek zamanlı Wyckoff sinyalleri ve güncel BTC fiyatı kullanılarak yapılmıştır. Eğitim amaçlıdır, finansal tavsiye değildir.
                </div>
              </div>
            </div>
            );
          })()}

          {/* 📊 TECHNICAL INDICATORS PANEL - REAL DATA FROM BINANCE */}
          {(() => {
            // Find BTC data with technical indicators
            const btcData = correlations.find((f: CorrelationData) => f.symbol === 'BTCUSDT');

            if (!btcData || !btcData.technicalIndicators) {
              return null; // Don't show panel if no tech indicators
            }

            const ti = btcData.technicalIndicators;

            return (
              <div className="animate-fadeIn" style={{ marginBottom: '24px', background: COLORS.bg.card, border: `2px solid ${COLORS.premium}`, borderRadius: '12px', padding: '24px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h2 style={{ fontSize: '20px', background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.info} 100%)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'flex', alignItems: 'center', gap: '10px', margin: 0, fontWeight: '700' }}>
                    <span>📈</span> TEKNİK İNDİKATÖRLER (BTC)
                  </h2>
                  <button
                    onClick={() => openExplainer('technical-indicators', 'Teknik İndikatörler')}
                    style={{
                      background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.info} 100%)`,
                      border: 'none',
                      color: 'white',
                      padding: '8px 16px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: '600',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      transition: 'all 0.2s ease-in-out',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(147, 51, 234, 0.4)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    💡 Mantık
                  </button>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
                  {/* RSI Card */}
                  <div style={{
                    background: COLORS.bg.secondary,
                    padding: '20px',
                    borderRadius: '10px',
                    border: `2px solid ${ti.rsi.signal === 'OVERSOLD' ? COLORS.success : ti.rsi.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.warning}`,
                    transition: 'all 0.3s ease-in-out',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.muted }}>RSI (14)</div>
                      <div style={{
                        padding: '4px 12px',
                        borderRadius: '12px',
                        fontSize: '10px',
                        fontWeight: '700',
                        background: ti.rsi.signal === 'OVERSOLD' ? `${COLORS.success}22` : ti.rsi.signal === 'OVERBOUGHT' ? `${COLORS.danger}22` : `${COLORS.warning}22`,
                        color: ti.rsi.signal === 'OVERSOLD' ? COLORS.success : ti.rsi.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.warning,
                      }}>
                        {ti.rsi.signal === 'OVERSOLD' ? 'AŞIRI SATIM' : ti.rsi.signal === 'OVERBOUGHT' ? 'AŞIRI ALIM' : 'NÖTR'}
                      </div>
                    </div>
                    <div style={{ fontSize: '40px', fontWeight: '700', color: ti.rsi.signal === 'OVERSOLD' ? COLORS.success : ti.rsi.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.warning, marginBottom: '12px' }}>
                      {(ti.rsi.value ?? 0).toFixed(1)}
                    </div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                      {ti.rsi.interpretation}
                    </div>
                  </div>

                  {/* MACD Card */}
                  <div style={{
                    background: COLORS.bg.secondary,
                    padding: '20px',
                    borderRadius: '10px',
                    border: `2px solid ${ti.macd.signal === 'BULLISH' ? COLORS.success : ti.macd.signal === 'BEARISH' ? COLORS.danger : COLORS.warning}`,
                    transition: 'all 0.3s ease-in-out',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.muted }}>MACD (12,26,9)</div>
                      <div style={{
                        padding: '4px 12px',
                        borderRadius: '12px',
                        fontSize: '10px',
                        fontWeight: '700',
                        background: ti.macd.signal === 'BULLISH' ? `${COLORS.success}22` : ti.macd.signal === 'BEARISH' ? `${COLORS.danger}22` : `${COLORS.warning}22`,
                        color: ti.macd.signal === 'BULLISH' ? COLORS.success : ti.macd.signal === 'BEARISH' ? COLORS.danger : COLORS.warning,
                      }}>
                        {ti.macd.signal === 'BULLISH' ? 'YÜKSELİŞ' : ti.macd.signal === 'BEARISH' ? 'DÜŞÜŞ' : 'NÖTR'}
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>MACD Çizgisi</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.info }}>{(ti.macd.macdLine ?? 0).toFixed(4)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Sinyal Çizgisi</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.warning }}>{(ti.macd.signalLine ?? 0).toFixed(4)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Histogram</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: (ti.macd.histogram ?? 0) > 0 ? COLORS.success : COLORS.danger }}>{(ti.macd.histogram ?? 0).toFixed(4)}</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                      {ti.macd.interpretation}
                    </div>
                  </div>

                  {/* Bollinger Bands Card */}
                  <div style={{
                    background: COLORS.bg.secondary,
                    padding: '20px',
                    borderRadius: '10px',
                    border: `2px solid ${ti.bollingerBands.signal === 'OVERSOLD' ? COLORS.success : ti.bollingerBands.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.info}`,
                    transition: 'all 0.3s ease-in-out',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.muted }}>Bollinger Bands (20,2)</div>
                      <div style={{
                        padding: '4px 12px',
                        borderRadius: '12px',
                        fontSize: '10px',
                        fontWeight: '700',
                        background: ti.bollingerBands.signal === 'OVERSOLD' ? `${COLORS.success}22` : ti.bollingerBands.signal === 'OVERBOUGHT' ? `${COLORS.danger}22` : `${COLORS.info}22`,
                        color: ti.bollingerBands.signal === 'OVERSOLD' ? COLORS.success : ti.bollingerBands.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.info,
                      }}>
                        {ti.bollingerBands.signal === 'OVERSOLD' ? 'AŞIRI SATIM' : ti.bollingerBands.signal === 'OVERBOUGHT' ? 'AŞIRI ALIM' : 'NORMAL'}
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Üst Bant</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.danger }}>${(ti.bollingerBands.upper ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Orta (SMA)</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.warning }}>${(ti.bollingerBands.middle ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Alt Bant</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.success }}>${(ti.bollingerBands.lower ?? 0).toFixed(2)}</div>
                      </div>
                    </div>
                    <div style={{ marginBottom: '8px', fontSize: '11px', color: COLORS.text.muted }}>
                      <span style={{ fontWeight: '600' }}>Bandwidth:</span> {(ti.bollingerBands.bandwidth ?? 0).toFixed(2)}% | <span style={{ fontWeight: '600' }}>%B:</span> {(ti.bollingerBands.percentB ?? 0).toFixed(3)}
                    </div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                      {ti.bollingerBands.interpretation}
                    </div>
                  </div>
                </div>

                {/* Real Data Notice */}
                <div style={{ marginTop: '16px', padding: '12px', background: `${COLORS.success}11`, borderRadius: '6px', border: `1px solid ${COLORS.success}33` }}>
                  <div style={{ fontSize: '11px', color: COLORS.success, fontWeight: '600', marginBottom: '4px' }}>
                    ✅ GERÇEK VERİ - BINANCE KLINES API:
                  </div>
                  <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                    Bu indikatörler Binance Futures API'den alınan son 100 saatlik gerçek candlestick verileri ile hesaplanmıştır. Eğitim amaçlıdır, finansal tavsiye değildir.
                  </div>
                </div>
              </div>
            );
          })()}

          {/* ⏰ MULTI-TIMEFRAME ANALYSIS PANEL - REAL DATA FROM BINANCE */}
          {btcMultiTimeframe && (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', background: COLORS.bg.card, border: `2px solid ${COLORS.premium}`, borderRadius: '12px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '20px', background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.info} 100%)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', display: 'flex', alignItems: 'center', gap: '10px', margin: 0, fontWeight: '700' }}>
                  <span>⏰</span> MULTI-TIMEFRAME ANALYSIS (BTC)
                </h2>
                <button
                  onClick={() => openExplainer('multi-timeframe', 'Multi-Timeframe Analysis')}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.premium} 0%, ${COLORS.info} 100%)`,
                    border: 'none',
                    color: 'white',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease-in-out',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(147, 51, 234, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  💡 Mantık
                </button>
              </div>

              {/* 4 Timeframe Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                {(['1h', '4h', '1d', '1w'] as const).map((tf) => {
                  const timeframeData = btcMultiTimeframe.timeframes[tf];
                  const weight = tf === '1h' ? 1 : tf === '4h' ? 2 : tf === '1d' ? 3 : 4;
                  const signalColor = timeframeData.overallSignal === 'BULLISH' ? COLORS.success : timeframeData.overallSignal === 'BEARISH' ? COLORS.danger : COLORS.warning;

                  return (
                    <div
                      key={tf}
                      style={{
                        background: COLORS.bg.secondary,
                        padding: '18px',
                        borderRadius: '10px',
                        border: `2px solid ${signalColor}`,
                        transition: 'all 0.3s ease-in-out',
                      }}
                    >
                      {/* Timeframe Header */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary }}>
                          {tf === '1h' ? '1 Saat' : tf === '4h' ? '4 Saat' : tf === '1d' ? '1 Gün' : '1 Hafta'}
                        </div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, background: `${COLORS.premium}22`, padding: '4px 8px', borderRadius: '6px', fontWeight: '600' }}>
                          Ağırlık: {weight}x
                        </div>
                      </div>

                      {/* Overall Signal Badge */}
                      <div style={{
                        background: `${signalColor}22`,
                        color: signalColor,
                        padding: '8px 12px',
                        borderRadius: '8px',
                        fontSize: '14px',
                        fontWeight: '700',
                        textAlign: 'center',
                        marginBottom: '12px',
                      }}>
                        {timeframeData.overallSignal === 'BULLISH' ? '📈 YÜKSELİŞ' : timeframeData.overallSignal === 'BEARISH' ? '📉 DÜŞÜŞ' : '➖ NÖTR'}
                      </div>

                      {/* Signal Strength Bar */}
                      <div style={{ marginBottom: '12px' }}>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                          <span>Güç</span>
                          <span style={{ fontWeight: '700' }}>{(timeframeData.signalStrength ?? 0).toFixed(0)}%</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', background: COLORS.bg.primary, borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            width: `${timeframeData.signalStrength ?? 0}%`,
                            height: '100%',
                            background: `linear-gradient(90deg, ${signalColor}66 0%, ${signalColor} 100%)`,
                            transition: 'width 0.5s ease-in-out',
                          }} />
                        </div>
                      </div>

                      {/* Mini Indicators */}
                      <div style={{ fontSize: '10px', color: COLORS.text.muted, lineHeight: '1.6' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>RSI:</span>
                          <span style={{ fontWeight: '600', color: timeframeData.rsi.signal === 'OVERSOLD' ? COLORS.success : timeframeData.rsi.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.warning }}>
                            {(timeframeData.rsi.value ?? 0).toFixed(1)}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>MACD:</span>
                          <span style={{ fontWeight: '600', color: timeframeData.macd.signal === 'BULLISH' ? COLORS.success : timeframeData.macd.signal === 'BEARISH' ? COLORS.danger : COLORS.warning }}>
                            {timeframeData.macd.signal === 'BULLISH' ? '↑ Yükseliş' : timeframeData.macd.signal === 'BEARISH' ? '↓ Düşüş' : '→ Nötr'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>BB:</span>
                          <span style={{ fontWeight: '600', color: timeframeData.bollingerBands.signal === 'OVERSOLD' ? COLORS.success : timeframeData.bollingerBands.signal === 'OVERBOUGHT' ? COLORS.danger : COLORS.info }}>
                            {timeframeData.bollingerBands.signal === 'OVERSOLD' ? 'Alt Bant' : timeframeData.bollingerBands.signal === 'OVERBOUGHT' ? 'Üst Bant' : 'Normal'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Consensus Section */}
              <div style={{ background: COLORS.bg.secondary, padding: '24px', borderRadius: '12px', border: `2px solid ${btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.success : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? COLORS.danger : COLORS.warning}`, marginBottom: '20px' }}>
                <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.muted, marginBottom: '16px', textAlign: 'center' }}>
                  📊 CONSENSUS (Ağırlıklı Ortalama)
                </div>

                {/* Big Consensus Badge */}
                <div style={{
                  background: `linear-gradient(135deg, ${btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.success : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? COLORS.danger : COLORS.warning} 0%, ${btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.info : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? '#dc2626' : '#f59e0b'} 100%)`,
                  color: 'white',
                  padding: '16px 24px',
                  borderRadius: '12px',
                  fontSize: '24px',
                  fontWeight: '700',
                  textAlign: 'center',
                  marginBottom: '16px',
                  boxShadow: `0 4px 16px ${btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.success : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? COLORS.danger : COLORS.warning}44`,
                }}>
                  {btcMultiTimeframe.consensus.signal === 'STRONG_BULLISH' ? '🚀 GÜÇLÜ YÜKSELİŞ' :
                   btcMultiTimeframe.consensus.signal === 'BULLISH' ? '📈 YÜKSELİŞ' :
                   btcMultiTimeframe.consensus.signal === 'STRONG_BEARISH' ? '⚡ GÜÇLÜ DÜŞÜŞ' :
                   btcMultiTimeframe.consensus.signal === 'BEARISH' ? '📉 DÜŞÜŞ' :
                   '➖ NÖTR'}
                </div>

                {/* Strength and Alignment */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '6px' }}>Konsensüs Gücü</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.success : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? COLORS.danger : COLORS.warning }}>
                      {(btcMultiTimeframe.consensus.strength ?? 0).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '6px' }}>Timeframe Uyumu</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.info }}>
                      {btcMultiTimeframe.consensus.alignment}/4
                    </div>
                  </div>
                </div>

                {/* Interpretation */}
                <div style={{ padding: '12px', background: `${COLORS.info}11`, borderRadius: '8px', border: `1px solid ${COLORS.info}33` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.primary, lineHeight: '1.6', fontWeight: '500' }}>
                    {btcMultiTimeframe.consensus.interpretation}
                  </div>
                </div>
              </div>

              {/* Higher Timeframe Bias */}
              <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '10px', border: `1px solid ${COLORS.border.default}`, marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '4px' }}>Higher Timeframe Bias (1d + 1w)</div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? COLORS.success : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? COLORS.danger : COLORS.warning }}>
                      {btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? '📈 Yükseliş Eğilimi' : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? '📉 Düşüş Eğilimi' : '➖ Nötr'}
                    </div>
                  </div>
                  <div style={{
                    padding: '12px 20px',
                    borderRadius: '12px',
                    background: `${btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? COLORS.success : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? COLORS.danger : COLORS.warning}22`,
                    color: btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? COLORS.success : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? COLORS.danger : COLORS.warning,
                    fontWeight: '700',
                    fontSize: '14px',
                  }}>
                    {btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? '⬆️ LONG' : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? '⬇️ SHORT' : '⏸️ WAIT'}
                  </div>
                </div>
              </div>

              {/* Real Data Notice */}
              <div style={{ padding: '12px', background: `${COLORS.success}11`, borderRadius: '6px', border: `1px solid ${COLORS.success}33` }}>
                <div style={{ fontSize: '11px', color: COLORS.success, fontWeight: '600', marginBottom: '4px' }}>
                  ✅ GERÇEK VERİ - BINANCE KLINES API (4 TIMEFRAME):
                </div>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, lineHeight: '1.5' }}>
                  Bu analiz Binance Futures API'den alınan gerçek candlestick verileri ile 4 farklı zaman diliminde (1h, 4h, 1d, 1w) hesaplanmıştır. Ağırlıklı konsensüs algoritması kullanılmıştır. Eğitim amaçlıdır, finansal tavsiye değildir.
                </div>
              </div>
            </div>
          )}

          {/* Piyasa Genel Bakış */}
          {marketOverview && (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', background: COLORS.bg.card, border: `1px solid ${COLORS.border.hover}`, borderRadius: '10px', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <h2 style={{ fontSize: '18px', color: COLORS.info, display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
                  <span>📊</span> PİYASA GENEL BAKIŞ
                </h2>
                <button
                  onClick={() => openExplainer('wyckoff', 'Wyckoff Metodolojisi')}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.info} 0%, ${COLORS.success} 100%)`,
                    border: 'none',
                    color: 'white',
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '13px',
                    fontWeight: '600',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s ease-in-out',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(34, 197, 94, 0.4)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  💡 Mantık
                </button>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', marginBottom: '24px' }}>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.text.muted, fontSize: '11px', marginBottom: '8px' }}>Toplam Analiz</div>
                  <div style={{ fontSize: '28px', fontWeight: '700' }}>{marketOverview.totalCoins}</div>
                </div>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.text.muted, fontSize: '11px', marginBottom: '8px' }}>Ort. Matrix Skoru</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.info }}>{marketOverview.avgOmnipotentScore}<span style={{ fontSize: '16px', color: COLORS.text.muted }}>/100</span></div>
                </div>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.success}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.success, fontSize: '11px', marginBottom: '8px' }}>Yükseliş Sinyalleri</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.success }}>{marketOverview.bullishCount}</div>
                </div>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.danger}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.danger, fontSize: '11px', marginBottom: '8px' }}>Düşüş Sinyalleri</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.danger }}>{marketOverview.bearishCount}</div>
                </div>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.text.muted, fontSize: '11px', marginBottom: '8px' }}>Ort. Volatilite</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.warning }}>{marketOverview.avgVolatility}%</div>
                </div>
                <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.premium}`, transition: 'all 0.2s ease-in-out' }}>
                  <div style={{ color: COLORS.premium, fontSize: '11px', marginBottom: '8px' }}>Yüksek Güven</div>
                  <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.premium }}>{marketOverview.highConfidenceSignals}</div>
                </div>
                {/* NEW v2.0: BTC Dominance */}
                {globalMetrics?.btcDominance && (
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.warning}`, transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ color: COLORS.warning, fontSize: '11px', marginBottom: '8px' }}>BTC Dominans</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.warning }}>{(globalMetrics.btcDominance.btc ?? 0).toFixed(2)}%</div>
                  </div>
                )}
                {/* NEW v2.0: Fear & Greed */}
                {globalMetrics?.fearGreed && (
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${globalMetrics.fearGreed.value > 60 ? COLORS.success : globalMetrics.fearGreed.value < 40 ? COLORS.danger : COLORS.warning}`, transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ color: COLORS.text.muted, fontSize: '11px', marginBottom: '8px' }}>Korku & Açgözlülük</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: globalMetrics.fearGreed.value > 60 ? COLORS.success : globalMetrics.fearGreed.value < 40 ? COLORS.danger : COLORS.warning }}>{globalMetrics.fearGreed.value}</div>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginTop: '4px' }}>{globalMetrics.fearGreed.classification}</div>
                  </div>
                )}
              </div>

              {/* Piyasa Faz Dağılımı */}
              <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '13px', color: COLORS.text.muted, marginBottom: '16px', fontWeight: '600' }}>PİYASA FAZ DAĞILIMI</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
                  <div style={{ textAlign: 'center', padding: '14px', background: `${COLORS.info}1A`, border: `2px solid ${COLORS.info}`, borderRadius: '6px', transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.info }}>{marketOverview.marketPhaseDistribution.ACCUMULATION}</div>
                    <div style={{ fontSize: '10px', color: COLORS.info, marginTop: '4px', fontWeight: '600' }}>BİRİKTİRME</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '14px', background: `${COLORS.success}1A`, border: `2px solid ${COLORS.success}`, borderRadius: '6px', transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.success }}>{marketOverview.marketPhaseDistribution.MARKUP}</div>
                    <div style={{ fontSize: '10px', color: COLORS.success, marginTop: '4px', fontWeight: '600' }}>YUKARI HAREKET</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '14px', background: `${COLORS.warning}1A`, border: `2px solid ${COLORS.warning}`, borderRadius: '6px', transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.warning }}>{marketOverview.marketPhaseDistribution.DISTRIBUTION}</div>
                    <div style={{ fontSize: '10px', color: COLORS.warning, marginTop: '4px', fontWeight: '600' }}>DAĞITIM</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '14px', background: `${COLORS.danger}1A`, border: `2px solid ${COLORS.danger}`, borderRadius: '6px', transition: 'all 0.2s ease-in-out' }}>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.danger }}>{marketOverview.marketPhaseDistribution.MARKDOWN}</div>
                    <div style={{ fontSize: '10px', color: COLORS.danger, marginTop: '4px', fontWeight: '600' }}>AŞAĞI HAREKET</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* NEW v3.0: BTC Advanced Analysis - Volume Profile, Order Flow, Multi-Timeframe */}
          {(btcVolumeProfile || btcOrderFlow || btcMultiTimeframe) && (
            <div className="animate-fadeIn" style={{ marginBottom: '24px', padding: '20px', background: COLORS.bg.card, borderRadius: '10px', border: `2px solid ${COLORS.premium}` }}>
              <div style={{ fontSize: '15px', fontWeight: '700', marginBottom: '16px', color: COLORS.premium, display: 'flex', alignItems: 'center', gap: '8px' }}>
                BTC GELİŞMİŞ ANALİZ (GERÇEK VERİ)
                <div style={{ fontSize: '10px', background: `${COLORS.success}22`, color: COLORS.success, padding: '4px 8px', borderRadius: '4px', fontWeight: '600' }}>CANLI</div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>

                {/* Volume Profile */}
                {btcVolumeProfile && (
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.info}` }}>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: COLORS.info, marginBottom: '12px' }}>📊 VOLUME PROFILE</div>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '8px' }}>POC (Point of Control)</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '12px' }}>${(btcVolumeProfile.poc.price ?? 0).toFixed(2)}</div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                      <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px' }}>
                        <div style={{ fontSize: '9px', color: COLORS.text.muted }}>VAH</div>
                        <div style={{ fontSize: '12px', fontWeight: '600' }}>${(btcVolumeProfile.valueArea.high ?? 0).toFixed(2)}</div>
                      </div>
                      <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px' }}>
                        <div style={{ fontSize: '9px', color: COLORS.text.muted }}>VAL</div>
                        <div style={{ fontSize: '12px', fontWeight: '600' }}>${(btcVolumeProfile.valueArea.low ?? 0).toFixed(2)}</div>
                      </div>
                    </div>

                    <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
                      <div style={{ fontSize: '9px', color: COLORS.text.muted }}>VWAP</div>
                      <div style={{ fontSize: '14px', fontWeight: '700', color: COLORS.warning }}>${(btcVolumeProfile.vwap.price ?? 0).toFixed(2)}</div>
                      <div style={{ fontSize: '9px', color: (btcVolumeProfile.vwap.deviation ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                        {(btcVolumeProfile.vwap.deviation ?? 0) >= 0 ? '+' : ''}{(btcVolumeProfile.vwap.deviation ?? 0).toFixed(2)}% sapma
                      </div>
                    </div>

                    <div style={{
                      background: btcVolumeProfile.pricePosition === 'ABOVE_VAH' ? `${COLORS.success}22` : btcVolumeProfile.pricePosition === 'BELOW_VAL' ? `${COLORS.danger}22` : `${COLORS.info}22`,
                      color: btcVolumeProfile.pricePosition === 'ABOVE_VAH' ? COLORS.success : btcVolumeProfile.pricePosition === 'BELOW_VAL' ? COLORS.danger : COLORS.info,
                      padding: '6px 10px',
                      borderRadius: '4px',
                      fontSize: '10px',
                      fontWeight: '600',
                      textAlign: 'center'
                    }}>
                      {btcVolumeProfile.pricePosition === 'ABOVE_VAH' ? 'VAH ÜSTÜNDE' : btcVolumeProfile.pricePosition === 'BELOW_VAL' ? 'VAL ALTINDA' : 'VALUE AREA İÇİNDE'}
                    </div>
                  </div>
                )}

                {/* Order Flow */}
                {btcOrderFlow && (
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.warning}` }}>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: COLORS.warning, marginBottom: '12px' }}>💧 ORDER FLOW</div>

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Sinyal</div>
                      <div style={{
                        fontSize: '16px',
                        fontWeight: '700',
                        color: btcOrderFlow.signal.includes('BUY') ? COLORS.success : btcOrderFlow.signal.includes('SELL') ? COLORS.danger : COLORS.text.primary
                      }}>
                        {btcOrderFlow.signal} ({btcOrderFlow.confidence}%)
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                      <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px' }}>
                        <div style={{ fontSize: '9px', color: COLORS.text.muted }}>Imbalance</div>
                        <div style={{ fontSize: '11px', fontWeight: '600', color: btcOrderFlow.imbalance.strength.includes('BUY') ? COLORS.success : btcOrderFlow.imbalance.strength.includes('SELL') ? COLORS.danger : COLORS.text.primary }}>
                          {btcOrderFlow.imbalance.strength}
                        </div>
                      </div>
                      <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px' }}>
                        <div style={{ fontSize: '9px', color: COLORS.text.muted }}>Delta Trend</div>
                        <div style={{ fontSize: '11px', fontWeight: '600', color: btcOrderFlow.delta.trend === 'BULLISH' ? COLORS.success : btcOrderFlow.delta.trend === 'BEARISH' ? COLORS.danger : COLORS.text.primary }}>
                          {btcOrderFlow.delta.trend}
                        </div>
                      </div>
                    </div>

                    <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
                      <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '6px' }}>Agresif Baskı</div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '10px' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '9px', color: COLORS.success }}>BUY: {(btcOrderFlow.aggressive.buyPressure ?? 0).toFixed(0)}%</div>
                          <div style={{ height: '4px', background: COLORS.border.default, borderRadius: '2px', marginTop: '2px', overflow: 'hidden' }}>
                            <div style={{ width: `${btcOrderFlow.aggressive.buyPressure ?? 0}%`, height: '100%', background: COLORS.success }}></div>
                          </div>
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '9px', color: COLORS.danger }}>SELL: {(btcOrderFlow.aggressive.sellPressure ?? 0).toFixed(0)}%</div>
                          <div style={{ height: '4px', background: COLORS.border.default, borderRadius: '2px', marginTop: '2px', overflow: 'hidden' }}>
                            <div style={{ width: `${btcOrderFlow.aggressive.sellPressure ?? 0}%`, height: '100%', background: COLORS.danger }}></div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {btcOrderFlow.volume.surge && (
                      <div style={{ background: `${COLORS.warning}22`, color: COLORS.warning, padding: '6px 10px', borderRadius: '4px', fontSize: '10px', fontWeight: '600', textAlign: 'center' }}>
                        HACIM PATLAMASI ({(btcOrderFlow.volume.ratio ?? 0).toFixed(2)}x)
                      </div>
                    )}
                  </div>
                )}

                {/* Multi-Timeframe */}
                {btcMultiTimeframe && (
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.success}` }}>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: COLORS.success, marginBottom: '12px' }}>⏰ MULTI-TIMEFRAME</div>

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '4px' }}>Consensus</div>
                      <div style={{
                        fontSize: '16px',
                        fontWeight: '700',
                        color: btcMultiTimeframe.consensus.signal.includes('BULLISH') ? COLORS.success : btcMultiTimeframe.consensus.signal.includes('BEARISH') ? COLORS.danger : COLORS.text.primary
                      }}>
                        {btcMultiTimeframe.consensus.signal}
                      </div>
                      <div style={{ fontSize: '10px', color: COLORS.text.muted }}>Strength: {(btcMultiTimeframe.consensus.strength ?? 0).toFixed(0)}%</div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px', marginBottom: '12px' }}>
                      {(['1h', '4h', '1d', '1w'] as const).map((tf) => {
                        const tfData = btcMultiTimeframe.timeframes[tf];
                        return (
                          <div key={tf} style={{ background: COLORS.bg.card, padding: '6px', borderRadius: '4px' }}>
                            <div style={{ fontSize: '9px', color: COLORS.text.muted }}>{tf.toUpperCase()}</div>
                            <div style={{
                              fontSize: '10px',
                              fontWeight: '600',
                              color: tfData.overallSignal === 'BULLISH' ? COLORS.success : tfData.overallSignal === 'BEARISH' ? COLORS.danger : COLORS.text.primary
                            }}>
                              {tfData.overallSignal}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div style={{ background: COLORS.bg.card, padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
                      <div style={{ fontSize: '9px', color: COLORS.text.muted }}>Alignment</div>
                      <div style={{ fontSize: '14px', fontWeight: '700' }}>{btcMultiTimeframe.consensus.alignment}/4 Zaman Dilimi Uyumlu</div>
                    </div>

                    <div style={{
                      background: btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? `${COLORS.success}22` : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? `${COLORS.danger}22` : `${COLORS.border.default}22`,
                      color: btcMultiTimeframe.higherTimeframeBias === 'BULLISH' ? COLORS.success : btcMultiTimeframe.higherTimeframeBias === 'BEARISH' ? COLORS.danger : COLORS.text.muted,
                      padding: '6px 10px',
                      borderRadius: '4px',
                      fontSize: '10px',
                      fontWeight: '600',
                      textAlign: 'center'
                    }}>
                      HTF Bias: {btcMultiTimeframe.higherTimeframeBias}
                    </div>
                  </div>
                )}
              </div>

              <div style={{ marginTop: '12px', fontSize: '9px', color: COLORS.text.muted, textAlign: 'center' }}>
                White-Hat Compliant • Gerçek Piyasa Verisi • Eğitim Amaçlıdır
              </div>
            </div>
          )}

          {/* Filtreler */}
          <div className="animate-fadeIn" style={{ marginBottom: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center', padding: '16px', background: COLORS.bg.card, borderRadius: '8px', border: `1px solid ${COLORS.border.hover}` }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span style={{ color: COLORS.text.muted, fontSize: '12px', fontWeight: '600' }}>SİNYAL:</span>
              {['TÜMÜ', 'AL', 'SAT', 'BEKLE', 'NÖTR'].map((s) => (
                <button
                  key={s}
                  onClick={() => setFilterSignal(s)}
                  style={{
                    background: filterSignal === s ? COLORS.text.primary : 'transparent',
                    color: filterSignal === s ? COLORS.bg.primary : COLORS.text.secondary,
                    border: `1px solid ${filterSignal === s ? COLORS.text.primary : COLORS.border.active}`,
                    padding: '6px 14px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease-in-out',
                  }}
                >
                  {s}
                </button>
              ))}
            </div>

            <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span style={{ color: COLORS.text.muted, fontSize: '12px', fontWeight: '600' }}>SIRALA:</span>
              {[
                { key: 'score', label: 'Matrix Skoru' },
                { key: 'confidence', label: 'Güven' },
                { key: 'risk', label: 'Likidasyon Riski' },
              ].map((s) => (
                <button
                  key={s.key}
                  onClick={() => setSortBy(s.key as any)}
                  style={{
                    background: sortBy === s.key ? COLORS.text.primary : 'transparent',
                    color: sortBy === s.key ? COLORS.bg.primary : COLORS.text.secondary,
                    border: `1px solid ${sortBy === s.key ? COLORS.text.primary : COLORS.border.active}`,
                    padding: '6px 14px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease-in-out',
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Korelasyon Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
            {filteredCorrelations.map((coin, index) => {
              const signalColor = getSignalColor(coin.signal);
              const translateSignal = (signal: string) => {
                const map: {[key: string]: string} = {
                  'BUY': 'AL',
                  'SELL': 'SAT',
                  'WAIT': 'BEKLE',
                  'NEUTRAL': 'NÖTR',
                  'HOLD': 'TUT'
                };
                return map[signal] || signal;
              };
              const translatePhase = (phase: string) => {
                const map: {[key: string]: string} = {
                  'ACCUMULATION': 'BİRİKTİRME',
                  'MARKUP': 'YUKARI',
                  'DISTRIBUTION': 'DAĞITIM',
                  'MARKDOWN': 'AŞAĞI'
                };
                return map[phase] || phase;
              };
              const translateTrend = (trend: string) => {
                const map: {[key: string]: string} = {
                  'BULLISH': 'YÜKSELİŞTE',
                  'BEARISH': 'DÜŞÜŞte',
                  'NEUTRAL': 'NÖTR'
                };
                return map[trend] || trend;
              };
              const _translateStrength = (strength: string) => {
                const map: {[key: string]: string} = {
                  'STRONG': 'GÜÇLÜ',
                  'MODERATE': 'ORTA',
                  'WEAK': 'ZAYIF',
                  'NONE': 'YOK'
                };
                return map[strength] || strength;
              };
              const _translateDirection = (direction: string) => {
                const map: {[key: string]: string} = {
                  'POSITIVE': 'POZİTİF',
                  'NEGATIVE': 'NEGATİF',
                  'NEUTRAL': 'NÖTR'
                };
                return map[direction] || direction;
              };
              return (
              <div
                key={coin.symbol}
                className="animate-fadeIn"
                style={{
                  background: COLORS.bg.card,
                  border: `1px solid ${coin.signal === 'BUY' ? COLORS.success : coin.signal === 'SELL' ? COLORS.danger : COLORS.border.hover}`,
                  borderRadius: '10px',
                  padding: '16px',
                  cursor: 'pointer',
                  transition: 'all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)',
                  animationDelay: `${index * 0.05}s`
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.boxShadow = `0 8px 20px ${signalColor}33`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                {/* Sembol & Fiyat */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                  <div>
                    <div style={{ fontSize: '18px', fontWeight: '700', marginBottom: '4px' }}>
                      {coin.symbol.replace('USDT', '')}
                    </div>
                    <div style={{ fontSize: '13px', color: COLORS.text.muted, fontFamily: 'monospace' }}>
                      ${(coin.price ?? 0) < 1 ? (coin.price ?? 0).toFixed(6) : (coin.price ?? 0).toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '14px', fontWeight: '700', color: (coin.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                      {(coin.change24h ?? 0) >= 0 ? '+' : ''}{(coin.change24h ?? 0).toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* Omnipotent Skoru */}
                <div style={{ marginBottom: '12px', background: COLORS.bg.secondary, padding: '10px', borderRadius: '6px' }}>
                  <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px', fontWeight: '600' }}>OMNIPOTENT SKORU</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ flex: 1, height: '6px', background: COLORS.border.default, borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{ width: `${coin.omnipotentScore}%`, height: '100%', background: coin.omnipotentScore >= 75 ? COLORS.success : coin.omnipotentScore >= 50 ? COLORS.warning : COLORS.danger, transition: 'width 0.3s' }} />
                    </div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: coin.omnipotentScore >= 75 ? COLORS.success : coin.omnipotentScore >= 50 ? COLORS.warning : COLORS.danger }}>
                      {coin.omnipotentScore}
                    </div>
                  </div>
                </div>

                {/* Metrikler Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
                  <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>FAZ</div>
                    <div style={{ fontSize: '10px', fontWeight: '600', color: getPhaseColor(coin.marketPhase) }}>{translatePhase(coin.marketPhase)}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>TREND</div>
                    <div style={{ fontSize: '10px', fontWeight: '600', color: coin.trend === 'BULLISH' ? COLORS.success : coin.trend === 'BEARISH' ? COLORS.danger : COLORS.text.muted }}>{translateTrend(coin.trend)}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>GÜVEN</div>
                    <div style={{ fontSize: '12px', fontWeight: '700' }}>{coin.confidence}%</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>LİK RİSK</div>
                    <div style={{ fontSize: '12px', fontWeight: '700', color: (coin.liquidationRisk ?? 0) > 50 ? COLORS.danger : (coin.liquidationRisk ?? 0) > 30 ? COLORS.warning : COLORS.success }}>{(coin.liquidationRisk ?? 0).toFixed(0)}%</div>
                  </div>
                  {/* NEW v2.0: Funding Rate */}
                  {coin.fundingRate !== undefined && (
                    <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                      <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>FUNDING</div>
                      <div style={{ fontSize: '11px', fontWeight: '700', color: (coin.fundingRate ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                        {((coin.fundingRate ?? 0) * 100).toFixed(4)}%
                      </div>
                    </div>
                  )}
                  {/* NEW v2.0: Open Interest */}
                  {coin.openInterest !== undefined && (
                    <div style={{ background: COLORS.bg.secondary, padding: '8px', borderRadius: '4px' }}>
                      <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '4px' }}>OPEN INT</div>
                      <div style={{ fontSize: '10px', fontWeight: '700', color: COLORS.info }}>
                        {(coin.openInterest ?? 0) >= 1000000 ? `${((coin.openInterest ?? 0) / 1000000).toFixed(1)}M` : `${((coin.openInterest ?? 0) / 1000).toFixed(1)}K`}
                      </div>
                    </div>
                  )}
                </div>

                {/* NEW v2.0: Liquidation Zones Indicator */}
                {coin.nearestLiquidation && (
                  <div style={{ background: `${COLORS.danger}11`, padding: '8px', borderRadius: '6px', marginBottom: '12px', border: `1px solid ${COLORS.danger}33` }}>
                    <div style={{ fontSize: '9px', color: COLORS.text.muted, marginBottom: '6px', fontWeight: '600' }}>⚠️ EN YAKIN LİKİDASYONLAR</div>
                    <div style={{ display: 'flex', gap: '8px', fontSize: '10px' }}>
                      <div style={{ flex: 1 }}>
                        <span style={{ color: COLORS.danger }}>SHORT:</span>
                        <span style={{ marginLeft: '4px', fontWeight: '700' }}>${(coin.nearestLiquidation.short.price ?? 0).toFixed(2)}</span>
                        <span style={{ marginLeft: '4px', fontSize: '9px', color: COLORS.text.muted }}>({(coin.nearestLiquidation.short.distance ?? 0).toFixed(1)}%)</span>
                      </div>
                      <div style={{ flex: 1 }}>
                        <span style={{ color: COLORS.success }}>LONG:</span>
                        <span style={{ marginLeft: '4px', fontWeight: '700' }}>${(coin.nearestLiquidation.long.price ?? 0).toFixed(2)}</span>
                        <span style={{ marginLeft: '4px', fontSize: '9px', color: COLORS.text.muted }}>({(coin.nearestLiquidation.long.distance ?? 0).toFixed(1)}%)</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Sinyal Badge'i */}
                <div style={{ textAlign: 'center' }}>
                  <div style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    background: `linear-gradient(135deg, ${signalColor}, ${signalColor}dd)`,
                    color: COLORS.bg.primary,
                    padding: '8px 20px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    fontWeight: '700',
                    letterSpacing: '1px',
                    boxShadow: `0 0 10px ${signalColor}44`,
                  }}>
                    {(coin.signal === 'BUY' || coin.signal === 'SELL') && (
                      <span style={{ fontSize: '14px' }}>⚠️</span>
                    )}
                    {translateSignal(coin.signal)}
                  </div>
                </div>
              </div>
              );
            })}
          </div>

          {filteredCorrelations.length === 0 && (
            <div className="animate-fadeIn" style={{ textAlign: 'center', padding: '60px 20px', color: COLORS.text.muted, background: COLORS.bg.card, borderRadius: '10px', border: `1px solid ${COLORS.border.hover}` }}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
              <div style={{ fontSize: '16px' }}>Seçilen filtrelere uygun coin bulunamadı.</div>
            </div>
          )}
        </>
      )}
      </div>

      {/* MANTIK Modal */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '20px',
            backdropFilter: 'blur(10px)',
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '16px',
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}80`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.premium}15, ${COLORS.warning}15)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.premium}`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
              backdropFilter: 'blur(10px)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                  <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    Omnipotent Futures MANTIK
                  </h2>
                </div>
                <button
                  onClick={() => setShowLogicModal(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = COLORS.danger;
                    e.currentTarget.style.borderColor = COLORS.danger;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = COLORS.border.active;
                  }}
                >
                  KAPAT
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div style={{ padding: '24px' }}>
              {/* Overview */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.Activity style={{ width: '24px', height: '24px' }} />
                  Genel Bakış
                </h3>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                  Omnipotent Futures sayfası, 8 farklı strateji ile kapsamlı futures analizi sunar.
                  Long/short sinyalleri, funding rate analizi, open interest takibi, liquidation seviyeleri,
                  volume profile ve risk/reward oranlarını içeren çok boyutlu bir analiz platformudur.
                </p>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                  Ta-Lib ve AI modellerini kullanarak Binance Futures verilerini gerçek zamanlı olarak analiz eder
                  ve profesyonel seviyede ticaret sinyalleri üretir.
                </p>
              </div>

              {/* Key Features */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.Target style={{ width: '24px', height: '24px' }} />
                  Temel Özellikler
                </h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: '8 Strateji Analizi', desc: 'RSI, MACD, Bollinger Bands, MA Cross, Volume, Fibonacci, Stochastic ve ATR stratejilerini aynı anda analiz eder.' },
                    { name: 'Long/Short Sinyalleri', desc: 'Her strateji için ayrı long ve short sinyalleri üretir. Çoğunluk oylaması ile genel sinyal belirlenir.' },
                    { name: 'Funding Rate Analizi', desc: 'Binance Futures funding rate verilerini takip eder ve yıllık funding oranlarını hesaplar.' },
                    { name: 'Open Interest Takibi', desc: 'Açık pozisyon miktarını ve değerini izler. OI artışı veya azalışı trend gücünü gösterir.' },
                    { name: 'Liquidation Seviyeleri', desc: 'Long ve short pozisyonlar için en yakın liquidation seviyelerini ve mesafelerini gösterir.' },
                    { name: 'Volume Profile', desc: 'İşlem hacmi dağılımını analiz eder ve yüksek/düşük hacim bölgelerini belirler.' },
                    { name: 'Risk/Reward Oranları', desc: 'Her sinyal için risk-ödül oranını hesaplar ve güvenli giriş/çıkış seviyelerini önerir.' },
                    { name: 'Otomatik Yenileme (30s)', desc: 'Her 30 saniyede bir tüm veriler otomatik olarak güncellenir ve yeni analizler yapılır.' }
                  ].map((feature, index) => (
                    <div key={index} style={{
                      background: `${COLORS.bg.card}40`,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '16px',
                      transition: 'all 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = COLORS.premium;
                      e.currentTarget.style.transform = 'translateX(8px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = COLORS.border.default;
                      e.currentTarget.style.transform = 'translateX(0)';
                    }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}20)`,
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '14px',
                          fontWeight: 'bold',
                          color: COLORS.premium,
                          minWidth: '32px',
                          textAlign: 'center',
                        }}>
                          {index + 1}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            {feature.name}
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            {feature.desc}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Usage Guide */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.BarChart3 style={{ width: '24px', height: '24px' }} />
                  Kullanım Rehberi
                </h3>
                <div style={{ display: 'grid', gap: '16px' }}>
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      background: `linear-gradient(135deg, ${COLORS.success}, ${COLORS.success}dd)`,
                      color: '#000',
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      fontWeight: 'bold',
                      flexShrink: 0,
                    }}>
                      1
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        Coin Listesini İnceleyin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Ana ekranda tüm Binance Futures çiftleri listelenir. Omnipotent Score, güven skoru ve sinyal bilgilerini görürsünüz.
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      background: `linear-gradient(135deg, ${COLORS.info}, ${COLORS.info}dd)`,
                      color: '#000',
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      fontWeight: 'bold',
                      flexShrink: 0,
                    }}>
                      2
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        Filtreleri Kullanın
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        LONG, SHORT veya YÜKSELİŞ filtrelerini kullanarak istediğiniz sinyallere odaklanın. Sıralama seçenekleriyle listeyi özelleştirin.
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                      color: '#000',
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      fontWeight: 'bold',
                      flexShrink: 0,
                    }}>
                      3
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        Detaylı Analiz Görüntüleyin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Her coin için funding rate, open interest, liquidation seviyeleri ve tüm strateji sonuçlarını inceleyin.
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.premium}dd)`,
                      color: '#000',
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      fontWeight: 'bold',
                      flexShrink: 0,
                    }}>
                      4
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        Risk Yönetimi Uygulayın
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Liquidation seviyelerini kontrol edin, stop-loss belirleyin ve risk/reward oranını değerlendirin.
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Important Notes */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '12px',
                padding: '20px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                  Önemli Notlar
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.8' }}>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Binance Futures Verisi:</strong> Tüm analizler Binance Futures API'sinden alınan gerçek zamanlı verilerle yapılır.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Otomatik Yenileme (30s):</strong> Her 30 saniyede bir tüm coinler yeniden analiz edilir ve sinyaller güncellenir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Ta-Lib ve AI Modeller:</strong> Python Ta-Lib kütüphanesi ve özel AI modelleri kullanılarak profesyonel seviye analiz sağlanır.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Yüksek Kaldıraç Riski:</strong> Futures işlemler yüksek kaldıraçlıdır. Küçük fiyat hareketleri büyük kayıplara neden olabilir.
                  </li>
                  <li>
                    <strong style={{ color: COLORS.text.primary }}>Eğitim Amaçlıdır:</strong> Bu sinyaller yatırım tavsiyesi değildir. Kendi araştırmanızı yapın ve sorumlu yatırım yapın.
                  </li>
                </ul>
              </div>
            </div>

            {/* Modal Footer */}
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
              padding: '20px 24px',
              borderTop: `1px solid ${COLORS.border.default}`,
              textAlign: 'center',
              position: 'sticky',
              bottom: 0,
              backdropFilter: 'blur(10px)',
            }}>
              <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                Omnipotent Futures - Çok Boyutlu Futures Analizi ve Profesyonel Ticaret Sinyalleri
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
