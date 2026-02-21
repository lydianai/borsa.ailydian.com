'use client';

/**
 * 🪜 QUANTUM MERDİVEN STRATEJİ SAYFASI
 * Premium Fibonacci Merdiveni + MA 7-25-99 Dip Avcısı + ZigZag Analizi
 *
 * Özellikler:
 * - Canlı fiyatlarla gerçek zamanlı coin listesi
 * - Çoklu zaman dilimi Fibonacci merdiveni görselleştirmesi
 * - MA 7-25-99 Dip Avcısı puanlama sistemi
 * - Güç puanlarıyla birleşme bölgesi tespiti
 * - ZigZag salınım noktası analizi
 * - Profesyonel premium tasarım
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS, getChangeColor } from '@/lib/colors';
import { DecisionPanel } from '@/components/DecisionPanel';

interface CoinData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
}

interface FibonacciLadder {
  timeframe: string;
  direction: string;
  levels: { [key: string]: number };
  range: number;
  swing_high: number;
  swing_low: number;
}

interface MABottomHunter {
  bottom_ma: string;
  bottom_value: number;
  mas: {
    ma7: number;
    ma25: number;
    ma99: number;
  };
  score: number;
  signal: string;
  confidence: number;
  crossover_imminent: boolean;
  ma_alignment: string;
  distance_to_next: number;
}

interface ConfluenceZone {
  price: number;
  timeframes: number;
  levels: string[];
  power_score: number;
  confluence_count: number;
}

interface GoldenCrossCrossover {
  crossover_detected: boolean;
  crossover_type: 'golden' | 'death' | null;
  crossover_candles_ago: number | null;
  retest_detected: boolean;
  retest_touches: number;
  distance_to_cross_pct: number;
  ma_short: number;
  ma_long: number;
  ma_short_period: number;
  ma_long_period: number;
  score: number;
  current_price: number;
}

interface GoldenCrossTimeframe {
  ma7_x_ma25: GoldenCrossCrossover;
  ma25_x_ma99: GoldenCrossCrossover;
  overall_score: number;
  signal: string;
}

interface GoldenCrossData {
  symbol: string;
  timeframes: { [key: string]: GoldenCrossTimeframe };
  best_timeframe: string;
  best_score: number;
  timestamp: string;
}

interface QuantumLadderData {
  symbol: string;
  signal: string;
  confidence: number;
  current_price: number;
  fibonacci_ladders: FibonacciLadder[];
  ma_bottom_hunter: MABottomHunter;
  golden_cross: GoldenCrossData;
  confluence_zones: ConfluenceZone[];
  nearest_resistance: ConfluenceZone | null;
  nearest_support: ConfluenceZone | null;
  timeframes_analyzed: number;
  total_confluence_zones: number;
  timestamp: string;
}

export default function QuantumLadderPage() {
  const [coins, setCoins] = useState<CoinData[]>([]);
  const [selectedCoin, setSelectedCoin] = useState<string>('BTCUSDT');
  const [data, setData] = useState<QuantumLadderData | null>(null);
  const [loading, setLoading] = useState(true);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [countdown, setCountdown] = useState(900); // 15 dakika = 900 saniye
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [buyCoins, setBuyCoins] = useState<string[]>([]); // BUY sinyali veren coinler
  const [coinIndex, setCoinIndex] = useState(0); // Rotation için index
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Sadece istemci tarafı render için
  useEffect(() => {
    setMounted(true);
  }, []);

  // BUY sinyali veren coinleri getir
  const fetchBuySignalCoins = async () => {
    try {
      const response = await fetch('/api/buy-signals-coins');
      if (!response.ok) {
        console.warn('[Quantum Ladder] Failed to fetch BUY signal coins');
        return;
      }

      const result = await response.json();
      if (result.success && result.data.coins && result.data.coins.length > 0) {
        const coinSymbols = result.data.coins.map((c: any) => c.symbol);
        setBuyCoins(coinSymbols);

        // İlk coin'i seç
        if (coinSymbols.length > 0) {
          setSelectedCoin(coinSymbols[0]);
          setCoinIndex(0);
        }

        console.log(`[Quantum Ladder] Loaded ${coinSymbols.length} BUY signal coins:`, coinSymbols);
      } else {
        // BUY sinyali yoksa BTC ile devam et
        console.log('[Quantum Ladder] No BUY signals found - falling back to BTCUSDT');
        setBuyCoins([]);
        setSelectedCoin('BTCUSDT');
        setCoinIndex(0);
      }
    } catch (error) {
      console.warn('[Quantum Ladder] Error fetching BUY signal coins:', error);
      // Hata durumunda da BTC ile devam et
      setBuyCoins([]);
      setSelectedCoin('BTCUSDT');
      setCoinIndex(0);
    }
  };

  // Binance'den yeniden deneme ile coin listesi getir
  const fetchCoins = async (retries = 3) => {
    for (let i = 0; i < retries; i++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          try {
            controller.abort();
          } catch (err) {
            // Suppress AbortError during timeout
          }
        }, 10000); // 10 second timeout

        const response = await fetch('/api/binance/futures', {
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success && result.data?.all) {
          // Tüm 600+ koinleri göster
          setCoins(result.data.all);
          setLoading(false);
          return; // Success - exit function
        }
      } catch (err) {
        if (i === retries - 1) {
          // Last retry failed - silently handle error
          console.warn('Coin listesi yüklenemedi, varsayılan liste kullanılıyor');
        } else {
          // Wait before retry
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }

    setLoading(false);
  };

  // Yeniden deneme ile Quantum Merdiven analizi getir
  const fetchAnalysis = async (symbol: string, retries = 2) => {
    setAnalysisLoading(true);
    setError(null);

    console.log(`[Quantum Ladder] Fetching analysis for ${symbol}...`);

    for (let i = 0; i < retries; i++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
          try {
            controller.abort();
          } catch (err) {
            // Suppress AbortError during timeout
          }
        }, 30000); // 30 second timeout for analysis

        console.log(`[Quantum Ladder] Attempt ${i + 1}/${retries} for ${symbol}`);

        const response = await fetch('/api/quantum-ladder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            symbol,
            timeframes: ['15m', '1h', '4h'],
            limit: 500
          }),
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        console.log(`[Quantum Ladder] Response status: ${response.status} for ${symbol}`);

        if (!response.ok) {
          // Try to get error details from response
          let errorDetail = '';
          try {
            const errorData = await response.json();
            errorDetail = errorData.error || '';
            // Don't log error to console - only to internal logs
            console.log(`[Quantum Ladder] Server returned error for ${symbol}:`, response.status);
          } catch (e) {
            console.log(`[Quantum Ladder] Response parsing issue for ${symbol}`);
          }

          if (response.status === 404) {
            throw new Error('Quantum Ladder servisi bulunamadı');
          } else if (response.status === 400) {
            // Python service error - show user-friendly message
            throw new Error('Analiz şu anda kullanılamıyor - daha sonra tekrar deneyin');
          } else if (response.status === 500) {
            throw new Error(errorDetail || 'Sunucu hatası - Analiz yapılamadı');
          } else if (response.status === 429) {
            throw new Error('Çok fazla istek - Lütfen bekleyin');
          } else {
            throw new Error(`Analiz şu anda kullanılamıyor`);
          }
        }

        const result = await response.json();

        // Check if this is a successful response (including fallback)
        if (result.success && result.data) {
          // Success! Use the data whether it's from Python service or Binance fallback
          if (result.fallback === true) {
            console.log(`[Quantum Ladder] Using Binance fallback data for ${symbol} (Python service unavailable)`);
          } else {
            console.log(`[Quantum Ladder] Analysis successful for ${symbol}`);
          }
          setData(result.data);
          setAnalysisLoading(false);
          return; // Success - exit function
        } else {
          const errorMsg = result.error || 'Analiz verisi alınamadı';
          console.log(`[Quantum Ladder] API returned error for ${symbol}:`, response.status);
          if (i === retries - 1) {
            // Show user-friendly message instead of technical error
            if (errorMsg.includes('Python service') || errorMsg.includes('BAD REQUEST') || errorMsg.includes('400')) {
              setError('Analiz servisi geçici olarak kullanılamıyor');
            } else {
              setError('Analiz verisi alınamadı');
            }
          }
        }
      } catch (err: any) {
        // Log only - don't use console.error to avoid showing in UI
        console.log(`[Quantum Ladder] Attempt ${i + 1}/${retries} for ${symbol}:`, err.message);

        if (i === retries - 1) {
          // Last retry failed - show user-friendly error
          if (err.name === 'AbortError') {
            setError(`Analiz süresi doldu - Lütfen daha sonra tekrar deneyin`);
          } else if (err.message) {
            // Show the error message without symbol suffix
            setError(err.message);
          } else {
            setError(`Analiz şu anda kullanılamıyor`);
          }
        } else {
          // Wait before retry
          console.log(`[Quantum Ladder] Retrying in 1.5s for ${symbol}...`);
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      }
    }

    setAnalysisLoading(false);
  };

  // Sayfa yüklendiğinde BUY sinyali veren coinleri getir
  useEffect(() => {
    if (mounted) {
      fetchCoins();
      fetchBuySignalCoins(); // BUY sinyali veren coinleri çek
    }
  }, [mounted]);

  // Yüklendikten sonra her 30 saniyede bir coinleri otomatik yenile
  useEffect(() => {
    if (!mounted) return;

    const interval = setInterval(fetchCoins, 30000);
    return () => clearInterval(interval);
  }, [mounted]);

  // BUY sinyali veren coinler arasında her 15 dakikada bir otomatik dön
  useEffect(() => {
    if (!mounted || buyCoins.length === 0) return;

    const rotationInterval = setInterval(() => {
      // Bir sonraki coin'e geç
      setCoinIndex((prevIndex) => {
        const nextIndex = (prevIndex + 1) % buyCoins.length;
        setSelectedCoin(buyCoins[nextIndex]);
        setCountdown(900); // Reset countdown to 15 minutes
        console.log(`[Quantum Ladder] Rotating to coin ${nextIndex + 1}/${buyCoins.length}: ${buyCoins[nextIndex]}`);
        return nextIndex;
      });
    }, 900000); // 15 dakika = 900000 ms

    return () => clearInterval(rotationInterval);
  }, [mounted, buyCoins]);

  useEffect(() => {
    if (selectedCoin) {
      fetchAnalysis(selectedCoin);
    }
  }, [selectedCoin]);

  // Countdown timer - 15 dakika (900 saniye)
  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prev) => (prev > 0 ? prev - 1 : 900));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const filteredCoins = coins.filter(coin =>
    coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatPrice = (price: number | undefined | null) => {
    if (!price && price !== 0) return '$0.00'; // Handle undefined, null, NaN
    if (price >= 1000) return `$${price.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(6)}`;
  };

  const getSignalColor = (signal: string) => {
    if (signal === 'AL' || signal === 'BUY') return COLORS.success;
    if (signal === 'SAT' || signal === 'SELL') return COLORS.danger;
    return COLORS.warning;
  };

  // Giriş fiyatı hesaplama fonksiyonu
  const calculateEntryPrice = (data: QuantumLadderData | null): string => {
    if (!data) return 'N/A';

    const signal = data.signal;
    const currentPrice = data.current_price;
    const nearest_support = data.nearest_support;
    const nearest_resistance = data.nearest_resistance;
    const ma_bottom = data.ma_bottom_hunter?.bottom_value;

    if (signal === 'BUY' || signal === 'AL') {
      // AL sinyali: En yakın destek veya MA bottom'a giriş öner
      if (nearest_support && nearest_support.price < currentPrice) {
        const entryPrice = nearest_support.price * 1.002; // %0.2 üstünden gir
        return `${formatPrice(entryPrice)} (Destek Üstü)`;
      } else if (ma_bottom && ma_bottom < currentPrice) {
        const entryPrice = ma_bottom * 1.005; // %0.5 üstünden gir
        return `${formatPrice(entryPrice)} (MA Dibi)`;
      } else {
        // Mevcut fiyattan %1 aşağı limit order
        const entryPrice = currentPrice * 0.99;
        return `${formatPrice(entryPrice)} (Limit Order)`;
      }
    } else if (signal === 'SELL' || signal === 'SAT') {
      // SAT sinyali: En yakın direnç altında çıkış öner
      if (nearest_resistance && nearest_resistance.price > currentPrice) {
        const exitPrice = nearest_resistance.price * 0.998; // %0.2 altından sat
        return `${formatPrice(exitPrice)} (Direnç Altı)`;
      } else {
        // Mevcut fiyattan %1 yukarı limit order
        const exitPrice = currentPrice * 1.01;
        return `${formatPrice(exitPrice)} (Limit Order)`;
      }
    } else {
      // HOLD - İki taraflı limit öner
      const buyLimit = currentPrice * 0.98;
      const sellLimit = currentPrice * 1.02;
      return `AL: ${formatPrice(buyLimit)} / SAT: ${formatPrice(sellLimit)}`;
    }
  };

  // Hidrasyon hatasını önlemek için istemci tarafı yüklenmeyi bekle
  if (!mounted || (loading && !data)) {
    return (
      <div
        suppressHydrationWarning
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          backgroundColor: '#0A0A0A'
        }}
      >
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div className="dashboard-container" suppressHydrationWarning style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
      {/* Sidebar */}
      <SharedSidebar
        currentPage="quantum-ladder"
        notificationCounts={notificationCounts}
      />

      {/* Main Content */}
      <div className="dashboard-main">
        <main className="dashboard-content" style={{ padding: '20px' }}>
          {/* Page Header */}
          <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <Icons.TrendingUp style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  Quantum Merdiven Stratejisi
                </h1>
              </div>
              <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
                Fibonacci Merdivenleri + MA 7-25-99 Dip Avcısı + ZigZag Swing Analizi
              </p>
            </div>

            {/* MANTIK Button - Responsive */}
            <div>
              <style>{`
                @media (max-width: 768px) {
                  .mantik-button-ladder {
                    padding: 10px 20px !important;
                    fontSize: 13px !important;
                    height: 42px !important;
                  }
                  .mantik-button-ladder svg {
                    width: 18px !important;
                    height: 18px !important;
                  }
                }
                @media (max-width: 480px) {
                  .mantik-button-ladder {
                    padding: 8px 16px !important;
                    fontSize: 12px !important;
                    height: 40px !important;
                  }
                  .mantik-button-ladder svg {
                    width: 16px !important;
                    height: 16px !important;
                  }
                }
              `}</style>
              <button
                onClick={() => setShowLogicModal(true)}
                className="mantik-button-ladder"
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
                  boxShadow: `0 4px 20px ${COLORS.premium}40`,
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = `0 6px 25px ${COLORS.premium}60`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = `0 4px 20px ${COLORS.premium}40`;
                }}
              >
                <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
                MANTIK
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '20px' }}>
            {/* Left Panel - Coin List */}
            <div className="neon-card" style={{ padding: '0', maxHeight: 'calc(100vh - 160px)', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              {/* Search */}
              <div style={{ padding: '16px', borderBottom: `1px solid ${COLORS.border.default}` }}>
                <div style={{ position: 'relative' }}>
                  <Icons.Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', width: '18px', height: '18px', color: COLORS.text.secondary }} />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Coin ara..."
                    style={{
                      width: '100%',
                      padding: '12px 12px 12px 40px',
                      background: COLORS.bg.secondary,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      color: COLORS.text.primary,
                      fontSize: '14px',
                      outline: 'none'
                    }}
                  />
                </div>
              </div>

              {/* Coin List */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                {filteredCoins.map((coin) => (
                  <div
                    key={coin.symbol}
                    onClick={() => setSelectedCoin(coin.symbol)}
                    style={{
                      padding: '16px',
                      borderBottom: `1px solid ${COLORS.border.default}`,
                      cursor: 'pointer',
                      background: selectedCoin === coin.symbol ? COLORS.bg.hover : 'transparent',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (selectedCoin !== coin.symbol) {
                        e.currentTarget.style.background = COLORS.bg.secondary;
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedCoin !== coin.symbol) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary }}>
                        {coin.symbol.replace('USDT', '')}/USDT
                      </div>
                      <div style={{
                        fontSize: '13px',
                        fontWeight: '600',
                        color: getChangeColor(coin.changePercent24h),
                        padding: '4px 8px',
                        borderRadius: '4px',
                        background: coin.changePercent24h >= 0 ? `${COLORS.success}20` : `${COLORS.danger}20`
                      }}>
                        {coin.changePercent24h >= 0 ? '+' : ''}{coin.changePercent24h.toFixed(2)}%
                      </div>
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.text.primary }}>
                      {formatPrice(coin.price)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Panel - Analysis */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {analysisLoading ? (
                <div className="neon-card" style={{ padding: '60px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <LoadingAnimation />
                </div>
              ) : error ? (
                <div className="neon-card" style={{
                  padding: '40px',
                  textAlign: 'center',
                  background: `${COLORS.danger}0D`,
                  border: `1px solid ${COLORS.danger}4D`
                }}>
                  <Icons.AlertTriangle style={{ width: '48px', height: '48px', color: COLORS.danger, margin: '0 auto 16px' }} />
                  <div style={{ fontSize: '18px', color: COLORS.danger, marginBottom: '8px', fontWeight: '600' }}>
                    Analiz Hatası
                  </div>
                  <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '20px' }}>
                    {error}
                  </div>
                  <button
                    onClick={() => fetchAnalysis(selectedCoin)}
                    className="premium-button"
                    style={{
                      padding: '12px 24px',
                      background: COLORS.premium,
                      color: '#000',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '14px',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    Tekrar Dene
                  </button>
                </div>
              ) : data ? (
                <>
                  {/* KARAR PANELİ - Decision Engine */}
                  <DecisionPanel symbol={selectedCoin} />

                  {/* Overview Cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                    {/* Price Card */}
                    <div className="neon-card" style={{ padding: '20px', background: `${COLORS.premium}0D`, border: `1px solid ${COLORS.premium}4D` }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        {selectedCoin.replace('USDT', '')}/USDT
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '4px' }}>
                        {formatPrice(data.current_price)}
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        Güncel Fiyat
                      </div>
                    </div>

                    {/* Signal Card */}
                    <div className="neon-card" style={{ padding: '20px', background: `${getSignalColor(data.signal)}0D`, border: `1px solid ${getSignalColor(data.signal)}4D` }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        Strateji Sinyali
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: getSignalColor(data.signal), marginBottom: '4px' }}>
                        {data.signal}
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        {data.confidence}% Güven
                      </div>
                    </div>

                    {/* Confluence Card */}
                    <div className="neon-card" style={{ padding: '20px', background: `${COLORS.warning}0D`, border: `1px solid ${COLORS.warning}4D` }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        Confluence Bölgeleri
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '4px' }}>
                        {data.total_confluence_zones}
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        Aktif Seviye
                      </div>
                    </div>

                    {/* Timeframes Card */}
                    <div className="neon-card" style={{ padding: '20px', background: `${COLORS.info}0D`, border: `1px solid ${COLORS.info}4D` }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        Analiz Edilen
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.info, marginBottom: '4px' }}>
                        {data.timeframes_analyzed}
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        Zaman Dilimi
                      </div>
                    </div>

                    {/* Entry Price Card */}
                    <div className="neon-card" style={{ padding: '20px', background: `linear-gradient(135deg, ${COLORS.premium}10, ${COLORS.success}10)`, border: `1px solid ${COLORS.success}4D`, gridColumn: 'span 2' }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        Önerilen Giriş Fiyatı
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success, marginBottom: '4px' }}>
                        {calculateEntryPrice(data)}
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        Analiz Bazlı Strateji Önerisi
                      </div>
                    </div>
                  </div>

                  {/* MA Bottom Hunter */}
                  {data.ma_bottom_hunter && (
                    <div className="neon-card" style={{ padding: '24px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                        <Icons.Activity style={{ width: '28px', height: '28px', color: COLORS.premium }} />
                        <div>
                          <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                            MA 7-25-99 Bottom Hunter
                          </h2>
                          <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                            Moving Average Tabanlı Dip Av Sistemi
                          </p>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: COLORS.bg.secondary,
                          border: `1px solid ${COLORS.border.default}`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>MA 7</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.info }}>
                            {formatPrice(data.ma_bottom_hunter.mas.ma7)}
                          </div>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: COLORS.bg.secondary,
                          border: `1px solid ${COLORS.border.default}`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>MA 25</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.info }}>
                            {formatPrice(data.ma_bottom_hunter.mas.ma25)}
                          </div>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: COLORS.bg.secondary,
                          border: `1px solid ${COLORS.border.default}`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>MA 99</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.info }}>
                            {formatPrice(data.ma_bottom_hunter.mas.ma99)}
                          </div>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: `${COLORS.premium}20`,
                          border: `1px solid ${COLORS.premium}4D`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Bottom MA</div>
                          <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.premium }}>
                            {data.ma_bottom_hunter.bottom_ma}
                          </div>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: `${getSignalColor(data.ma_bottom_hunter.signal)}20`,
                          border: `1px solid ${getSignalColor(data.ma_bottom_hunter.signal)}4D`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Sinyal</div>
                          <div style={{ fontSize: '20px', fontWeight: 'bold', color: getSignalColor(data.ma_bottom_hunter.signal) }}>
                            {data.ma_bottom_hunter.signal}
                          </div>
                          <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                            {data.ma_bottom_hunter.confidence}% Güven
                          </div>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: `${COLORS.success}20`,
                          border: `1px solid ${COLORS.success}4D`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Skor</div>
                          <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                            {data.ma_bottom_hunter.score}/120
                          </div>
                          <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                            Dip Gücü
                          </div>
                        </div>
                        <div style={{
                          padding: '16px',
                          borderRadius: '8px',
                          background: data.ma_bottom_hunter.crossover_imminent ? `${COLORS.danger}20` : `${COLORS.bg.secondary}`,
                          border: data.ma_bottom_hunter.crossover_imminent ? `1px solid ${COLORS.danger}4D` : `1px solid ${COLORS.border.default}`
                        }}>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Crossover</div>
                          <div style={{ fontSize: '16px', fontWeight: 'bold', color: data.ma_bottom_hunter.crossover_imminent ? COLORS.danger : COLORS.text.primary }}>
                            {data.ma_bottom_hunter.crossover_imminent ? 'YAKIN!' : 'Stable'}
                          </div>
                          <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                            Kesişme Durumu
                          </div>
                        </div>
                      </div>

                      <div style={{
                        padding: '12px 16px',
                        borderRadius: '8px',
                        background: COLORS.bg.secondary,
                        fontSize: '13px',
                        color: COLORS.text.secondary,
                        fontStyle: 'italic'
                      }}>
                        {data.ma_bottom_hunter.ma_alignment}
                      </div>
                    </div>
                  )}

                  {/* Golden Cross Analysis */}
                  {data.golden_cross && data.golden_cross.timeframes && (
                    <div className="neon-card" style={{ padding: '24px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                        <Icons.TrendingUp style={{ width: '28px', height: '28px', color: COLORS.success }} />
                        <div>
                          <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                            Golden Cross Analizi
                          </h2>
                          <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                            MA7x25 ve MA25x99 Kesişim Tespiti + Retest Analizi
                          </p>
                        </div>
                      </div>

                      {/* Best Timeframe Summary */}
                      <div style={{
                        padding: '20px',
                        borderRadius: '12px',
                        background: `linear-gradient(135deg, ${COLORS.success}20, ${COLORS.premium}20)`,
                        border: `1px solid ${COLORS.success}40`,
                        marginBottom: '20px'
                      }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
                          <div>
                            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>En İyi Zaman Dilimi</div>
                            <div style={{ fontSize: '22px', fontWeight: 'bold', color: COLORS.success }}>
                              {data.golden_cross.best_timeframe?.toUpperCase() || 'N/A'}
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Skor</div>
                            <div style={{ fontSize: '22px', fontWeight: 'bold', color: COLORS.premium }}>
                              {data.golden_cross.best_score}/100
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Timeframe Analysis */}
                      {Object.entries(data.golden_cross.timeframes).map(([tf, tfData]) => (
                        <div key={tf} style={{ marginBottom: '16px' }}>
                          <div style={{
                            padding: '16px',
                            borderRadius: '12px',
                            background: COLORS.bg.secondary,
                            border: `1px solid ${COLORS.border.default}`
                          }}>
                            {/* Timeframe Header */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                              <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.text.primary }}>
                                {tf.toUpperCase()} Timeframe
                              </div>
                              <div style={{
                                background: tfData.signal === 'STRONG_BUY' || tfData.signal === 'BUY' ? `${COLORS.success}30` :
                                           tfData.signal === 'NEUTRAL' ? `${COLORS.warning}30` : `${COLORS.danger}30`,
                                color: tfData.signal === 'STRONG_BUY' || tfData.signal === 'BUY' ? COLORS.success :
                                       tfData.signal === 'NEUTRAL' ? COLORS.warning : COLORS.danger,
                                fontSize: '12px',
                                fontWeight: '700',
                                padding: '6px 12px',
                                borderRadius: '6px'
                              }}>
                                {tfData.signal}
                              </div>
                            </div>

                            {/* MA7 x MA25 Crossover */}
                            {tfData.ma7_x_ma25 && (
                              <div style={{
                                padding: '14px',
                                borderRadius: '8px',
                                background: `${COLORS.info}10`,
                                border: `1px solid ${COLORS.info}30`,
                                marginBottom: '12px'
                              }}>
                                <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.info, marginBottom: '10px' }}>
                                  MA7 x MA25 Kesişimi
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Kesişim Durumu</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: tfData.ma7_x_ma25.crossover_detected ? COLORS.success : COLORS.text.secondary }}>
                                      {tfData.ma7_x_ma25.crossover_detected ?
                                        `${tfData.ma7_x_ma25.crossover_type === 'golden' ? '🟢 Golden' : '🔴 Death'}` :
                                        '⚪ Yok'}
                                    </div>
                                  </div>
                                  {tfData.ma7_x_ma25.crossover_detected && (
                                    <>
                                      <div>
                                        <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Kaç Mum Önce</div>
                                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.text.primary }}>
                                          {tfData.ma7_x_ma25.crossover_candles_ago} mum
                                        </div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Retest</div>
                                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: tfData.ma7_x_ma25.retest_detected ? COLORS.success : COLORS.text.secondary }}>
                                          {tfData.ma7_x_ma25.retest_detected ?
                                            `✅ ${tfData.ma7_x_ma25.retest_touches}x` :
                                            '❌ Yok'}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Skor</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.premium }}>
                                      {tfData.ma7_x_ma25.score}/100
                                    </div>
                                  </div>
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Mesafe</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.warning }}>
                                      %{tfData.ma7_x_ma25.distance_to_cross_pct.toFixed(2)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* MA25 x MA99 Crossover */}
                            {tfData.ma25_x_ma99 && (
                              <div style={{
                                padding: '14px',
                                borderRadius: '8px',
                                background: `${COLORS.premium}10`,
                                border: `1px solid ${COLORS.premium}30`
                              }}>
                                <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '10px' }}>
                                  MA25 x MA99 Kesişimi
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Kesişim Durumu</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: tfData.ma25_x_ma99.crossover_detected ? COLORS.success : COLORS.text.secondary }}>
                                      {tfData.ma25_x_ma99.crossover_detected ?
                                        `${tfData.ma25_x_ma99.crossover_type === 'golden' ? '🟢 Golden' : '🔴 Death'}` :
                                        '⚪ Yok'}
                                    </div>
                                  </div>
                                  {tfData.ma25_x_ma99.crossover_detected && (
                                    <>
                                      <div>
                                        <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Kaç Mum Önce</div>
                                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.text.primary }}>
                                          {tfData.ma25_x_ma99.crossover_candles_ago} mum
                                        </div>
                                      </div>
                                      <div>
                                        <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Retest</div>
                                        <div style={{ fontSize: '13px', fontWeight: 'bold', color: tfData.ma25_x_ma99.retest_detected ? COLORS.success : COLORS.text.secondary }}>
                                          {tfData.ma25_x_ma99.retest_detected ?
                                            `✅ ${tfData.ma25_x_ma99.retest_touches}x` :
                                            '❌ Yok'}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Skor</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.premium }}>
                                      {tfData.ma25_x_ma99.score}/100
                                    </div>
                                  </div>
                                  <div>
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>Mesafe</div>
                                    <div style={{ fontSize: '13px', fontWeight: 'bold', color: COLORS.warning }}>
                                      %{tfData.ma25_x_ma99.distance_to_cross_pct.toFixed(2)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Overall Score */}
                            <div style={{
                              marginTop: '12px',
                              padding: '12px',
                              borderRadius: '8px',
                              background: `${COLORS.success}15`,
                              border: `1px solid ${COLORS.success}30`,
                              textAlign: 'center'
                            }}>
                              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>Genel Skor</div>
                              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                                {tfData.overall_score}/100
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Confluence Zones */}
                  {data.confluence_zones && data.confluence_zones.length > 0 && (
                    <div className="neon-card" style={{ padding: '24px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                        <Icons.Layers style={{ width: '28px', height: '28px', color: COLORS.warning }} />
                        <div>
                          <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                            Fibonacci Confluence Bölgeleri
                          </h2>
                          <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                            Çoklu Zaman Dilimi Fibonacci Kesişim Noktaları
                          </p>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '12px' }}>
                        {data.confluence_zones.map((zone, idx) => (
                          <div
                            key={idx}
                            style={{
                              padding: '16px',
                              borderRadius: '12px',
                              background: `linear-gradient(135deg, ${COLORS.premium}10, ${COLORS.warning}10)`,
                              border: `1px solid ${COLORS.premium}40`
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium }}>
                                {formatPrice(zone.price)}
                              </div>
                              <div style={{
                                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                                color: '#000',
                                fontSize: '11px',
                                fontWeight: '700',
                                padding: '4px 10px',
                                borderRadius: '6px'
                              }}>
                                ⚡ {zone.power_score}
                              </div>
                            </div>
                            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '10px' }}>
                              {zone.confluence_count} zaman dilimi birleşimi
                            </div>
                            <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                              {zone.levels.map((level, i) => (
                                <span key={i} style={{
                                  background: `${COLORS.info}30`,
                                  color: COLORS.info,
                                  fontSize: '10px',
                                  fontWeight: '600',
                                  padding: '3px 8px',
                                  borderRadius: '4px'
                                }}>
                                  {level}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Fibonacci Ladders */}
                  {data.fibonacci_ladders && data.fibonacci_ladders.length > 0 && (
                    <div className="neon-card" style={{ padding: '24px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                        <Icons.BarChart3 style={{ width: '28px', height: '28px', color: COLORS.info }} />
                        <div>
                          <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                            Fibonacci Merdivenleri
                          </h2>
                          <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                            Zaman Dilimlerine Göre Fibonacci Retracement Seviyeleri
                          </p>
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '20px' }}>
                        {data.fibonacci_ladders.map((ladder, idx) => (
                          <div
                            key={idx}
                            style={{
                              padding: '20px',
                              borderRadius: '12px',
                              background: COLORS.bg.secondary,
                              border: `1px solid ${COLORS.border.default}`
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                              <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.premium }}>
                                {ladder.timeframe.toUpperCase()}
                              </div>
                              <div style={{
                                background: ladder.direction === 'bullish' ? `${COLORS.success}30` : `${COLORS.danger}30`,
                                color: ladder.direction === 'bullish' ? COLORS.success : COLORS.danger,
                                fontSize: '12px',
                                fontWeight: '600',
                                padding: '4px 12px',
                                borderRadius: '6px'
                              }}>
                                {ladder.direction === 'bullish' ? '🐂 BULLISH' : '🐻 BEARISH'}
                              </div>
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                              <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '10px', fontWeight: '600' }}>
                                FİBONACCİ SEVİYELERİ
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                                {Object.entries(ladder.levels).sort((a, b) => {
                                  const orderMap: { [key: string]: number } = {
                                    'fib_0': 0, 'fib_23': 1, 'fib_38': 2, 'fib_50': 3,
                                    'fib_61': 4, 'fib_78': 5, 'fib_100': 6
                                  };
                                  return orderMap[a[0]] - orderMap[b[0]];
                                }).map(([level, price]) => (
                                  <div
                                    key={level}
                                    style={{
                                      padding: '8px 10px',
                                      borderRadius: '6px',
                                      background: COLORS.bg.hover,
                                      border: `1px solid ${COLORS.border.default}`
                                    }}
                                  >
                                    <div style={{ fontSize: '10px', color: COLORS.text.secondary, marginBottom: '2px' }}>
                                      {level.replace('fib_', '')}%
                                    </div>
                                    <div style={{ fontSize: '14px', color: COLORS.text.primary, fontWeight: '600' }}>
                                      {formatPrice(price)}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                              <div style={{
                                padding: '10px',
                                borderRadius: '6px',
                                background: `${COLORS.success}20`,
                                border: `1px solid ${COLORS.success}40`
                              }}>
                                <div style={{ fontSize: '10px', color: COLORS.text.secondary, marginBottom: '4px' }}>Swing High</div>
                                <div style={{ fontSize: '14px', color: COLORS.success, fontWeight: '600' }}>
                                  {formatPrice(ladder.swing_high)}
                                </div>
                              </div>
                              <div style={{
                                padding: '10px',
                                borderRadius: '6px',
                                background: `${COLORS.danger}20`,
                                border: `1px solid ${COLORS.danger}40`
                              }}>
                                <div style={{ fontSize: '10px', color: COLORS.text.secondary, marginBottom: '4px' }}>Swing Low</div>
                                <div style={{ fontSize: '14px', color: COLORS.danger, fontWeight: '600' }}>
                                  {formatPrice(ladder.swing_low)}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="neon-card" style={{ padding: '60px', textAlign: 'center' }}>
                  <Icons.TrendingUp style={{ width: '64px', height: '64px', color: COLORS.text.secondary, margin: '0 auto 20px', opacity: 0.5 }} />
                  <div style={{ fontSize: '16px', color: COLORS.text.secondary }}>
                    Sol taraftan bir coin seçin
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* MANTIK Modal - Strategy Explanation Popup */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
            backdropFilter: 'blur(8px)'
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            className="neon-card"
            style={{
              maxWidth: '900px',
              maxHeight: '85vh',
              overflowY: 'auto',
              padding: '32px',
              background: COLORS.bg.primary,
              border: `2px solid ${COLORS.premium}`,
              boxShadow: `0 0 40px ${COLORS.premium}60`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                <h2 style={{ fontSize: '26px', fontWeight: 'bold', color: COLORS.premium, margin: 0 }}>
                  QUANTUM MERDIVEN MANTIĞI
                </h2>
              </div>
              <button
                onClick={() => setShowLogicModal(false)}
                style={{
                  background: 'transparent',
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '8px',
                  padding: '8px 16px',
                  color: COLORS.text.secondary,
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600'
                }}
              >
                ✕ Kapat
              </button>
            </div>

            {/* Strategy Overview */}
            <div style={{ marginBottom: '28px' }}>
              <div style={{
                padding: '20px',
                borderRadius: '12px',
                background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.info}20)`,
                border: `1px solid ${COLORS.premium}40`,
                marginBottom: '20px'
              }}>
                <p style={{ fontSize: '15px', lineHeight: '1.7', color: COLORS.text.primary, margin: 0 }}>
                  <strong style={{ color: COLORS.premium }}>Quantum Merdiven Stratejisi</strong>, kripto para piyasalarında
                  en güçlü destek ve direnç seviyelerini tespit etmek için 3 ileri seviye teknik analiz yöntemini birleştirir.
                  Bu sistem, profesyonel traderların kullandığı matematiksel ve istatistiksel yaklaşımları otomatikleştirir.
                </p>
              </div>
            </div>

            {/* Component 1: Fibonacci Ladders */}
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <Icons.BarChart3 style={{ width: '24px', height: '24px', color: COLORS.info }} />
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  1️⃣ Fibonacci Merdivenleri
                </h3>
              </div>
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                background: COLORS.bg.secondary,
                border: `1px solid ${COLORS.border.default}`,
                marginBottom: '12px'
              }}>
                <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                  <strong style={{ color: COLORS.text.primary }}>Ne Yapar?</strong><br />
                  Fibonacci retracement seviyeleri, fiyat hareketlerinin doğal geri çekilme noktalarını matematiksel olarak hesaplar.
                  Her zaman diliminde (15m, 1h, 4h) ayrı "merdiven basamakları" oluşturur.
                </p>
                <div style={{
                  padding: '12px',
                  borderRadius: '8px',
                  background: `${COLORS.info}10`,
                  border: `1px solid ${COLORS.info}30`
                }}>
                  <p style={{ fontSize: '13px', lineHeight: '1.6', color: COLORS.text.primary, margin: 0 }}>
                    <strong>Fibonacci Seviyeleri:</strong><br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>0%</code> → En düşük nokta (swing low)<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>23.6%</code> → Hafif düzeltme bölgesi<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>38.2%</code> → Orta seviye geri çekilme<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>50%</code> → Psikolojik yarı nokta<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>61.8%</code> → Altın oran (en kritik seviye)<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>78.6%</code> → Derin düzeltme<br />
                    • <code style={{ background: COLORS.bg.hover, padding: '2px 6px', borderRadius: '4px' }}>100%</code> → En yüksek nokta (swing high)
                  </p>
                </div>
              </div>
            </div>

            {/* Component 2: MA 7-25-99 Bottom Hunter */}
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <Icons.Activity style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  2️⃣ MA 7-25-99 Dip Avcısı (Bottom Hunter)
                </h3>
              </div>
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                background: COLORS.bg.secondary,
                border: `1px solid ${COLORS.border.default}`,
                marginBottom: '12px'
              }}>
                <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                  <strong style={{ color: COLORS.text.primary }}>Ne Yapar?</strong><br />
                  3 hareketli ortalamayı (MA 7, MA 25, MA 99) analiz ederek fiyatın dip noktalarını tespit eder ve
                  0-120 arası güç skoru verir.
                </p>
                <div style={{
                  padding: '12px',
                  borderRadius: '8px',
                  background: `${COLORS.premium}10`,
                  border: `1px solid ${COLORS.premium}30`
                }}>
                  <p style={{ fontSize: '13px', lineHeight: '1.6', color: COLORS.text.primary, margin: 0 }}>
                    <strong>Skor Sistemi (0-120 puan):</strong><br />
                    • <strong style={{ color: COLORS.success }}>90-120 puan:</strong> Çok güçlü dip sinyali (AL bölgesi)<br />
                    • <strong style={{ color: COLORS.warning }}>60-89 puan:</strong> Orta seviye dip (dikkatli AL)<br />
                    • <strong style={{ color: COLORS.danger }}>0-59 puan:</strong> Zayıf sinyal (BEKLE/SAT)<br /><br />
                    <strong>Nasıl Hesaplanır?</strong><br />
                    • Fiyat en düşük MA'ya yakınsa → yüksek puan<br />
                    • MA'lar birbirine yakınsa (crossover yakın) → bonus puan<br />
                    • MA sıralaması doğruysa (MA7 {'<'} MA25 {'<'} MA99) → ek puan<br />
                    • Fiyat MA'ların altındaysa → dip sinyali güçlenir
                  </p>
                </div>
              </div>
            </div>

            {/* Component 3: Confluence Zones */}
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                <Icons.Layers style={{ width: '24px', height: '24px', color: COLORS.warning }} />
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  3️⃣ Confluence (Kesişim) Bölgeleri
                </h3>
              </div>
              <div style={{
                padding: '16px',
                borderRadius: '10px',
                background: COLORS.bg.secondary,
                border: `1px solid ${COLORS.border.default}`,
                marginBottom: '12px'
              }}>
                <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                  <strong style={{ color: COLORS.text.primary }}>Ne Yapar?</strong><br />
                  Farklı zaman dilimlerindeki Fibonacci seviyelerinin kesiştiği noktaları bulur. Birden fazla zaman diliminde
                  aynı fiyat seviyesi çıkarsa, o seviye "güç bölgesi" olur.
                </p>
                <div style={{
                  padding: '12px',
                  borderRadius: '8px',
                  background: `${COLORS.warning}10`,
                  border: `1px solid ${COLORS.warning}30`
                }}>
                  <p style={{ fontSize: '13px', lineHeight: '1.6', color: COLORS.text.primary, margin: 0 }}>
                    <strong>Güç Skoru (Power Score):</strong><br />
                    • <strong style={{ color: COLORS.premium }}>80-100:</strong> Kritik seviye! Çok güçlü destek/direnç<br />
                    • <strong style={{ color: COLORS.warning }}>50-79:</strong> Orta güçte seviye<br />
                    • <strong style={{ color: COLORS.text.secondary }}>0-49:</strong> Zayıf seviye<br /><br />
                    <strong>Örnek:</strong> 15m'de Fib 61.8%, 1h'de Fib 50%, 4h'de Fib 38.2% aynı fiyatta
                    kesişirse → çok güçlü confluence bölgesi oluşur.
                  </p>
                </div>
              </div>
            </div>

            {/* How To Use */}
            <div style={{
              padding: '20px',
              borderRadius: '12px',
              background: `linear-gradient(135deg, ${COLORS.success}20, ${COLORS.info}20)`,
              border: `1px solid ${COLORS.success}40`
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.success, marginBottom: '12px' }}>
                📊 Nasıl Kullanılır?
              </h3>
              <ol style={{ fontSize: '14px', lineHeight: '1.8', color: COLORS.text.primary, paddingLeft: '20px', margin: 0 }}>
                <li><strong>Sol Listeden Coin Seç:</strong> 600+ coin arasından istediğini seç</li>
                <li><strong>MA Skoruna Bak:</strong> 90+ puan varsa güçlü dip bölgesindesin</li>
                <li><strong>Confluence Bölgelerini İncele:</strong> Power score yüksek olanlar en kritik seviyeler</li>
                <li><strong>Fibonacci Merdivenleri Kontrol Et:</strong> 61.8% ve 50% seviyeleri en önemli basamaklar</li>
                <li><strong>Önerilen Giriş Fiyatını Kullan:</strong> Sistem otomatik olarak en iyi giriş fiyatını hesaplar</li>
              </ol>
            </div>

            {/* Warning */}
            <div style={{
              marginTop: '20px',
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.danger}10`,
              border: `1px solid ${COLORS.danger}40`
            }}>
              <p style={{ fontSize: '13px', lineHeight: '1.6', color: COLORS.text.primary, margin: 0 }}>
                <strong style={{ color: COLORS.danger }}>⚠️ Uyarı:</strong> Bu analiz otomatik bir teknik analiz aracıdır.
                Yatırım kararlarınızı vermeden önce mutlaka kendi araştırmanızı yapın. Risk yönetimini asla ihmal etmeyin
                ve kaybetmeyi göze alamayacağınız parayla işlem yapmayın.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
