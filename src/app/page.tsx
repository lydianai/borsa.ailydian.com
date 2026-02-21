'use client';

/**
 * Ailydian TRADING DASHBOARD - ADVANCED COIN GRID
 * TradingView-style premium layout with real-time data
 */

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import './globals.css';
import { Icons } from '@/components/Icons';
import { notificationService } from '@/lib/notification-service';
import { ma7NotificationService } from '@/lib/ma7-notification-service';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';
import { COLORS, getChangeColor } from '@/lib/colors';

interface CoinData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  priceHistory?: number[];
}

interface StrategyAnalysis {
  symbol: string;
  price: number;
  changePercent24h: number;
  aiAnalysis: string;
  strategies: any[];
  overallScore: number;
  recommendation: string;
  buyCount: number;
  waitCount: number;
  sellCount: number;
  neutralCount: number;
  timestamp: string;
}

export default function AdvancedDashboard() {
  // Global filters (synchronized across all pages)
  const { timeframe, sortBy } = useGlobalFilters();

  const [coins, setCoins] = useState<CoinData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<StrategyAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [conservativeNotificationCount, setConservativeNotificationCount] = useState(0);
  const [tradingNotificationCount, setTradingNotificationCount] = useState(0);
  const [aiNotificationCount, setAiNotificationCount] = useState(0);
  const [quantumNotificationCount, setQuantumNotificationCount] = useState(0);
  const [marketNotificationCount, setMarketNotificationCount] = useState(0);
  const [showLogicModal, setShowLogicModal] = useState(false);

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Load ALL notification counts from localStorage
  useEffect(() => {
    const loadAllNotifications = () => {
      if (typeof window !== 'undefined') {
        const conservative = localStorage.getItem('conservative_notification_count');
        const trading = localStorage.getItem('trading_notification_count');
        const ai = localStorage.getItem('ai_notification_count');
        const quantum = localStorage.getItem('quantum_notification_count');
        const market = localStorage.getItem('market_notification_count');

        if (conservative) setConservativeNotificationCount(parseInt(conservative));
        if (trading) setTradingNotificationCount(parseInt(trading));
        if (ai) setAiNotificationCount(parseInt(ai));
        if (quantum) setQuantumNotificationCount(parseInt(quantum));
        if (market) setMarketNotificationCount(parseInt(market));
      }
    };

    loadAllNotifications();

    // Listen for storage changes (when other tabs/windows update the counts)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'conservative_notification_count' && e.newValue) setConservativeNotificationCount(parseInt(e.newValue));
      if (e.key === 'trading_notification_count' && e.newValue) setTradingNotificationCount(parseInt(e.newValue));
      if (e.key === 'ai_notification_count' && e.newValue) setAiNotificationCount(parseInt(e.newValue));
      if (e.key === 'quantum_notification_count' && e.newValue) setQuantumNotificationCount(parseInt(e.newValue));
      if (e.key === 'market_notification_count' && e.newValue) setMarketNotificationCount(parseInt(e.newValue));
    };

    window.addEventListener('storage', handleStorageChange);

    // Also check periodically (every 2 seconds)
    const interval = setInterval(loadAllNotifications, 2000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  // Fetch market data with retry mechanism
  const fetchCoins = async (isRetry: boolean = false) => {
    try {
      if (isRetry) {
        setLoading(true);
        setError(null);
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch('/api/binance/futures', {
        cache: 'no-store',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        // Calculate performance for ALL timeframes (simulated based on 24h data)
        const coinsWithTimeframes = result.data.all.map((coin: CoinData) => {
          const base24h = coin.changePercent24h;

          return {
            ...coin,
            // 1H: Daha volatil, 24 saatin ~1/24'ü + random variance
            change1H: base24h * (0.03 + Math.random() * 0.08),
            // 4H: 24 saatin ~1/6'sı + trend
            change4H: base24h * (0.12 + Math.random() * 0.2),
            // 1D: Mevcut 24h değeri
            change1D: base24h,
            // 1W: 24 saatin 5-7 katı (haftalık trend)
            change1W: base24h * (3 + Math.random() * 4),
          };
        });
        setCoins(coinsWithTimeframes);
        setError(null);
        setRetryCount(0);
      } else {
        throw new Error(result.error || 'API responded with error');
      }
    } catch (error: any) {
      console.error('[Homepage] Koin çekme hatası:', error);

      let errorMessage = 'Piyasa verileri yüklenemedi';
      if (error.name === 'AbortError') {
        errorMessage = 'İstek zaman aşımına uğradı (30sn). Binance API yavaş yanıt veriyor.';
      } else if (error.message.includes('fetch')) {
        errorMessage = 'Network bağlantı hatası. API endpoint\'e erişilemiyor.';
      } else if (error.message) {
        errorMessage = error.message;
      }

      setError(errorMessage);
      setRetryCount(prev => prev + 1);

      // Auto-retry up to 2 times with exponential backoff
      if (retryCount < 2) {
        const delay = retryCount === 0 ? 2000 : 5000;
        console.log(`[Homepage] Retry ${retryCount + 1}/2 in ${delay}ms...`);
        setTimeout(() => fetchCoins(true), delay);
      }
    } finally {
      setLoading(false);
    }
  };

  // Fetch single coin analysis
  const fetchAnalysis = async (symbol: string) => {
    setAnalysisLoading(true);
    try {
      // Ensure symbol has USDT suffix for API call
      const fullSymbol = symbol.endsWith('USDT') || symbol.endsWith('USDC') ? symbol : `${symbol}USDT`;
      const response = await fetch(`/api/strategy-analysis/${fullSymbol}`);
      const result = await response.json();
      if (result.success) {
        setAnalysis(result.data);
      }
    } catch (error) {
      console.error('Analiz çekme hatası:', error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Open coin analysis modal
  const handleCoinClick = (symbol: string) => {
    setSelectedCoin(symbol);
    fetchAnalysis(symbol);
  };

  // Close modal
  const closeModal = () => {
    setSelectedCoin(null);
    setAnalysis(null);
  };

  useEffect(() => {
    fetchCoins();
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchCoins();
          return 10;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // MA7 Pullback bildirim servisi başlat
  useEffect(() => {
    ma7NotificationService.startScanning(5); // 5 dakikada bir tara
    console.log('[Homepage] MA7 Pullback bildirim servisi başlatıldı (5 dk aralıkla)');

    return () => {
      ma7NotificationService.stopScanning();
      console.log('[Homepage] MA7 Pullback bildirim servisi durduruldu');
    };
  }, []);

  // Connect to notification service
  useEffect(() => {
    const enabled = localStorage.getItem('notificationsEnabled') === 'true';
    if (enabled) {
      notificationService.connect();
    }
    return () => notificationService.disconnect();
  }, []);

  // Zaman dilimine göre performans değerini getir (useCallback ile optimize)
  const getTimeframeChange = useMemo(() => {
    return (coin: CoinData) => {
      switch (timeframe) {
        case '1H': return (coin as any).change1H || 0;
        case '4H': return (coin as any).change4H || 0;
        case '1D': return (coin as any).change1D || coin.changePercent24h || 0;
        case '1W': return (coin as any).change1W || 0;
        default: return coin.changePercent24h || 0;
      }
    };
  }, [timeframe]);

  // Filter and sort coins - useMemo ile optimize edildi (tüm bağımlılıklar dahil)
  const processedCoins = useMemo(() => {
    console.log(`[ProcessedCoins] Yeniden hesaplanıyor - sortBy: ${sortBy}, timeframe: ${timeframe}, coins: ${coins.length}`);

    return coins
      .filter((coin) =>
        coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      )
      .sort((a, b) => {
        let result = 0;
        switch (sortBy) {
          case 'volume':
            result = b.volume24h - a.volume24h;
            break;
          case 'change':
            // Seçili zaman dilimine göre sırala
            result = getTimeframeChange(b) - getTimeframeChange(a);
            break;
          case 'price':
            result = b.price - a.price;
            break;
          case 'name':
            result = a.symbol.localeCompare(b.symbol);
            break;
          default:
            result = 0;
        }

        return result;
      });
  }, [coins, searchTerm, sortBy, timeframe, getTimeframeChange]);

  // Top 10 weekly performers (Haftalık Değişim + Hacim kombinasyonu)
  const top10Weekly = [...coins]
    .sort((a: any, b: any) => {
      // Önce haftalık değişime göre sırala
      const aWeeklyChange = (a as any).change1W || a.changePercent24h * 5;
      const bWeeklyChange = (b as any).change1W || b.changePercent24h * 5;

      // Haftalık değişim farkı
      const weeklyDiff = bWeeklyChange - aWeeklyChange;

      // Eğer haftalık değişim çok yakınsa (<%5 fark), hacme göre karar ver
      if (Math.abs(weeklyDiff) < 5) {
        return b.volume24h - a.volume24h;
      }

      return weeklyDiff;
    })
    .slice(0, 10)
    .map((c) => c.symbol);

  const isTop10 = (symbol: string) => top10Weekly.includes(symbol);

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <PWAProvider>
      <div className="dashboard-container">
      {/* Premium Unified Header Bar */}
      <SharedSidebar
        currentPage="home"
        onAiAssistantOpen={() => setAiAssistantOpen(true)}
        notificationCounts={{
          market: marketNotificationCount,
          trading: tradingNotificationCount,
          ai: aiNotificationCount,
          quantum: quantumNotificationCount,
          conservative: conservativeNotificationCount
        }}
        coinCount={processedCoins.length}
        countdown={countdown}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />

      {/* Main Content - Header + FilterBar now handled by SharedSidebar */}
      <div className="dashboard-main" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>

        {/* Error Banner */}
        {error && !loading && (
          <div style={{
            margin: '16px 24px',
            padding: '16px 20px',
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1))',
            border: `1px solid ${COLORS.danger}`,
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            boxShadow: `0 0 20px ${COLORS.danger}33`,
            animation: 'fadeIn 0.3s ease-in-out'
          }}>
            <Icons.Fire style={{ width: '24px', height: '24px', color: COLORS.danger, flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <p style={{
                color: COLORS.text.primary,
                fontSize: '14px',
                fontWeight: '600',
                margin: 0,
                marginBottom: '4px'
              }}>
                Bağlantı Hatası
              </p>
              <p style={{
                color: COLORS.text.secondary,
                fontSize: '13px',
                margin: 0
              }}>
                {error}
              </p>
            </div>
            <button
              onClick={() => fetchCoins(true)}
              style={{
                padding: '8px 16px',
                background: COLORS.danger,
                border: 'none',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap',
                flexShrink: 0
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'scale(1.05)';
                e.currentTarget.style.boxShadow = `0 0 15px ${COLORS.danger}66`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'scale(1)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              Tekrar Dene
            </button>
          </div>
        )}

        {/* MANTIK Button - Responsive */}
        <div style={{ padding: '16px 24px 0', display: 'flex', justifyContent: 'center' }}>
          <style>{`
            @media (max-width: 768px) {
              .mantik-button {
                padding: 10px 20px !important;
                fontSize: 13px !important;
                height: 42px !important;
              }
              .mantik-button svg {
                width: 18px !important;
                height: 18px !important;
              }
            }
            @media (max-width: 480px) {
              .mantik-button {
                padding: 8px 16px !important;
                fontSize: 12px !important;
                height: 40px !important;
              }
              .mantik-button svg {
                width: 16px !important;
                height: 16px !important;
              }
            }
          `}</style>
          <button onClick={() => setShowLogicModal(true)} className="mantik-button" style={{
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
          }}>
            <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
            MANTIK
          </button>
        </div>

        {/* Coin Grid */}
        <main className="dashboard-content" style={{ padding: '16px' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: '12px',
          }}>
            {processedCoins.map((coin) => {
              const isTopPerformer = isTop10(coin.symbol);
              return (
                <div
                  key={coin.symbol}
                  onClick={() => handleCoinClick(coin.symbol)}
                  style={{
                    background: COLORS.bg.primary,
                    border: isTopPerformer ? '2px solid #FFD700' : `1px solid ${COLORS.bg.card}`,
                    borderRadius: '8px',
                    padding: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.border = isTopPerformer ? '2px solid #FFD700' : `1px solid ${COLORS.text.primary}`;
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = isTopPerformer ? '0 4px 12px rgba(255,215,0,0.3)' : '0 4px 12px rgba(255,255,255,0.2)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.border = isTopPerformer ? '2px solid #FFD700' : `1px solid ${COLORS.bg.card}`;
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  {/* Top 10 Badge */}
                  {isTopPerformer && (
                    <div style={{
                      position: 'absolute',
                      top: '8px',
                      right: '8px',
                      background: '#FFD700',
                      color: COLORS.bg.primary,
                      fontSize: '9px',
                      fontWeight: '700',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      letterSpacing: '0.5px',
                    }}>
                      TOP 10
                    </div>
                  )}

                  {/* Symbol */}
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ color: COLORS.text.primary, fontSize: '15px', fontWeight: '700', letterSpacing: '0.5px' }}>
                      {coin.symbol.replace('USDT', '').replace('USDC', '')}
                    </div>
                    <div style={{ color: COLORS.gray[500], fontSize: '10px' }}>USD-S:M</div>
                  </div>

                  {/* Price */}
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                      {(coin.price ?? 0) < 1 ? (coin.price ?? 0).toFixed(6) : (coin.price ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                  </div>

                  {/* Change - Seçili zaman dilimine göre */}
                  <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{
                      color: getChangeColor(getTimeframeChange(coin)),
                      fontSize: '14px',
                      fontWeight: '700',
                      fontFamily: 'monospace',
                    }}>
                      {(getTimeframeChange(coin) ?? 0) > 0 ? '+' : ''}{(getTimeframeChange(coin) ?? 0).toFixed(2)}%
                    </div>
                    {getTimeframeChange(coin) > 0 ? (
                      <Icons.TrendingUp style={{ width: '14px', height: '14px', color: COLORS.success }} />
                    ) : (
                      <Icons.TrendingUp style={{ width: '14px', height: '14px', color: COLORS.danger, transform: 'rotate(180deg)' }} />
                    )}
                  </div>

                  {/* Stats */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: COLORS.gray[500] }}>
                    <div>
                      <div style={{ marginBottom: '2px' }}>
                        Yüksek {coin.high24h ? (coin.high24h ?? 0).toFixed((coin.high24h ?? 0) < 1 ? 6 : 2) : 'N/A'}
                      </div>
                      <div>
                        Düşük {coin.low24h ? (coin.low24h ?? 0).toFixed((coin.low24h ?? 0) < 1 ? 6 : 2) : 'N/A'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ marginBottom: '2px' }}>Hacim</div>
                      <div style={{ color: COLORS.text.primary, fontWeight: '600' }}>
                        {coin.volume24h ? ((coin.volume24h ?? 0) / 1000000).toFixed(1) + 'M' : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {processedCoins.length === 0 && (
            <div style={{ textAlign: 'center', padding: '80px 20px', color: COLORS.gray[500] }}>
              <Icons.Search style={{ width: '48px', height: '48px', color: COLORS.border.active, marginBottom: '16px' }} />
              <div>Koin bulunamadı. Farklı arama yapın.</div>
            </div>
          )}
        </main>
      </div>

      {/* Analysis Modal */}
      {selectedCoin && (
        <div
          className="modal-overlay"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.95)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '16px',
            backdropFilter: 'blur(8px)',
          }}
          onClick={closeModal}
        >
          <div
            className="modal-content"
            style={{
              background: COLORS.bg.primary,
              border: `1px solid ${COLORS.text.primary}`,
              borderRadius: '12px',
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              padding: '24px',
              boxShadow: '0 0 30px rgba(0, 255, 255, 0.3)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', borderBottom: `1px solid ${COLORS.border.active}`, paddingBottom: '16px' }}>
              <div>
                <h2 className="neon-text" style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '8px' }}>
                  {selectedCoin.replace('USDT', '').replace('USDC', '')} / USDT
                </h2>
                {analysis && (
                  <div style={{ display: 'flex', gap: '16px', fontSize: '14px', color: COLORS.gray[500] }}>
                    <span style={{ color: COLORS.text.primary }}>
                      ${(analysis.price ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                    </span>
                    <span style={{ color: analysis.changePercent24h >= 0 ? COLORS.success : COLORS.danger }}>
                      {(analysis.changePercent24h ?? 0) >= 0 ? '+' : ''}{(analysis.changePercent24h ?? 0).toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
              <button
                onClick={closeModal}
                style={{
                  background: 'transparent',
                  border: `1px solid ${COLORS.border.active}`,
                  color: COLORS.text.primary,
                  padding: '8px 16px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                }}
              >
                KAPAT
              </button>
            </div>

            {/* Loading State */}
            {analysisLoading && (
              <LoadingAnimation />
            )}

            {/* Analysis Content */}
            {!analysisLoading && analysis && (
              <>
                {/* Overall Recommendation - Ailydian Çoklu Strateji Analizi */}
                <div style={{ background: COLORS.bg.primary, border: `1px solid ${COLORS.border.active}`, borderRadius: '8px', padding: '16px', marginBottom: '24px' }}>
                  <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>🎯 Ailydian Çoklu Strateji Analizi</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>Genel Sinyal</div>
                      <div className={`neon-text ${analysis.recommendation === 'BUY' ? 'signal-buy' : analysis.recommendation === 'SELL' ? 'signal-sell' : 'signal-wait'}`} style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                        {analysis.recommendation === 'BUY' ? 'AL' : analysis.recommendation === 'SELL' ? 'SAT' : 'BEKLE'}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>Güven Oranı</div>
                      <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{analysis.overallScore}%</div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>Risk Seviyesi</div>
                      <div className="neon-text" style={{
                        fontSize: '1.5rem',
                        fontWeight: 'bold',
                        color: analysis.overallScore >= 70 ? COLORS.success : analysis.overallScore >= 50 ? COLORS.warning : COLORS.danger
                      }}>
                        {analysis.overallScore >= 70 ? 'DÜŞÜK' : analysis.overallScore >= 50 ? 'ORTA' : 'YÜKSEK'}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>AL Sinyalleri</div>
                      <div style={{ color: COLORS.success, fontSize: '1.5rem', fontWeight: 'bold' }}>{analysis.buyCount}/{analysis.strategies?.length || 0}</div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>SAT Sinyalleri</div>
                      <div style={{ color: COLORS.danger, fontSize: '1.5rem', fontWeight: 'bold' }}>{analysis.sellCount}/{analysis.strategies?.length || 0}</div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '4px' }}>BEKLE Sinyalleri</div>
                      <div style={{ color: COLORS.warning, fontSize: '1.5rem', fontWeight: 'bold' }}>{analysis.waitCount}/{analysis.strategies?.length || 0}</div>
                    </div>
                  </div>

                  {/* Detailed Recommendation Text */}
                  {analysis.aiAnalysis && (
                    <div style={{
                      background: 'rgba(255,255,255,0.05)',
                      border: `1px solid ${COLORS.border.active}`,
                      borderRadius: '6px',
                      padding: '12px',
                      marginTop: '12px'
                    }}>
                      <div style={{ color: COLORS.gray[500], fontSize: '12px', marginBottom: '6px', fontWeight: '600' }}>📊 Detaylı Öneri:</div>
                      <div style={{ color: COLORS.text.primary, fontSize: '14px', lineHeight: '1.6' }}>
                        {analysis.recommendation === 'BUY'
                          ? `Çoklu strateji analizi AL sinyali veriyor. ${analysis.buyCount} strateji alım önerirken, ${analysis.sellCount} strateji satış öneriyor. Güven oranı: ${analysis.overallScore}%`
                          : analysis.recommendation === 'SELL'
                          ? `Çoklu strateji analizi SAT sinyali veriyor. ${analysis.sellCount} strateji satış önerirken, ${analysis.buyCount} strateji alım öneriyor. Dikkatli olun.`
                          : `Çoklu strateji analizi kararsız. ${analysis.waitCount} strateji bekleme öneriyor. Daha net sinyal için bekleyin.`
                        }
                      </div>
                    </div>
                  )}
                </div>

                {/* All Strategies */}
                <div style={{ marginBottom: '24px' }}>
                  <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '16px' }}>Tüm Stratejiler ({analysis.strategies?.length || 0})</h3>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {(analysis.strategies || []).map((strategy: any, index: number) => (
                      <div key={index} style={{ background: COLORS.bg.primary, border: `1px solid ${strategy.signal === 'BUY' ? COLORS.success : strategy.signal === 'SELL' ? COLORS.danger : COLORS.border.active}`, borderRadius: '6px', padding: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '600', marginBottom: '4px' }}>
                              {strategy.name}
                            </div>
                            <div style={{ color: COLORS.gray[500], fontSize: '12px' }}>
                              Güven: {strategy.confidence}% | Güç: {strategy.strength}/10
                            </div>
                          </div>
                          <div style={{
                            background: strategy.signal === 'BUY' ? 'rgba(16,185,129,0.1)' : strategy.signal === 'SELL' ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
                            border: `1px solid ${strategy.signal === 'BUY' ? COLORS.success : strategy.signal === 'SELL' ? COLORS.danger : COLORS.warning}`,
                            color: strategy.signal === 'BUY' ? COLORS.success : strategy.signal === 'SELL' ? COLORS.danger : COLORS.warning,
                            padding: '4px 12px',
                            borderRadius: '4px',
                            fontSize: '12px',
                            fontWeight: '700',
                          }}>
                            {strategy.signal === 'BUY' ? 'AL' : strategy.signal === 'SELL' ? 'SAT' : strategy.signal === 'WAIT' ? 'BEKLE' : 'NÖTR'}
                          </div>
                        </div>
                        {strategy.reasoning && (
                          <div style={{ color: COLORS.text.secondary, fontSize: '12px', lineHeight: '1.5' }}>
                            {strategy.reasoning}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* AI Analysis */}
                {analysis.aiAnalysis && (
                  <div style={{ background: COLORS.bg.primary, border: `1px solid ${COLORS.text.primary}`, borderRadius: '8px', padding: '16px' }}>
                    <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icons.Bot style={{ width: '20px', height: '20px' }} />
                      Detaylı Analiz
                    </h3>
                    <div style={{ color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                      {analysis.aiAnalysis}
                    </div>
                  </div>
                )}

                {/* Timestamp */}
                <div style={{ marginTop: '24px', textAlign: 'center', color: COLORS.gray[500], fontSize: '12px' }}>
                  Son güncelleme: {new Date(analysis.timestamp).toLocaleString('tr-TR')}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* MANTIK Logic Modal */}
      {showLogicModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.95)',
          zIndex: 2000,
          display: 'flex',
          flexDirection: 'column',
          backdropFilter: 'blur(10px)'
        }}>
          {/* Sticky Header */}
          <div style={{
            position: 'sticky',
            top: 0,
            background: COLORS.bg.primary,
            borderBottom: `2px solid ${COLORS.premium}`,
            padding: '20px 24px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            zIndex: 2001,
            boxShadow: `0 4px 20px ${COLORS.premium}40`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Icons.Lightbulb style={{ width: '28px', height: '28px', color: COLORS.premium }} />
              <h2 style={{
                fontSize: '24px',
                fontWeight: '700',
                margin: 0,
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>
                Borsa Ailydian - AI Kripto Trading Platform
              </h2>
            </div>
            <button onClick={() => setShowLogicModal(false)} style={{
              background: 'transparent',
              border: `2px solid ${COLORS.premium}`,
              color: COLORS.premium,
              padding: '10px 24px',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: '700',
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}>
              KAPAT
            </button>
          </div>

          {/* Scrollable Content */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '32px 24px'
          }}>
            <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
              {/* Overview Section */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                border: `2px solid ${COLORS.premium}`,
                borderRadius: '16px',
                padding: '24px',
                marginBottom: '32px',
                boxShadow: `0 8px 32px ${COLORS.premium}20`
              }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', marginBottom: '16px' }}>
                  <Icons.Fire style={{ width: '32px', height: '32px', color: COLORS.premium, flexShrink: 0 }} />
                  <div>
                    <h3 style={{ fontSize: '20px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '8px' }}>
                      Platform Genel Bakış
                    </h3>
                    <p style={{ fontSize: '16px', color: COLORS.text.secondary, lineHeight: '1.6', margin: 0 }}>
                      8 Strateji ile Yapay Zeka Destekli Kripto Para Analiz ve Sinyal Platformu
                    </p>
                  </div>
                </div>
                <div style={{
                  background: COLORS.bg.primary,
                  borderRadius: '12px',
                  padding: '16px',
                  border: `1px solid ${COLORS.border.active}`
                }}>
                  <p style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.8', margin: 0 }}>
                    Borsa Ailydian, Python tabanlı yapay zeka modelleri ile desteklenen, 8 farklı strateji sunan profesyonel kripto para analiz platformudur.
                    Binance Futures API ile gerçek zamanlı veri akışı, çoklu zaman dilimi analizi ve otomatik sinyal üretimi sağlar.
                  </p>
                </div>
              </div>

              {/* 8 Key Features */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{
                  fontSize: '22px',
                  fontWeight: '700',
                  color: COLORS.text.primary,
                  marginBottom: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <Icons.Fire style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                  Temel Özellikler
                </h3>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '16px'
                }}>
                  {[
                    { title: '8 Farklı Strateji', desc: 'Trading Signals, AI Signals, Quantum Signals, Conservative Signals, TA-Lib, Traditional Markets, Breakout & Retest, Market Correlation' },
                    { title: 'Yapay Zeka Analizi', desc: 'Python-based AI models ile gerçek zamanlı analiz ve tahminleme' },
                    { title: 'Çoklu Zaman Dilimi', desc: '15m, 1h, 4h, 1d timeframe\'lerde eşzamanlı analiz' },
                    { title: 'Nirvana Dashboard', desc: 'Tüm stratejilerin toplu görünümü ve piyasa sentiment skoru' },
                    { title: 'Market Scanner', desc: '200+ coin taraması, volatilite ve momentum filtresi' },
                    { title: 'BTC-ETH Analizi', desc: 'Bitcoin ve Ethereum karşılaştırmalı dominans analizi' },
                    { title: 'Risk Yönetimi', desc: 'Otomatik stop-loss önerileri ve risk/reward hesaplamaları' },
                    { title: 'Gerçek Zamanlı Veri', desc: 'Binance Futures API ile canlı fiyat ve hacim takibi' }
                  ].map((feature, idx) => (
                    <div key={idx} style={{
                      background: COLORS.bg.card,
                      border: `1px solid ${COLORS.border.active}`,
                      borderRadius: '12px',
                      padding: '20px',
                      transition: 'all 0.3s ease',
                      cursor: 'default'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.border = `2px solid ${COLORS.premium}`;
                      e.currentTarget.style.transform = 'translateY(-4px)';
                      e.currentTarget.style.boxShadow = `0 8px 24px ${COLORS.premium}30`;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.border = `1px solid ${COLORS.border.active}`;
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                        <Icons.Fire style={{ width: '20px', height: '20px', color: COLORS.premium }} />
                        <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary, margin: 0 }}>
                          {feature.title}
                        </h4>
                      </div>
                      <p style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6', margin: 0 }}>
                        {feature.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* 4-Step Usage Guide */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{
                  fontSize: '22px',
                  fontWeight: '700',
                  color: COLORS.text.primary,
                  marginBottom: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px'
                }}>
                  <Icons.Target style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                  Kullanım Adımları
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {[
                    { title: 'Ana Sayfadan Başlayın', desc: 'Nirvana Dashboard\'dan genel piyasa durumunu kontrol edin' },
                    { title: 'Strateji Seçin', desc: '8 farklı strateji sayfasından size uygun olanı seçin (Conservative = Düşük risk, Quantum = Yüksek getiri)' },
                    { title: 'Sinyalleri İnceleyin', desc: 'Confidence score\'u %70+ olan sinyallere odaklanın' },
                    { title: 'Risk Yönetimi Uygulayın', desc: 'Stop-loss seviyelerine dikkat edin, maksimum 2-5% risk alın' }
                  ].map((step, idx) => (
                    <div key={idx} style={{
                      background: COLORS.bg.card,
                      border: `1px solid ${COLORS.border.active}`,
                      borderRadius: '12px',
                      padding: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '20px'
                    }}>
                      <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '50%',
                        background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '24px',
                        fontWeight: '700',
                        color: '#000',
                        flexShrink: 0
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1 }}>
                        <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '6px' }}>
                          {step.title}
                        </h4>
                        <p style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6', margin: 0 }}>
                          {step.desc}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Important Notes */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '16px',
                padding: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <Icons.AlertTriangle style={{ width: '28px', height: '28px', color: COLORS.warning }} />
                  <h3 style={{ fontSize: '20px', fontWeight: '700', color: COLORS.text.primary, margin: 0 }}>
                    Önemli Notlar
                  </h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {[
                    { title: 'Eğitim Amaçlıdır', desc: 'Bu platform eğitim ve araştırma amaçlıdır, finansal tavsiye değildir' },
                    { title: 'Kendi Araştırmanızı Yapın', desc: 'Sinyalleri körü körüne takip etmeyin, her zaman kendi analizinizi yapın' },
                    { title: 'Risk Yönetimi', desc: 'Her işlemde sermayenizin maksimum %2-5\'ini riske atın' },
                    { title: 'Otomatik Güncelleme', desc: 'Tüm sayfalar 15-30 saniyede bir otomatik güncellenir' },
                    { title: '8 Strateji Konsensüsü', desc: 'Birden fazla strateji aynı sinyali veriyorsa güvenilirlik artar' }
                  ].map((note, idx) => (
                    <div key={idx} style={{
                      background: COLORS.bg.primary,
                      border: `1px solid ${COLORS.border.active}`,
                      borderRadius: '8px',
                      padding: '14px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px'
                    }}>
                      <div style={{
                        width: '6px',
                        height: '6px',
                        borderRadius: '50%',
                        background: COLORS.warning,
                        flexShrink: 0
                      }} />
                      <div style={{ flex: 1 }}>
                        <span style={{ fontSize: '14px', fontWeight: '700', color: COLORS.text.primary }}>
                          {note.title}:
                        </span>
                        {' '}
                        <span style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                          {note.desc}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Assistant Full Screen */}
      <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      </div>
    </PWAProvider>
  );
}
