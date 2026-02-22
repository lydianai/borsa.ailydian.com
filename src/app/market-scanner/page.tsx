'use client';

/**
 * 🎯 LyTrade MARKET SCANNER - Ultra Premium Edition
 * Real-time market scanner + Heatmap + Multi-Timeframe + Advanced Analytics
 * Unique Features:
 * - Live Heatmap Visualization
 * - 4 Timeframe Comparison (1H/4H/1D/1W)
 * - Momentum Scoring
 * - Volume Profile Analysis
 * - AI-Powered Opportunity Detection
 * - Real-time Notifications
 */

import { useState, useEffect, useMemo } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { COLORS, getChangeColor } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface MarketCoin {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  change1H: number;
  change4H: number;
  change1D: number;
  change1W: number;
}

interface StrategyAnalysis {
  symbol: string;
  price: number;
  changePercent24h: number;
  groqAnalysis: string;
  strategies: any[];
  overallScore: number;
  recommendation: string;
  buyCount: number;
  waitCount: number;
  sellCount: number;
  neutralCount: number;
  timestamp: string;
}

type ViewMode = 'table' | 'grid' | 'heatmap';
type MarketCapFilter = 'all' | 'large' | 'mid' | 'small';
type PerformanceFilter = 'all' | 'gainers' | 'losers' | 'volatile';

export default function MarketScannerPage() {
  const [coins, setCoins] = useState<MarketCoin[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<StrategyAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [countdown, setCountdown] = useState(30);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [conservativeNotificationCount, setConservativeNotificationCount] = useState(0);
  const [notificationCount, setNotificationCount] = useState(0);
  const [previousHighPerformers, setPreviousHighPerformers] = useState<string[]>([]);
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [marketCapFilter, setMarketCapFilter] = useState<MarketCapFilter>('all');
  const [performanceFilter, setPerformanceFilter] = useState<PerformanceFilter>('all');
  const [minVolume, setMinVolume] = useState(0);

  const { timeframe, sortBy } = useGlobalFilters();

  // Request notification permission on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission();
      }
    }
    const savedCount = localStorage.getItem('market_notification_count');
    if (savedCount) setNotificationCount(parseInt(savedCount));
  }, []);

  // Auto-clear notification count
  useEffect(() => {
    const timer = setTimeout(() => {
      localStorage.setItem('market_notification_count', '0');
      setNotificationCount(0);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  // Load conservative notification count
  useEffect(() => {
    const loadNotificationCount = () => {
      if (typeof window !== 'undefined') {
        const savedCount = localStorage.getItem('conservative_notification_count');
        if (savedCount) {
          setConservativeNotificationCount(parseInt(savedCount));
        }
      }
    };

    loadNotificationCount();
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'conservative_notification_count' && e.newValue) {
        setConservativeNotificationCount(parseInt(e.newValue));
      }
    };
    window.addEventListener('storage', handleStorageChange);
    const interval = setInterval(loadNotificationCount, 2000);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  // Fetch all coins with real multi-timeframe data
  const fetchCoins = async () => {
    try {
      const response = await fetch('/api/binance/multi-timeframe');
      const result = await response.json();

      if (result.success) {
        const coinsWithTimeframes = result.data.map((coin: any) => ({
          symbol: coin.symbol,
          price: coin.price,
          change24h: (coin.change1D / 100) * coin.price,
          changePercent24h: coin.change1D,
          volume24h: coin.volume24h,
          change1H: coin.change1H,
          change4H: coin.change4H,
          change1D: coin.change1D,
          change1W: coin.change1W,
        }));

        // Detect high performers
        const highPerformers = coinsWithTimeframes
          .filter((c: MarketCoin) => c.changePercent24h > 10)
          .map((c: MarketCoin) => c.symbol);

        if (previousHighPerformers.length > 0) {
          const newHighPerformers = highPerformers.filter(
            (symbol: string) => !previousHighPerformers.includes(symbol)
          );

          if (newHighPerformers.length > 0) {
            const currentCount = parseInt(localStorage.getItem('market_notification_count') || '0');
            const updatedCount = currentCount + newHighPerformers.length;
            localStorage.setItem('market_notification_count', updatedCount.toString());
            setNotificationCount(updatedCount);

            if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
              const topCoin = coinsWithTimeframes.find((c: MarketCoin) => c.symbol === newHighPerformers[0]);
              new Notification('High Performance Detected', {
                body: `${newHighPerformers.length} coins showing 10%+ gains. E.g: ${topCoin?.symbol} +${(topCoin?.changePercent24h ?? 0).toFixed(2)}%`,
                icon: '/icons/icon-192x192.png',
              });
            }
          }
        }

        setPreviousHighPerformers(highPerformers);
        setCoins(coinsWithTimeframes);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Coin fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch single coin analysis
  const fetchAnalysis = async (symbol: string) => {
    setAnalysisLoading(true);
    try {
      const fullSymbol = symbol.endsWith('USDT') || symbol.endsWith('USDC') ? symbol : `${symbol}USDT`;
      const response = await fetch(`/api/strategy-analysis/${fullSymbol}`);
      const result = await response.json();
      if (result.success) {
        setAnalysis(result.data);
      }
    } catch (error) {
      console.error('Analysis fetch error:', error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // 30-second auto-refresh
  useEffect(() => {
    fetchCoins();

    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchCoins();
          return 30;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Modal auto-refresh
  useEffect(() => {
    if (!selectedCoin) return;
    fetchAnalysis(selectedCoin);
    const interval = setInterval(() => fetchAnalysis(selectedCoin), 10000);
    return () => clearInterval(interval);
  }, [selectedCoin]);

  // Get timeframe change
  const getTimeframeChange = (coin: MarketCoin) => {
    switch (timeframe) {
      case '1H': return coin.change1H || 0;
      case '4H': return coin.change4H || 0;
      case '1D': return coin.change1D || coin.changePercent24h || 0;
      case '1W': return coin.change1W || 0;
      default: return coin.changePercent24h || 0;
    }
  };

  // Calculate momentum score (0-100)
  const calculateMomentumScore = (coin: MarketCoin): number => {
    const h1 = coin.change1H || 0;
    const h4 = coin.change4H || 0;
    const d1 = coin.change1D || 0;
    const w1 = coin.change1W || 0;

    // Weighted average with recency bias
    const momentum = (h1 * 0.4) + (h4 * 0.3) + (d1 * 0.2) + (w1 * 0.1);

    // Normalize to 0-100 scale
    return Math.min(100, Math.max(0, 50 + momentum * 2));
  };

  // Filter and sort coins
  const processedCoins = useMemo(() => {
    return coins
      .filter((coin) => {
        // Search filter
        if (searchTerm && !coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())) {
          return false;
        }

        // Volume filter
        if (minVolume > 0 && coin.volume24h < minVolume * 1_000_000) {
          return false;
        }

        // Market cap filter (based on volume as proxy)
        if (marketCapFilter !== 'all') {
          const volumeM = coin.volume24h / 1_000_000;
          if (marketCapFilter === 'large' && volumeM < 100) return false;
          if (marketCapFilter === 'mid' && (volumeM < 10 || volumeM > 100)) return false;
          if (marketCapFilter === 'small' && volumeM > 10) return false;
        }

        // Performance filter
        if (performanceFilter !== 'all') {
          const change = getTimeframeChange(coin);
          if (performanceFilter === 'gainers' && change < 5) return false;
          if (performanceFilter === 'losers' && change > -5) return false;
          if (performanceFilter === 'volatile' && Math.abs(change) < 3) return false;
        }

        return true;
      })
      .sort((a, b) => {
        switch (sortBy) {
          case 'volume':
            return b.volume24h - a.volume24h;
          case 'change':
            return getTimeframeChange(b) - getTimeframeChange(a);
          case 'price':
            return b.price - a.price;
          case 'name':
            return a.symbol.localeCompare(b.symbol);
          default:
            return 0;
        }
      });
  }, [coins, searchTerm, minVolume, marketCapFilter, performanceFilter, timeframe, sortBy]);

  // Top performers
  const topGainers = [...coins].sort((a, b) => getTimeframeChange(b) - getTimeframeChange(a)).slice(0, 10);
  const _topLosers = [...coins].sort((a, b) => getTimeframeChange(a) - getTimeframeChange(b)).slice(0, 10);
  const highVolume = [...coins].sort((a, b) => b.volume24h - a.volume24h).slice(0, 10);

  // Market stats
  const avgChange = coins.length > 0 ? coins.reduce((sum, c) => sum + getTimeframeChange(c), 0) / coins.length : 0;
  const gainersCount = coins.filter(c => getTimeframeChange(c) > 0).length;
  const losersCount = coins.filter(c => getTimeframeChange(c) < 0).length;
  const totalVolume = coins.reduce((sum, c) => sum + c.volume24h, 0);

  const getSignalColor = (change: number) => {
    if (change > 5) return COLORS.success;
    if (change > 0) return COLORS.info;
    if (change > -5) return COLORS.warning;
    return COLORS.danger;
  };

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.primary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: COLORS.bg.primary }}>
      {/* Sidebar */}
      <SharedSidebar
        currentPage="market-scanner"
        notificationCounts={{
          market: notificationCount,
          conservative: conservativeNotificationCount
        }}
      />

      {/* Main Content */}
      <div style={{ flex: 1, marginLeft: '280px', padding: '32px 48px', paddingTop: isLocalhost ? '116px' : '60px', overflowY: 'auto', maxWidth: '1920px', margin: '0 auto', width: '100%' }}>

        {/* Page Header */}
        <div style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '24px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                padding: '12px',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: `0 8px 24px ${COLORS.premium}40`
              }}>
                <Icons.Target style={{ width: '32px', height: '32px', color: '#000' }} />
              </div>
              <div>
                <h1 style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0, background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  Piyasa Tarayıcı Ultra
                </h1>
                <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: '4px 0 0 0' }}>
                  {processedCoins.length} coin taranıyor • {countdown}s içinde yenileniyor
                </p>
              </div>
            </div>
          </div>

          <div>
            <style>{`
              @media (max-width: 768px) {
                .mantik-button-scanner {
                  padding: 10px 20px !important;
                  fontSize: 13px !important;
                  height: 42px !important;
                }
                .mantik-button-scanner svg {
                  width: 18px !important;
                  height: 18px !important;
                }
              }
              @media (max-width: 480px) {
                .mantik-button-scanner {
                  padding: 8px 16px !important;
                  fontSize: 12px !important;
                  height: 40px !important;
                }
                .mantik-button-scanner svg {
                  width: 16px !important;
                  height: 16px !important;
                }
              }
            `}</style>
            <button
              onClick={() => setShowLogicModal(true)}
              className="mantik-button-scanner"
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

        {/* Market Stats Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px', marginBottom: '32px' }}>
          {[
            { label: 'Toplam Hacim', value: `$${(totalVolume / 1_000_000_000).toFixed(2)}B`, icon: Icons.TrendingUp, color: COLORS.premium, change: null },
            { label: 'Ortalama Değişim', value: `${avgChange >= 0 ? '+' : ''}${avgChange.toFixed(2)}%`, icon: Icons.Activity, color: avgChange >= 0 ? COLORS.success : COLORS.danger, change: null },
            { label: 'Yükselenler', value: gainersCount.toString(), icon: Icons.ArrowUp, color: COLORS.success, change: `${((gainersCount / coins.length) * 100).toFixed(0)}%` },
            { label: 'Düşenler', value: losersCount.toString(), icon: Icons.ArrowDown, color: COLORS.danger, change: `${((losersCount / coins.length) * 100).toFixed(0)}%` },
            { label: 'En Yüksek Hacim', value: highVolume[0]?.symbol.replace('USDT', '') || '-', icon: Icons.BarChart, color: COLORS.info, change: `$${(highVolume[0]?.volume24h / 1_000_000).toFixed(1)}M` },
            { label: 'En Çok Kazanan', value: topGainers[0]?.symbol.replace('USDT', '') || '-', icon: Icons.Zap, color: COLORS.warning, change: `+${topGainers[0] ? getTimeframeChange(topGainers[0]).toFixed(2) : '0'}%` },
          ].map((stat, index) => {
            const IconComponent = stat.icon;
            return (
              <div key={index} style={{
                background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '16px',
                padding: '20px',
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.3s ease',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = stat.color;
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = `0 12px 24px ${stat.color}30`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = COLORS.border.default;
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}>
                <div style={{
                  position: 'absolute',
                  top: '-20px',
                  right: '-20px',
                  width: '100px',
                  height: '100px',
                  background: `radial-gradient(circle, ${stat.color}15, transparent)`,
                  borderRadius: '50%'
                }} />
                <div style={{ position: 'relative', zIndex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                    <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0, fontWeight: '500' }}>{stat.label}</p>
                    <IconComponent style={{ width: '20px', height: '20px', color: stat.color, opacity: 0.8 }} />
                  </div>
                  <p style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 4px 0' }}>{stat.value}</p>
                  {stat.change && (
                    <p style={{ fontSize: '12px', color: COLORS.text.secondary, margin: 0 }}>{stat.change}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Filter Panel */}
        <div style={{
          background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
          border: `1px solid ${COLORS.border.default}`,
          borderRadius: '16px',
          padding: '24px',
          marginBottom: '32px'
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', alignItems: 'end' }}>

            {/* Search */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Coin Ara
              </label>
              <div style={{ position: 'relative' }}>
                <Icons.Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', width: '18px', height: '18px', color: COLORS.text.secondary }} />
                <input
                  type="text"
                  placeholder="BTC, ETH, SOL..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px 12px 12px 40px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '10px',
                    color: COLORS.text.primary,
                    fontSize: '14px',
                    outline: 'none',
                    transition: 'all 0.2s ease'
                  }}
                  onFocus={(e) => e.currentTarget.style.borderColor = COLORS.premium}
                  onBlur={(e) => e.currentTarget.style.borderColor = COLORS.border.default}
                />
              </div>
            </div>

            {/* View Mode */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Görünüm
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[
                  { mode: 'grid' as ViewMode, icon: Icons.Dashboard, label: 'Izgara' },
                  { mode: 'table' as ViewMode, icon: Icons.Menu, label: 'Tablo' },
                  { mode: 'heatmap' as ViewMode, icon: Icons.BarChart3, label: 'Isı Haritası' }
                ].map((view) => {
                  const IconComponent = view.icon;
                  return (
                    <button
                      key={view.mode}
                      onClick={() => setViewMode(view.mode)}
                      style={{
                        flex: 1,
                        padding: '10px',
                        background: viewMode === view.mode ? `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})` : COLORS.bg.primary,
                        border: `1px solid ${viewMode === view.mode ? COLORS.premium : COLORS.border.default}`,
                        borderRadius: '10px',
                        color: viewMode === view.mode ? '#000' : COLORS.text.secondary,
                        fontSize: '13px',
                        fontWeight: viewMode === view.mode ? '600' : '500',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '6px',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      <IconComponent style={{ width: '16px', height: '16px' }} />
                      {view.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Market Cap Filter */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Piyasa Değeri
              </label>
              <select
                value={marketCapFilter}
                onChange={(e) => setMarketCapFilter(e.target.value as MarketCapFilter)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.bg.primary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '10px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="all">Tümü</option>
                <option value="large">Büyük (&gt;100M)</option>
                <option value="mid">Orta (10M-100M)</option>
                <option value="small">Küçük (&lt;10M)</option>
              </select>
            </div>

            {/* Performance Filter */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Performans
              </label>
              <select
                value={performanceFilter}
                onChange={(e) => setPerformanceFilter(e.target.value as PerformanceFilter)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.bg.primary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '10px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="all">Tümü</option>
                <option value="gainers">Kazananlar (+5%)</option>
                <option value="losers">Kaybedenler (-5%)</option>
                <option value="volatile">Değişkenler (±3%)</option>
              </select>
            </div>

            {/* Min Volume */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Min Hacim: ${minVolume}M
              </label>
              <input
                type="range"
                min="0"
                max="100"
                step="10"
                value={minVolume}
                onChange={(e) => setMinVolume(Number(e.target.value))}
                style={{
                  width: '100%',
                  height: '8px',
                  borderRadius: '4px',
                  background: `linear-gradient(to right, ${COLORS.premium} 0%, ${COLORS.premium} ${minVolume}%, ${COLORS.border.default} ${minVolume}%, ${COLORS.border.default} 100%)`,
                  outline: 'none',
                  cursor: 'pointer',
                  WebkitAppearance: 'none',
                  appearance: 'none'
                }}
              />
              <style>{`
                input[type="range"]::-webkit-slider-thumb {
                  -webkit-appearance: none;
                  appearance: none;
                  width: 18px;
                  height: 18px;
                  border-radius: 50%;
                  background: ${COLORS.premium};
                  cursor: pointer;
                  box-shadow: 0 0 10px ${COLORS.premium}80;
                }
                input[type="range"]::-moz-range-thumb {
                  width: 18px;
                  height: 18px;
                  border-radius: 50%;
                  background: ${COLORS.premium};
                  cursor: pointer;
                  border: none;
                  box-shadow: 0 0 10px ${COLORS.premium}80;
                }
              `}</style>
            </div>

          </div>
        </div>

        {/* Content Area - Grid View */}
        {viewMode === 'grid' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
            {processedCoins.map((coin) => {
              const change = getTimeframeChange(coin);
              const momentum = calculateMomentumScore(coin);
              const isTopGainer = topGainers.slice(0, 5).some(c => c.symbol === coin.symbol);

              return (
                <div
                  key={coin.symbol}
                  onClick={() => setSelectedCoin(coin.symbol)}
                  style={{
                    background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
                    border: `2px solid ${isTopGainer ? COLORS.premium : COLORS.border.default}`,
                    borderRadius: '16px',
                    padding: '20px',
                    cursor: 'pointer',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-6px)';
                    e.currentTarget.style.boxShadow = `0 12px 24px ${getSignalColor(change)}40`;
                    e.currentTarget.style.borderColor = getSignalColor(change);
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = isTopGainer ? COLORS.premium : COLORS.border.default;
                  }}
                >
                  {/* Top Badge */}
                  {isTopGainer && (
                    <div style={{
                      position: 'absolute',
                      top: '12px',
                      right: '12px',
                      background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                      color: '#000',
                      fontSize: '10px',
                      fontWeight: '700',
                      padding: '4px 8px',
                      borderRadius: '6px',
                      letterSpacing: '0.5px'
                    }}>
                      TOP 5
                    </div>
                  )}

                  {/* Symbol */}
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 4px 0' }}>
                    {coin.symbol.replace('USDT', '')}
                    <span style={{ fontSize: '14px', color: COLORS.text.secondary, fontWeight: '400' }}>/USDT</span>
                  </h3>

                  {/* Price */}
                  <p style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: '8px 0' }}>
                    ${coin.price < 1 ? coin.price.toFixed(6) : coin.price.toFixed(2)}
                  </p>

                  {/* Change */}
                  <div style={{
                    display: 'inline-block',
                    padding: '6px 12px',
                    background: `${getSignalColor(change)}20`,
                    border: `1px solid ${getSignalColor(change)}`,
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    color: getSignalColor(change),
                    marginBottom: '16px'
                  }}>
                    {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                  </div>

                  {/* Multi-Timeframe */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
                    {[
                      { label: '1H', value: coin.change1H },
                      { label: '4H', value: coin.change4H },
                      { label: '1D', value: coin.change1D },
                      { label: '1W', value: coin.change1W }
                    ].map((tf) => (
                      <div key={tf.label} style={{
                        background: COLORS.bg.primary,
                        padding: '8px',
                        borderRadius: '8px',
                        textAlign: 'center'
                      }}>
                        <p style={{ fontSize: '11px', color: COLORS.text.secondary, margin: '0 0 2px 0' }}>{tf.label}</p>
                        <p style={{ fontSize: '13px', fontWeight: '600', color: getSignalColor(tf.value), margin: 0 }}>
                          {tf.value >= 0 ? '+' : ''}{tf.value.toFixed(1)}%
                        </p>
                      </div>
                    ))}
                  </div>

                  {/* Momentum Bar */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span style={{ fontSize: '12px', color: COLORS.text.secondary, fontWeight: '500' }}>Momentum</span>
                      <span style={{ fontSize: '12px', color: COLORS.text.primary, fontWeight: '600' }}>{momentum.toFixed(0)}/100</span>
                    </div>
                    <div style={{
                      width: '100%',
                      height: '6px',
                      background: COLORS.bg.primary,
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${momentum}%`,
                        height: '100%',
                        background: `linear-gradient(90deg, ${COLORS.danger}, ${COLORS.warning}, ${COLORS.success})`,
                        borderRadius: '3px',
                        transition: 'width 0.5s ease'
                      }} />
                    </div>
                  </div>

                  {/* Volume */}
                  <p style={{ fontSize: '12px', color: COLORS.text.secondary, margin: '12px 0 0 0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Icons.Activity style={{ width: '14px', height: '14px' }} />
                    Volume: ${(coin.volume24h / 1_000_000).toFixed(1)}M
                  </p>
                </div>
              );
            })}
          </div>
        )}

        {/* Content Area - Table View */}
        {viewMode === 'table' && (
          <div style={{
            background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
            border: `1px solid ${COLORS.border.default}`,
            borderRadius: '16px',
            overflow: 'hidden'
          }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: COLORS.bg.primary, borderBottom: `2px solid ${COLORS.border.default}` }}>
                    <th style={{ padding: '16px', textAlign: 'left', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>SYMBOL</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>PRICE</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>1H</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>4H</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>24H</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>1W</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>VOLUME</th>
                    <th style={{ padding: '16px', textAlign: 'right', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>MOMENTUM</th>
                    <th style={{ padding: '16px', textAlign: 'center', fontSize: '13px', color: COLORS.text.secondary, fontWeight: '600' }}>ANALYZE</th>
                  </tr>
                </thead>
                <tbody>
                  {processedCoins.map((coin, index) => {
                    const momentum = calculateMomentumScore(coin);
                    return (
                      <tr
                        key={coin.symbol}
                        onClick={() => setSelectedCoin(coin.symbol)}
                        style={{
                          borderBottom: `1px solid ${COLORS.border.default}`,
                          background: index % 2 === 0 ? 'transparent' : `${COLORS.bg.primary}40`,
                          cursor: 'pointer',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = `${COLORS.premium}10`;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = index % 2 === 0 ? 'transparent' : `${COLORS.bg.primary}40`;
                        }}
                      >
                        <td style={{ padding: '16px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary }}>
                              {coin.symbol.replace('USDT', '')}
                            </span>
                            <span style={{ fontSize: '13px', color: COLORS.text.secondary }}>/USDT</span>
                          </div>
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '15px', fontWeight: '500', color: COLORS.text.primary }}>
                          ${coin.price < 1 ? coin.price.toFixed(6) : coin.price.toFixed(2)}
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '14px', fontWeight: '600', color: getSignalColor(coin.change1H) }}>
                          {coin.change1H >= 0 ? '+' : ''}{coin.change1H.toFixed(2)}%
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '14px', fontWeight: '600', color: getSignalColor(coin.change4H) }}>
                          {coin.change4H >= 0 ? '+' : ''}{coin.change4H.toFixed(2)}%
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '14px', fontWeight: '600', color: getSignalColor(coin.change1D) }}>
                          {coin.change1D >= 0 ? '+' : ''}{coin.change1D.toFixed(2)}%
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '14px', fontWeight: '600', color: getSignalColor(coin.change1W) }}>
                          {coin.change1W >= 0 ? '+' : ''}{coin.change1W.toFixed(2)}%
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontSize: '14px', color: COLORS.text.secondary }}>
                          ${(coin.volume24h / 1_000_000).toFixed(1)}M
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px' }}>
                            <div style={{
                              width: '60px',
                              height: '6px',
                              background: COLORS.bg.primary,
                              borderRadius: '3px',
                              overflow: 'hidden'
                            }}>
                              <div style={{
                                width: `${momentum}%`,
                                height: '100%',
                                background: `linear-gradient(90deg, ${COLORS.danger}, ${COLORS.warning}, ${COLORS.success})`,
                                borderRadius: '3px'
                              }} />
                            </div>
                            <span style={{ fontSize: '13px', color: COLORS.text.secondary, minWidth: '35px' }}>
                              {momentum.toFixed(0)}
                            </span>
                          </div>
                        </td>
                        <td style={{ padding: '16px', textAlign: 'center' }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedCoin(coin.symbol);
                            }}
                            style={{
                              padding: '8px 16px',
                              background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                              color: '#000',
                              border: 'none',
                              borderRadius: '8px',
                              fontSize: '12px',
                              fontWeight: '600',
                              cursor: 'pointer',
                              transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.transform = 'scale(1.05)';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.transform = 'scale(1)';
                            }}
                          >
                            ANALYZE
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Content Area - Heatmap View */}
        {viewMode === 'heatmap' && (
          <div>
            <div style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `1px solid ${COLORS.border.default}`,
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '20px'
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Icons.BarChart3 style={{ width: '20px', height: '20px', color: COLORS.premium }} />
                Market Heatmap - {timeframe} Performance
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <span style={{ fontSize: '13px', color: COLORS.text.secondary }}>Color Scale:</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '40px', height: '16px', background: COLORS.danger, borderRadius: '4px' }} />
                  <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>-10%</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '40px', height: '16px', background: COLORS.warning, borderRadius: '4px' }} />
                  <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>0%</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{ width: '40px', height: '16px', background: COLORS.success, borderRadius: '4px' }} />
                  <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>+10%</span>
                </div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' }}>
              {processedCoins.map((coin) => {
                const change = getTimeframeChange(coin);
                const intensity = Math.min(Math.abs(change) / 15, 1); // Max 15% for full color

                let bgColor;
                if (change > 0) {
                  bgColor = `rgba(0, 255, 127, ${0.2 + intensity * 0.8})`;
                } else if (change < 0) {
                  bgColor = `rgba(255, 69, 58, ${0.2 + intensity * 0.8})`;
                } else {
                  bgColor = `rgba(255, 204, 0, 0.3)`;
                }

                return (
                  <div
                    key={coin.symbol}
                    onClick={() => setSelectedCoin(coin.symbol)}
                    style={{
                      background: bgColor,
                      border: `2px solid ${change > 0 ? COLORS.success : change < 0 ? COLORS.danger : COLORS.warning}`,
                      borderRadius: '12px',
                      padding: '16px',
                      cursor: 'pointer',
                      position: 'relative',
                      transition: 'all 0.3s ease',
                      minHeight: '100px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'scale(1.08)';
                      e.currentTarget.style.zIndex = '10';
                      e.currentTarget.style.boxShadow = `0 8px 24px ${change > 0 ? COLORS.success : COLORS.danger}60`;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'scale(1)';
                      e.currentTarget.style.zIndex = '1';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <div>
                      <p style={{ fontSize: '15px', fontWeight: 'bold', color: '#000', margin: '0 0 4px 0' }}>
                        {coin.symbol.replace('USDT', '')}
                      </p>
                      <p style={{ fontSize: '11px', color: '#00000080', margin: 0 }}>
                        ${coin.price < 1 ? coin.price.toFixed(4) : coin.price.toFixed(2)}
                      </p>
                    </div>
                    <p style={{ fontSize: '20px', fontWeight: 'bold', color: '#000', margin: '8px 0 0 0' }}>
                      {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

      </div>

      {/* Strategy Modal */}
      {selectedCoin && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 50,
            padding: '16px',
            backdropFilter: 'blur(10px)'
          }}
          onClick={() => setSelectedCoin(null)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '20px',
              maxWidth: '1000px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}60`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {analysisLoading && !analysis ? (
              <div style={{ textAlign: 'center', padding: '80px 0' }}>
                <LoadingAnimation />
                <p style={{ fontSize: '18px', color: COLORS.text.secondary, marginTop: '20px' }}>
                  Analyzing Strategies...
                </p>
              </div>
            ) : analysis ? (
              <div style={{ padding: '32px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '32px' }}>
                  <div>
                    <h2 style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 8px 0', background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                      {analysis.symbol.replace('USDT', '')}/USDT
                    </h2>
                    <div style={{ display: 'flex', gap: '16px', fontSize: '16px' }}>
                      <span style={{ color: COLORS.text.primary, fontWeight: '500' }}>
                        ${analysis.price < 1 ? analysis.price.toFixed(6) : analysis.price.toFixed(2)}
                      </span>
                      <span style={{ fontWeight: 'bold', color: getSignalColor(analysis.changePercent24h) }}>
                        {analysis.changePercent24h >= 0 ? '+' : ''}
                        {analysis.changePercent24h.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedCoin(null)}
                    style={{
                      padding: '12px 24px',
                      background: 'transparent',
                      border: `2px solid ${COLORS.danger}`,
                      borderRadius: '12px',
                      color: COLORS.danger,
                      fontSize: '14px',
                      fontWeight: '600',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = COLORS.danger;
                      e.currentTarget.style.color = '#fff';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.color = COLORS.danger;
                    }}
                  >
                    CLOSE
                  </button>
                </div>

                {analysis.groqAnalysis && !analysis.groqAnalysis.includes('kullanılamıyor') && (
                  <div style={{
                    background: `linear-gradient(145deg, ${COLORS.info}10, ${COLORS.premium}10)`,
                    border: `1px solid ${COLORS.info}`,
                    borderRadius: '16px',
                    padding: '24px',
                    marginBottom: '24px'
                  }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <Icons.Bot style={{ width: '24px', height: '24px', color: COLORS.info }} />
                      AI ANALYSIS
                    </h3>
                    <p style={{ color: COLORS.text.primary, lineHeight: '1.8', fontSize: '15px' }}>{analysis.groqAnalysis}</p>
                  </div>
                )}

                <div style={{
                  background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.card})`,
                  border: `2px solid ${COLORS.premium}`,
                  borderRadius: '16px',
                  padding: '24px',
                  marginBottom: '24px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <Icons.Target style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                        OVERALL SIGNAL
                      </h3>
                      <div style={{ display: 'flex', gap: '16px', fontSize: '15px' }}>
                        <span style={{ color: COLORS.success, fontWeight: '600' }}>{analysis.buyCount} BUY</span>
                        <span style={{ color: COLORS.warning, fontWeight: '600' }}>{analysis.waitCount} WAIT</span>
                        <span style={{ color: COLORS.danger, fontWeight: '600' }}>{analysis.sellCount} SELL</span>
                        <span style={{ color: COLORS.text.muted, fontWeight: '600' }}>{analysis.neutralCount} NEUTRAL</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '48px', fontWeight: 'bold', color: COLORS.premium, lineHeight: '1' }}>
                        {analysis.overallScore}/100
                      </div>
                      <div style={{
                        fontSize: '18px',
                        fontWeight: 'bold',
                        marginTop: '8px',
                        color: analysis.recommendation.includes('BUY') ? COLORS.success :
                               analysis.recommendation === 'WAIT' ? COLORS.warning :
                               analysis.recommendation === 'SELL' ? COLORS.danger : COLORS.text.muted
                      }}>
                        {analysis.recommendation.replace('_', ' ')}
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Icons.BarChart style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                    STRATEGY SIGNALS ({analysis.strategies.length})
                  </h3>
                  <div style={{ maxHeight: '400px', overflowY: 'auto', display: 'grid', gap: '16px' }}>
                    {analysis.strategies.map((strategy, index) => (
                      <div key={index} style={{
                        background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.card})`,
                        border: `1px solid ${COLORS.border.default}`,
                        borderRadius: '12px',
                        padding: '20px',
                        transition: 'all 0.2s ease'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = COLORS.premium;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = COLORS.border.default;
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                          <h4 style={{ color: COLORS.text.primary, fontWeight: 'bold', fontSize: '16px' }}>{strategy.name}</h4>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span style={{
                              padding: '6px 16px',
                              borderRadius: '8px',
                              fontSize: '14px',
                              fontWeight: 'bold',
                              background: strategy.signal === 'BUY' ? `${COLORS.success}20` :
                                         strategy.signal === 'WAIT' ? `${COLORS.warning}20` :
                                         strategy.signal === 'SELL' ? `${COLORS.danger}20` : `${COLORS.text.muted}20`,
                              color: strategy.signal === 'BUY' ? COLORS.success :
                                    strategy.signal === 'WAIT' ? COLORS.warning :
                                    strategy.signal === 'SELL' ? COLORS.danger : COLORS.text.muted,
                              border: `1px solid ${strategy.signal === 'BUY' ? COLORS.success :
                                                   strategy.signal === 'WAIT' ? COLORS.warning :
                                                   strategy.signal === 'SELL' ? COLORS.danger : COLORS.text.muted}`
                            }}>
                              {strategy.signal === 'BUY' ? 'BUY' :
                               strategy.signal === 'WAIT' ? 'WAIT' :
                               strategy.signal === 'SELL' ? 'SELL' : 'NEUTRAL'}
                            </span>
                            <span style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '600' }}>
                              %{strategy.confidence}
                            </span>
                          </div>
                        </div>
                        <p style={{ color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.6' }}>{strategy.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ marginTop: '24px', textAlign: 'center', color: COLORS.text.muted, fontSize: '13px' }}>
                  <Icons.AlertTriangle style={{ width: '14px', height: '14px', display: 'inline', marginRight: '6px' }} />
                  This is not investment advice. Make your own decisions at your own risk.
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '80px 0', color: COLORS.text.muted }}>
                Analysis could not be loaded.
              </div>
            )}
          </div>
        </div>
      )}

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
                    Market Scanner Ultra - MANTIK
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
                  CLOSE
                </button>
              </div>
            </div>

            <div style={{ padding: '32px' }}>
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.Target style={{ width: '24px', height: '24px' }} />
                  Unique Ultra Features
                </h3>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                  Market Scanner Ultra is one of the most advanced scanning and analysis tools on the market.
                  50+ coins are scanned in real-time and visualized with multiple view modes.
                </p>
              </div>

              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.Zap style={{ width: '24px', height: '24px' }} />
                  New Features
                </h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: '3 View Modes', desc: 'Switch between Grid, Table and Heatmap views. Each mode is optimized for different analysis needs.' },
                    { name: 'Multi-Timeframe Analysis', desc: 'Compare 1H, 4H, 1D and 1W timeframes simultaneously. See trend consistency instantly.' },
                    { name: 'Momentum Scoring', desc: 'A momentum score between 0-100 is calculated for each coin. Weighted average from all timeframes.' },
                    { name: 'Advanced Filters', desc: 'Find coins instantly with market cap, performance, volume and search filters.' },
                    { name: 'Live Heatmap', desc: 'See the market at a glance. Color intensity varies based on performance.' },
                    { name: 'Premium Cards', desc: 'Detailed card view for each coin. Multi-timeframe, momentum bar and volume info.' },
                    { name: 'TOP 5 Marking', desc: 'Top 5 performing coins are automatically highlighted.' },
                    { name: '30s Auto-Refresh', desc: 'Data automatically refreshes every 30 seconds. Track the market live.' }
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

              <div style={{
                background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '12px',
                padding: '20px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                  Performance Improvements
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.8' }}>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Optimized API Calls:</strong> Multi-timeframe data optimized with 5-minute cache.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>useMemo Usage:</strong> Filtering and sorting operations memoized, preventing unnecessary calculations.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Smart Rendering:</strong> Only necessary components are rendered based on view mode.
                  </li>
                  <li>
                    <strong style={{ color: COLORS.text.primary }}>30s Refresh:</strong> Refresh interval increased from 10s to 30s to reduce unnecessary API load.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
    </div>
  );
}
