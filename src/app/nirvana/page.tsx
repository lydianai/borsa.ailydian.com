'use client';

/**
 * 🌟 NIRVANA DASHBOARD - ULTIMATE MARKET ANALYSIS
 * Tüm stratejilerin birleşik analizi - Trading, AI, Quantum, Conservative, Breakout, Omnipotent
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS, getSignalColor } from '@/lib/colors';

interface UnifiedSignal {
  symbol: string;
  price: number;
  change24h: number;
  signal: string;
  confidence: number;
  strategies: {
    trading?: string;
    ai?: string;
    quantum?: string;
    conservative?: string;
    breakout?: string;
    omnipotent?: string;
  };
  consensusScore: number;
  volume24h: number;
  marketCap?: number;
  timestamp: string;
}

interface MarketOverview {
  totalCoins: number;
  bullishSignals: number;
  bearishSignals: number;
  neutralSignals: number;
  avgConsensus: number;
  highConfidenceCount: number;
  topPerformers: UnifiedSignal[];
}

export default function NirvanaPage() {
  const [signals, setSignals] = useState<UnifiedSignal[]>([]);
  const [marketOverview, setMarketOverview] = useState<MarketOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(60);
  const [filterSignal, setFilterSignal] = useState<'ALL' | 'BUY' | 'SELL' | 'WAIT'>('ALL');
  const [sortBy, setSortBy] = useState<'consensus' | 'confidence' | 'volume'>('consensus');
  const [selectedCoin, setSelectedCoin] = useState<UnifiedSignal | null>(null);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [timeRange, setTimeRange] = useState<'ALL' | '5m' | '15m' | '1h' | '4h'>('ALL');
  const [minConsensus, setMinConsensus] = useState(0);
  const [minConfidence, setMinConfidence] = useState(0);
  const [showLogicModal, setShowLogicModal] = useState(false);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Safe fetch that returns empty result on error - with built-in timeout
  const safeFetch = async (url: string, timeoutMs: number = 10000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        cache: 'no-store',
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        console.warn(`[Nirvana] HTTP ${response.status} for ${url}`);
        return { success: false, data: null };
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      // Handle timeout gracefully - this is expected behavior
      if (error instanceof Error && error.name === 'AbortError') {
        // Don't log timeout as error - it's normal when APIs are slow
        return { success: false, data: null };
      }

      console.warn(`[Nirvana] Failed to fetch ${url}`);
      return { success: false, data: null };
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('[Nirvana] Starting to fetch all strategy signals...');

      // Fetch all signals with timeout protection
      const [trading, ai, quantum, conservative, breakout, omnipotent] = await Promise.all([
        safeFetch('/api/signals', 10000),
        safeFetch('/api/ai-signals', 15000), // AI needs more time
        safeFetch('/api/quantum-signals', 10000),
        safeFetch('/api/conservative-signals', 10000),
        safeFetch('/api/breakout-signals', 10000),
        safeFetch('/api/market-correlation', 12000)
      ]);

      console.log('[Nirvana] Fetch complete. Processing signals...');

      // Merge all signals by symbol
      const mergedSignals = new Map<string, UnifiedSignal>();

      // Helper to add signal
      const addSignal = (symbol: string, data: any, strategyName: string, signalValue: string) => {
        if (!mergedSignals.has(symbol)) {
          mergedSignals.set(symbol, {
            symbol,
            price: data.price || 0,
            change24h: data.change24h || 0,
            signal: signalValue,
            confidence: data.confidence || 0,
            strategies: {},
            consensusScore: 0,
            volume24h: data.volume24h || 0,
            marketCap: data.marketCap,
            timestamp: data.timestamp || new Date().toISOString()
          });
        }
        const existing = mergedSignals.get(symbol)!;
        existing.strategies[strategyName as keyof typeof existing.strategies] = signalValue;
      };

      // Process trading signals
      if (trading.success && trading.data) {
        const signals = Array.isArray(trading.data) ? trading.data : trading.data.signals || [];
        signals.forEach((s: any) => {
          if (s.symbol && s.type) {
            addSignal(s.symbol, s, 'trading', s.type);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} trading signals`);
      }

      // Process AI signals
      if (ai.success && ai.data) {
        const signals = ai.data.signals || [];
        signals.forEach((s: any) => {
          if (s.symbol && s.type) {
            addSignal(s.symbol, s, 'ai', s.type);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} AI signals`);
      }

      // Process quantum signals
      if (quantum.success && quantum.data) {
        const signals = quantum.data.signals || [];
        signals.forEach((s: any) => {
          if (s.symbol && s.type) {
            addSignal(s.symbol, s, 'quantum', s.type);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} quantum signals`);
      }

      // Process conservative signals
      if (conservative.success && conservative.data) {
        const signals = conservative.data.signals || [];
        signals.forEach((s: any) => {
          if (s.symbol && (s.type || s.signal)) {
            addSignal(s.symbol, s, 'conservative', s.type || s.signal);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} conservative signals`);
      }

      // Process breakout signals
      if (breakout.success && breakout.data) {
        const signals = breakout.data.signals || [];
        signals.forEach((s: any) => {
          if (s.symbol && (s.direction || s.signal)) {
            addSignal(s.symbol, s, 'breakout', s.direction || s.signal);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} breakout signals`);
      }

      // Process omnipotent signals
      if (omnipotent.success && omnipotent.data?.correlations) {
        const signals = omnipotent.data.correlations;
        signals.forEach((s: any) => {
          if (s.symbol && s.signal) {
            addSignal(s.symbol, s, 'omnipotent', s.signal);
          }
        });
        console.log(`[Nirvana] Processed ${signals.length} omnipotent signals`);
      }

      console.log(`[Nirvana] Total merged symbols: ${mergedSignals.size}`);

      // Calculate consensus score
      const finalSignals = Array.from(mergedSignals.values()).map(signal => {
        const strategies = Object.values(signal.strategies);
        const buyCount = strategies.filter(s => s === 'BUY' || s === 'LONG').length;
        const sellCount = strategies.filter(s => s === 'SELL' || s === 'SHORT').length;
        const totalStrategies = strategies.length;

        let consensusSignal = 'WAIT';
        let consensusScore = 0;

        if (totalStrategies > 0) {
          if (buyCount > sellCount && buyCount >= totalStrategies * 0.5) {
            consensusSignal = 'BUY';
            consensusScore = Math.round((buyCount / totalStrategies) * 100);
          } else if (sellCount > buyCount && sellCount >= totalStrategies * 0.5) {
            consensusSignal = 'SELL';
            consensusScore = Math.round((sellCount / totalStrategies) * 100);
          } else {
            consensusScore = Math.round(Math.max(buyCount, sellCount) / totalStrategies * 100);
          }
        }

        return {
          ...signal,
          signal: consensusSignal,
          consensusScore
        };
      });

      setSignals(finalSignals);

      // Calculate market overview
      const avgConsensus = finalSignals.length > 0
        ? Math.round(finalSignals.reduce((acc, s) => acc + s.consensusScore, 0) / finalSignals.length)
        : 0;

      const overview: MarketOverview = {
        totalCoins: finalSignals.length,
        bullishSignals: finalSignals.filter(s => s.signal === 'BUY').length,
        bearishSignals: finalSignals.filter(s => s.signal === 'SELL').length,
        neutralSignals: finalSignals.filter(s => s.signal === 'WAIT').length,
        avgConsensus,
        highConfidenceCount: finalSignals.filter(s => s.confidence >= 70 && s.consensusScore >= 70).length,
        topPerformers: finalSignals
          .sort((a, b) => b.consensusScore - a.consensusScore)
          .slice(0, 5)
      };

      setMarketOverview(overview);

      // Check if we got any data
      if (finalSignals.length === 0) {
        console.warn('[Nirvana] No signals received from any API');
        setError('Hiçbir API\u0027den sinyal alınamadı. Lütfen API\u0027lerin çalıştığından emin olun.');
      } else {
        setError(null);
        console.log(`[Nirvana] Successfully loaded ${finalSignals.length} signals`);
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
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

  // Filter and sort
  const filteredSignals = signals
    .filter((s) => {
      const matchesFilter = filterSignal === 'ALL' || s.signal === filterSignal;
      const matchesSearch = s.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesConsensus = s.consensusScore >= minConsensus;
      const matchesConfidence = s.confidence >= minConfidence;

      let matchesTimeRange = true;
      if (timeRange !== 'ALL') {
        const signalTime = new Date(s.timestamp).getTime();
        const now = Date.now();
        const timeRanges: Record<string, number> = {
          '5m': 5 * 60 * 1000,
          '15m': 15 * 60 * 1000,
          '1h': 60 * 60 * 1000,
          '4h': 4 * 60 * 60 * 1000,
        };
        matchesTimeRange = (now - signalTime) <= timeRanges[timeRange];
      }

      return matchesFilter && matchesSearch && matchesConsensus && matchesConfidence && matchesTimeRange;
    })
    .sort((a, b) => {
      if (sortBy === 'consensus') return b.consensusScore - a.consensusScore;
      if (sortBy === 'confidence') return b.confidence - a.confidence;
      if (sortBy === 'volume') return b.volume24h - a.volume24h;
      return 0;
    });


  const getStrategyLabel = (strategyKey: string) => {
    const labels: Record<string, string> = {
      'trading': 'İşlem Sinyali',
      'ai': 'Yapay Zeka',
      'quantum': 'Kuantum',
      'conservative': 'Muhafazakâr',
      'breakout': 'Kırılım',
      'omnipotent': 'Omnipotent'
    };
    return labels[strategyKey] || strategyKey;
  };

  return (
    <div className="dashboard-container">
      {/* AI Assistant */}
      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}

      {/* Sidebar */}
      <SharedSidebar
        currentPage="nirvana"
        notificationCounts={notificationCounts}
      />

      {/* Main Content */}
      <div className="dashboard-main">
        {/* Page Header with MANTIK Button */}
        <div style={{ margin: '16px 24px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Icons.Fire style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                Nirvana Dashboard
              </h1>
            </div>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
              8 Strateji Birleşik Görünümü - Tüm Sinyaller Tek Ekranda
            </p>
          </div>

          {/* MANTIK Button - Responsive */}
          <div>
            <style>{`
              @media (max-width: 768px) {
                .mantik-button-nirvana {
                  padding: 10px 20px !important;
                  fontSize: 13px !important;
                  height: 42px !important;
                }
                .mantik-button-nirvana svg {
                  width: 18px !important;
                  height: 18px !important;
                }
              }
              @media (max-width: 480px) {
                .mantik-button-nirvana {
                  padding: 8px 16px !important;
                  fontSize: 12px !important;
                  height: 40px !important;
                }
                .mantik-button-nirvana svg {
                  width: 16px !important;
                  height: 16px !important;
                }
              }
            `}</style>
            <button
              onClick={() => setShowLogicModal(true)}
              className="mantik-button-nirvana"
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

        <main className="dashboard-content" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
          {loading && signals.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '100px 20px', color: COLORS.text.muted }}>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
            <div style={{ fontSize: '18px' }}>Tüm stratejiler analiz ediliyor...</div>
          </div>
        ) : (
          <>
            {/* Market Overview */}
            {marketOverview && (
              <div style={{ marginBottom: '24px', background: COLORS.bg.card, border: `1px solid ${COLORS.border.default}`, borderRadius: '10px', padding: '24px' }}>
                <h2 style={{ fontSize: '18px', marginBottom: '20px', color: COLORS.cyan, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>📊</span> PİYASA GENEL BAKIŞ
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px' }}>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                    <div style={{ color: COLORS.text.muted, fontSize: '11px', marginBottom: '8px' }}>Toplam Analiz</div>
                    <div style={{ fontSize: '28px', fontWeight: '700' }}>{marketOverview.totalCoins}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.success}` }}>
                    <div style={{ color: COLORS.success, fontSize: '11px', marginBottom: '8px' }}>Yükseliş</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.success }}>{marketOverview.bullishSignals}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.danger}` }}>
                    <div style={{ color: COLORS.danger, fontSize: '11px', marginBottom: '8px' }}>Düşüş</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.danger }}>{marketOverview.bearishSignals}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.warning}` }}>
                    <div style={{ color: COLORS.warning, fontSize: '11px', marginBottom: '8px' }}>Nötr</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.warning }}>{marketOverview.neutralSignals}</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.info}` }}>
                    <div style={{ color: COLORS.info, fontSize: '11px', marginBottom: '8px' }}>Ort. Uzlaşma</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.info }}>{marketOverview.avgConsensus}%</div>
                  </div>
                  <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.premium}` }}>
                    <div style={{ color: COLORS.premium, fontSize: '11px', marginBottom: '8px' }}>Yüksek Güven</div>
                    <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.premium }}>{marketOverview.highConfidenceCount}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Filters */}
            <div style={{ marginBottom: '20px', display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center', padding: '16px', background: COLORS.bg.card, borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
              {/* Signal Filter */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: COLORS.text.secondary, fontWeight: '600' }}>SİNYAL:</span>
                {(['ALL', 'BUY', 'SELL', 'WAIT'] as const).map((type) => (
                  <button
                    key={type}
                    onClick={() => setFilterSignal(type)}
                    style={{
                      background: filterSignal === type ? COLORS.text.primary : 'transparent',
                      color: filterSignal === type ? COLORS.bg.primary : type === 'BUY' ? COLORS.success : type === 'SELL' ? COLORS.danger : type === 'WAIT' ? COLORS.warning : COLORS.text.secondary,
                      border: `1px solid ${filterSignal === type ? COLORS.text.primary : COLORS.border.active}`,
                      padding: '6px 14px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      fontWeight: '600',
                      transition: 'all 0.2s',
                    }}
                  >
                    {type === 'ALL' ? 'Tümü' : type}
                  </button>
                ))}
              </div>

              {/* Time Range */}
              <div style={{ display: 'flex', gap: '6px', alignItems: 'center', borderLeft: `1px solid ${COLORS.border.default}`, paddingLeft: '12px' }}>
                <span style={{ fontSize: '12px', color: COLORS.text.secondary, marginRight: '4px' }}>Zaman:</span>
                {(['ALL', '5m', '15m', '1h', '4h'] as const).map((range) => (
                  <button
                    key={range}
                    onClick={() => setTimeRange(range)}
                    style={{
                      background: timeRange === range ? COLORS.success : COLORS.bg.hover,
                      color: timeRange === range ? COLORS.bg.primary : COLORS.text.secondary,
                      border: `1px solid ${timeRange === range ? COLORS.success : COLORS.border.default}`,
                      padding: '4px 10px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      fontWeight: '600',
                      transition: 'all 0.2s',
                    }}
                  >
                    {range}
                  </button>
                ))}
              </div>

              {/* Search */}
              <div className="header-search" style={{ flex: 1, maxWidth: '300px' }}>
                <Icons.Search style={{ width: '16px', height: '16px', color: COLORS.text.secondary }} />
                <input
                  type="text"
                  placeholder="Coin ara..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="search-input"
                />
                {searchTerm && (
                  <button className="search-clear" onClick={() => setSearchTerm('')}>
                    <Icons.X style={{ width: '14px', height: '14px', color: COLORS.text.secondary }} />
                  </button>
                )}
              </div>

              {/* Min Consensus */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: COLORS.text.secondary, fontWeight: '600' }}>MİN UZLAŞMA:</span>
                <select
                  value={minConsensus}
                  onChange={(e) => setMinConsensus(Number(e.target.value))}
                  style={{
                    padding: '6px 12px',
                    fontSize: '11px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.active}`,
                    borderRadius: '4px',
                    color: COLORS.text.primary,
                    cursor: 'pointer'
                  }}
                >
                  <option value="0">Tümü (0)</option>
                  <option value="50">≥50%</option>
                  <option value="70">≥70%</option>
                  <option value="80">≥80%</option>
                </select>
              </div>

              {/* Min Confidence */}
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: COLORS.text.secondary, fontWeight: '600' }}>MIN GÜVEN:</span>
                <select
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(Number(e.target.value))}
                  style={{
                    padding: '6px 12px',
                    fontSize: '11px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.active}`,
                    borderRadius: '4px',
                    color: COLORS.text.primary,
                    cursor: 'pointer'
                  }}
                >
                  <option value="0">Tümü (0)</option>
                  <option value="50">≥50%</option>
                  <option value="70">≥70%</option>
                  <option value="80">≥80%</option>
                </select>
              </div>

              {/* Sort */}
              <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ color: COLORS.text.muted, fontSize: '12px', fontWeight: '600' }}>SIRALA:</span>
                {[
                  { key: 'consensus', label: 'Uzlaşma' },
                  { key: 'confidence', label: 'Güven' },
                  { key: 'volume', label: 'Hacim' },
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
                      transition: 'all 0.2s',
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Signals Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
              {filteredSignals.map((coin) => (
                <div
                  key={coin.symbol}
                  onClick={() => setSelectedCoin(coin)}
                  style={{
                    background: COLORS.bg.card,
                    border: `1px solid ${getSignalColor(coin.signal)}`,
                    borderRadius: '10px',
                    padding: '16px',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    const signalColor = getSignalColor(coin.signal);
                    const alpha = coin.signal === 'WAIT' ? '0.2' : '0.3';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${signalColor}${alpha}`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                    <div>
                      <div style={{ fontSize: '18px', fontWeight: '700', marginBottom: '4px' }}>
                        {coin.symbol.replace('USDT', '')}
                      </div>
                      <div style={{ fontSize: '13px', color: COLORS.text.muted, fontFamily: 'monospace' }}>
                        ${(coin.price ?? 0) < 1 ? (coin.price ?? 0).toFixed(6) : (coin.price ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '14px', fontWeight: '700', color: (coin.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                        {(coin.change24h ?? 0) >= 0 ? '+' : ''}{(coin.change24h ?? 0).toFixed(2)}%
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.muted }}>24h</div>
                    </div>
                  </div>

                  {/* Consensus Score */}
                  <div style={{ marginBottom: '12px', background: COLORS.bg.secondary, padding: '10px', borderRadius: '6px' }}>
                    <div style={{ fontSize: '10px', color: COLORS.text.muted, marginBottom: '6px', fontWeight: '600' }}>CONSENSUS SCORE</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ flex: 1, height: '6px', background: COLORS.border.default, borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${coin.consensusScore}%`, height: '100%', background: coin.consensusScore >= 75 ? COLORS.success : coin.consensusScore >= 50 ? COLORS.warning : COLORS.danger, transition: 'width 0.3s' }} />
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: coin.consensusScore >= 75 ? COLORS.success : coin.consensusScore >= 50 ? COLORS.warning : COLORS.danger }}>
                        {coin.consensusScore}%
                      </div>
                    </div>
                  </div>

                  {/* Strategies */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '6px', marginBottom: '12px' }}>
                    {Object.entries(coin.strategies).map(([name, signal]) => (
                      <div key={name} style={{ background: COLORS.bg.secondary, padding: '6px', borderRadius: '4px', textAlign: 'center' }}>
                        <div style={{ fontSize: '8px', color: COLORS.text.muted, marginBottom: '2px', textTransform: 'uppercase' }}>{name}</div>
                        <div style={{ fontSize: '10px', fontWeight: '600', color: signal === 'BUY' || signal === 'LONG' ? COLORS.success : signal === 'SELL' || signal === 'SHORT' ? COLORS.danger : COLORS.warning }}>
                          {signal}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Signal Badge */}
                  <div style={{ textAlign: 'center' }}>
                    <div style={{
                      display: 'inline-block',
                      background: getSignalColor(coin.signal),
                      color: COLORS.bg.primary,
                      padding: '8px 20px',
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: '700',
                      letterSpacing: '1px',
                    }}>
                      {coin.signal}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {filteredSignals.length === 0 && (
              <div style={{ textAlign: 'center', padding: '60px 20px', color: COLORS.text.muted, background: COLORS.bg.card, borderRadius: '10px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
                <div style={{ fontSize: '16px' }}>Seçilen filtrelere uygun sinyal bulunamadı.</div>
              </div>
            )}
          </>
        )}

        {/* Detail Modal */}
        {selectedCoin && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.95)',
              zIndex: 9999,
              padding: '20px',
              overflow: 'auto',
              backdropFilter: 'blur(10px)',
            }}
            onClick={() => setSelectedCoin(null)}
          >
            <div
              style={{
                maxWidth: '900px',
                margin: '0 auto',
                background: COLORS.bg.secondary,
                border: `2px solid ${COLORS.success}`,
                borderRadius: '16px',
                padding: '32px',
                boxShadow: `0 0 60px ${COLORS.success}4D`,
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
                <div>
                  <h2 style={{ fontSize: '32px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '8px' }}>
                    {selectedCoin.symbol.replace('USDT', '')}
                  </h2>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: (selectedCoin.change24h ?? 0) >= 0 ? COLORS.success : COLORS.danger }}>
                    {(selectedCoin.change24h ?? 0) >= 0 ? '+' : ''}{(selectedCoin.change24h ?? 0).toFixed(2)}%
                  </div>
                </div>
                <button
                  onClick={() => setSelectedCoin(null)}
                  style={{
                    background: COLORS.danger,
                    color: COLORS.text.primary,
                    border: 'none',
                    padding: '12px 24px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '16px',
                    fontWeight: '600',
                  }}
                >
                  ✕ KAPAT
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '32px' }}>
                <div style={{ background: COLORS.bg.card, border: `2px solid ${COLORS.border.default}`, borderRadius: '12px', padding: '24px', textAlign: 'center' }}>
                  <div style={{ fontSize: '13px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>GÜNCEL FİYAT</div>
                  <div style={{ fontSize: '36px', fontWeight: '700', color: COLORS.info, fontFamily: 'monospace' }}>
                    ${(selectedCoin.price ?? 0) < 1 ? (selectedCoin.price ?? 0).toFixed(6) : (selectedCoin.price ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </div>
                </div>

                <div style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%)', border: `2px solid ${getSignalColor(selectedCoin.signal)}`, borderRadius: '12px', padding: '24px', textAlign: 'center' }}>
                  <div style={{ fontSize: '13px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>UZLAŞMA SİNYALİ</div>
                  <div style={{ fontSize: '40px', fontWeight: '700', color: getSignalColor(selectedCoin.signal), letterSpacing: '2px' }}>
                    {selectedCoin.signal}
                  </div>
                  <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginTop: '8px' }}>
                    Uzlaşma: {selectedCoin.consensusScore}% | Güven: {selectedCoin.confidence}%
                  </div>
                </div>
              </div>

              <div style={{ background: COLORS.bg.card, border: `2px solid ${COLORS.success}`, borderRadius: '12px', padding: '24px' }}>
                <h3 style={{ fontSize: '18px', color: COLORS.success, marginBottom: '20px', fontWeight: '700', textAlign: 'center' }}>
                  📊 TÜM STRATEJİLER
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                  {Object.entries(selectedCoin.strategies).map(([name, signal]) => (
                    <div key={name} style={{ background: COLORS.bg.secondary, border: `2px solid ${signal === 'BUY' || signal === 'LONG' ? COLORS.success : signal === 'SELL' || signal === 'SHORT' ? COLORS.danger : COLORS.warning}`, borderRadius: '10px', padding: '20px', textAlign: 'center' }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px', fontWeight: '600' }}>
                        {getStrategyLabel(name)}
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: signal === 'BUY' || signal === 'LONG' ? COLORS.success : signal === 'SELL' || signal === 'SHORT' ? COLORS.danger : COLORS.warning }}>
                        {signal}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
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
                      Nirvana Dashboard MANTIK
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
                    <Icons.Fire style={{ width: '24px', height: '24px' }} />
                    Genel Bakış
                  </h3>
                  <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                    Nirvana Dashboard, tüm stratejilerin birleştirilmiş analizini sunan merkezi kontrol paneldir.
                    Trading, AI, Quantum, Conservative, Breakout ve Omnipotent stratejilerinin sinyalleri tek bir ekranda görüntülenir.
                  </p>
                  <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                    Consensus skoru ile stratejiler arasındaki uyumu görebilir ve en güvenilir sinyalleri anında tespit edebilirsiniz.
                  </p>
                </div>

                {/* Key Features */}
                <div style={{ marginBottom: '32px' }}>
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Icons.Zap style={{ width: '24px', height: '24px' }} />
                    Temel Özellikler
                  </h3>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {[
                      { name: 'Tüm 8 Strateji Birleşimi', desc: 'Trading, AI, Quantum, Conservative, Breakout, Omnipotent ve daha fazla strateji tek ekranda.' },
                      { name: 'Birleşik Sinyal Görünümü', desc: 'Her coin için tüm stratejilerin sinyalleri yan yana görüntülenir ve karşılaştırılır.' },
                      { name: 'Piyasa Sentiment Analizi', desc: 'Toplu yükseliş, düşüş ve nötr sinyal sayıları ile genel piyasa durumu görülür.' },
                      { name: 'Aktif Strateji Sayısı', desc: 'Her coin için kaç stratejinin aktif sinyal verdiği takip edilir.' },
                      { name: 'Toplam Sinyal Takibi', desc: 'Buy, sell ve wait sinyallerinin toplam sayıları anlık güncellenir.' },
                      { name: 'Güven Skorlaması', desc: 'Her strateji için ayrı güven yüzdesi hesaplanır ve görüntülenir.' },
                      { name: 'Otomatik Yenileme', desc: 'Tüm stratejilerin verileri 60 saniyede bir otomatik olarak güncellenir.' }
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
                          Piyasa Genel Bakış
                        </div>
                        <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                          Üst panelde toplam analiz sayısı, yükseliş/düşüş sinyalleri ve ortalama uzlaşma yüzdesini görürsünüz.
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
                          Sinyal Filtreleme
                        </div>
                        <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                          BUY, SELL veya WAIT filtrelerini kullanarak sadece ilgilendiğiniz sinyalleri görebilirsiniz.
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
                          Coin Detayı
                        </div>
                        <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                          Herhangi bir coin kartına tıklayarak tüm stratejilerin detaylı sinyallerini ve uzlaşma skorunu görüntüleyin.
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
                          Consensus Takibi
                        </div>
                        <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                          Consensus skoru %70 ve üzeri olan coinler yüksek güvenilirliğe sahiptir.
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
                      <strong style={{ color: COLORS.text.primary }}>Tüm Stratejileri Toplar:</strong> 8 farklı stratejinin sinyalleri tek bir sayfada birleştirilir.
                    </li>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>30 Saniye Güncelleme:</strong> Gerçek zamanlı veri akışı ile tüm stratejiler her 30 saniyede güncellenir.
                    </li>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>Buy/Sell Sinyal Sayıları:</strong> Her coin için kaç buy ve kaç sell sinyali olduğu gösterilir.
                    </li>
                    <li>
                      <strong style={{ color: COLORS.text.primary }}>Strateji Durum İzleme:</strong> Hangi stratejilerin aktif olduğunu ve hangilerinin sinyal verdiğini takip edebilirsiniz.
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
              }}>
                <p style={{ margin: 0, fontSize: '13px', color: COLORS.text.secondary }}>
                  Nirvana Dashboard ile tüm stratejileri tek bakışta görün
                </p>
              </div>
            </div>
          </div>
        )}
        </main>
      </div>
    </div>
  );
}
