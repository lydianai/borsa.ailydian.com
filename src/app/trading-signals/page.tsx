'use client';

/**
 * 🎯 TRADING SIGNALS PAGE - Premium Modern Design
 *
 * Features:
 * - AI-Powered Trading Signals with Ta-Lib Analysis
 * - Real-time Signal Generation (10s auto-refresh)
 * - Multi-Strategy Analysis (Momentum, Volume, Reversal)
 * - Advanced Filtering & Sorting System
 * - Signal Detail Modal with Full Analysis
 * - MANTIK Educational Modal
 * - Premium Gradient UI with Glassmorphism
 *
 * White Hat Compliance:
 * - Public Binance API only
 * - Rate limit protection with caching
 * - Educational purpose signals
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { SharedSidebar } from '@/components/SharedSidebar';
import { COLORS } from '@/lib/colors';
import { PWAProvider } from '@/components/PWAProvider';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';

interface TradingSignal {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  price: number;
  confidence: number;
  strength: number;
  strategy: string;
  targets?: string[];
  timestamp: string;
  reasoning?: string;
}

interface AILearning {
  totalAnalyzed: number;
  successRate: string;
  buyToSellTransitions: number;
  topRiskyCoins: Array<{
    symbol: string;
    transitions: number;
    risk: number;
  }>;
}

interface SignalsData {
  signals: TradingSignal[];
  totalSignals: number;
  lastUpdate: string;
  aiLearning: AILearning;
  marketStats: {
    totalMarkets: number;
    avgChange: string;
    topGainer: string;
    topLoser: string;
  };
}

export default function TradingSignalsPage() {
  const [data, setData] = useState<SignalsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefreshCountdown, setAutoRefreshCountdown] = useState(10);

  // Filters
  const [filterType, setFilterType] = useState<'ALL' | 'BUY' | 'SELL'>('ALL');
  const [sortBy, setSortBy] = useState<'confidence' | 'strength' | 'time'>('confidence');
  const [searchTerm, setSearchTerm] = useState('');
  const [minConfidence, setMinConfidence] = useState(60);
  const [strategyFilter, setStrategyFilter] = useState<'ALL' | string>('ALL');

  // Modals
  const [selectedSignal, setSelectedSignal] = useState<TradingSignal | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showLogicModal, setShowLogicModal] = useState(false);

  // Notification counts
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  const fetchSignals = async () => {
    try {
      setError(null);
      const response = await fetch('/api/signals');
      const result = await response.json();

      if (result.success) {
        setData(result.data);
      } else {
        throw new Error(result.error || 'Failed to fetch signals');
      }
    } catch (err) {
      console.error('Error fetching signals:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch signals');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
  }, []);

  // Auto-refresh countdown
  useEffect(() => {
    const timer = setInterval(() => {
      setAutoRefreshCountdown((prev) => {
        if (prev <= 1) {
          fetchSignals();
          return 10;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Get unique strategies for filter
  const uniqueStrategies = data
    ? Array.from(new Set(data.signals.map(s => s.strategy)))
    : [];

  // Apply filters
  const filteredSignals = data?.signals.filter(signal => {
    const matchesType = filterType === 'ALL' || signal.type === filterType;
    const matchesSearch = signal.symbol.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesConfidence = signal.confidence >= minConfidence;
    const matchesStrategy = strategyFilter === 'ALL' || signal.strategy === strategyFilter;

    return matchesType && matchesSearch && matchesConfidence && matchesStrategy;
  }).sort((a, b) => {
    if (sortBy === 'confidence') return b.confidence - a.confidence;
    if (sortBy === 'strength') return b.strength - a.strength;
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  }) || [];

  const handleSignalClick = (signal: TradingSignal) => {
    setSelectedSignal(signal);
    setShowDetailModal(true);
  };

  if (loading) {
    return (
      <PWAProvider>
        <div style={{
          minHeight: '100vh',
          background: '#0A0A0A',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontSize: '60px',
              marginBottom: '24px',
              animation: 'pulse 2s infinite'
            }}>
              🎯
            </div>
            <div style={{
              fontSize: '20px',
              color: '#FFFFFF',
              fontWeight: '600',
              marginBottom: '12px'
            }}>
              YZ Ticaret Sinyalleri Oluşturuluyor...
            </div>
            <div style={{
              fontSize: '14px',
              color: 'rgba(255, 255, 255, 0.6)'
            }}>
              Piyasa Ta-Lib teknik analizi ile taranıyor
            </div>
          </div>
        </div>
      </PWAProvider>
    );
  }

  if (error) {
    return (
      <PWAProvider>
        <div style={{
          minHeight: '100vh',
          background: '#0A0A0A',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px'
        }}>
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.danger}15, rgba(26, 26, 26, 0.98))`,
            border: `2px solid ${COLORS.danger}`,
            borderRadius: '16px',
            padding: '32px',
            maxWidth: '500px',
            textAlign: 'center'
          }}>
            <Icons.AlertTriangle style={{
              width: '48px',
              height: '48px',
              color: COLORS.danger,
              margin: '0 auto 16px'
            }} />
            <h2 style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: COLORS.text.primary,
              marginBottom: '12px'
            }}>
              Hata Oluştu
            </h2>
            <p style={{
              fontSize: '14px',
              color: COLORS.text.secondary,
              marginBottom: '24px'
            }}>
              {error}
            </p>
            <button
              onClick={() => {
                setLoading(true);
                setError(null);
                fetchSignals();
              }}
              style={{
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                color: '#000',
                border: 'none',
                borderRadius: '8px',
                padding: '12px 24px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              Tekrar Dene
            </button>
          </div>
        </div>
      </PWAProvider>
    );
  }

  const buySignals = filteredSignals.filter(s => s.type === 'BUY').length;
  const sellSignals = filteredSignals.filter(s => s.type === 'SELL').length;
  const avgConfidence = filteredSignals.length > 0
    ? (filteredSignals.reduce((sum, s) => sum + s.confidence, 0) / filteredSignals.length).toFixed(1)
    : '0';
  const highConfidenceCount = filteredSignals.filter(s => s.confidence >= 75).length;

  return (
    <PWAProvider>
      <div style={{
        minHeight: '100vh',
        background: '#0A0A0A',
        display: 'flex'
      }}>
        <SharedSidebar
          currentPage="trading-signals"
          notificationCounts={notificationCounts}
        />

        <div style={{
          flex: 1,
          marginLeft: '280px',
          padding: '32px 48px',
          paddingTop: isLocalhost ? '116px' : '60px',
          overflowY: 'auto',
          maxWidth: '1920px',
          margin: '0 auto',
          width: '100%'
        }}>
          {/* Premium Header */}
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.premium}10, ${COLORS.info}10)`,
            backdropFilter: 'blur(20px)',
            borderRadius: '20px',
            padding: '32px',
            marginBottom: '24px',
            border: `1px solid ${COLORS.premium}40`,
            position: 'relative',
            overflow: 'hidden'
          }}>
            {/* Decorative Elements */}
            <div style={{
              position: 'absolute',
              top: '-50px',
              right: '-50px',
              width: '200px',
              height: '200px',
              background: `radial-gradient(circle, ${COLORS.premium}20, transparent)`,
              borderRadius: '50%',
              pointerEvents: 'none'
            }} />

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              flexWrap: 'wrap',
              gap: '20px',
              position: 'relative',
              zIndex: 1
            }}>
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  marginBottom: '12px'
                }}>
                  <div style={{
                    background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                    borderRadius: '12px',
                    padding: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Icons.TrendingUp style={{ width: '28px', height: '28px', color: '#000' }} />
                  </div>
                  <div>
                    <h1 style={{
                      fontSize: '32px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      margin: 0,
                      lineHeight: 1.2
                    }}>
                      AI Trading Sinyalleri
                    </h1>
                    <p style={{
                      fontSize: '14px',
                      color: COLORS.text.secondary,
                      margin: '4px 0 0 0'
                    }}>
                      Ta-Lib Teknik Analiz ile Gerçek Zamanlı Sinyal Üretimi
                    </p>
                  </div>
                </div>

                {/* Auto-refresh countdown */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginTop: '16px'
                }}>
                  <div style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: COLORS.success,
                    animation: 'pulse 2s infinite'
                  }} />
                  <span style={{
                    fontSize: '13px',
                    color: COLORS.text.muted,
                    fontFamily: 'monospace'
                  }}>
                    Otomatik yenileme: {autoRefreshCountdown}s
                  </span>
                </div>
              </div>

              <div>
                <style>{`
                  @media (max-width: 768px) {
                    .mantik-button-trading {
                      padding: 10px 20px !important;
                      fontSize: 13px !important;
                      height: 42px !important;
                    }
                    .mantik-button-trading svg {
                      width: 18px !important;
                      height: 18px !important;
                    }
                  }
                  @media (max-width: 480px) {
                    .mantik-button-trading {
                      padding: 8px 16px !important;
                      fontSize: 12px !important;
                      height: 40px !important;
                    }
                    .mantik-button-trading svg {
                      width: 16px !important;
                      height: 16px !important;
                    }
                  }
                `}</style>
                <button
                  onClick={() => setShowLogicModal(true)}
                  className="mantik-button-trading"
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                    color: '#000',
                    border: 'none',
                    borderRadius: '10px',
                    padding: '12px 24px',
                    fontSize: '14px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    height: '44px',
                    boxShadow: `0 8px 24px ${COLORS.premium}40`,
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = `0 12px 32px ${COLORS.premium}60`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${COLORS.premium}40`;
                  }}
                >
                  <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
                  MANTIK
                </button>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '16px',
            marginBottom: '24px'
          }}>
            {[
              { label: 'Toplam Sinyal', value: filteredSignals.length, icon: Icons.Target, color: COLORS.info },
              { label: 'AL Sinyalleri', value: buySignals, icon: Icons.TrendingUp, color: COLORS.success },
              { label: 'SAT Sinyalleri', value: sellSignals, icon: Icons.TrendingDown, color: COLORS.danger },
              { label: 'Ort. Güven', value: `${avgConfidence}%`, icon: Icons.Activity, color: COLORS.premium },
              { label: 'Yüksek Güven', value: highConfidenceCount, icon: Icons.Award, color: COLORS.warning },
              { label: 'YZ Başarı', value: data?.aiLearning?.successRate || '0%', icon: Icons.Bot, color: COLORS.cyan }
            ].map((stat, index) => (
              <div
                key={index}
                style={{
                  background: `linear-gradient(135deg, ${stat.color}08 0%, rgba(26, 26, 26, 0.98) 100%)`,
                  backdropFilter: 'blur(30px)',
                  border: `2px solid ${stat.color}40`,
                  borderRadius: '16px',
                  padding: '24px',
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-4px)';
                  e.currentTarget.style.borderColor = `${stat.color}80`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.borderColor = `${stat.color}40`;
                }}
              >
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '12px'
                }}>
                  <span style={{
                    fontSize: '13px',
                    color: COLORS.text.muted,
                    fontWeight: '600',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px'
                  }}>
                    {stat.label}
                  </span>
                  <stat.icon style={{ width: '20px', height: '20px', color: stat.color }} />
                </div>
                <div style={{
                  fontSize: '32px',
                  fontWeight: 'bold',
                  color: stat.color,
                  fontFamily: 'monospace'
                }}>
                  {stat.value}
                </div>
              </div>
            ))}
          </div>

          {/* Filters Panel */}
          <div style={{
            background: 'rgba(26, 26, 26, 0.95)',
            backdropFilter: 'blur(30px)',
            borderRadius: '16px',
            padding: '20px',
            marginBottom: '24px',
            border: `1px solid ${COLORS.border.active}`
          }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
              gap: '16px',
              alignItems: 'end'
            }}>
              {/* Type Filter */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  color: COLORS.text.muted,
                  marginBottom: '8px',
                  fontWeight: '600'
                }}>
                  SIGNAL TYPE
                </label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  {['ALL', 'BUY', 'SELL'].map(type => (
                    <button
                      key={type}
                      onClick={() => setFilterType(type as any)}
                      style={{
                        flex: 1,
                        padding: '10px',
                        background: filterType === type
                          ? `linear-gradient(135deg, ${type === 'BUY' ? COLORS.success : type === 'SELL' ? COLORS.danger : COLORS.info}, ${type === 'BUY' ? COLORS.success : type === 'SELL' ? COLORS.danger : COLORS.info}dd)`
                          : 'rgba(255, 255, 255, 0.05)',
                        border: `2px solid ${filterType === type ? (type === 'BUY' ? COLORS.success : type === 'SELL' ? COLORS.danger : COLORS.info) : COLORS.border.default}`,
                        borderRadius: '8px',
                        color: filterType === type ? '#000' : COLORS.text.secondary,
                        fontSize: '12px',
                        fontWeight: '700',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      {type === 'ALL' ? 'ALL' : type === 'BUY' ? 'BUY' : 'SELL'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sort By */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  color: COLORS.text.muted,
                  marginBottom: '8px',
                  fontWeight: '600'
                }}>
                  SORT BY
                </label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    color: COLORS.text.primary,
                    fontSize: '13px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    outline: 'none'
                  }}
                >
                  <option value="confidence">By Confidence Score</option>
                  <option value="strength">By Strength Level</option>
                  <option value="time">By Time</option>
                </select>
              </div>

              {/* Min Confidence */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  color: COLORS.text.muted,
                  marginBottom: '8px',
                  fontWeight: '600'
                }}>
                  MIN. CONFIDENCE: {minConfidence}%
                </label>
                <div style={{ position: 'relative', paddingTop: '8px' }}>
                  <input
                    type="range"
                    min="0"
                    max="95"
                    step="5"
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                    style={{
                      width: '100%',
                      height: '8px',
                      borderRadius: '4px',
                      background: `linear-gradient(to right, ${COLORS.premium} 0%, ${COLORS.premium} ${(minConfidence / 95) * 100}%, ${COLORS.border.default} ${(minConfidence / 95) * 100}%, ${COLORS.border.default} 100%)`,
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
                      width: 20px;
                      height: 20px;
                      border-radius: 50%;
                      background: ${COLORS.premium};
                      cursor: pointer;
                      box-shadow: 0 0 10px ${COLORS.premium}80;
                    }
                    input[type="range"]::-moz-range-thumb {
                      width: 20px;
                      height: 20px;
                      border-radius: 50%;
                      background: ${COLORS.premium};
                      cursor: pointer;
                      border: none;
                      box-shadow: 0 0 10px ${COLORS.premium}80;
                    }
                  `}</style>
                </div>
              </div>

              {/* Strategy Filter */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  color: COLORS.text.muted,
                  marginBottom: '8px',
                  fontWeight: '600'
                }}>
                  STRATEGY
                </label>
                <select
                  value={strategyFilter}
                  onChange={(e) => setStrategyFilter(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    color: COLORS.text.primary,
                    fontSize: '13px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    outline: 'none'
                  }}
                >
                  <option value="ALL">All Strategies</option>
                  {uniqueStrategies.map(strategy => (
                    <option key={strategy} value={strategy}>
                      {strategy.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
              </div>

              {/* Search */}
              <div>
                <label style={{
                  display: 'block',
                  fontSize: '12px',
                  color: COLORS.text.muted,
                  marginBottom: '8px',
                  fontWeight: '600'
                }}>
                  SEARCH
                </label>
                <input
                  type="text"
                  placeholder="Search coin..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: `2px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    color: COLORS.text.primary,
                    fontSize: '13px',
                    fontWeight: '600',
                    outline: 'none'
                  }}
                />
              </div>
            </div>

            <div style={{
              marginTop: '16px',
              padding: '12px',
              background: `${COLORS.info}10`,
              border: `1px solid ${COLORS.info}30`,
              borderRadius: '8px',
              fontSize: '13px',
              color: COLORS.text.secondary
            }}>
              <strong style={{ color: COLORS.info }}>{filteredSignals.length}</strong> signals displayed
              {' • '}
              Last update: {new Date(data?.lastUpdate || '').toLocaleTimeString('en-US')}
            </div>
          </div>

          {/* Signals Grid */}
          {filteredSignals.length === 0 ? (
            <div style={{
              background: 'rgba(26, 26, 26, 0.95)',
              backdropFilter: 'blur(30px)',
              borderRadius: '16px',
              padding: '64px 32px',
              textAlign: 'center',
              border: `1px solid ${COLORS.border.active}`
            }}>
              <Icons.AlertTriangle style={{
                width: '48px',
                height: '48px',
                color: COLORS.warning,
                margin: '0 auto 16px'
              }} />
              <h3 style={{
                fontSize: '20px',
                fontWeight: 'bold',
                color: COLORS.text.primary,
                marginBottom: '8px'
              }}>
                No Signals Found
              </h3>
              <p style={{
                fontSize: '14px',
                color: COLORS.text.secondary
              }}>
                No signals found matching your filters. Adjust filters or wait for new signals.
              </p>
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
              gap: '20px'
            }}>
              {filteredSignals.map(signal => {
                const isLong = signal.type === 'BUY';
                const cardColor = isLong ? COLORS.success : COLORS.danger;

                return (
                  <div
                    key={signal.id}
                    onClick={() => handleSignalClick(signal)}
                    style={{
                      background: `linear-gradient(135deg, ${cardColor}08 0%, rgba(26, 26, 26, 0.98) 100%)`,
                      backdropFilter: 'blur(30px)',
                      border: `2px solid ${cardColor}60`,
                      borderRadius: '16px',
                      padding: '24px',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      position: 'relative',
                      overflow: 'hidden'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-4px)';
                      e.currentTarget.style.boxShadow = `0 16px 48px ${cardColor}40`;
                      e.currentTarget.style.borderColor = `${cardColor}`;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                      e.currentTarget.style.borderColor = `${cardColor}60`;
                    }}
                  >
                    {/* Decorative corner */}
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '60px',
                      height: '60px',
                      background: `linear-gradient(135deg, ${cardColor}40, transparent)`,
                      borderRadius: '0 16px 0 100%'
                    }} />

                    {/* Header */}
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                      marginBottom: '16px'
                    }}>
                      <div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: 'bold',
                          color: COLORS.text.primary,
                          marginBottom: '4px'
                        }}>
                          {signal.symbol.replace('USDT', '')}
                          <span style={{
                            fontSize: '14px',
                            color: COLORS.text.muted,
                            fontWeight: '400'
                          }}>
                            /USDT
                          </span>
                        </div>
                        <div style={{
                          display: 'inline-block',
                          padding: '4px 12px',
                          background: cardColor,
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: '700',
                          color: '#000'
                        }}>
                          {signal.type === 'BUY' ? 'BUY' : 'SELL'}
                        </div>
                      </div>

                      <div style={{
                        textAlign: 'right'
                      }}>
                        <div style={{
                          fontSize: '20px',
                          fontWeight: 'bold',
                          color: COLORS.text.primary,
                          fontFamily: 'monospace'
                        }}>
                          ${signal.price < 1 ? signal.price.toFixed(6) : signal.price.toFixed(2)}
                        </div>
                      </div>
                    </div>

                    {/* Stats */}
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '12px',
                      marginBottom: '16px'
                    }}>
                      <div style={{
                        padding: '12px',
                        background: 'rgba(255, 255, 255, 0.03)',
                        borderRadius: '8px'
                      }}>
                        <div style={{
                          fontSize: '11px',
                          color: COLORS.text.muted,
                          marginBottom: '4px',
                          fontWeight: '600'
                        }}>
                          CONFIDENCE
                        </div>
                        <div style={{
                          fontSize: '20px',
                          fontWeight: 'bold',
                          color: cardColor,
                          fontFamily: 'monospace'
                        }}>
                          {signal.confidence}%
                        </div>
                      </div>

                      <div style={{
                        padding: '12px',
                        background: 'rgba(255, 255, 255, 0.03)',
                        borderRadius: '8px'
                      }}>
                        <div style={{
                          fontSize: '11px',
                          color: COLORS.text.muted,
                          marginBottom: '4px',
                          fontWeight: '600'
                        }}>
                          POWER LEVEL
                        </div>
                        <div style={{
                          display: 'flex',
                          gap: '3px',
                          marginTop: '6px'
                        }}>
                          {Array.from({ length: 10 }).map((_, i) => (
                            <div
                              key={i}
                              style={{
                                flex: 1,
                                height: '8px',
                                background: i < signal.strength ? cardColor : 'rgba(255, 255, 255, 0.1)',
                                borderRadius: '2px'
                              }}
                            />
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* Strategy & Targets */}
                    <div style={{
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.02)',
                      borderRadius: '8px',
                      border: `1px solid ${COLORS.border.default}`
                    }}>
                      <div style={{
                        fontSize: '11px',
                        color: COLORS.text.muted,
                        marginBottom: '8px',
                        fontWeight: '600'
                      }}>
                        STRATEGY
                      </div>
                      <div style={{
                        fontSize: '13px',
                        color: COLORS.text.primary,
                        fontWeight: '600',
                        marginBottom: '12px'
                      }}>
                        {signal.strategy.replace(/_/g, ' ')}
                      </div>

                      {signal.targets && signal.targets.length > 0 && (
                        <>
                          <div style={{
                            fontSize: '11px',
                            color: COLORS.text.muted,
                            marginBottom: '6px',
                            fontWeight: '600'
                          }}>
                            TARGETS
                          </div>
                          <div style={{
                            display: 'flex',
                            gap: '8px'
                          }}>
                            {signal.targets.map((target, i) => (
                              <div
                                key={i}
                                style={{
                                  flex: 1,
                                  padding: '6px',
                                  background: `${cardColor}15`,
                                  border: `1px solid ${cardColor}40`,
                                  borderRadius: '6px',
                                  fontSize: '11px',
                                  fontWeight: '600',
                                  color: cardColor,
                                  textAlign: 'center',
                                  fontFamily: 'monospace'
                                }}
                              >
                                ${parseFloat(target) < 1 ? parseFloat(target).toFixed(6) : parseFloat(target).toFixed(2)}
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>

                    {/* Time */}
                    <div style={{
                      marginTop: '12px',
                      fontSize: '11px',
                      color: COLORS.text.muted,
                      textAlign: 'right',
                      fontFamily: 'monospace'
                    }}>
                      {new Date(signal.timestamp).toLocaleString('en-US')}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Signal Detail Modal */}
        {showDetailModal && selectedSignal && (
          <div
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0, 0, 0, 0.92)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 9999,
              padding: '20px',
              backdropFilter: 'blur(10px)'
            }}
            onClick={() => setShowDetailModal(false)}
          >
            <div
              style={{
                background: `linear-gradient(145deg, rgba(26, 26, 26, 0.98), rgba(10, 10, 10, 0.98))`,
                border: `2px solid ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}`,
                borderRadius: '20px',
                maxWidth: '800px',
                width: '100%',
                maxHeight: '90vh',
                overflow: 'auto',
                boxShadow: `0 20px 60px ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}40`
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div style={{
                background: `linear-gradient(135deg, ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}20, transparent)`,
                padding: '32px',
                borderBottom: `1px solid ${COLORS.border.active}`
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  marginBottom: '16px'
                }}>
                  <div>
                    <h2 style={{
                      fontSize: '32px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      margin: '0 0 8px 0'
                    }}>
                      {selectedSignal.symbol.replace('USDT', '')}/USDT
                    </h2>
                    <div style={{
                      display: 'inline-block',
                      padding: '8px 16px',
                      background: selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger,
                      borderRadius: '8px',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: '#000'
                    }}>
                      {selectedSignal.type === 'BUY' ? 'BUY SIGNAL' : 'SELL SIGNAL'}
                    </div>
                  </div>

                  <button
                    onClick={() => setShowDetailModal(false)}
                    style={{
                      background: 'transparent',
                      border: `2px solid ${COLORS.border.active}`,
                      color: COLORS.text.muted,
                      width: '40px',
                      height: '40px',
                      borderRadius: '8px',
                      fontSize: '20px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = COLORS.danger;
                      e.currentTarget.style.borderColor = COLORS.danger;
                      e.currentTarget.style.color = '#000';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.borderColor = COLORS.border.active;
                      e.currentTarget.style.color = COLORS.text.muted;
                    }}
                  >
                    ✕
                  </button>
                </div>

                <div style={{
                  fontSize: '28px',
                  fontWeight: 'bold',
                  color: COLORS.text.primary,
                  fontFamily: 'monospace'
                }}>
                  ${selectedSignal.price < 1 ? selectedSignal.price.toFixed(6) : selectedSignal.price.toFixed(2)}
                </div>
              </div>

              {/* Modal Content */}
              <div style={{ padding: '32px' }}>
                {/* Stats Grid */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '16px',
                  marginBottom: '24px'
                }}>
                  <div style={{
                    padding: '20px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '12px',
                    border: `1px solid ${COLORS.border.default}`
                  }}>
                    <div style={{
                      fontSize: '12px',
                      color: COLORS.text.muted,
                      marginBottom: '8px',
                      fontWeight: '600'
                    }}>
                      GÜVENİLİRLİK
                    </div>
                    <div style={{
                      fontSize: '32px',
                      fontWeight: 'bold',
                      color: selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger,
                      fontFamily: 'monospace'
                    }}>
                      {selectedSignal.confidence}%
                    </div>
                  </div>

                  <div style={{
                    padding: '20px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '12px',
                    border: `1px solid ${COLORS.border.default}`
                  }}>
                    <div style={{
                      fontSize: '12px',
                      color: COLORS.text.muted,
                      marginBottom: '8px',
                      fontWeight: '600'
                    }}>
                      GÜÇ SEVİYESİ
                    </div>
                    <div style={{
                      fontSize: '32px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      fontFamily: 'monospace'
                    }}>
                      {selectedSignal.strength}/10
                    </div>
                  </div>

                  <div style={{
                    padding: '20px',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '12px',
                    border: `1px solid ${COLORS.border.default}`
                  }}>
                    <div style={{
                      fontSize: '12px',
                      color: COLORS.text.muted,
                      marginBottom: '8px',
                      fontWeight: '600'
                    }}>
                      STRATEJİ
                    </div>
                    <div style={{
                      fontSize: '14px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary
                    }}>
                      {selectedSignal.strategy.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>

                {/* Targets */}
                {selectedSignal.targets && selectedSignal.targets.length > 0 && (
                  <div style={{
                    padding: '24px',
                    background: `linear-gradient(135deg, ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}10, rgba(255, 255, 255, 0.02))`,
                    border: `1px solid ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}40`,
                    borderRadius: '12px',
                    marginBottom: '24px'
                  }}>
                    <h3 style={{
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      marginBottom: '16px'
                    }}>
                      🎯 Target Levels
                    </h3>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                      gap: '12px'
                    }}>
                      {selectedSignal.targets.map((target, i) => (
                        <div
                          key={i}
                          style={{
                            padding: '16px',
                            background: `${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}15`,
                            border: `2px solid ${selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger}`,
                            borderRadius: '10px',
                            textAlign: 'center'
                          }}
                        >
                          <div style={{
                            fontSize: '12px',
                            color: COLORS.text.muted,
                            marginBottom: '6px',
                            fontWeight: '600'
                          }}>
                            TARGET {i + 1}
                          </div>
                          <div style={{
                            fontSize: '18px',
                            fontWeight: 'bold',
                            color: selectedSignal.type === 'BUY' ? COLORS.success : COLORS.danger,
                            fontFamily: 'monospace'
                          }}>
                            ${parseFloat(target) < 1 ? parseFloat(target).toFixed(6) : parseFloat(target).toFixed(2)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Reasoning */}
                {selectedSignal.reasoning && (
                  <div style={{
                    padding: '24px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '12px',
                    marginBottom: '24px'
                  }}>
                    <h3 style={{
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      marginBottom: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <Icons.Bot style={{ width: '20px', height: '20px', color: COLORS.cyan }} />
                      AI Analiz
                    </h3>
                    <p style={{
                      fontSize: '14px',
                      color: COLORS.text.secondary,
                      lineHeight: '1.8',
                      margin: 0,
                      whiteSpace: 'pre-wrap'
                    }}>
                      {selectedSignal.reasoning}
                    </p>
                  </div>
                )}

                {/* Binance Link */}
                <a
                  href={`https://www.binance.com/tr/futures/${selectedSignal.symbol}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    display: 'block',
                    padding: '16px',
                    background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                    borderRadius: '12px',
                    textAlign: 'center',
                    color: '#000',
                    fontSize: '16px',
                    fontWeight: '700',
                    textDecoration: 'none',
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${COLORS.warning}60`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  📊 View on Binance
                </a>
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
              zIndex: 9999,
              padding: '20px',
              backdropFilter: 'blur(10px)'
            }}
            onClick={() => setShowLogicModal(false)}
          >
            <div
              style={{
                background: `linear-gradient(145deg, rgba(26, 26, 26, 0.98), rgba(10, 10, 10, 0.98))`,
                border: `2px solid ${COLORS.premium}`,
                borderRadius: '20px',
                maxWidth: '900px',
                width: '100%',
                maxHeight: '90vh',
                overflow: 'auto',
                boxShadow: `0 0 60px ${COLORS.premium}80`
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}15)`,
                padding: '32px',
                borderBottom: `2px solid ${COLORS.premium}`,
                position: 'sticky',
                top: 0,
                zIndex: 10,
                backdropFilter: 'blur(10px)'
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px'
                  }}>
                    <Icons.Lightbulb style={{
                      width: '32px',
                      height: '32px',
                      color: COLORS.premium
                    }} />
                    <h2 style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: COLORS.text.primary,
                      margin: 0
                    }}>
                      Trading Signals LOGIC
                    </h2>
                  </div>

                  <button
                    onClick={() => setShowLogicModal(false)}
                    style={{
                      background: 'transparent',
                      border: `2px solid ${COLORS.border.active}`,
                      color: COLORS.text.muted,
                      width: '40px',
                      height: '40px',
                      borderRadius: '8px',
                      fontSize: '20px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = COLORS.danger;
                      e.currentTarget.style.borderColor = COLORS.danger;
                      e.currentTarget.style.color = '#000';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.borderColor = COLORS.border.active;
                      e.currentTarget.style.color = COLORS.text.muted;
                    }}
                  >
                    ✕
                  </button>
                </div>
              </div>

              {/* Modal Content */}
              <div style={{ padding: '32px' }}>
                {/* Overview */}
                <div style={{ marginBottom: '32px' }}>
                  <h3 style={{
                    fontSize: '22px',
                    fontWeight: 'bold',
                    color: COLORS.premium,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <Icons.Bot style={{ width: '24px', height: '24px' }} />
                    Overview
                  </h3>
                  <p style={{
                    fontSize: '15px',
                    color: COLORS.text.secondary,
                    lineHeight: '1.8',
                    margin: 0
                  }}>
                    AI Trading Signals page generates real-time BUY/SELL signals by performing technical analysis with Python Ta-Lib library. Each signal is created by analyzing RSI, MACD, Bollinger Bands and other technical indicators. The system tracks past performance with AI memory system and continuously improves signal quality.
                  </p>
                </div>

                {/* Key Features */}
                <div style={{ marginBottom: '32px' }}>
                  <h3 style={{
                    fontSize: '22px',
                    fontWeight: 'bold',
                    color: COLORS.premium,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <Icons.Target style={{ width: '24px', height: '24px' }} />
                    Key Features
                  </h3>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {[
                      { name: 'Ta-Lib Technical Analysis', desc: 'Professional technical analysis with Python Ta-Lib library. RSI, MACD, Bollinger Bands and 50+ indicators.' },
                      { name: 'AI Memory System', desc: 'Tracks past signals, calculates success rate and optimizes signal quality.' },
                      { name: 'Multi-Strategy', desc: 'Momentum Breakout, Volume Surge, Downtrend Reversal and Technical Analysis strategies.' },
                      { name: 'Confidence Scores', desc: 'Reliability score from 0-100 for each signal. Dynamic adjustment with AI learning.' },
                      { name: 'Power Levels', desc: 'Power level from 1-10. Higher power = more reliable signal.' },
                      { name: 'Target Levels', desc: '2-3 target level suggestions for each signal. Risk/Reward ratio calculated.' },
                      { name: 'Auto-Refresh', desc: 'Signals automatically update every 10 seconds. Real-time data feed.' },
                      { name: 'Advanced Filters', desc: 'Customize signals with signal type, strategy, minimum confidence and search filters.' }
                    ].map((feature, index) => (
                      <div
                        key={index}
                        style={{
                          background: 'rgba(255, 255, 255, 0.02)',
                          border: `1px solid ${COLORS.border.default}`,
                          borderRadius: '10px',
                          padding: '16px',
                          transition: 'all 0.3s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = COLORS.premium;
                          e.currentTarget.style.transform = 'translateX(8px)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = COLORS.border.default;
                          e.currentTarget.style.transform = 'translateX(0)';
                        }}
                      >
                        <div style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '12px'
                        }}>
                          <div style={{
                            background: `linear-gradient(135deg, ${COLORS.premium}30, ${COLORS.warning}30)`,
                            padding: '8px 12px',
                            borderRadius: '6px',
                            fontSize: '14px',
                            fontWeight: 'bold',
                            color: COLORS.premium,
                            minWidth: '32px',
                            textAlign: 'center'
                          }}>
                            {index + 1}
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{
                              fontSize: '15px',
                              fontWeight: '600',
                              color: COLORS.text.primary,
                              marginBottom: '6px'
                            }}>
                              {feature.name}
                            </div>
                            <div style={{
                              fontSize: '14px',
                              color: COLORS.text.secondary,
                              lineHeight: '1.6'
                            }}>
                              {feature.desc}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* How to Use */}
                <div style={{ marginBottom: '32px' }}>
                  <h3 style={{
                    fontSize: '22px',
                    fontWeight: 'bold',
                    color: COLORS.premium,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <Icons.BarChart3 style={{ width: '24px', height: '24px' }} />
                    How to Use?
                  </h3>
                  <div style={{ display: 'grid', gap: '16px' }}>
                    {[
                      { step: '1', title: 'Review Signals', desc: 'View all active signals on the main page. Pay attention to BUY or SELL signals.' },
                      { step: '2', title: 'Check Confidence and Power', desc: 'Each signal has a confidence score (0-100%) and power level (1-10). Higher scores = more reliable.' },
                      { step: '3', title: 'Use Filters', desc: 'Find desired signals with signal type, strategy, minimum confidence and search filters.' },
                      { step: '4', title: 'Detailed Analysis', desc: 'Click on a signal to view full analysis, target levels and AI explanation.' },
                      { step: '5', title: 'Risk Management', desc: 'Consider target levels and stop-loss suggestions for each signal. Manage risk.' }
                    ].map((item, index) => (
                      <div
                        key={index}
                        style={{
                          display: 'flex',
                          gap: '16px',
                          alignItems: 'flex-start'
                        }}
                      >
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
                          flexShrink: 0
                        }}>
                          {item.step}
                        </div>
                        <div>
                          <div style={{
                            fontSize: '16px',
                            fontWeight: '600',
                            color: COLORS.text.primary,
                            marginBottom: '6px'
                          }}>
                            {item.title}
                          </div>
                          <div style={{
                            fontSize: '14px',
                            color: COLORS.text.secondary,
                            lineHeight: '1.6'
                          }}>
                            {item.desc}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Important Notes */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                  border: `2px solid ${COLORS.warning}`,
                  borderRadius: '12px',
                  padding: '24px'
                }}>
                  <h3 style={{
                    fontSize: '20px',
                    fontWeight: 'bold',
                    color: COLORS.warning,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                    Important Notes
                  </h3>
                  <ul style={{
                    margin: 0,
                    paddingLeft: '20px',
                    color: COLORS.text.secondary,
                    fontSize: '14px',
                    lineHeight: '1.8'
                  }}>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>White Hat Compliant:</strong> All signals are generated with real data from Binance public API. No manipulation.
                    </li>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>Auto-Update:</strong> Signals automatically update every 10 seconds.
                    </li>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>Ta-Lib Analysis:</strong> Technical analysis performed with professional Python Ta-Lib library.
                    </li>
                    <li style={{ marginBottom: '8px' }}>
                      <strong style={{ color: COLORS.text.primary }}>AI Learning:</strong> System tracks past signals and improves success rate.
                    </li>
                    <li>
                      <strong style={{ color: COLORS.text.primary }}>For Educational Purposes:</strong> These signals are not investment advice. Do your own research and trade responsibly.
                    </li>
                  </ul>
                </div>
              </div>

              {/* Modal Footer */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.02)',
                padding: '20px 32px',
                borderTop: `1px solid ${COLORS.border.default}`,
                textAlign: 'center',
                position: 'sticky',
                bottom: 0,
                backdropFilter: 'blur(10px)'
              }}>
                <p style={{
                  fontSize: '13px',
                  color: COLORS.text.secondary,
                  margin: 0
                }}>
                  AI Trading Signals - Professional Technical Analysis with Ta-Lib
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </PWAProvider>
  );
}
