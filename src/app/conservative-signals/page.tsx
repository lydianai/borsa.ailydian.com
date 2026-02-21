'use client';

/**
 * 🎯 MUHAFAZAKÂR ALIM SİNYALLERİ SAYFASI
 * Ultra-sıkı alım sinyalleri - TÜM koşullar karşılanmalı
 * Sarı çerçeve vurgusu | 15 dakikalık tarama aralığı
 * İşlem başına maksimum %2 risk | Maksimum 5x kaldıraç
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { calculateTop10, isTop10 as checkTop10 } from '@/lib/top10-helper';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { COLORS } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface ConservativeSignal {
  symbol: string;
  price: number;
  changePercent24h: number;
  volume24h: number;
  signal: string;
  confidence: number;
  reason: string;
  targets?: number[];
  stopLoss?: number;
  indicators?: Record<string, number>;
  highlightYellow: boolean;
  priority: number;
  timestamp?: string;
}

export default function ConservativeSignalsPage() {
  const [signals, setSignals] = useState<ConservativeSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [countdown, setCountdown] = useState(900); // 15 minutes = 900 seconds
  const [stats, setStats] = useState<any>(null);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState<ConservativeSignal | null>(null);
  const [top10List, setTop10List] = useState<string[]>([]);
  const [notificationCount, setNotificationCount] = useState(0);
  const [previousSignalCount, setPreviousSignalCount] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [timeRange, setTimeRange] = useState<'ALL' | '5m' | '15m' | '1h' | '4h'>('ALL');
  const [minConfidence, setMinConfidence] = useState(0);
  const [minPriority, setMinPriority] = useState(0);
  const [showLogicModal, setShowLogicModal] = useState(false);

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Global filters (synchronized across all pages)
  const { timeframe, sortBy } = useGlobalFilters();

  // Request notification permission on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission();
      }
    }

    // Load notification count from localStorage
    const savedCount = localStorage.getItem('conservative_notification_count');
    if (savedCount) {
      setNotificationCount(parseInt(savedCount));
    }
  }, []);

  // Clear notification count when user visits this page
  useEffect(() => {
    // After 2 seconds on the page, clear notification count
    const timer = setTimeout(() => {
      localStorage.setItem('conservative_notification_count', '0');
      setNotificationCount(0);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const fetchSignals = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/conservative-signals');
      const result = await response.json();
      if (result.success) {
        const newSignals = result.data.signals || [];
        const newSignalCount = newSignals.length;

        // Yeni sinyal geldi mi kontrol et
        if (previousSignalCount > 0 && newSignalCount > previousSignalCount) {
          const newSignalsCount = newSignalCount - previousSignalCount;

          // LocalStorage'da notification count güncelle
          const currentCount = parseInt(localStorage.getItem('conservative_notification_count') || '0');
          const updatedCount = currentCount + newSignalsCount;
          localStorage.setItem('conservative_notification_count', updatedCount.toString());
          setNotificationCount(updatedCount);

          // Browser notification göster
          if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
            new Notification('🎯 Yeni Muhafazakâr Alım Sinyali!', {
              body: `${newSignalsCount} yeni sinyal tespit edildi. Toplam: ${newSignalCount}`,
              icon: '/icon-192x192.png',
              badge: '/icon-96x96.png',
              tag: 'conservative-signal',
              requireInteraction: true,
            });
          }
        }

        setPreviousSignalCount(newSignalCount);
        setSignals(newSignals);
        setStats(result.data.stats || null);
        if (result.data.stats?.nextScanIn) {
          setCountdown(result.data.stats.nextScanIn);
        }
      }
    } catch (error) {
      console.error('[Conservative Signals] Fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();

    // Countdown timer
    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchSignals();
          return 900; // Reset to 15 minutes
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdownInterval);
  }, []);

  // Fetch TOP 10 (background)
  useEffect(() => {
    const fetchTop10 = async () => {
      try {
        const res = await fetch('/api/binance/futures');
        const data = await res.json();
        if (data.success) {
          const top10 = calculateTop10(data.data.all);
          setTop10List(top10);
        }
      } catch (err) {
        console.error('[TOP 10] fetch error:', err);
      }
    };
    fetchTop10();
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatPrice = (price: number) => {
    const p = price ?? 0;
    if (p >= 1000) return `$${p.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
    if (p >= 1) return `$${p.toFixed(2)}`;
    if (p >= 0.01) return `$${p.toFixed(4)}`;
    return `$${p.toFixed(6)}`;
  };

  const filteredSignals = signals.filter((signal) => {
    const matchesSearch = signal.symbol.toLowerCase().includes(searchTerm.toLowerCase());

    // Time range filter
    let matchesTimeRange = true;
    if (timeRange !== 'ALL') {
      const signalTime = new Date(signal.timestamp || Date.now()).getTime();
      const now = Date.now();
      const timeRanges: Record<string, number> = {
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
      };
      matchesTimeRange = (now - signalTime) <= timeRanges[timeRange];
    }

    // Confidence and priority filters
    const matchesConfidence = signal.confidence >= minConfidence;
    const matchesPriority = signal.priority >= minPriority;

    return matchesSearch && matchesTimeRange && matchesConfidence && matchesPriority;
  });

  const formatVolume = (volume: number) => {
    const v = volume ?? 0;
    if (v >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(2)}B`;
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`;
    if (v >= 1_000) return `${(v / 1_000).toFixed(2)}K`;
    return v.toFixed(2);
  };

  if (loading && signals.length === 0) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <SharedSidebar
        currentPage="conservative-signals"
        notificationCounts={{
          conservative: notificationCount
        }}
      />

      {/* Main Content */}
      <div className="dashboard-main" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
        {/* Page Header with MANTIK Button */}
        <div style={{ margin: '16px 24px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Icons.Shield style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                Konservatif Sinyaller
              </h1>
            </div>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
              Düşük Riskli Yatırım Sinyalleri - Güvenli Strateji Önerileri
            </p>
          </div>

          {/* MANTIK Button - Responsive */}
          <div>
            <style>{`
              @media (max-width: 768px) {
                .mantik-button-conservative {
                  padding: 10px 20px !important;
                  fontSize: 13px !important;
                  height: 42px !important;
                }
                .mantik-button-conservative svg {
                  width: 18px !important;
                  height: 18px !important;
                }
              }
              @media (max-width: 480px) {
                .mantik-button-conservative {
                  padding: 8px 16px !important;
                  fontSize: 12px !important;
                  height: 40px !important;
                }
                .mantik-button-conservative svg {
                  width: 16px !important;
                  height: 16px !important;
                }
              }
            `}</style>
            <button
              onClick={() => setShowLogicModal(true)}
              className="mantik-button-conservative"
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

        {/* Stats Bar */}
        {stats && (
          <div style={{ padding: '20px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            <div className="neon-card" style={{ padding: '16px', background: `${COLORS.warning}0D`, border: `1px solid ${COLORS.warning}33` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Taranan Coin</span>
                <Icons.Layers style={{ width: '16px', height: '16px', color: COLORS.warning }} />
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.warning }}>{stats.totalCoinsScanned}</div>
            </div>

            <div className="neon-card" style={{ padding: '16px', background: `${COLORS.success}0D`, border: `1px solid ${COLORS.success}33` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>AL Sinyali</span>
                <Icons.TrendingUp style={{ width: '16px', height: '16px', color: COLORS.success }} />
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.success }}>{stats.buySignalsFound}</div>
            </div>

            <div className="neon-card" style={{ padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Başarı Oranı</span>
                <Icons.Percent style={{ width: '16px', height: '16px', color: COLORS.text.primary }} />
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary }}>{stats.scanRate}%</div>
            </div>

            <div className="neon-card" style={{ padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Ort. Güven</span>
                <Icons.Award style={{ width: '16px', height: '16px', color: COLORS.text.primary }} />
              </div>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary }}>{stats.avgConfidence}%</div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <main className="dashboard-content" style={{ padding: '20px' }}>
          {signals.length === 0 ? (
            <div className="neon-card" style={{ padding: '40px', textAlign: 'center' }}>
              <Icons.AlertTriangle style={{ width: '48px', height: '48px', color: COLORS.warning, margin: '0 auto 16px' }} />
              <h2 style={{ color: COLORS.warning, fontSize: '1.5rem', marginBottom: '8px' }}>
                ⚠️ HİÇBİR MUHAFAZAKÂR ALIM SİNYALİ YOK
              </h2>
              <p style={{ color: COLORS.text.secondary, fontSize: '14px', maxWidth: '600px', margin: '0 auto' }}>
                Şu anda TÜM koşulları karşılayan sinyal bulunamadı. Piyasa koşulları uygun değil.
                <br /><br />
                <strong style={{ color: COLORS.warning }}>Sabırlı olun!</strong> Kaliteli fırsatlar için beklemeye değer.
                <br />
                Sonraki tarama: <strong style={{ color: COLORS.success }}>{formatTime(countdown)}</strong>
              </p>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: '16px' }}>
              {filteredSignals.map((signal, index) => (
                <div
                  key={`${signal.symbol}-${index}`}
                  className="neon-card"
                  style={{
                    padding: '20px',
                    background: `${COLORS.warning}08`,
                    border: `2px solid ${COLORS.warning}`,
                    boxShadow: `0 0 20px ${COLORS.warning}4D`,
                    cursor: 'pointer',
                    transition: 'all 0.3s ease'
                  }}
                  onClick={() => setSelectedSignal(signal)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    e.currentTarget.style.boxShadow = `0 0 30px ${COLORS.warning}80`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = `0 0 20px ${COLORS.warning}4D`;
                  }}
                >
                  {/* Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '50%',
                        background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}CC)`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '20px',
                        fontWeight: 'bold',
                        color: COLORS.bg.primary
                      }}>
                        #{index + 1}
                      </div>
                      <div>
                        <h3 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: COLORS.warning, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                          {checkTop10(signal.symbol, top10List) && (
                            <span style={{
                              background: COLORS.warning,
                              color: COLORS.bg.primary,
                              fontSize: '10px',
                              fontWeight: '700',
                              padding: '3px 6px',
                              borderRadius: '4px',
                              letterSpacing: '0.5px',
                            }}>
                              TOP10
                            </span>
                          )}
                          {signal.symbol.replace('USDT', '')}/USDT
                        </h3>
                        <p style={{ color: COLORS.text.secondary, fontSize: '12px', margin: 0 }}>Binance Futures USDT-M</p>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{
                        background: `${COLORS.success}33`,
                        border: `1px solid ${COLORS.success}80`,
                        borderRadius: '8px',
                        padding: '8px 16px',
                        display: 'inline-block'
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '4px' }}>GÜVEN SKORU</div>
                        <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.success }}>
                          {signal.confidence}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Price Info */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                    <div>
                      <div style={{ color: COLORS.text.secondary, fontSize: '12px', marginBottom: '4px' }}>Mevcut Fiyat</div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary }}>
                        {formatPrice(signal.price)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.text.secondary, fontSize: '12px', marginBottom: '4px' }}>24s Değişim</div>
                      <div style={{
                        fontSize: '18px',
                        fontWeight: 'bold',
                        color: signal.changePercent24h >= 0 ? COLORS.success : COLORS.danger
                      }}>
                        {(signal.changePercent24h ?? 0) >= 0 ? '+' : ''}{(signal.changePercent24h ?? 0).toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.text.secondary, fontSize: '12px', marginBottom: '4px' }}>24s Hacim</div>
                      <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary }}>
                        {formatVolume(signal.volume24h)}
                      </div>
                    </div>
                    {signal.indicators?.stopLossPercent && (
                      <div>
                        <div style={{ color: COLORS.text.secondary, fontSize: '12px', marginBottom: '4px' }}>Max Risk</div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.danger }}>
                          -{(signal.indicators.stopLossPercent ?? 0).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Targets */}
                  {signal.targets && signal.targets.length > 0 && (
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ color: COLORS.warning, fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>
                        🎯 HEDEF FİYATLAR
                      </div>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {signal.targets.map((target, i) => (
                          <div
                            key={i}
                            style={{
                              padding: '8px 16px',
                              background: `${COLORS.success}1A`,
                              border: `1px solid ${COLORS.success}4D`,
                              borderRadius: '6px',
                              fontSize: '14px',
                              color: COLORS.success
                            }}
                          >
                            TP{i + 1}: {formatPrice(target)} (+{((((target ?? 0) - (signal.price ?? 0)) / (signal.price ?? 0)) * 100).toFixed(1)}%)
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Stop Loss */}
                  {signal.stopLoss && (
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{
                        padding: '12px',
                        background: `${COLORS.danger}1A`,
                        border: `1px solid ${COLORS.danger}4D`,
                        borderRadius: '6px'
                      }}>
                        <div style={{ color: COLORS.danger, fontSize: '14px', fontWeight: 'bold', marginBottom: '4px' }}>
                          🛡️ STOP LOSS
                        </div>
                        <div style={{ fontSize: '16px', color: COLORS.danger }}>
                          {formatPrice(signal.stopLoss)} (-{((((signal.price ?? 0) - (signal.stopLoss ?? 0)) / (signal.price ?? 0)) * 100).toFixed(1)}%)
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Indicators */}
                  {signal.indicators && (
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ color: COLORS.warning, fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>
                        📊 TEKNİK GÖSTERGELER
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '8px' }}>
                        {signal.indicators.rsi && (
                          <div style={{ padding: '8px', background: `${COLORS.text.primary}0D`, borderRadius: '6px' }}>
                            <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>RSI</div>
                            <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: 'bold' }}>
                              {(signal.indicators.rsi ?? 0).toFixed(1)}
                            </div>
                          </div>
                        )}
                        {signal.indicators.volumeRatio && (
                          <div style={{ padding: '8px', background: `${COLORS.text.primary}0D`, borderRadius: '6px' }}>
                            <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>Hacim</div>
                            <div style={{ color: COLORS.success, fontSize: '14px', fontWeight: 'bold' }}>
                              {(signal.indicators.volumeRatio ?? 0).toFixed(2)}x
                            </div>
                          </div>
                        )}
                        {signal.indicators.riskRewardRatio && (
                          <div style={{ padding: '8px', background: `${COLORS.text.primary}0D`, borderRadius: '6px' }}>
                            <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>R:R</div>
                            <div style={{ color: COLORS.warning, fontSize: '14px', fontWeight: 'bold' }}>
                              {(signal.indicators.riskRewardRatio ?? 0).toFixed(1)}:1
                            </div>
                          </div>
                        )}
                        {signal.indicators.leverageMax && (
                          <div style={{ padding: '8px', background: `${COLORS.text.primary}0D`, borderRadius: '6px' }}>
                            <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>Maks. Kaldıraç</div>
                            <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: 'bold' }}>
                              {signal.indicators.leverageMax}x
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Reason (collapsed by default) */}
                  <div style={{
                    padding: '12px',
                    background: `${COLORS.warning}0D`,
                    border: `1px solid ${COLORS.warning}33`,
                    borderRadius: '6px',
                    fontSize: '13px',
                    color: COLORS.text.secondary,
                    lineHeight: '1.6',
                    maxHeight: '100px',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis'
                  }}>
                    {signal.reason.split('\n')[0]}...
                    <div style={{ marginTop: '8px', color: COLORS.warning, fontSize: '12px', cursor: 'pointer' }}>
                      Detayları görmek için tıklayın →
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>

      {/* Signal Detail Modal */}
      {selectedSignal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `${COLORS.bg.primary}E6`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '20px'
          }}
          onClick={() => setSelectedSignal(null)}
        >
          <div
            className="neon-card"
            style={{
              maxWidth: '800px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              padding: '24px',
              background: COLORS.bg.secondary,
              border: `2px solid ${COLORS.warning}`,
              boxShadow: `0 0 40px ${COLORS.warning}80`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '2rem', color: COLORS.warning, margin: 0 }}>
                {selectedSignal.symbol.replace('USDT', '')}/USDT
              </h2>
              <button
                onClick={() => setSelectedSignal(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  padding: '8px'
                }}
              >
                <Icons.X style={{ width: '24px', height: '24px', color: COLORS.text.primary }} />
              </button>
            </div>

            <div style={{ whiteSpace: 'pre-wrap', color: COLORS.text.primary, fontSize: '14px', lineHeight: '1.8' }}>
              {selectedSignal.reason}
            </div>

            <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
              <button
                className="neon-button"
                style={{ flex: 1, padding: '12px', fontSize: '14px', background: `${COLORS.success}33`, border: `1px solid ${COLORS.success}` }}
                onClick={() => window.open(`https://www.binance.com/en/futures/${selectedSignal.symbol}`, '_blank')}
              >
                🚀 Binance'de Aç
              </button>
              <button
                className="neon-button"
                style={{ flex: 1, padding: '12px', fontSize: '14px' }}
                onClick={() => setSelectedSignal(null)}
              >
                Kapat
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Assistant */}
      <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />

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
                    Konservatif Sinyaller MANTIK
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
                  <Icons.Shield style={{ width: '24px', height: '24px' }} />
                  Genel Bakış
                </h3>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                  Konservatif Sinyaller sayfası, düşük riskli yatırım fırsatlarını seçici kriterlerle filtreler.
                  Sadece güvenli ve yüksek güvenilirlik skoruna sahip sinyaller gösterilir.
                </p>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                  Bu sayfa, sermaye korumasını öncelik olarak belirleyen yatırımcılar için tasarlanmıştır.
                  Her 15 dakikada bir tarama yapılır ve sadece tüm koşulları karşılayan sinyaller listelenir.
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
                    { name: 'Düşük Riskli Stratejiler', desc: 'Sadece kanıtlanmış ve güvenli stratejiler kullanılır. Risk minimizasyonu önceliktir.' },
                    { name: 'Konservatif Giriş Noktaları', desc: 'Giriş fiyatları muhafazakar seviyelerden belirlenir, agresif alımlar yapılmaz.' },
                    { name: 'Yüksek Güven Sadece', desc: 'Sadece %70+ güvenilirlik skoruna sahip sinyaller gösterilir.' },
                    { name: 'Sıkı Stop-Loss Seviyeleri', desc: 'Her sinyal için maksimum %2 risk limiti ile stop-loss belirlenir.' },
                    { name: 'Sermaye Koruma Odaklı', desc: 'Yüksek getiri yerine sermaye güvenliği ve istikrar öncelenir.' },
                    { name: 'Uzun Vadeli Yönelimli', desc: 'Kısa vadeli spekülatif hareketler yerine sağlam trendler takip edilir.' }
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
                        Risk Seviyesine Göre Filtreleyin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Sinyaller otomatik olarak risk seviyesine göre sıralanır. En düşük risk olanlar en üstte gösterilir.
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
                        Konservatif Stratejileri İnceleyin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Her sinyalin detayında hangi muhafazakar kriterleri karşıladığı açıklanır.
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
                        Stop-Loss Seviyelerini Kontrol Edin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Her sinyal için önerilen stop-loss seviyesi maksimum %2 risk ile belirlenmiştir.
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
                        Sermaye Güvenliğine Öncelik Verin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Yüksek getiri peşinde koşmak yerine sermayenizi korumayı hedefleyin. Küçük ama istikrarlı kazançlar uzun vadede daha değerlidir.
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
                    <strong style={{ color: COLORS.text.primary }}>15 Dakika Tarama:</strong> Sinyaller her 15 dakikada bir güncellenir. Hızlı sinyal yerine kaliteli sinyal önceliklidir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Maksimum %2 Risk:</strong> Her işlem için portföyünüzün maksimum %2'si riske atılır.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Maksimum 5x Kaldıraç:</strong> Yüksek kaldıraç kullanımı önerilmez, maksimum 5x ile sınırlı kalın.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Sıkı Filtreleme:</strong> Binlerce coin taranır ancak sadece çok az sinyal koşulları sağlar.
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
                Konservatif Sinyaller - Düşük Risk, Yüksek Güvenilirlik ile Güvenli Yatırım Stratejisi
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
