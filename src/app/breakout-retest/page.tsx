'use client';

/**
 * 🚀 BREAKOUT & RETEST - PREMIUM EDITION
 *
 * Modern premium UI ile geliştirilmiş breakout-retest sinyal sayfası
 * - Destek/Direnç kırılımlarını tespit eder
 * - Retest fırsatlarını yakalar
 * - Profesyonel risk yönetimi
 * - Real-time sinyal bildirimleri
 *
 * BEYAZ ŞAPKA: Sadece eğitim ve analiz amaçlı
 */

import { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { PWAProvider } from '@/components/PWAProvider';
import { COLORS } from '@/lib/colors';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';

interface BreakoutRetestSignal {
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
  direction?: string;
  highlightGreen: boolean;
  priority: number;
  timestamp?: string;
}

interface Stats {
  totalScanned: number;
  signalsFound: number;
  avgConfidence: string;
  nextScanIn: number;
  lastUpdate: string;
  longSignals?: number;
  shortSignals?: number;
  highConfidenceCount?: number;
  totalCoinsScanned?: number;
}

export default function BreakoutRetestPage() {
  const [signals, setSignals] = useState<BreakoutRetestSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(600);
  const [stats, setStats] = useState<Stats | null>(null);
  const [selectedSignal, setSelectedSignal] = useState<BreakoutRetestSignal | null>(null);
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [filterDirection, setFilterDirection] = useState<'ALL' | 'LONG' | 'SHORT'>('ALL');
  const [minConfidence, setMinConfidence] = useState(70);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'confidence' | 'volume' | 'change'>('confidence');
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          fetchSignals();
          return 600;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [minConfidence]);

  const fetchSignals = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/breakout-retest?minConfidence=${minConfidence}`);
      const result = await response.json();

      if (result.success) {
        const newSignals = result.data.signals || [];
        setSignals(newSignals);

        // Calculate additional stats
        const longSignals = newSignals.filter((s: BreakoutRetestSignal) => s.direction === 'LONG').length;
        const shortSignals = newSignals.filter((s: BreakoutRetestSignal) => s.direction === 'SHORT').length;
        const highConfidenceCount = newSignals.filter((s: BreakoutRetestSignal) => s.confidence >= 85).length;

        setStats({
          totalScanned: result.data.stats.totalScanned || 0,
          signalsFound: newSignals.length,
          avgConfidence: result.data.stats.avgConfidence || '0',
          nextScanIn: result.data.stats.nextScanIn || 600,
          lastUpdate: result.data.stats.lastUpdate || new Date().toISOString(),
          longSignals,
          shortSignals,
          highConfidenceCount,
          totalCoinsScanned: result.data.stats.totalScanned || 0
        });

        setCountdown(result.data.stats.nextScanIn || 600);
      } else {
        setError(result.error || 'Veri alınamadı');
      }
    } catch (err: any) {
      setError(err.message || 'Bağlantı hatası');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatPrice = (price: number) => {
    if (price >= 1000) return `$${price.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    if (price >= 0.01) return `$${price.toFixed(4)}`;
    return `$${price.toFixed(6)}`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1_000_000_000) return `$${(volume / 1_000_000_000).toFixed(2)}B`;
    if (volume >= 1_000_000) return `$${(volume / 1_000_000).toFixed(2)}M`;
    if (volume >= 1_000) return `$${(volume / 1_000).toFixed(2)}K`;
    return `$${volume.toFixed(2)}`;
  };

  // Filter and sort signals
  const filteredSignals = signals
    .filter(signal => {
      const matchesDirection = filterDirection === 'ALL' || signal.direction === filterDirection;
      const matchesSearch = signal.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      return matchesDirection && matchesSearch;
    })
    .sort((a, b) => {
      if (sortBy === 'confidence') return b.confidence - a.confidence;
      if (sortBy === 'volume') return b.volume24h - a.volume24h;
      return Math.abs(b.changePercent24h) - Math.abs(a.changePercent24h);
    });

  if (loading && signals.length === 0) {
    return (
      <PWAProvider>
        <div style={{ minHeight: '100vh', background: '#0A0A0A', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '60px', marginBottom: '24px', animation: 'pulse 2s infinite' }}>🚀</div>
            <div style={{ fontSize: '20px', color: '#FFFFFF', fontWeight: '600', marginBottom: '12px' }}>
              Breakout & Retest Sinyalleri Taranıyor...
            </div>
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)' }}>
              Piyasa analiz ediliyor, formasyonlar tespit ediliyor
            </div>
          </div>
        </div>
      </PWAProvider>
    );
  }

  if (error) {
    return (
      <PWAProvider>
        <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
          <SharedSidebar currentPage="breakout-retest" notificationCounts={notificationCounts} />
          <main style={{ padding: '80px 24px 24px 24px', maxWidth: '1400px', margin: '0 auto' }}>
            <div style={{ textAlign: 'center', padding: '60px 20px' }}>
              <div style={{ fontSize: '60px', marginBottom: '24px' }}>⚠️</div>
              <div style={{ fontSize: '24px', color: COLORS.danger, fontWeight: '700', marginBottom: '16px' }}>
                Hata Oluştu
              </div>
              <div style={{ fontSize: '16px', color: 'rgba(255,255,255,0.7)', marginBottom: '32px' }}>
                {error}
              </div>
              <button
                onClick={fetchSignals}
                style={{
                  padding: '14px 32px',
                  background: COLORS.premium,
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '12px',
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  boxShadow: `0 4px 20px ${COLORS.premium}40`
                }}
              >
                Yeniden Dene
              </button>
            </div>
          </main>
        </div>
      </PWAProvider>
    );
  }

  return (
    <PWAProvider>
      <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
        <SharedSidebar currentPage="breakout-retest" notificationCounts={notificationCounts} />

        <main style={{ padding: `${isLocalhost ? '116px' : '60px'} 24px 24px 24px`, maxWidth: '1600px', margin: '0 auto' }}>
          {/* Premium Header */}
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.premium}20 0%, ${COLORS.info}15 100%)`,
            backdropFilter: 'blur(40px)',
            border: `2px solid ${COLORS.premium}60`,
            borderRadius: '24px',
            padding: '40px',
            marginBottom: '32px',
            boxShadow: `0 20px 60px ${COLORS.premium}30`
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', flexWrap: 'wrap', gap: '20px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '16px',
                    background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '32px',
                    boxShadow: `0 8px 24px ${COLORS.premium}40`
                  }}>
                    🚀
                  </div>
                  <div>
                    <h1 style={{
                      fontSize: '42px',
                      fontWeight: '900',
                      background: 'linear-gradient(135deg, #FFFFFF 0%, #A78BFA 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      margin: 0,
                      marginBottom: '4px'
                    }}>
                      Breakout & Retest
                    </h1>
                    <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.8)', margin: 0 }}>
                      Profesyonel Destek/Direnç Kırılım Analizi
                    </p>
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <div>
                  <style>{`
                    @media (max-width: 768px) {
                      .mantik-button-breakout {
                        padding: 10px 20px !important;
                        fontSize: 13px !important;
                        height: 42px !important;
                      }
                      .mantik-button-breakout span {
                        fontSize: 18px !important;
                      }
                    }
                    @media (max-width: 480px) {
                      .mantik-button-breakout {
                        padding: 8px 16px !important;
                        fontSize: 12px !important;
                        height: 40px !important;
                      }
                      .mantik-button-breakout span {
                        fontSize: 16px !important;
                      }
                    }
                  `}</style>
                  <button
                    onClick={() => setShowLogicModal(true)}
                    className="mantik-button-breakout"
                    style={{
                      background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                      border: `2px solid ${COLORS.premium}80`,
                      borderRadius: '10px',
                      padding: '12px 24px',
                      color: '#000000',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      boxShadow: `0 4px 16px ${COLORS.premium}30`,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      height: '44px'
                    }}
                  >
                    <span style={{ fontSize: '18px' }}>🧠</span>
                    MANTIK
                  </button>
                </div>

                <div style={{
                  background: 'rgba(255,255,255,0.1)',
                  backdropFilter: 'blur(10px)',
                  border: '2px solid rgba(255,255,255,0.2)',
                  borderRadius: '12px',
                  padding: '12px 20px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
                    SONRAKİ GÜNCELLEME
                  </div>
                  <div style={{ fontSize: '20px', fontWeight: '900', color: COLORS.success }}>
                    ⏱️ {formatTime(countdown)}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stats Grid */}
          {stats && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
              marginBottom: '32px'
            }}>
              <StatCard
                icon="🎯"
                label="Taranan Coin"
                value={stats.totalCoinsScanned || stats.totalScanned}
                color={COLORS.info}
              />
              <StatCard
                icon="📊"
                label="Sinyal Bulundu"
                value={stats.signalsFound}
                color={COLORS.cyan}
              />
              <StatCard
                icon="📈"
                label="LONG Sinyali"
                value={stats.longSignals || 0}
                color={COLORS.success}
              />
              <StatCard
                icon="📉"
                label="SHORT Sinyali"
                value={stats.shortSignals || 0}
                color={COLORS.danger}
              />
              <StatCard
                icon="⭐"
                label="Yüksek Güven"
                value={stats.highConfidenceCount || 0}
                color={COLORS.warning}
              />
              <StatCard
                icon="💯"
                label="Ort. Güven"
                value={`${stats.avgConfidence}%`}
                color={COLORS.premium}
              />
            </div>
          )}

          {/* Filters */}
          <div style={{
            background: 'rgba(26, 26, 26, 0.95)',
            backdropFilter: 'blur(30px)',
            border: `2px solid ${COLORS.premium}30`,
            borderRadius: '20px',
            padding: '24px',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', alignItems: 'center' }}>
              {/* Direction Filter */}
              <div style={{ display: 'flex', gap: '8px' }}>
                {(['ALL', 'LONG', 'SHORT'] as const).map(dir => (
                  <button
                    key={dir}
                    onClick={() => setFilterDirection(dir)}
                    style={{
                      padding: '12px 24px',
                      background: filterDirection === dir ? COLORS.premium : 'rgba(255,255,255,0.05)',
                      border: `2px solid ${filterDirection === dir ? COLORS.premium : 'rgba(255,255,255,0.1)'}`,
                      borderRadius: '12px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.3s'
                    }}
                  >
                    {dir === 'ALL' ? `TÜM (${signals.length})` :
                     dir === 'LONG' ? `LONG (${stats?.longSignals || 0})` :
                     `SHORT (${stats?.shortSignals || 0})`}
                  </button>
                ))}
              </div>

              {/* Sort By */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
                <span style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)' }}>Sırala:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  style={{
                    padding: '10px 16px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '2px solid rgba(255,255,255,0.1)',
                    borderRadius: '10px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  <option value="confidence">Güven Skoru</option>
                  <option value="volume">Hacim</option>
                  <option value="change">Değişim %</option>
                </select>
              </div>

              {/* Min Confidence */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)' }}>Min Güven:</span>
                <select
                  value={minConfidence}
                  onChange={(e) => setMinConfidence(parseInt(e.target.value))}
                  style={{
                    padding: '10px 16px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '2px solid rgba(255,255,255,0.1)',
                    borderRadius: '10px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  <option value="60">60%</option>
                  <option value="70">70%</option>
                  <option value="75">75%</option>
                  <option value="80">80%</option>
                  <option value="85">85%</option>
                  <option value="90">90%</option>
                </select>
              </div>

              {/* Search */}
              <div style={{ flex: '1', minWidth: '250px', maxWidth: '400px' }}>
                <input
                  type="text"
                  placeholder="🔍 Coin ara (ör: BTC, ETH)..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px 20px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '2px solid rgba(255,255,255,0.1)',
                    borderRadius: '12px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                    fontWeight: '500',
                    outline: 'none'
                  }}
                />
              </div>
            </div>
          </div>

          {/* Signals Grid */}
          {filteredSignals.length === 0 ? (
            <div style={{
              background: 'rgba(26, 26, 26, 0.95)',
              backdropFilter: 'blur(30px)',
              border: `2px solid ${COLORS.warning}40`,
              borderRadius: '24px',
              padding: '80px 40px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '80px', marginBottom: '24px', opacity: 0.3 }}>🔍</div>
              <h2 style={{ fontSize: '28px', fontWeight: '800', color: COLORS.warning, marginBottom: '16px' }}>
                Sinyal Bulunamadı
              </h2>
              <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.6)', maxWidth: '600px', margin: '0 auto' }}>
                Breakout & Retest stratejisi sıkı doğrulama gerektirir. Kaliteli sinyaller için sabırla bekleyin.
              </p>
              <div style={{ marginTop: '32px', fontSize: '18px', color: COLORS.info }}>
                Sonraki tarama: <strong>{formatTime(countdown)}</strong>
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: '20px' }}>
              {filteredSignals.map((signal, index) => (
                <SignalCard
                  key={`${signal.symbol}-${index}`}
                  signal={signal}
                  onClick={() => setSelectedSignal(signal)}
                  formatPrice={formatPrice}
                  formatVolume={formatVolume}
                />
              ))}
            </div>
          )}
        </main>

        {/* Signal Detail Modal */}
        {selectedSignal && (
          <SignalModal
            signal={selectedSignal}
            onClose={() => setSelectedSignal(null)}
            formatPrice={formatPrice}
          />
        )}

        {/* Logic Modal */}
        {showLogicModal && (
          <LogicModal onClose={() => setShowLogicModal(false)} />
        )}
      </div>
    </PWAProvider>
  );
}

// Helper Components
function StatCard({ icon, label, value, color }: { icon: string; label: string; value: number | string; color: string }) {
  return (
    <div style={{
      background: `${color}10`,
      backdropFilter: 'blur(30px)',
      border: `2px solid ${color}40`,
      borderRadius: '16px',
      padding: '24px',
      boxShadow: `0 8px 24px ${color}20`,
      transition: 'all 0.3s'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <span style={{ fontSize: '32px' }}>{icon}</span>
        <span style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', fontWeight: '600' }}>
          {label}
        </span>
      </div>
      <div style={{ fontSize: '32px', fontWeight: '900', color }}>
        {value}
      </div>
    </div>
  );
}

function SignalCard({ signal, onClick, formatPrice, formatVolume }: any) {
  const isLong = signal.direction === 'LONG';
  const cardColor = isLong ? COLORS.success : COLORS.danger;

  return (
    <div
      onClick={onClick}
      style={{
        background: `linear-gradient(135deg, ${cardColor}08 0%, rgba(26, 26, 26, 0.98) 100%)`,
        backdropFilter: 'blur(30px)',
        border: `2px solid ${cardColor}60`,
        borderRadius: '20px',
        padding: '32px',
        boxShadow: `0 12px 40px ${cardColor}30`,
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        position: 'relative',
        overflow: 'hidden'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-4px)';
        e.currentTarget.style.boxShadow = `0 20px 60px ${cardColor}50`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = `0 12px 40px ${cardColor}30`;
      }}
    >
      {/* Decorative Corner */}
      <div style={{
        position: 'absolute',
        top: 0,
        right: 0,
        width: '120px',
        height: '120px',
        background: `linear-gradient(135deg, ${cardColor}20, transparent)`,
        borderRadius: '0 20px 0 100%'
      }} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '24px', position: 'relative' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '56px',
            height: '56px',
            borderRadius: '16px',
            background: `linear-gradient(135deg, ${cardColor}, ${isLong ? '#0a9f6b' : '#cc2020'})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '28px',
            boxShadow: `0 8px 24px ${cardColor}40`
          }}>
            {isLong ? '📈' : '📉'}
          </div>
          <div>
            <h3 style={{ fontSize: '28px', fontWeight: '900', color: cardColor, margin: 0, marginBottom: '4px' }}>
              {signal.symbol.replace('USDT', '')}/USDT
            </h3>
            <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', margin: 0 }}>
              {signal.direction} • Binance Futures
            </p>
          </div>
        </div>

        <div style={{
          background: `${cardColor}20`,
          border: `2px solid ${cardColor}80`,
          borderRadius: '12px',
          padding: '12px 24px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
            GÜVEN SKORU
          </div>
          <div style={{ fontSize: '28px', fontWeight: '900', color: cardColor }}>
            {signal.confidence}%
          </div>
        </div>
      </div>

      {/* Price Info Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '16px',
        marginBottom: '24px'
      }}>
        <InfoBox label="Giriş Fiyatı" value={formatPrice(signal.price)} />
        <InfoBox
          label="24s Değişim"
          value={`${signal.changePercent24h >= 0 ? '+' : ''}${signal.changePercent24h.toFixed(2)}%`}
          color={signal.changePercent24h >= 0 ? COLORS.success : COLORS.danger}
        />
        <InfoBox label="24s Hacim" value={formatVolume(signal.volume24h)} color={COLORS.cyan} />
        {signal.indicators?.riskRewardRatio && (
          <InfoBox label="R:R Oranı" value={`${signal.indicators.riskRewardRatio.toFixed(2)}:1`} color={COLORS.warning} />
        )}
      </div>

      {/* Targets */}
      {signal.targets && signal.targets.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ fontSize: '14px', fontWeight: '700', color: cardColor, marginBottom: '12px' }}>
            🎯 HEDEF FİYATLAR
          </div>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {signal.targets.map((target, i) => (
              <div key={i} style={{
                padding: '10px 16px',
                background: `${cardColor}15`,
                border: `1px solid ${cardColor}40`,
                borderRadius: '10px',
                fontSize: '13px',
                fontWeight: '600',
                color: cardColor
              }}>
                TP{i + 1}: {formatPrice(target)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stop Loss */}
      {signal.stopLoss && (
        <div style={{
          padding: '14px',
          background: `${COLORS.danger}15`,
          border: `1px solid ${COLORS.danger}40`,
          borderRadius: '10px',
          marginBottom: '20px'
        }}>
          <span style={{ fontSize: '14px', fontWeight: '700', color: COLORS.danger }}>
            🛡️ STOP LOSS: {formatPrice(signal.stopLoss)}
          </span>
        </div>
      )}

      {/* Preview Reason */}
      <div style={{
        padding: '16px',
        background: `${cardColor}10`,
        border: `1px solid ${cardColor}30`,
        borderRadius: '12px',
        fontSize: '14px',
        color: 'rgba(255,255,255,0.8)',
        lineHeight: '1.6',
        maxHeight: '100px',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {signal.reason.split('\n')[0]}...
        <div style={{
          marginTop: '12px',
          color: COLORS.info,
          fontSize: '13px',
          fontWeight: '600',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span>Tam analiz için tıklayın</span>
          <span>→</span>
        </div>
      </div>
    </div>
  );
}

function InfoBox({ label, value, color = '#FFFFFF' }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginBottom: '6px', fontWeight: '600' }}>
        {label}
      </div>
      <div style={{ fontSize: '18px', fontWeight: '800', color }}>
        {value}
      </div>
    </div>
  );
}

function SignalModal({ signal, onClose, _formatPrice }: any) {
  const isLong = signal.direction === 'LONG';
  const modalColor = isLong ? COLORS.success : COLORS.danger;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.92)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
        padding: '24px'
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
          backdropFilter: 'blur(20px)',
          border: `2px solid ${modalColor}60`,
          borderRadius: '24px',
          maxWidth: '900px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          boxShadow: `0 20px 60px ${modalColor}40`
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{
          background: `linear-gradient(135deg, ${modalColor}15, rgba(26, 26, 26, 0.95))`,
          padding: '32px',
          borderBottom: `2px solid ${modalColor}40`,
          position: 'sticky',
          top: 0,
          zIndex: 10
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ fontSize: '32px', fontWeight: '900', color: modalColor, margin: 0, marginBottom: '8px' }}>
                {signal.symbol.replace('USDT', '')}/USDT
              </h2>
              <p style={{ fontSize: '16px', color: 'rgba(255,255,255,0.7)', margin: 0 }}>
                {signal.direction} Sinyali • Güven Skoru: {signal.confidence}%
              </p>
            </div>
            <button
              onClick={onClose}
              style={{
                background: 'rgba(255,255,255,0.1)',
                border: '2px solid rgba(255,255,255,0.2)',
                borderRadius: '12px',
                width: '48px',
                height: '48px',
                color: '#FFFFFF',
                fontSize: '24px',
                cursor: 'pointer',
                transition: 'all 0.3s'
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div style={{ padding: '32px' }}>
          <div style={{
            whiteSpace: 'pre-wrap',
            color: '#FFFFFF',
            fontSize: '15px',
            lineHeight: '1.8',
            fontFamily: 'ui-monospace, monospace',
            background: 'rgba(0,0,0,0.3)',
            padding: '24px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.1)'
          }}>
            {signal.reason}
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '16px', marginTop: '32px' }}>
            <button
              onClick={() => window.open(`https://www.binance.com/en/futures/${signal.symbol}`, '_blank')}
              style={{
                flex: 1,
                padding: '16px',
                background: `linear-gradient(135deg, ${modalColor}, ${isLong ? '#0a9f6b' : '#cc2020'})`,
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: '700',
                cursor: 'pointer',
                boxShadow: `0 4px 20px ${modalColor}40`
              }}
            >
              🚀 Binance'de Aç
            </button>
            <button
              onClick={onClose}
              style={{
                flex: 1,
                padding: '16px',
                background: 'rgba(255,255,255,0.1)',
                color: '#FFFFFF',
                border: '2px solid rgba(255,255,255,0.2)',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: '700',
                cursor: 'pointer'
              }}
            >
              Kapat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function LogicModal({ onClose }: { onClose: () => void }) {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.92)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
        padding: '24px'
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
          backdropFilter: 'blur(20px)',
          border: `2px solid ${COLORS.premium}60`,
          borderRadius: '24px',
          maxWidth: '900px',
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          boxShadow: `0 20px 60px ${COLORS.premium}40`
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          background: `linear-gradient(135deg, ${COLORS.premium}15, rgba(26, 26, 26, 0.95))`,
          padding: '32px',
          borderBottom: `2px solid ${COLORS.premium}40`,
          position: 'sticky',
          top: 0,
          zIndex: 10
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '40px' }}>🧠</span>
              <h2 style={{ fontSize: '32px', fontWeight: '900', color: COLORS.premium, margin: 0 }}>
                Breakout & Retest MANTIĞI
              </h2>
            </div>
            <button
              onClick={onClose}
              style={{
                background: 'rgba(255,255,255,0.1)',
                border: '2px solid rgba(255,255,255,0.2)',
                borderRadius: '12px',
                width: '48px',
                height: '48px',
                color: '#FFFFFF',
                fontSize: '24px',
                cursor: 'pointer'
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div style={{ padding: '32px' }}>
          {/* Overview */}
          <Section title="📊 Genel Bakış" icon="📊">
            <p style={{ fontSize: '15px', color: 'rgba(255,255,255,0.8)', lineHeight: '1.8', margin: 0 }}>
              Breakout & Retest stratejisi, kritik destek ve direnç seviyelerinin kırılmasını tespit eder.
              Seviye kırıldıktan sonra, fiyatın bu seviyelere geri dönüp onaylaması (yeniden test) güçlü giriş fırsatları yaratır.
            </p>
          </Section>

          {/* Features */}
          <Section title="⚡ Temel Özellikler" icon="⚡">
            <FeatureList features={[
              { title: 'Kırılım Tespiti', desc: 'Önemli destek/direnç seviyelerinin kırılmasını otomatik tespit eder' },
              { title: 'Yeniden Test Onayı', desc: 'Kırılan seviyeye geri dönüş fırsatlarını yakalar' },
              { title: 'Hacim Analizi', desc: 'Kırılım sırasında hacim artışını kontrol eder' },
              { title: 'Risk/Kazanç Hesaplama', desc: 'Her sinyal için otomatik zarar kes ve hedef seviyeler belirlenir' },
              { title: 'Fibonacci Seviyeleri', desc: 'Hedefler Fibonacci genişleme seviyelerine göre hesaplanır' }
            ]} />
          </Section>

          {/* Usage Guide */}
          <Section title="📖 Kullanım Rehberi" icon="📖">
            <StepList steps={[
              { title: 'Sinyalleri İncele', desc: 'Yüksek güven skorlu sinyalleri önceliklendirin (%85+)' },
              { title: 'Formasyon Metrikleri', desc: 'Konsolidasyon aralığı, kırılım hacim oranı ve RSI değerlerini kontrol edin' },
              { title: 'Hedef Seviyeleri', desc: 'TP1, TP2, TP3 hedeflerini ve zarar kes seviyesini not alın' },
              { title: 'Binance\'de Aç', desc: 'Karta tıklayıp detaylı analizi görün, ardından Binance\'de işlem yapın' }
            ]} />
          </Section>

          {/* Warning */}
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
            border: `2px solid ${COLORS.warning}60`,
            borderRadius: '16px',
            padding: '24px',
            marginTop: '24px'
          }}>
            <h3 style={{ fontSize: '20px', fontWeight: '800', color: COLORS.warning, marginBottom: '16px' }}>
              ⚠️ Önemli Notlar
            </h3>
            <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255,255,255,0.8)', fontSize: '14px', lineHeight: '1.8' }}>
              <li>4 saat ve 1 gün zaman dilimlerinde çalışan orta-uzun vadeli strateji</li>
              <li>Mutlaka zarar kes kullanın ve risk yönetimi kurallarına uyun</li>
              <li>Yüksek başarı oranı (%70+) ancak dikkatli kullanım gerektirir</li>
              <li>Bu sayfa sadece eğitim amaçlıdır, finansal tavsiye değildir</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ title, icon, children }: { title: string; icon: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '32px' }}>
      <h3 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontSize: '28px' }}>{icon}</span>
        {title}
      </h3>
      {children}
    </div>
  );
}

function FeatureList({ features }: { features: Array<{ title: string; desc: string }> }) {
  return (
    <div style={{ display: 'grid', gap: '12px' }}>
      {features.map((feature, i) => (
        <div key={i} style={{
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: '12px',
          padding: '16px',
          transition: 'all 0.3s'
        }}>
          <div style={{ fontSize: '15px', fontWeight: '700', color: '#FFFFFF', marginBottom: '6px' }}>
            • {feature.title}
          </div>
          <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', lineHeight: '1.6' }}>
            {feature.desc}
          </div>
        </div>
      ))}
    </div>
  );
}

function StepList({ steps }: { steps: Array<{ title: string; desc: string }> }) {
  return (
    <div style={{ display: 'grid', gap: '16px' }}>
      {steps.map((step, i) => (
        <div key={i} style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.success}, ${COLORS.info})`,
            color: '#000',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px',
            fontWeight: '900',
            flexShrink: 0
          }}>
            {i + 1}
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '6px' }}>
              {step.title}
            </div>
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', lineHeight: '1.6' }}>
              {step.desc}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
