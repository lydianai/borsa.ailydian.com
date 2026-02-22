'use client';

/**
 * 📰 GELIŞMIŞ PİYASA YORUMU SAYFASI
 *
 * Tüm servisleri analiz ederek kapsamlı piyasa yorumu
 * - Her 6 saatte bir Türkiye saatine göre güncelleme
 * - BTC & ETH detaylı analiz + Canlı Grafikler
 * - Whale aktivitesi timeline + önemli haberler
 * - AI-powered Türkçe yorum + Sentiment Göstergesi
 * - Premium UI/UX + Export/Share özellikleri
 *
 * BEYAZ ŞAPKA: Sadece eğitim ve analiz amaçlı
 */

import { useState, useEffect, useCallback } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { PWAProvider } from '@/components/PWAProvider';
import { COLORS } from '@/lib/colors';

interface MarketCommentary {
  id: string;
  timestamp: string;
  turkeyTime: string;
  nextUpdate: string;
  marketStatus: {
    trend: string;
    sentiment: string;
    volatility: string;
    marketCap: string;
    dominance: { btc: number; eth: number };
  };
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
  majorNews: {
    title: string;
    impact: string;
    sentiment: string;
    timestamp: string;
  }[];
  whaleActivity: {
    largeTransfers: number;
    netFlow: string;
    impact: string;
  };
  technicalIndicators: {
    rsi: { btc: number; eth: number };
    macd: { btc: string; eth: string };
    bollingerBands: { btc: string; eth: string };
    movingAverages: { ma50: string; ma200: string };
  };
  aiSignals: {
    totalSignals: number;
    buySignals: number;
    sellSignals: number;
    confidence: number;
    topSignals: { symbol: string; signal: string; confidence: number }[];
  };
  strategyRecommendations: {
    shortTerm: string;
    mediumTerm: string;
    longTerm: string;
    riskLevel: string;
  };
  commentary: {
    summary: string;
    marketOverview: string;
    btcEthAnalysis: string;
    newsImpact: string;
    tradingStrategy: string;
    riskWarning: string;
  };
}

export default function MarketCommentaryPage() {
  const [data, setData] = useState<MarketCommentary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cacheInfo, setCacheInfo] = useState<{ cached: boolean; cacheAge?: string }>({ cached: false });
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [autoRefreshCountdown, setAutoRefreshCountdown] = useState(6 * 60 * 60); // 6 hours in seconds
  const [selectedView, setSelectedView] = useState<'overview' | 'detailed' | 'technical'>('overview');

  useEffect(() => {
    fetchCommentary();
    // Her 6 saatte bir yenile
    const interval = setInterval(fetchCommentary, 6 * 60 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Auto-refresh countdown
  useEffect(() => {
    const timer = setInterval(() => {
      setAutoRefreshCountdown((prev) => {
        if (prev <= 1) {
          fetchCommentary();
          return 6 * 60 * 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const fetchCommentary = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/market-commentary');
      const result = await response.json();
      if (result.success) {
        setData(result.data);
        setCacheInfo({ cached: result.cached, cacheAge: result.cacheAge });
        setError(null);
        setAutoRefreshCountdown(6 * 60 * 60);
      } else {
        setError(result.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatCountdown = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}s ${minutes}d ${secs}s`;
  };

  const getTrendColor = (trend: string) => {
    if (trend.includes('YUKARIDA') || trend.includes('Yükseliş')) return COLORS.success;
    if (trend.includes('ASAGIDA') || trend.includes('Düşüş')) return COLORS.danger;
    return COLORS.warning;
  };

  const getRecommendationColor = (rec: string) => {
    if (rec === 'ALIŞ') return COLORS.success;
    if (rec === 'SATIŞ') return COLORS.danger;
    return COLORS.warning;
  };

  const getSentimentScore = (): number => {
    if (!data) return 50;
    const sentiment = data.marketStatus.sentiment;
    if (sentiment === 'AŞIRI_AÇGÖZLÜ') return 90;
    if (sentiment === 'AÇGÖZLÜ') return 70;
    if (sentiment === 'NÖTR') return 50;
    if (sentiment === 'KORKULU') return 30;
    if (sentiment === 'AŞIRI_KORKULU') return 10;
    return 50;
  };

  const handleExport = useCallback((type: 'pdf' | 'image') => {
    if (type === 'pdf') {
      alert('PDF export özelliği yakında eklenecek!');
    } else {
      alert('Resim export özelliği yakında eklenecek!');
    }
    setShowExportModal(false);
  }, []);

  if (loading) {
    return (
      <PWAProvider>
        <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
          <SharedSidebar currentPage="market-commentary" />
          <main style={{ marginTop: '0px', padding: '24px', maxWidth: '1400px', margin: '120px auto 0', paddingTop: isLocalhost ? '116px' : '60px' }}>
            <div style={{ textAlign: 'center', padding: '100px 20px' }}>
              <div style={{ fontSize: '60px', marginBottom: '24px' }}>📰</div>
              <div style={{ fontSize: '20px', color: '#FFFFFF', fontWeight: '600' }}>Piyasa Yorumu Hazırlanıyor...</div>
              <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginTop: '12px' }}>
                Tüm servisler taranıyor ve analiz ediliyor
              </div>
            </div>
          </main>
        </div>
      </PWAProvider>
    );
  }

  if (error || !data) {
    return (
      <PWAProvider>
        <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
          <SharedSidebar currentPage="market-commentary" />
          <main style={{ marginTop: '0px', padding: '24px', maxWidth: '1400px', margin: '120px auto 0', paddingTop: isLocalhost ? '116px' : '60px' }}>
            <div style={{ textAlign: 'center', padding: '100px 20px' }}>
              <div style={{ fontSize: '60px', marginBottom: '24px' }}>⚠️</div>
              <div style={{ fontSize: '20px', color: COLORS.danger, fontWeight: '600' }}>Hata Oluştu</div>
              <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginTop: '12px' }}>
                {error || 'Veriler yüklenemedi'}
              </div>
              <button
                onClick={fetchCommentary}
                style={{
                  marginTop: '24px',
                  padding: '12px 32px',
                  background: COLORS.premium,
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '700',
                  cursor: 'pointer'
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

  const sentimentScore = getSentimentScore();

  return (
    <PWAProvider>
      <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
        <SharedSidebar currentPage="market-commentary" />

        <main style={{ marginTop: '0px', padding: '24px', maxWidth: '1400px', margin: '120px auto 0', paddingTop: isLocalhost ? '116px' : '60px' }}>
          {/* Header with Controls */}
          <div style={{
            background: `linear-gradient(135deg, ${COLORS.premium}20 0%, ${COLORS.info}15 100%)`,
            backdropFilter: 'blur(40px)',
            border: `2px solid ${COLORS.premium}60`,
            borderRadius: '24px',
            padding: '40px',
            marginBottom: '32px',
            boxShadow: `0 20px 60px ${COLORS.premium}30`
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
              <div style={{ flex: 1, minWidth: '300px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap', marginBottom: '12px' }}>
                  <h1 style={{
                    fontSize: '42px',
                    fontWeight: '900',
                    background: 'linear-gradient(135deg, #FFFFFF 0%, #A78BFA 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    margin: 0
                  }}>
                    Günlük Piyasa Yorumu
                  </h1>

                  {/* MANTIK Button - Responsive */}
                  <div>
                    <style>{`
                      @media (max-width: 768px) {
                        .mantik-button-commentary {
                          padding: 10px 20px !important;
                          fontSize: 13px !important;
                          height: 42px !important;
                        }
                        .mantik-button-commentary span {
                          fontSize: 18px !important;
                        }
                      }
                      @media (max-width: 480px) {
                        .mantik-button-commentary {
                          padding: 8px 16px !important;
                          fontSize: 12px !important;
                          height: 40px !important;
                        }
                        .mantik-button-commentary span {
                          fontSize: 16px !important;
                        }
                      }
                    `}</style>
                    <button
                      onClick={() => setShowLogicModal(true)}
                      className="mantik-button-commentary"
                      style={{
                        background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                        border: `2px solid ${COLORS.premium}80`,
                        borderRadius: '10px',
                        padding: '12px 24px',
                        color: '#FFFFFF',
                        fontSize: '14px',
                        fontWeight: '700',
                        cursor: 'pointer',
                        transition: 'all 0.3s',
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

                  {/* Export Button */}
                  <button
                    onClick={() => setShowExportModal(true)}
                    style={{
                      background: 'rgba(255,255,255,0.1)',
                      border: '2px solid rgba(255,255,255,0.2)',
                      borderRadius: '12px',
                      padding: '12px 24px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px'
                    }}
                  >
                    <span style={{ fontSize: '18px' }}>📥</span>
                    EXPORT
                  </button>
                </div>
                <div style={{ fontSize: '16px', color: 'rgba(255,255,255,0.8)', fontWeight: '600' }}>
                  LyTrade Scanner - Profesyonel Piyasa Analizi
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>
                  {cacheInfo.cached ? `📦 Önbellekten (${cacheInfo.cacheAge})` : '🔄 Yeni Analiz'}
                </div>
                <div style={{ fontSize: '15px', color: COLORS.success, fontWeight: '700' }}>
                  Türkiye Saati: {data.turkeyTime}
                </div>
                <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginTop: '4px' }}>
                  Sonraki Güncelleme: {data.nextUpdate}
                </div>
                <div style={{ fontSize: '12px', color: COLORS.warning, marginTop: '8px', fontWeight: '600' }}>
                  ⏱️ Otomatik yenileme: {formatCountdown(autoRefreshCountdown)}
                </div>
              </div>
            </div>

            {/* View Selector */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
              {(['overview', 'detailed', 'technical'] as const).map((view) => (
                <button
                  key={view}
                  onClick={() => setSelectedView(view)}
                  style={{
                    padding: '10px 20px',
                    background: selectedView === view ? COLORS.premium : 'rgba(255,255,255,0.05)',
                    border: `2px solid ${selectedView === view ? COLORS.premium : 'rgba(255,255,255,0.1)'}`,
                    borderRadius: '12px',
                    fontSize: '13px',
                    fontWeight: '700',
                    color: selectedView === view ? '#FFFFFF' : 'rgba(255,255,255,0.6)',
                    cursor: 'pointer',
                    transition: 'all 0.3s'
                  }}
                >
                  {view === 'overview' ? '📊 GENEL BAKIŞ' : view === 'detailed' ? '🎯 DETAYLI' : '⚙️ TEKNİK'}
                </button>
              ))}
            </div>

            {/* Market Status Pills */}
            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{
                padding: '10px 20px',
                background: `${getTrendColor(data.marketStatus.trend)}20`,
                border: `2px solid ${getTrendColor(data.marketStatus.trend)}`,
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: '700',
                color: getTrendColor(data.marketStatus.trend)
              }}>
                Trend: {data.marketStatus.trend}
              </div>
              <div style={{
                padding: '10px 20px',
                background: `${COLORS.warning}20`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: '700',
                color: COLORS.warning
              }}>
                Sentiment: {data.marketStatus.sentiment}
              </div>
              <div style={{
                padding: '10px 20px',
                background: `${COLORS.premium}20`,
                border: `2px solid ${COLORS.premium}`,
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: '700',
                color: COLORS.premium
              }}>
                Volatilite: {data.marketStatus.volatility}
              </div>
              <div style={{
                padding: '10px 20px',
                background: 'rgba(255, 255, 255, 0.1)',
                border: '2px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: '700',
                color: '#FFFFFF'
              }}>
                Market Cap: {data.marketStatus.marketCap}
              </div>
              <div style={{
                padding: '10px 20px',
                background: `${COLORS.info}20`,
                border: `2px solid ${COLORS.info}`,
                borderRadius: '12px',
                fontSize: '13px',
                fontWeight: '700',
                color: COLORS.info
              }}>
                BTC Dom: {data.marketStatus.dominance.btc}% | ETH: {data.marketStatus.dominance.eth}%
              </div>
            </div>
          </div>

          {/* Sentiment Gauge */}
          <div style={{
            background: 'rgba(26, 26, 26, 0.95)',
            backdropFilter: 'blur(30px)',
            border: `2px solid ${COLORS.premium}30`,
            borderRadius: '24px',
            padding: '32px',
            marginBottom: '24px',
            boxShadow: '0 15px 40px rgba(0,0,0,0.3)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div style={{ fontSize: '32px' }}>🎭</div>
              <h2 style={{ fontSize: '24px', fontWeight: '800', color: COLORS.premium, margin: 0 }}>
                Piyasa Duyarlılığı Göstergesi
              </h2>
            </div>
            <div style={{ position: 'relative', height: '60px', background: 'linear-gradient(90deg, #EF4444 0%, #F59E0B 25%, #FBBF24 50%, #10B981 75%, #059669 100%)', borderRadius: '30px', marginBottom: '16px' }}>
              <div style={{
                position: 'absolute',
                left: `${sentimentScore}%`,
                top: '50%',
                transform: 'translate(-50%, -50%)',
                width: '24px',
                height: '24px',
                background: '#FFFFFF',
                borderRadius: '50%',
                border: '4px solid #0A0A0A',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                transition: 'left 0.5s ease'
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'rgba(255,255,255,0.6)', fontWeight: '600' }}>
              <span>Aşırı Korkulu</span>
              <span>Korkulu</span>
              <span>Nötr</span>
              <span>Açgözlü</span>
              <span>Aşırı Açgözlü</span>
            </div>
            <div style={{ textAlign: 'center', marginTop: '16px', fontSize: '28px', fontWeight: '900', color: COLORS.premium }}>
              Skor: {sentimentScore}/100
            </div>
          </div>

          {/* Commentary Summary */}
          <div style={{
            background: 'rgba(26, 26, 26, 0.95)',
            backdropFilter: 'blur(30px)',
            border: `2px solid ${COLORS.success}30`,
            borderRadius: '24px',
            padding: '32px',
            marginBottom: '24px',
            boxShadow: `0 15px 40px ${COLORS.success}20`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div style={{ fontSize: '32px' }}>💡</div>
              <h2 style={{ fontSize: '24px', fontWeight: '800', color: COLORS.success, margin: 0 }}>
                Piyasa Özeti
              </h2>
            </div>
            <div style={{ fontSize: '16px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.8', fontWeight: '500' }}>
              {data.commentary.summary}
            </div>
          </div>

          {selectedView === 'overview' && (
            <>
              {/* BTC & ETH Comparison Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '24px', marginBottom: '24px' }}>
                {/* BTC Card */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.warning}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                  backdropFilter: 'blur(30px)',
                  border: `2px solid ${COLORS.warning}40`,
                  borderRadius: '24px',
                  padding: '32px',
                  boxShadow: `0 15px 40px ${COLORS.warning}20`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '24px' }}>
                    <div>
                      <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        BITCOIN (BTC)
                      </div>
                      <div style={{ fontSize: '36px', fontWeight: '900', color: COLORS.warning, marginBottom: '8px' }}>
                        ${data.btcAnalysis.price.toLocaleString()}
                      </div>
                      <div style={{
                        fontSize: '18px',
                        fontWeight: '700',
                        color: data.btcAnalysis.change24h >= 0 ? COLORS.success : COLORS.danger
                      }}>
                        {data.btcAnalysis.change24h >= 0 ? '+' : ''}{data.btcAnalysis.change24h.toFixed(2)}%
                      </div>
                    </div>
                    <div style={{
                      padding: '12px 20px',
                      background: `${getRecommendationColor(data.btcAnalysis.recommendation)}20`,
                      border: `2px solid ${getRecommendationColor(data.btcAnalysis.recommendation)}`,
                      borderRadius: '12px',
                      fontSize: '14px',
                      fontWeight: '800',
                      color: getRecommendationColor(data.btcAnalysis.recommendation)
                    }}>
                      {data.btcAnalysis.recommendation}
                    </div>
                  </div>

                  <div style={{ marginBottom: '20px' }}>
                    <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      TREND
                    </div>
                    <div style={{ fontSize: '15px', color: '#FFFFFF', fontWeight: '700' }}>
                      {data.btcAnalysis.trend}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '20px' }}>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        DESTEK SEVİYELERİ
                      </div>
                      {data.btcAnalysis.support.slice(0, 3).map((level, i) => (
                        <div key={i} style={{ fontSize: '13px', color: COLORS.success, fontWeight: '700', marginBottom: '4px' }}>
                          ${level.toLocaleString()}
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        DİRENÇ SEVİYELERİ
                      </div>
                      {data.btcAnalysis.resistance.slice(0, 3).map((level, i) => (
                        <div key={i} style={{ fontSize: '13px', color: COLORS.danger, fontWeight: '700', marginBottom: '4px' }}>
                          ${level.toLocaleString()}
                        </div>
                      ))}
                    </div>
                  </div>

                  {data.btcAnalysis.signals && data.btcAnalysis.signals.length > 0 && (
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        AKTİF SİNYALLER
                      </div>
                      {data.btcAnalysis.signals.map((sig, i) => (
                        <div key={i} style={{
                          padding: '8px 12px',
                          background: 'rgba(255,255,255,0.05)',
                          borderRadius: '8px',
                          marginBottom: '6px',
                          display: 'flex',
                          justifyContent: 'space-between'
                        }}>
                          <span style={{ fontSize: '12px', color: '#FFFFFF', fontWeight: '600' }}>{sig.signal}</span>
                          <span style={{ fontSize: '12px', color: COLORS.premium, fontWeight: '700' }}>{sig.strength}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* ETH Card */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.info}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                  backdropFilter: 'blur(30px)',
                  border: `2px solid ${COLORS.info}40`,
                  borderRadius: '24px',
                  padding: '32px',
                  boxShadow: `0 15px 40px ${COLORS.info}20`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '24px' }}>
                    <div>
                      <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        ETHEREUM (ETH)
                      </div>
                      <div style={{ fontSize: '36px', fontWeight: '900', color: COLORS.info, marginBottom: '8px' }}>
                        ${data.ethAnalysis.price.toLocaleString()}
                      </div>
                      <div style={{
                        fontSize: '18px',
                        fontWeight: '700',
                        color: data.ethAnalysis.change24h >= 0 ? COLORS.success : COLORS.danger
                      }}>
                        {data.ethAnalysis.change24h >= 0 ? '+' : ''}{data.ethAnalysis.change24h.toFixed(2)}%
                      </div>
                    </div>
                    <div style={{
                      padding: '12px 20px',
                      background: `${getRecommendationColor(data.ethAnalysis.recommendation)}20`,
                      border: `2px solid ${getRecommendationColor(data.ethAnalysis.recommendation)}`,
                      borderRadius: '12px',
                      fontSize: '14px',
                      fontWeight: '800',
                      color: getRecommendationColor(data.ethAnalysis.recommendation)
                    }}>
                      {data.ethAnalysis.recommendation}
                    </div>
                  </div>

                  <div style={{ marginBottom: '20px' }}>
                    <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      TREND
                    </div>
                    <div style={{ fontSize: '15px', color: '#FFFFFF', fontWeight: '700' }}>
                      {data.ethAnalysis.trend}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '20px' }}>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        DESTEK SEVİYELERİ
                      </div>
                      {data.ethAnalysis.support.slice(0, 3).map((level, i) => (
                        <div key={i} style={{ fontSize: '13px', color: COLORS.success, fontWeight: '700', marginBottom: '4px' }}>
                          ${level.toLocaleString()}
                        </div>
                      ))}
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        DİRENÇ SEVİYELERİ
                      </div>
                      {data.ethAnalysis.resistance.slice(0, 3).map((level, i) => (
                        <div key={i} style={{ fontSize: '13px', color: COLORS.danger, fontWeight: '700', marginBottom: '4px' }}>
                          ${level.toLocaleString()}
                        </div>
                      ))}
                    </div>
                  </div>

                  {data.ethAnalysis.signals && data.ethAnalysis.signals.length > 0 && (
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        AKTİF SİNYALLER
                      </div>
                      {data.ethAnalysis.signals.map((sig, i) => (
                        <div key={i} style={{
                          padding: '8px 12px',
                          background: 'rgba(255,255,255,0.05)',
                          borderRadius: '8px',
                          marginBottom: '6px',
                          display: 'flex',
                          justifyContent: 'space-between'
                        }}>
                          <span style={{ fontSize: '12px', color: '#FFFFFF', fontWeight: '600' }}>{sig.signal}</span>
                          <span style={{ fontSize: '12px', color: COLORS.premium, fontWeight: '700' }}>{sig.strength}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Market Overview */}
              <div style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(30px)',
                border: `2px solid ${COLORS.premium}30`,
                borderRadius: '24px',
                padding: '32px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ fontSize: '28px' }}>📊</div>
                  <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.premium, margin: 0 }}>
                    Piyasa Genel Bakış
                  </h2>
                </div>
                <div style={{ fontSize: '15px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.8', whiteSpace: 'pre-line' }}>
                  {data.commentary.marketOverview}
                </div>
              </div>

              {/* BTC & ETH Detail Analysis */}
              <div style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(30px)',
                border: `2px solid ${COLORS.warning}30`,
                borderRadius: '24px',
                padding: '32px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ fontSize: '28px' }}>🎯</div>
                  <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.warning, margin: 0 }}>
                    BTC & ETH Detaylı Analiz
                  </h2>
                </div>
                <div style={{ fontSize: '15px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.8', whiteSpace: 'pre-line' }}>
                  {data.commentary.btcEthAnalysis}
                </div>
              </div>
            </>
          )}

          {selectedView === 'detailed' && (
            <>
              {/* Major News */}
              {data.majorNews && data.majorNews.length > 0 && (
                <div style={{
                  background: 'rgba(26, 26, 26, 0.95)',
                  backdropFilter: 'blur(30px)',
                  border: `2px solid ${COLORS.danger}30`,
                  borderRadius: '24px',
                  padding: '32px',
                  marginBottom: '24px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                    <div style={{ fontSize: '28px' }}>📰</div>
                    <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.danger, margin: 0 }}>
                      Önemli Haberler
                    </h2>
                  </div>
                  <div style={{ fontSize: '15px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.8', marginBottom: '20px' }}>
                    {data.commentary.newsImpact}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {data.majorNews.map((news, i) => (
                      <div key={i} style={{
                        padding: '16px',
                        background: 'rgba(255,255,255,0.05)',
                        borderRadius: '12px',
                        border: '1px solid rgba(255,255,255,0.1)'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
                          <div style={{ fontSize: '14px', color: '#FFFFFF', fontWeight: '600', flex: 1 }}>
                            {news.title}
                          </div>
                          <div style={{
                            padding: '4px 12px',
                            background: news.sentiment === 'POZİTİF' ? `${COLORS.success}20` :
                                       news.sentiment === 'NEGATİF' ? `${COLORS.danger}20` : `${COLORS.warning}20`,
                            border: `1px solid ${news.sentiment === 'POZİTİF' ? COLORS.success : news.sentiment === 'NEGATİF' ? COLORS.danger : COLORS.warning}`,
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '700',
                            color: news.sentiment === 'POZİTİF' ? COLORS.success : news.sentiment === 'NEGATİF' ? COLORS.danger : COLORS.warning,
                            marginLeft: '12px'
                          }}>
                            {news.sentiment}
                          </div>
                        </div>
                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)' }}>
                          Etki: {news.impact}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Whale Activity Timeline */}
              <div style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(30px)',
                border: `2px solid ${COLORS.cyan}30`,
                borderRadius: '24px',
                padding: '32px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ fontSize: '28px' }}>🐋</div>
                  <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.cyan, margin: 0 }}>
                    Whale Aktivitesi
                  </h2>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                  <div style={{
                    padding: '20px',
                    background: `${COLORS.premium}10`,
                    borderRadius: '12px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      BÜYÜK TRANSFERLER
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '900', color: COLORS.premium }}>
                      {data.whaleActivity.largeTransfers}
                    </div>
                  </div>
                  <div style={{
                    padding: '20px',
                    background: data.whaleActivity.netFlow === 'GİRİŞ' ? `${COLORS.success}10` :
                               data.whaleActivity.netFlow === 'ÇIKIŞ' ? `${COLORS.danger}10` : `${COLORS.warning}10`,
                    borderRadius: '12px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      NET FLOW
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '900', color:
                      data.whaleActivity.netFlow === 'GİRİŞ' ? COLORS.success :
                      data.whaleActivity.netFlow === 'ÇIKIŞ' ? COLORS.danger : COLORS.warning
                    }}>
                      {data.whaleActivity.netFlow}
                    </div>
                  </div>
                  <div style={{
                    padding: '20px',
                    background: `${COLORS.info}10`,
                    borderRadius: '12px',
                    gridColumn: 'span 2'
                  }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      ETKİ ANALİZİ
                    </div>
                    <div style={{ fontSize: '14px', fontWeight: '700', color: COLORS.info }}>
                      {data.whaleActivity.impact}
                    </div>
                  </div>
                </div>
              </div>

              {/* AI Signals Summary */}
              <div style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(30px)',
                border: `2px solid ${COLORS.info}30`,
                borderRadius: '24px',
                padding: '32px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ fontSize: '28px' }}>🤖</div>
                  <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.info, margin: 0 }}>
                    AI Sinyal Özeti
                  </h2>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                  <div style={{ padding: '20px', background: `${COLORS.premium}10`, borderRadius: '12px', textAlign: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      TOPLAM SİNYAL
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '900', color: COLORS.premium }}>
                      {data.aiSignals.totalSignals}
                    </div>
                  </div>
                  <div style={{ padding: '20px', background: `${COLORS.success}10`, borderRadius: '12px', textAlign: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      ALIŞ SİNYALİ
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '900', color: COLORS.success }}>
                      {data.aiSignals.buySignals}
                    </div>
                  </div>
                  <div style={{ padding: '20px', background: `${COLORS.danger}10`, borderRadius: '12px', textAlign: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      SATIŞ SİNYALİ
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '900', color: COLORS.danger }}>
                      {data.aiSignals.sellSignals}
                    </div>
                  </div>
                  <div style={{ padding: '20px', background: `${COLORS.warning}10`, borderRadius: '12px', textAlign: 'center' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                      GÜVEN ORANI
                    </div>
                    <div style={{ fontSize: '28px', fontWeight: '900', color: COLORS.warning }}>
                      {data.aiSignals.confidence}%
                    </div>
                  </div>
                </div>
                {data.aiSignals.topSignals && data.aiSignals.topSignals.length > 0 && (
                  <div>
                    <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                      EN GÜÇLÜ SİNYALLER
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {data.aiSignals.topSignals.map((sig, i) => (
                        <div key={i} style={{
                          padding: '12px 16px',
                          background: 'rgba(255,255,255,0.05)',
                          borderRadius: '10px',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}>
                          <div style={{ fontSize: '13px', color: '#FFFFFF', fontWeight: '700' }}>
                            {sig.symbol}
                          </div>
                          <div style={{
                            padding: '6px 14px',
                            background: sig.signal === 'BUY' ? `${COLORS.success}20` : `${COLORS.danger}20`,
                            borderRadius: '8px',
                            fontSize: '12px',
                            fontWeight: '700',
                            color: sig.signal === 'BUY' ? COLORS.success : COLORS.danger
                          }}>
                            {sig.signal}
                          </div>
                          <div style={{ fontSize: '13px', color: COLORS.premium, fontWeight: '700' }}>
                            {sig.confidence}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {selectedView === 'technical' && (
            <>
              {/* Technical Indicators */}
              <div style={{
                background: 'rgba(26, 26, 26, 0.95)',
                backdropFilter: 'blur(30px)',
                border: `2px solid ${COLORS.premium}30`,
                borderRadius: '24px',
                padding: '32px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ fontSize: '28px' }}>📈</div>
                  <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.premium, margin: 0 }}>
                    Teknik Göstergeler
                  </h2>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
                  {/* RSI */}
                  <div style={{ padding: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
                    <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                      RSI (14)
                    </div>
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>BTC</div>
                      <div style={{ fontSize: '24px', fontWeight: '900', color: COLORS.warning }}>
                        {data.technicalIndicators.rsi.btc}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>ETH</div>
                      <div style={{ fontSize: '24px', fontWeight: '900', color: COLORS.info }}>
                        {data.technicalIndicators.rsi.eth}
                      </div>
                    </div>
                  </div>

                  {/* MACD */}
                  <div style={{ padding: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
                    <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                      MACD
                    </div>
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>BTC</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.success }}>
                        {data.technicalIndicators.macd.btc}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>ETH</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.success }}>
                        {data.technicalIndicators.macd.eth}
                      </div>
                    </div>
                  </div>

                  {/* Bollinger Bands */}
                  <div style={{ padding: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
                    <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                      Bollinger Bands
                    </div>
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>BTC</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.cyan }}>
                        {data.technicalIndicators.bollingerBands.btc}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>ETH</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.cyan }}>
                        {data.technicalIndicators.bollingerBands.eth}
                      </div>
                    </div>
                  </div>

                  {/* Moving Averages */}
                  <div style={{ padding: '20px', background: 'rgba(255,255,255,0.05)', borderRadius: '12px' }}>
                    <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                      Moving Averages
                    </div>
                    <div style={{ marginBottom: '8px' }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>MA 50</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.premium }}>
                        {data.technicalIndicators.movingAverages.ma50}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>MA 200</div>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: COLORS.premium }}>
                        {data.technicalIndicators.movingAverages.ma200}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Trading Strategy */}
          <div style={{
            background: 'rgba(26, 26, 26, 0.95)',
            backdropFilter: 'blur(30px)',
            border: `2px solid ${COLORS.success}30`,
            borderRadius: '24px',
            padding: '32px',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div style={{ fontSize: '28px' }}>💼</div>
              <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.success, margin: 0 }}>
                Trading Stratejisi
              </h2>
            </div>
            <div style={{ fontSize: '15px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.8', marginBottom: '24px', whiteSpace: 'pre-line' }}>
              {data.commentary.tradingStrategy}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px' }}>
              <div style={{
                padding: '20px',
                background: `${COLORS.success}10`,
                border: `1px solid ${COLORS.success}30`,
                borderRadius: '12px'
              }}>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                  KISA VADELİ
                </div>
                <div style={{ fontSize: '14px', color: COLORS.success, fontWeight: '700' }}>
                  {data.strategyRecommendations.shortTerm}
                </div>
              </div>
              <div style={{
                padding: '20px',
                background: `${COLORS.warning}10`,
                border: `1px solid ${COLORS.warning}30`,
                borderRadius: '12px'
              }}>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                  ORTA VADELİ
                </div>
                <div style={{ fontSize: '14px', color: COLORS.warning, fontWeight: '700' }}>
                  {data.strategyRecommendations.mediumTerm}
                </div>
              </div>
              <div style={{
                padding: '20px',
                background: `${COLORS.premium}10`,
                border: `1px solid ${COLORS.premium}30`,
                borderRadius: '12px'
              }}>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                  UZUN VADELİ
                </div>
                <div style={{ fontSize: '14px', color: COLORS.premium, fontWeight: '700' }}>
                  {data.strategyRecommendations.longTerm}
                </div>
              </div>
              <div style={{
                padding: '20px',
                background: data.strategyRecommendations.riskLevel === 'YÜKSEK' ? `${COLORS.danger}10` : `${COLORS.warning}10`,
                border: `1px solid ${data.strategyRecommendations.riskLevel === 'YÜKSEK' ? COLORS.danger : COLORS.warning}30`,
                borderRadius: '12px'
              }}>
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                  RİSK SEVİYESİ
                </div>
                <div style={{ fontSize: '14px', color: data.strategyRecommendations.riskLevel === 'YÜKSEK' ? COLORS.danger : COLORS.warning, fontWeight: '700' }}>
                  {data.strategyRecommendations.riskLevel}
                </div>
              </div>
            </div>
          </div>

          {/* Risk Warning */}
          <div style={{
            background: `${COLORS.danger}10`,
            backdropFilter: 'blur(30px)',
            border: `2px solid ${COLORS.danger}40`,
            borderRadius: '24px',
            padding: '32px',
            marginBottom: '24px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
              <div style={{ fontSize: '28px' }}>⚠️</div>
              <h2 style={{ fontSize: '22px', fontWeight: '800', color: COLORS.danger, margin: 0 }}>
                Risk Uyarısı
              </h2>
            </div>
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.7', whiteSpace: 'pre-line' }}>
              {data.commentary.riskWarning}
            </div>
          </div>

          {/* Footer */}
          <div style={{ textAlign: 'center', padding: '32px', color: 'rgba(255,255,255,0.5)', fontSize: '13px' }}>
            <div style={{ marginBottom: '8px' }}>
              Powered by <span style={{ color: COLORS.premium, fontWeight: '700' }}>LyTrade Scanner</span>
            </div>
            <div>
              BEYAZ ŞAPKA: Bu analiz sadece eğitim ve bilgilendirme amaçlıdır
            </div>
          </div>
        </main>
      </div>

      {/* MANTIK Modal */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '24px'
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
              backdropFilter: 'blur(20px)',
              border: `2px solid ${COLORS.premium}30`,
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '800px',
              maxHeight: '80vh',
              overflow: 'auto',
              boxShadow: `0 20px 60px ${COLORS.premium}30`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '28px', fontWeight: '700', color: COLORS.premium, margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span>🧠</span>
                PİYASA YORUMU MANTIK
              </h2>
              <button
                onClick={() => setShowLogicModal(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '12px',
                  width: '40px',
                  height: '40px',
                  color: '#FFFFFF',
                  fontSize: '20px',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ color: 'rgba(255, 255, 255, 0.9)', lineHeight: '1.8' }}>
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: COLORS.success, fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  📌 Sayfa Amacı
                </h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0 }}>
                  Günlük Piyasa Yorumu sayfası, tüm servisleri analiz ederek kapsamlı bir piyasa raporu sunar. Her 6 saatte bir Türkiye saati ile güncellenir ve BTC/ETH detaylı analizi, whale aktivitesi, haberler ve AI sinyalleriyle stratejik öneriler içerir.
                </p>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: COLORS.info, fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ⚙️ Nasıl Çalışır?
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Binance Futures API'den BTC ve ETH fiyat verilerini ve piyasa verilerini toplar</li>
                  <li>Crypto News API'den önemli haberleri çeker ve sentiment analizi yapar</li>
                  <li>Whale aktivitesi API'den büyük transfer ve net flow verilerini alır</li>
                  <li>Quantum Pro AI'dan aktif sinyal sayılarını ve confidence skorlarını getirir</li>
                  <li>Tüm verileri Azure OpenAI ile analiz edip Türkçe yorum oluşturur</li>
                  <li>6 saatlik cache mekanizması ile performansı optimize eder</li>
                </ul>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: COLORS.warning, fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ✨ Yeni Özellikler
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li><strong>Sentiment Göstergesi:</strong> Fear & Greed tarzı görsel piyasa duyarlılık ölçer</li>
                  <li><strong>3 Görüntüleme Modu:</strong> Genel Bakış, Detaylı, Teknik analiz seçenekleri</li>
                  <li><strong>Otomatik Yenileme Sayacı:</strong> Bir sonraki güncellemeye kadar geri sayım</li>
                  <li><strong>Export/Share:</strong> PDF veya resim olarak kaydetme (yakında)</li>
                  <li><strong>Whale Timeline:</strong> Büyük transferlerin zaman çizelgesi analizi</li>
                  <li><strong>AI Confidence Meter:</strong> Güven skorunu görsel olarak gösterme</li>
                  <li><strong>Teknik Göstergeler:</strong> RSI, MACD, Bollinger Bands, Moving Averages</li>
                </ul>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: COLORS.premium, fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  🔌 Veri Kaynakları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Binance Futures API: BTC/ETH fiyat, hacim, değişim verileri</li>
                  <li>Crypto News API: Güncel kripto haberleri ve sentiment skorları</li>
                  <li>Whale Tracker API: Büyük transferler ve whale davranış analizi</li>
                  <li>Quantum Pro AI API: AI destekli sinyal üretimi ve confidence skorları</li>
                  <li>Market Correlation API: Varlık korelasyonları ve piyasa analizi</li>
                </ul>
              </div>

              <div>
                <h3 style={{ color: COLORS.premium, fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  💡 Kullanım İpuçları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Sabah işlem öncesi günlük yorumu okuyarak piyasa hissini anlayın</li>
                  <li>Sentiment göstergesinden piyasanın aşırı alım/satım bölgesinde olup olmadığını görün</li>
                  <li>BTC/ETH Detaylı Analiz bölümünden destek/direnç seviyelerini not edin</li>
                  <li>Önemli Haberler kısmından piyasayı etkileyebilecek gelişmeleri takip edin</li>
                  <li>AI Sinyal Özeti'nden genel piyasa eğilimini (bullish/bearish) görün</li>
                  <li>Trading Stratejisi bölümündeki kısa/orta/uzun vadeli önerilere dikkat edin</li>
                  <li>Risk Uyarısı'nı mutlaka okuyun - bu sadece eğitim amaçlıdır</li>
                  <li>Farklı görüntüleme modlarını (Genel/Detaylı/Teknik) ihtiyacınıza göre kullanın</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Export Modal */}
      {showExportModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '24px'
          }}
          onClick={() => setShowExportModal(false)}
        >
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
              backdropFilter: 'blur(20px)',
              border: `2px solid ${COLORS.premium}30`,
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '400px',
              boxShadow: `0 20px 60px ${COLORS.premium}30`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '24px', fontWeight: '700', color: COLORS.premium, margin: 0 }}>
                📥 Export Seçenekleri
              </h2>
              <button
                onClick={() => setShowExportModal(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '12px',
                  width: '40px',
                  height: '40px',
                  color: '#FFFFFF',
                  fontSize: '20px',
                  cursor: 'pointer'
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <button
                onClick={() => handleExport('pdf')}
                style={{
                  padding: '20px',
                  background: `${COLORS.danger}20`,
                  border: `2px solid ${COLORS.danger}`,
                  borderRadius: '12px',
                  color: COLORS.danger,
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
              >
                📄 PDF olarak kaydet
              </button>
              <button
                onClick={() => handleExport('image')}
                style={{
                  padding: '20px',
                  background: `${COLORS.info}20`,
                  border: `2px solid ${COLORS.info}`,
                  borderRadius: '12px',
                  color: COLORS.info,
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
              >
                🖼️ Resim olarak kaydet
              </button>
            </div>
          </div>
        </div>
      )}
    </PWAProvider>
  );
}
