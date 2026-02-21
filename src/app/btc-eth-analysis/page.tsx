'use client';

/**
 * 📊 BTC-ETH KORELASYON ANALİZİ SAYFASI
 * BTC-ETH ilişkisi için özel derin analiz sayfası
 *
 * Özellikler:
 * - Büyük 30 günlük korelasyon grafiği
 * - MA bindirmeleri ile ETH/BTC oranı
 * - Hacim korelasyonu görselleştirmesi
 * - Sapma göstergeleri
 * - Parite ticaret önerileri
 * - Tarihsel korelasyon verileri
 * - Korelasyon bozulma uyarıları
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS, getChangeColor } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface CorrelationData {
  btcPrice: number;
  ethPrice: number;
  btcChange24h: number;
  ethChange24h: number;
  btcVolume: number;
  ethVolume: number;
  correlation30d: number;
  correlation7d: number;
  ethBtcRatio: number;
  ethBtcRatioMA50: number;
  ethBtcRatioMA200: number;
  volumeCorrelation: number;
  divergence: number;
  dominance: number;
  trend: 'Rising' | 'Falling' | 'Stable';
  recommendation: string;
  leadLagAnalysis?: {
    leader: 'BTC' | 'ETH' | 'NEUTRAL';
    confidence: number;
    description: string;
  };
}

export default function BtcEthAnalysisPage() {
  const [data, setData] = useState<CorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [countdown, setCountdown] = useState(300);
  const [showLogicModal, setShowLogicModal] = useState(false);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Global filtreler (tüm sayfalarda senkronize)
  const { timeframe, sortBy } = useGlobalFilters();

  // Grafik için tarihsel veriler (simüle edilmiş - üretimde API'den al)
  const [historicalCorrelation, setHistoricalCorrelation] = useState<number[]>([]);
  const [historicalRatio, setHistoricalRatio] = useState<number[]>([]);

  const fetchData = async () => {
    try {
      setLoading(true);

      // Özel API'den gerçek BTC-ETH korelasyon analizi getir
      const response = await fetch('/api/btc-eth-analysis');
      const result = await response.json();

      if (!result.success || !result.data) {
        throw new Error(result.error || 'Failed to fetch BTC-ETH correlation analysis');
      }

      const apiData = result.data;

      // API yanıtını bileşen durumuna eşle
      const analysisData: CorrelationData = {
        btcPrice: apiData.btcPrice,
        ethPrice: apiData.ethPrice,
        btcChange24h: apiData.btcChange24h,
        ethChange24h: apiData.ethChange24h,
        btcVolume: apiData.btcVolume,
        ethVolume: apiData.ethVolume,
        correlation30d: apiData.correlation30d,
        correlation7d: apiData.correlation7d,
        ethBtcRatio: apiData.ethBtcRatio,
        ethBtcRatioMA50: apiData.ethBtcRatioMA50,
        ethBtcRatioMA200: apiData.ethBtcRatioMA200,
        volumeCorrelation: apiData.volumeCorrelation,
        divergence: apiData.divergence,
        dominance: apiData.dominance,
        trend: apiData.trend,
        recommendation: apiData.recommendation,
        leadLagAnalysis: apiData.leadLagAnalysis,
      };

      setData(analysisData);

      // API'den gerçek tarihsel verileri ayarla (30 günlük korelasyonlar ve oranlar)
      setHistoricalCorrelation(apiData.historicalCorrelation || []);
      setHistoricalRatio(apiData.historicalRatio || []);
    } catch (error) {
      console.error('[BTC-ETH Analysis] Fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchData();
          return 300;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdownInterval);
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
    return `$${p.toFixed(6)}`;
  };

  const formatVolume = (volume: number) => {
    const v = volume ?? 0;
    if (v >= 1_000_000_000) return `$${(v / 1_000_000_000).toFixed(2)}B`;
    if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
    return `$${v.toFixed(2)}`;
  };

  const getCorrelationColor = (corr: number) => {
    if (corr > 0.85) return COLORS.success;
    if (corr > 0.65) return COLORS.warning;
    if (corr > 0) return COLORS.premium;
    return COLORS.danger;
  };

  const getCorrelationLabel = (corr: number) => {
    if (corr > 0.85) return 'GÜÇLÜ';
    if (corr > 0.65) return 'ORTA';
    if (corr > 0) return 'ZAYIF';
    return 'NEGATİF';
  };

  if (loading && !data) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div className="dashboard-container" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
      {/* AI Assistant */}
      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}

      {/* Sidebar */}
      <SharedSidebar
        currentPage="btc-eth-analysis"
        notificationCounts={notificationCounts}
      />

      {/* Main Content */}
      <div className="dashboard-main">
        <main className="dashboard-content" style={{ padding: '20px' }}>
          {/* Page Header with MANTIK Button */}
          <div style={{ margin: '16px 24px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <Icons.TrendingUp style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  BTC vs ETH Analizi
                </h1>
              </div>
              <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
                Bitcoin ve Ethereum Karşılaştırmalı Analiz - Korelasyon, Dominans ve Performans
              </p>
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              {/* Auto-refresh Countdown Widget */}
              <div style={{
                padding: '12px 20px',
                background: 'rgba(16, 185, 129, 0.1)',
                border: `2px solid ${COLORS.success}`,
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}>
                <Icons.Clock style={{ width: '20px', height: '20px', color: COLORS.success }} />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.secondary, fontWeight: '600' }}>
                    Otomatik Yenileme
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.success }}>
                    {formatTime(countdown)}
                  </div>
                </div>
              </div>

              <div>
                <style>{`
                  @media (max-width: 768px) {
                    .mantik-button-btceth {
                      padding: 10px 20px !important;
                      fontSize: 13px !important;
                      height: 42px !important;
                    }
                    .mantik-button-btceth svg {
                      width: 18px !important;
                      height: 18px !important;
                    }
                  }
                  @media (max-width: 480px) {
                    .mantik-button-btceth {
                      padding: 8px 16px !important;
                      fontSize: 12px !important;
                      height: 40px !important;
                    }
                    .mantik-button-btceth svg {
                      width: 16px !important;
                      height: 16px !important;
                    }
                  }
                `}</style>
                <button onClick={() => setShowLogicModal(true)} className="mantik-button-btceth" style={{
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
            </div>
          </div>

          {data && (
            <>
            {/* Top Overview Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              {/* BTC Card */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.warning}0D`, border: `1px solid ${COLORS.warning}4D`, position: 'relative' }}>
                {/* Live Data Badge */}
                <div style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  padding: '4px 10px',
                  background: `linear-gradient(135deg, ${COLORS.success}, #059669)`,
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: '700',
                  color: '#000',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  animation: 'pulse 2s ease-in-out infinite'
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: '#000',
                    animation: 'blink 1.5s ease-in-out infinite'
                  }} />
                  CANLI
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                  <Icons.Bitcoin style={{ width: '32px', height: '32px', color: COLORS.warning }} />
                  <div>
                    <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>Bitcoin</div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.warning }}>BTC/USDT</div>
                  </div>
                </div>
                <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '8px' }}>
                  {formatPrice(data.btcPrice)}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                  <span style={{ fontSize: '16px', fontWeight: 'bold', color: getChangeColor(data.btcChange24h) }}>
                    {(data.btcChange24h ?? 0) >= 0 ? '+' : ''}{(data.btcChange24h ?? 0).toFixed(2)}%
                  </span>
                  <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>24h</span>
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                  Volume: {formatVolume(data.btcVolume)}
                </div>
                <div style={{ marginTop: '12px', padding: '8px', background: COLORS.bg.card, borderRadius: '6px', border: `1px solid ${COLORS.warning}33` }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>BTC Bias</div>
                  <div style={{ fontSize: '16px', fontWeight: 'bold', color: data.btcChange24h > 0 ? COLORS.success : data.btcChange24h < 0 ? COLORS.danger : COLORS.text.secondary }}>
                    {data.btcChange24h > 2 ? '🟢 BULLISH' : data.btcChange24h < -2 ? '🔴 BEARISH' : '🟡 NEUTRAL'}
                  </div>
                  <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '6px' }}>
                    Score: <span style={{ color: COLORS.warning, fontWeight: 'bold' }}>{Math.min(100, Math.max(0, 50 + data.btcChange24h * 10)).toFixed(0)}/100</span>
                  </div>
                </div>
              </div>

              {/* ETH Card */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.info}0D`, border: `1px solid ${COLORS.info}4D`, position: 'relative' }}>
                {/* Live Data Badge */}
                <div style={{
                  position: 'absolute',
                  top: '12px',
                  right: '12px',
                  padding: '4px 10px',
                  background: `linear-gradient(135deg, ${COLORS.success}, #059669)`,
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: '700',
                  color: '#000',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  animation: 'pulse 2s ease-in-out infinite'
                }}>
                  <div style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: '#000',
                    animation: 'blink 1.5s ease-in-out infinite'
                  }} />
                  CANLI
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                  <Icons.Activity style={{ width: '32px', height: '32px', color: COLORS.info }} />
                  <div>
                    <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>Ethereum</div>
                    <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.info }}>ETH/USDT</div>
                  </div>
                </div>
                <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '8px' }}>
                  {formatPrice(data.ethPrice)}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                  <span style={{ fontSize: '16px', fontWeight: 'bold', color: getChangeColor(data.ethChange24h) }}>
                    {(data.ethChange24h ?? 0) >= 0 ? '+' : ''}{(data.ethChange24h ?? 0).toFixed(2)}%
                  </span>
                  <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>24h</span>
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                  Volume: {formatVolume(data.ethVolume)}
                </div>
                <div style={{ marginTop: '12px', padding: '8px', background: COLORS.bg.card, borderRadius: '6px', border: `1px solid ${COLORS.info}33` }}>
                  <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>ETH Bias</div>
                  <div style={{ fontSize: '16px', fontWeight: 'bold', color: data.ethChange24h > 0 ? COLORS.success : data.ethChange24h < 0 ? COLORS.danger : COLORS.text.secondary }}>
                    {data.ethChange24h > 2 ? '🟢 BULLISH' : data.ethChange24h < -2 ? '🔴 BEARISH' : '🟡 NEUTRAL'}
                  </div>
                  <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '6px' }}>
                    Score: <span style={{ color: COLORS.info, fontWeight: 'bold' }}>{Math.min(100, Math.max(0, 50 + data.ethChange24h * 10)).toFixed(0)}/100</span>
                  </div>
                </div>
              </div>

              {/* ETH/BTC Ratio Card */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.premium}0D`, border: `1px solid ${COLORS.premium}4D` }}>
                <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '12px' }}>ETH/BTC Oranı</div>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '8px' }}>
                  {(data.ethBtcRatio ?? 0).toFixed(6)}
                </div>
                <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div>MA50: <span style={{ color: data.ethBtcRatio > data.ethBtcRatioMA50 ? COLORS.success : COLORS.danger, fontWeight: 'bold' }}>{(data.ethBtcRatioMA50 ?? 0).toFixed(6)}</span></div>
                  <div>MA200: <span style={{ color: data.ethBtcRatio > data.ethBtcRatioMA200 ? COLORS.success : COLORS.danger, fontWeight: 'bold' }}>{(data.ethBtcRatioMA200 ?? 0).toFixed(6)}</span></div>
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '12px' }}>
                  Trend: <span style={{ color: data.trend === 'Rising' ? COLORS.success : data.trend === 'Falling' ? COLORS.danger : COLORS.text.secondary, fontWeight: 'bold' }}>{data.trend === 'Rising' ? 'Yükseliş' : data.trend === 'Falling' ? 'Düşüş' : 'Kararlı'}</span>
                </div>
              </div>

              {/* BTC Dominance Card */}
              <div className="neon-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '12px' }}>BTC Dominansı</div>
                <div style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '8px' }}>
                  {(data.dominance ?? 0).toFixed(2)}%
                </div>
                <div style={{ width: '100%', height: '6px', background: COLORS.bg.card, borderRadius: '3px', marginTop: '12px' }}>
                  <div
                    style={{
                      width: `${data.dominance}%`,
                      height: '100%',
                      background: `linear-gradient(90deg, ${COLORS.warning}, ${COLORS.premium})`,
                      borderRadius: '3px',
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Confluence/Divergence Analysis */}
            <div className="neon-card" style={{ padding: '20px', marginBottom: '24px', background: `${COLORS.premium}0D`, border: `2px solid ${COLORS.premium}` }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Icons.TrendingUp style={{ width: '24px', height: '24px' }} />
                Confluence / Divergence Analizi
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>Durum</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', color: Math.abs(data.btcChange24h - data.ethChange24h) < 1 ? COLORS.success : COLORS.warning }}>
                    {Math.abs(data.btcChange24h - data.ethChange24h) < 1 ? '🤝 CONFLUENCE' : '⚡ DIVERGENCE'}
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                    {Math.abs(data.btcChange24h - data.ethChange24h) < 1
                      ? 'BTC ve ETH aynı yönde hareket ediyor'
                      : 'BTC ve ETH farklı yönlerde hareket ediyor'
                    }
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>Performans Farkı</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.premium }}>
                    {Math.abs(data.btcChange24h - data.ethChange24h).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                    {data.btcChange24h > data.ethChange24h ? '🟠 BTC daha güçlü' : '🔵 ETH daha güçlü'}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>Trading Sinyali</div>
                  <div style={{ fontSize: '18px', fontWeight: 'bold', color: data.divergence > 3 ? COLORS.warning : COLORS.success }}>
                    {data.divergence > 5
                      ? '🔥 Yüksek Fırsat'
                      : data.divergence > 3
                        ? '⚠️ Fırsat Var'
                        : '✅ Normal'
                    }
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                    {data.divergence > 3 ? 'Rotasyon ticareti düşün' : 'Normal hareket'}
                  </div>
                </div>
              </div>
            </div>

            {/* Correlation Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              {/* 30-Day Correlation */}
              <div className="neon-card" style={{ padding: '20px', background: `${getCorrelationColor(data.correlation30d)}10`, border: `1px solid ${getCorrelationColor(data.correlation30d)}` }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>30 Günlük Korelasyon</div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: getCorrelationColor(data.correlation30d), marginBottom: '8px' }}>
                  {((data.correlation30d ?? 0) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: getCorrelationColor(data.correlation30d) }}>
                  {getCorrelationLabel(data.correlation30d)}
                </div>
              </div>

              {/* 7-Day Correlation */}
              <div className="neon-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>7 Günlük Korelasyon</div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: getCorrelationColor(data.correlation7d), marginBottom: '8px' }}>
                  {((data.correlation7d ?? 0) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: getCorrelationColor(data.correlation7d) }}>
                  {getCorrelationLabel(data.correlation7d)}
                </div>
              </div>

              {/* Volume Correlation */}
              <div className="neon-card" style={{ padding: '20px' }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>Hacim Korelasyonu</div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: COLORS.cyan, marginBottom: '8px' }}>
                  {((data.volumeCorrelation ?? 0) * 100).toFixed(1)}%
                </div>
              </div>

              {/* Price Divergence */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.premium}0D`, border: `1px solid ${COLORS.premium}4D` }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>Fiyat Sapması</div>
                <div style={{ fontSize: '36px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '8px' }}>
                  {(data.divergence ?? 0).toFixed(2)}%
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                  {data.divergence > 5 ? '⚠️ Yüksek Sapma' : '✅ Normal Aralık'}
                </div>
              </div>

              {/* Lead-Lag Analysis */}
              {data.leadLagAnalysis && (
                <div className="neon-card" style={{ padding: '20px', background: `${COLORS.cyan}0D`, border: `1px solid ${COLORS.cyan}4D` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>Liderlik Analizi</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.cyan, marginBottom: '8px' }}>
                    {data.leadLagAnalysis.leader === 'BTC' ? '🟠 BTC' : data.leadLagAnalysis.leader === 'ETH' ? '🔵 ETH' : '⚖️ NEUTRAL'}
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                    Güven: <span style={{ color: COLORS.cyan, fontWeight: 'bold' }}>{data.leadLagAnalysis.confidence}%</span>
                  </div>
                </div>
              )}
            </div>

            {/* Correlation Chart (Simulated) */}
            <div className="neon-card" style={{ padding: '24px', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                📈 30 Günlük Korelasyon Geçmişi
              </h3>
              <div style={{ position: 'relative', height: '300px', background: COLORS.bg.card, borderRadius: '8px', padding: '20px' }}>
                <svg width="100%" height="100%" viewBox="0 0 800 260" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="correlationGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" style={{ stopColor: COLORS.success, stopOpacity: 0.3 }} />
                      <stop offset="100%" style={{ stopColor: COLORS.success, stopOpacity: 0 }} />
                    </linearGradient>
                  </defs>

                  {/* Grid lines */}
                  {[0, 25, 50, 75, 100].map((val) => (
                    <line
                      key={val}
                      x1="0"
                      y1={(100 - val) * 2.6}
                      x2="800"
                      y2={(100 - val) * 2.6}
                      stroke={COLORS.border.default}
                      strokeWidth="1"
                    />
                  ))}

                  {/* Correlation line */}
                  <polyline
                    fill="url(#correlationGradient)"
                    stroke={COLORS.success}
                    strokeWidth="2"
                    points={historicalCorrelation
                      .map((val, i) => `${(i / 29) * 800},${(1 - val) * 260}`)
                      .join(' ') + ` 800,260 0,260`}
                  />
                  <polyline
                    fill="none"
                    stroke={COLORS.success}
                    strokeWidth="3"
                    points={historicalCorrelation
                      .map((val, i) => `${(i / 29) * 800},${(1 - val) * 260}`)
                      .join(' ')}
                  />
                </svg>

                {/* Y-axis labels */}
                <div style={{ position: 'absolute', left: '10px', top: '10px', fontSize: '12px', color: COLORS.text.secondary }}>100%</div>
                <div style={{ position: 'absolute', left: '10px', top: '75px', fontSize: '12px', color: COLORS.text.secondary }}>75%</div>
                <div style={{ position: 'absolute', left: '10px', top: '140px', fontSize: '12px', color: COLORS.text.secondary }}>50%</div>
                <div style={{ position: 'absolute', left: '10px', bottom: '10px', fontSize: '12px', color: COLORS.text.secondary }}>0%</div>
              </div>
            </div>

            {/* ETH/BTC Ratio Chart */}
            <div className="neon-card" style={{ padding: '24px', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                📊 ETH/BTC Oranı (30 Gün)
              </h3>
              <div style={{ position: 'relative', height: '300px', background: COLORS.bg.card, borderRadius: '8px', padding: '20px' }}>
                <svg width="100%" height="100%" viewBox="0 0 800 260" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="ratioGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" style={{ stopColor: COLORS.info, stopOpacity: 0.3 }} />
                      <stop offset="100%" style={{ stopColor: COLORS.info, stopOpacity: 0 }} />
                    </linearGradient>
                  </defs>

                  {/* Grid lines */}
                  {[0, 25, 50, 75, 100].map((val) => (
                    <line
                      key={val}
                      x1="0"
                      y1={(val / 100) * 260}
                      x2="800"
                      y2={(val / 100) * 260}
                      stroke={COLORS.border.default}
                      strokeWidth="1"
                    />
                  ))}

                  {/* Ratio line */}
                  <polyline
                    fill="url(#ratioGradient)"
                    stroke={COLORS.info}
                    strokeWidth="2"
                    points={historicalRatio
                      .map((val, i) => {
                        const normalized = ((val - Math.min(...historicalRatio)) / (Math.max(...historicalRatio) - Math.min(...historicalRatio)));
                        return `${(i / 29) * 800},${(1 - normalized) * 260}`;
                      })
                      .join(' ') + ` 800,260 0,260`}
                  />
                  <polyline
                    fill="none"
                    stroke={COLORS.info}
                    strokeWidth="3"
                    points={historicalRatio
                      .map((val, i) => {
                        const normalized = ((val - Math.min(...historicalRatio)) / (Math.max(...historicalRatio) - Math.min(...historicalRatio)));
                        return `${(i / 29) * 800},${(1 - normalized) * 260}`;
                      })
                      .join(' ')}
                  />
                </svg>
              </div>
            </div>

            {/* Recommendation Panel */}
            <div className="neon-card" style={{ padding: '24px', background: `${COLORS.cyan}0D`, border: `2px solid ${COLORS.cyan}`, boxShadow: `0 0 20px ${COLORS.cyan}4D` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.warning }} />
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.warning, margin: 0 }}>
                  💡 TİCARET ÖNERİSİ
                </h3>
              </div>
              <div style={{ fontSize: '16px', color: COLORS.text.primary, lineHeight: '1.8', padding: '16px', background: COLORS.bg.card, borderRadius: '8px', border: `1px solid ${COLORS.cyan}33` }}>
                {data.recommendation}
              </div>

              {/* Trading Strategies */}
              <div style={{ marginTop: '24px' }}>
                <h4 style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.cyan, marginBottom: '12px' }}>
                  📋 ÖNERİLEN STRATEJİLER:
                </h4>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {data.correlation30d > 0.85 && (
                    <div style={{ padding: '12px', background: `${COLORS.success}1A`, border: `1px solid ${COLORS.success}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.success, marginBottom: '4px' }}>✅ Parite İşlemi</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        BTC ve ETH'yi birlikte işlem yapın. Güçlü korelasyon benzer fiyat hareketleri anlamına gelir.
                      </div>
                    </div>
                  )}
                  {data.leadLagAnalysis && data.leadLagAnalysis.leader !== 'NEUTRAL' && (
                    <div style={{ padding: '12px', background: `${COLORS.cyan}1A`, border: `1px solid ${COLORS.cyan}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.cyan, marginBottom: '4px' }}>
                        {data.leadLagAnalysis.leader === 'BTC' ? '🟠 BTC Liderlik Ediyor' : '🔵 ETH Liderlik Ediyor'}
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        {data.leadLagAnalysis.description}
                      </div>
                    </div>
                  )}
                  {data.correlation30d > 0.65 && data.correlation30d <= 0.85 && (
                    <div style={{ padding: '12px', background: `${COLORS.warning}1A`, border: `1px solid ${COLORS.warning}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.warning, marginBottom: '4px' }}>⚡ Rotasyon Oyunu</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        Liderlik değişimlerini izleyin. Biri öncülük ettiğinde, diğeri takip edebilir.
                      </div>
                    </div>
                  )}
                  {data.divergence > 5 && (
                    <div style={{ padding: '12px', background: `${COLORS.premium}1A`, border: `1px solid ${COLORS.premium}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.premium, marginBottom: '4px' }}>🎯 Sapma İşlemi</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        Yüksek sapma tespit edildi ({(data.divergence ?? 0).toFixed(2)}%). Geride kalanı toparlanma potansiyeli için işlem yapın.
                      </div>
                    </div>
                  )}
                  {data.ethBtcRatio > data.ethBtcRatioMA50 && data.ethBtcRatio > data.ethBtcRatioMA200 && (
                    <div style={{ padding: '12px', background: `${COLORS.info}1A`, border: `1px solid ${COLORS.info}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.info, marginBottom: '4px' }}>🔵 ETH Güçlü</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        ETH/BTC oranı MA50 ve MA200 üzerinde. ETH BTC'den daha iyi performans gösteriyor. ETH long pozisyon düşünülebilir.
                      </div>
                    </div>
                  )}
                  {data.ethBtcRatio < data.ethBtcRatioMA50 && data.ethBtcRatio < data.ethBtcRatioMA200 && (
                    <div style={{ padding: '12px', background: `${COLORS.warning}1A`, border: `1px solid ${COLORS.warning}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.warning, marginBottom: '4px' }}>🟠 BTC Güçlü</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        ETH/BTC oranı MA50 ve MA200 altında. BTC ETH'den daha iyi performans gösteriyor. BTC long pozisyon düşünülebilir.
                      </div>
                    </div>
                  )}
                  {data.correlation30d < 0.65 && (
                    <div style={{ padding: '12px', background: `${COLORS.danger}1A`, border: `1px solid ${COLORS.danger}4D`, borderRadius: '6px' }}>
                      <div style={{ fontWeight: 'bold', color: COLORS.danger, marginBottom: '4px' }}>⚠️ Zayıf Korelasyon Uyarısı</div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                        Piyasalar bağımsız hareket ediyor. Daha yüksek risk. Sadece bireysel tekniklere göre işlem yapın.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
            </>
          )}
        </main>
      </div>

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
                    BTC vs ETH Analizi MANTIK
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
                  BTC vs ETH Analizi sayfası, Bitcoin ve Ethereum arasındaki karşılaştırmalı analiz sunar.
                  İki en büyük kripto varlık arasındaki korelasyon, dominans ve performans metriklerini izleyerek
                  piyasa dinamiklerini anlamanızı sağlar.
                </p>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                  Bu sayfa, BTC ve ETH arasındaki fiyat hareketlerini, hacim korelasyonlarını ve oran değişimlerini
                  gerçek zamanlı olarak takip ederek, rotasyon ticareti ve parite işlemleri için önemli bilgiler sunar.
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
                    { name: 'Fiyat Karşılaştırması', desc: 'BTC ve ETH fiyatlarını yan yana görüntüler. 24 saatlik değişimleri, hacim bilgilerini ve bias (BULLISH/BEARISH/NEUTRAL) gösterir.' },
                    { name: 'Bireysel Bias ve Score', desc: 'Her coin için ayrı bias (piyasa eğilimi) ve score (performans puanı) gösterir. Anlık piyasa durumunu hızlıca anlamanızı sağlar.' },
                    { name: 'ETH/BTC Ratio + MA', desc: 'ETH/BTC oranını MA50 ve MA200 ile karşılaştırır. Hangi coinin daha güçlü performans gösterdiğini belirler.' },
                    { name: 'Korelasyon Analizi', desc: '7 günlük ve 30 günlük korelasyon metriklerini gösterir. Güçlü, orta veya zayıf korelasyon seviyelerini belirler.' },
                    { name: 'Lead-Lag Analysis', desc: 'Hangi coinin piyasaya liderlik ettiğini tespit eder (BTC, ETH veya NEUTRAL). Güven yüzdesini gösterir.' },
                    { name: 'Confluence/Divergence', desc: 'BTC ve ETH aynı yönde mi yoksa farklı yönlerde mi hareket ediyor? Performans farkını ve trading sinyallerini gösterir.' },
                    { name: 'Dominans Takibi', desc: 'BTC dominansını yüzde olarak gösterir ve piyasa liderliğinin değişimini takip eder.' },
                    { name: 'Performans Metrikleri', desc: 'ETH/BTC oranı, hacim korelasyonu ve fiyat sapması gibi detaylı performans göstergelerini sunar.' },
                    { name: 'Tarihsel Trendler', desc: '30 günlük korelasyon ve ETH/BTC oran grafiklerini görselleştirir.' },
                    { name: 'Akıllı Trade Önerileri', desc: 'Korelasyon, sapma, liderlik ve MA bazlı stratejiler sunar. Her strateji için detaylı açıklamalar gösterir.' }
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
                        BTC ve ETH Bias'larını Kontrol Edin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Her coin için BULLISH, BEARISH veya NEUTRAL bias gösterilir. Score değerleri anlık momentum gücünü gösterir.
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
                        Confluence/Divergence'ı İnceleyin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        BTC ve ETH aynı yönde mi hareket ediyor (confluence) yoksa farklı yönlerde mi (divergence)? Yüksek divergence rotasyon fırsatı sunar.
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
                        Lead-Lag Analizini Değerlendirin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        BTC mi yoksa ETH mi piyasaya liderlik ediyor? Lider coinde önce pozisyon açıp, takipçi coinde sonra işlem yapabilirsiniz.
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
                        ETH/BTC Ratio ve MA'lara Bakın
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Ratio MA50 ve MA200 üzerindeyse ETH güçlü, altındaysa BTC güçlü. Bu, hangi coinin daha iyi performans gösterdiğini belirler.
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                    <div style={{
                      background: `linear-gradient(135deg, ${COLORS.cyan}, ${COLORS.cyan}dd)`,
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
                      5
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        Akıllı Trade Önerilerini Uygulayın
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Tüm metrikleri birleştiren akıllı stratejiler sunar: Parite işlemi, liderlik-bazlı işlem, sapma işlemi ve daha fazlası.
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
                    <strong style={{ color: COLORS.text.primary }}>Gerçek Zamanlı Karşılaştırma:</strong> Tüm veriler gerçek zamanlı Binance API'sinden alınır ve sürekli güncellenir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Otomatik Yenileme (30 saniye):</strong> Sayfa her 30 saniyede bir otomatik olarak yenilenir ve güncel verileri gösterir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Market Cap Takibi:</strong> BTC dominansı toplam kripto piyasa değeri içindeki BTC payını gösterir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Korelasyon Aralığı:</strong> Korelasyon -1 ile +1 arasında değişir. +1 tamamen birlikte hareket, 0 bağımsız hareket anlamına gelir.
                  </li>
                  <li>
                    <strong style={{ color: COLORS.text.primary }}>Eğitim Amaçlıdır:</strong> Bu analizler yatırım tavsiyesi değildir. Kendi araştırmanızı yapın ve sorumlu yatırım yapın.
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
                BTC vs ETH Analizi - Bitcoin ve Ethereum Karşılaştırmalı Piyasa Analizi
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Global Animations */}
      <style jsx global>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.7;
          }
        }

        @keyframes blink {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.3;
          }
        }

        .neon-card {
          transition: all 0.3s ease;
        }

        .neon-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3);
        }

        .premium-button:hover {
          transform: scale(1.05);
          box-shadow: 0 6px 30px rgba(0, 212, 255, 0.5) !important;
        }
      `}</style>
    </div>
  );
}
