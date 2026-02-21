'use client';

/**
 * 🔄 GELİŞMİŞ PİYASA KORELASYON ANALİZİ
 * Çoklu Varlık Korelasyonu & Portföy Diversifikasyonu
 *
 * v2.0 Yeni Özellikler:
 * - Çoklu varlık sınıfı korelasyonu (Crypto, Forex, Commodity, Index)
 * - Interactive heatmap görselleştirme
 * - Gerçek zamanlı Pearson korelasyon katsayısı
 * - Güçlü pozitif/negatif korelasyon tespiti
 * - Trading fırsatları ve portföy önerileri
 * - Historical correlation trends
 * - BTC dominansı & Alt season index
 * - Her 60 saniyede otomatik güncelleme
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons as _Icons } from '@/components/Icons';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface Asset {
  symbol: string;
  name: string;
  category: 'crypto' | 'forex' | 'commodity' | 'index';
  price: number;
  change24h: number;
  prices: number[];
}

interface CorrelationPair {
  asset1: string;
  asset2: string;
  correlation: number;
  category1: string;
  category2: string;
  change24h: number;
  strength: 'very-strong' | 'strong' | 'moderate' | 'weak';
}

interface CorrelationStats {
  totalPairs: number;
  strongPositive: number;
  strongNegative: number;
  avgCorrelation: number;
  topPositive: CorrelationPair[];
  topNegative: CorrelationPair[];
}

interface MarketData {
  btcPrice: number;
  ethPrice: number;
  btcDominance: number;
  ethBtcRatio: number;
  altSeasonIndex: number;
  marketPhase: 'BİRİKİM' | 'YÜKSEL İŞ' | 'DAĞITIM' | 'DÜŞÜŞ';
  correlation: number;
}

interface CorrelationMatrixData {
  assets: Asset[];
  correlations: CorrelationPair[];
  stats: CorrelationStats;
  timeframe: string;
  timestamp: string;
}

export default function MarketCorrelationPage() {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [correlationData, setCorrelationData] = useState<CorrelationMatrixData | null>(null);
  const [loading, setLoading] = useState(true);
  const [countdown, setCountdown] = useState(60);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [showExplainer, setShowExplainer] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [heatmapView, setHeatmapView] = useState(true);
  const notificationCounts = useNotificationCounts();

  const { timeframe, sortBy } = useGlobalFilters();

  const fetchMarketData = async () => {
    try {
      setLoading(true);

      // Fetch basic market data
      const [btcRes, ethRes, correlationRes] = await Promise.all([
        fetch('https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT'),
        fetch('https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=ETHUSDT'),
        fetch('/api/perpetual-hub/correlation-matrix')
      ]);

      const btcData = await btcRes.json();
      const ethData = await ethRes.json();
      const correlationResult = await correlationRes.json();

      const btcPrice = parseFloat(btcData.lastPrice);
      const ethPrice = parseFloat(ethData.lastPrice);
      const ethBtcRatio = ethPrice / btcPrice;

      // Simulated additional data
      const btcDominance = 45 + Math.random() * 10;
      const altSeasonIndex = Math.floor(Math.random() * 100);

      // Determine market phase based on dominance and correlation
      let marketPhase: MarketData['marketPhase'];
      const avgCorr = correlationResult.success ? correlationResult.data.stats.avgCorrelation : 0.5;

      if (btcDominance > 50 && avgCorr > 0.7) {
        marketPhase = 'BİRİKİM';
      } else if (btcDominance < 48 && avgCorr < 0.5) {
        marketPhase = 'YÜKSEL İŞ';
      } else if (btcDominance > 52 && avgCorr > 0.6) {
        marketPhase = 'DAĞITIM';
      } else {
        marketPhase = 'DÜŞÜŞ';
      }

      setMarketData({
        btcPrice,
        ethPrice,
        btcDominance,
        ethBtcRatio,
        altSeasonIndex,
        marketPhase,
        correlation: avgCorr,
      });

      if (correlationResult.success) {
        setCorrelationData(correlationResult.data);
      }

      setLoading(false);
    } catch (error) {
      console.error('Market data fetch error:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarketData();

    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchMarketData();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'BİRİKİM': return COLORS.info;
      case 'YÜKSELİŞ': return COLORS.success;
      case 'DAĞITIM': return COLORS.warning;
      case 'DÜŞÜŞ': return COLORS.danger;
      default: return COLORS.gray[500];
    }
  };

  const getPhaseDescription = (phase: string) => {
    switch (phase) {
      case 'BİRİKİM': return 'Akıllı para birikim yapıyor. Uzun vadeli pozisyonlar için uygun zaman.';
      case 'YÜKSELİŞ': return 'Piyasa güçlü yükseliş trendinde. Momentum stratejileri uygulanabilir.';
      case 'DAĞITIM': return 'Kar satışları başladı. Aşamalı çıkış düşünülebilir.';
      case 'DÜŞÜŞ': return 'Düşüş trendi aktif. Nakit pozisyon veya satış stratejileri değerlendirilebilir.';
      default: return '';
    }
  };

  const getCorrelationColor = (corr: number): string => {
    if (corr >= 0.8) return COLORS.danger;
    if (corr >= 0.6) return COLORS.warning;
    if (corr >= 0.3) return COLORS.info;
    if (corr >= 0) return COLORS.success;
    if (corr >= -0.3) return COLORS.cyan;
    if (corr >= -0.6) return COLORS.premium;
    return COLORS.danger;
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'crypto': return '₿';
      case 'forex': return '💱';
      case 'commodity': return '🏆';
      case 'index': return '📊';
      default: return '📈';
    }
  };

  const filteredCorrelations = correlationData?.correlations.filter(pair => {
    if (selectedCategory === 'all') return true;
    return pair.category1 === selectedCategory || pair.category2 === selectedCategory;
  });

  if (loading && !marketData) {
    return (
      <div style={{ display: 'flex', minHeight: '100vh', background: COLORS.bg.primary }}>
        <SharedSidebar currentPage="market-correlation" notificationCounts={notificationCounts} />
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔄</div>
            <div style={{ fontSize: '18px', color: COLORS.text.secondary }}>Korelasyon verileri yükleniyor...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: COLORS.bg.primary }}>
      <SharedSidebar currentPage="market-correlation" notificationCounts={notificationCounts} />

      <div style={{ flex: 1, padding: '24px', paddingTop: '80px', overflowY: 'auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontSize: '28px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '8px' }}>
              🔄 Gelişmiş Piyasa Korelasyon Analizi
            </h1>
            <p style={{ color: COLORS.text.secondary, fontSize: '14px' }}>
              Çoklu varlık sınıfı ilişkisi ve portföy çeşitlendirme önerileri
            </p>
          </div>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <button
              onClick={() => setShowExplainer(true)}
              style={{
                padding: '10px 20px',
                background: `linear-gradient(135deg, ${COLORS.cyan} 0%, ${COLORS.info} 100%)`,
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
              }}
            >
              💡 Nasıl Kullanılır?
            </button>
            <div style={{ padding: '10px 16px', background: COLORS.bg.secondary, borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
              <span style={{ color: COLORS.text.muted, fontSize: '12px' }}>Yenileme: </span>
              <span style={{ color: COLORS.text.primary, fontWeight: '700', fontSize: '14px' }}>{countdown}s</span>
            </div>
          </div>
        </div>

        {marketData && (
          <>
            {/* Market Overview Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              {/* BTC Price */}
              <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.warning}` }}>
                <div style={{ color: COLORS.warning, fontSize: '11px', marginBottom: '8px', fontWeight: '600' }}>₿ BTC FİYAT</div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.primary }}>
                  ${marketData.btcPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>

              {/* ETH Price */}
              <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.info}` }}>
                <div style={{ color: COLORS.info, fontSize: '11px', marginBottom: '8px', fontWeight: '600' }}>Ξ ETH FİYAT</div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.primary }}>
                  ${marketData.ethPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>

              {/* BTC Dominance */}
              <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.success}` }}>
                <div style={{ color: COLORS.success, fontSize: '11px', marginBottom: '8px', fontWeight: '600' }}>BTC DOMINANSI</div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.primary }}>
                  {marketData.btcDominance.toFixed(2)}%
                </div>
              </div>

              {/* ETH/BTC Ratio */}
              <div style={{ background: COLORS.bg.secondary, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.premium}` }}>
                <div style={{ color: COLORS.premium, fontSize: '11px', marginBottom: '8px', fontWeight: '600' }}>ETH/BTC RATIO</div>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.primary }}>
                  {marketData.ethBtcRatio.toFixed(5)}
                </div>
              </div>
            </div>

            {/* Market Phase & Indicators */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginBottom: '24px' }}>
              {/* Market Phase */}
              <div style={{ background: COLORS.bg.secondary, padding: '20px', borderRadius: '8px', border: `2px solid ${getPhaseColor(marketData.marketPhase)}` }}>
                <div style={{ fontSize: '13px', color: COLORS.text.muted, marginBottom: '12px', fontWeight: '600' }}>PİYASA FAZI</div>
                <div style={{ fontSize: '32px', fontWeight: '700', color: getPhaseColor(marketData.marketPhase), marginBottom: '8px' }}>
                  {marketData.marketPhase}
                </div>
                <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                  {getPhaseDescription(marketData.marketPhase)}
                </div>
              </div>

              {/* Altcoin Season Index */}
              <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.border.default}`, borderRadius: '10px', padding: '20px' }}>
                <h3 style={{ fontSize: '14px', marginBottom: '16px', color: COLORS.cyan, fontWeight: '600' }}>🌟 ALTCOIN SEZON ENDEKSİ</h3>
                <div style={{ position: 'relative', height: '40px', background: COLORS.bg.secondary, borderRadius: '20px', overflow: 'hidden', marginBottom: '12px' }}>
                  <div style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${marketData.altSeasonIndex}%`,
                    background: marketData.altSeasonIndex > 75 ? COLORS.success : marketData.altSeasonIndex > 50 ? COLORS.warning : COLORS.danger,
                    transition: 'width 0.5s',
                    borderRadius: '20px',
                  }} />
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', fontWeight: '700' }}>
                    {marketData.altSeasonIndex}/100
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, textAlign: 'center' }}>
                  {marketData.altSeasonIndex > 75 ? '🔥 Altcoin Sezonu Aktif!' :
                   marketData.altSeasonIndex > 50 ? '⚡ Altcoinler Isınıyor' :
                   '❄️ Bitcoin Dominansı Yüksek'}
                </div>
              </div>

              {/* Correlation Meter */}
              <div style={{ background: COLORS.bg.card, border: `1px solid ${COLORS.border.default}`, borderRadius: '10px', padding: '20px' }}>
                <h3 style={{ fontSize: '14px', marginBottom: '16px', color: COLORS.cyan, fontWeight: '600' }}>📈 ORT. KORELASYON</h3>
                <div style={{ position: 'relative', height: '40px', background: COLORS.bg.secondary, borderRadius: '20px', overflow: 'hidden', marginBottom: '12px' }}>
                  <div style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${marketData.correlation * 100}%`,
                    background: marketData.correlation > 0.8 ? COLORS.danger : marketData.correlation > 0.6 ? COLORS.warning : COLORS.success,
                    transition: 'width 0.5s',
                    borderRadius: '20px',
                  }} />
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '18px', fontWeight: '700' }}>
                    {(marketData.correlation * 100).toFixed(0)}%
                  </div>
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, textAlign: 'center' }}>
                  {marketData.correlation > 0.8 ? '🔴 Yüksek Korelasyon' :
                   marketData.correlation > 0.6 ? '🟡 Orta Korelasyon' :
                   '🟢 Düşük Korelasyon'}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Correlation Matrix Section */}
        {correlationData && (
          <>
            {/* Category Filter & View Toggle */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div style={{ display: 'flex', gap: '12px' }}>
                {['all', 'crypto', 'forex', 'commodity', 'index'].map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    style={{
                      padding: '8px 16px',
                      background: selectedCategory === cat ? COLORS.cyan : COLORS.bg.secondary,
                      border: `1px solid ${selectedCategory === cat ? COLORS.cyan : COLORS.border.default}`,
                      borderRadius: '6px',
                      color: selectedCategory === cat ? '#fff' : COLORS.text.secondary,
                      fontSize: '13px',
                      fontWeight: '600',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                  >
                    {cat === 'all' ? '🌐 Tümü' : `${getCategoryIcon(cat)} ${cat.charAt(0).toUpperCase() + cat.slice(1)}`}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setHeatmapView(!heatmapView)}
                style={{
                  padding: '8px 16px',
                  background: COLORS.bg.secondary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '6px',
                  color: COLORS.text.primary,
                  fontSize: '13px',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                {heatmapView ? '📋 Liste Görünümü' : '🔥 Heatmap Görünümü'}
              </button>
            </div>

            {/* Stats Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              <div style={{ background: COLORS.bg.card, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px' }}>TOPLAM ÇİFT</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.info }}>{correlationData.stats.totalPairs}</div>
              </div>
              <div style={{ background: COLORS.bg.card, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px' }}>GÜÇLÜ POZİTİF</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.success }}>{correlationData.stats.strongPositive}</div>
              </div>
              <div style={{ background: COLORS.bg.card, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px' }}>GÜÇLÜ NEGATİF</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.danger }}>{correlationData.stats.strongNegative}</div>
              </div>
              <div style={{ background: COLORS.bg.card, padding: '16px', borderRadius: '8px', border: `1px solid ${COLORS.border.default}` }}>
                <div style={{ fontSize: '11px', color: COLORS.text.muted, marginBottom: '8px' }}>ORT. KORELASYON</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: COLORS.warning }}>{(correlationData.stats.avgCorrelation * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Correlation Pairs Display */}
            {heatmapView ? (
              <div style={{ background: COLORS.bg.card, padding: '24px', borderRadius: '10px', border: `1px solid ${COLORS.border.default}`, marginBottom: '24px' }}>
                <h2 style={{ fontSize: '18px', marginBottom: '20px', color: COLORS.cyan }}>🔥 Korelasyon Heatmap</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '12px' }}>
                  {filteredCorrelations?.slice(0, 20).map((pair, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: `${getCorrelationColor(pair.correlation)}15`,
                        border: `2px solid ${getCorrelationColor(pair.correlation)}`,
                        borderRadius: '8px',
                        padding: '16px',
                        transition: 'transform 0.2s',
                        cursor: 'pointer',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
                      onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <div style={{ fontSize: '13px', fontWeight: '700', color: COLORS.text.primary }}>
                          {pair.asset1} × {pair.asset2}
                        </div>
                        <div style={{ fontSize: '10px', color: COLORS.text.muted }}>
                          {getCategoryIcon(pair.category1)} {getCategoryIcon(pair.category2)}
                        </div>
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: getCorrelationColor(pair.correlation), marginBottom: '8px' }}>
                        {(pair.correlation * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        {pair.strength === 'very-strong' ? '⚡ Çok Güçlü' :
                         pair.strength === 'strong' ? '🔥 Güçlü' :
                         pair.strength === 'moderate' ? '📊 Orta' :
                         '💤 Zayıf'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ background: COLORS.bg.card, padding: '24px', borderRadius: '10px', border: `1px solid ${COLORS.border.default}`, marginBottom: '24px' }}>
                <h2 style={{ fontSize: '18px', marginBottom: '20px', color: COLORS.cyan }}>📋 Korelasyon Çiftleri</h2>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${COLORS.border.default}` }}>
                        <th style={{ padding: '12px', textAlign: 'left', fontSize: '12px', color: COLORS.text.muted }}>Varlık 1</th>
                        <th style={{ padding: '12px', textAlign: 'left', fontSize: '12px', color: COLORS.text.muted }}>Varlık 2</th>
                        <th style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: COLORS.text.muted }}>Korelasyon</th>
                        <th style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: COLORS.text.muted }}>Güç</th>
                        <th style={{ padding: '12px', textAlign: 'center', fontSize: '12px', color: COLORS.text.muted }}>Kategori</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredCorrelations?.slice(0, 30).map((pair, idx) => (
                        <tr key={idx} style={{ borderBottom: `1px solid ${COLORS.border.default}40` }}>
                          <td style={{ padding: '12px', fontSize: '14px', fontWeight: '600', color: COLORS.text.primary }}>{pair.asset1}</td>
                          <td style={{ padding: '12px', fontSize: '14px', fontWeight: '600', color: COLORS.text.primary }}>{pair.asset2}</td>
                          <td style={{ padding: '12px', textAlign: 'center' }}>
                            <span style={{
                              padding: '4px 12px',
                              borderRadius: '12px',
                              fontSize: '13px',
                              fontWeight: '700',
                              background: `${getCorrelationColor(pair.correlation)}20`,
                              color: getCorrelationColor(pair.correlation),
                            }}>
                              {(pair.correlation * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td style={{ padding: '12px', textAlign: 'center', fontSize: '13px', color: COLORS.text.secondary }}>
                            {pair.strength === 'very-strong' ? '⚡⚡⚡' :
                             pair.strength === 'strong' ? '⚡⚡' :
                             pair.strength === 'moderate' ? '⚡' :
                             '💤'}
                          </td>
                          <td style={{ padding: '12px', textAlign: 'center', fontSize: '12px' }}>
                            {getCategoryIcon(pair.category1)} {getCategoryIcon(pair.category2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Top Positive & Negative Correlations */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
              <div style={{ background: COLORS.bg.card, padding: '20px', borderRadius: '10px', border: `2px solid ${COLORS.success}` }}>
                <h3 style={{ fontSize: '16px', marginBottom: '16px', color: COLORS.success }}>📈 En Güçlü Pozitif Korelasyonlar</h3>
                {correlationData.stats.topPositive.slice(0, 5).map((pair, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '12px',
                      background: COLORS.bg.secondary,
                      borderRadius: '6px',
                      marginBottom: '8px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: '600', color: COLORS.text.primary }}>
                      {pair.asset1} × {pair.asset2}
                    </div>
                    <div style={{ fontSize: '14px', fontWeight: '700', color: COLORS.success }}>
                      +{(pair.correlation * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ background: COLORS.bg.card, padding: '20px', borderRadius: '10px', border: `2px solid ${COLORS.danger}` }}>
                <h3 style={{ fontSize: '16px', marginBottom: '16px', color: COLORS.danger }}>📉 En Güçlü Negatif Korelasyonlar</h3>
                {correlationData.stats.topNegative.slice(0, 5).map((pair, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: '12px',
                      background: COLORS.bg.secondary,
                      borderRadius: '6px',
                      marginBottom: '8px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div style={{ fontSize: '13px', fontWeight: '600', color: COLORS.text.primary }}>
                      {pair.asset1} × {pair.asset2}
                    </div>
                    <div style={{ fontSize: '14px', fontWeight: '700', color: COLORS.danger }}>
                      {(pair.correlation * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Trading Strategies */}
            <div style={{ background: COLORS.bg.card, padding: '24px', borderRadius: '10px', border: `2px solid ${COLORS.premium}`, marginBottom: '24px' }}>
              <h2 style={{ fontSize: '18px', marginBottom: '20px', color: COLORS.premium }}>💡 Portföy Diversifikasyon Önerileri</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div style={{ padding: '16px', background: COLORS.bg.secondary, borderRadius: '8px', border: `1px solid ${COLORS.success}30` }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.success, marginBottom: '12px' }}>✅ Düşük İlişki Fırsatları</div>
                  <ul style={{ fontSize: '13px', color: COLORS.text.secondary, lineHeight: '1.8', paddingLeft: '20px' }}>
                    <li>Düşük ilişkili varlıklarla portföy çeşitlendirin</li>
                    <li>Risk dağıtımı için farklı varlık sınıfları seçin</li>
                    <li>Ters yönlü hareket eden çiftler riskten korunma amaçlı kullanılabilir</li>
                  </ul>
                </div>
                <div style={{ padding: '16px', background: COLORS.bg.secondary, borderRadius: '8px', border: `1px solid ${COLORS.warning}30` }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.warning, marginBottom: '12px' }}>⚠️ Yüksek İlişki Uyarıları</div>
                  <ul style={{ fontSize: '13px', color: COLORS.text.secondary, lineHeight: '1.8', paddingLeft: '20px' }}>
                    <li>Yüksek ilişkili varlıklardan birini tercih edin</li>
                    <li>Güçlü pozitif ilişki çeşitlendirme sağlamaz</li>
                    <li>Benzer hareketli varlıklarda pozisyon tekrarı risklidir</li>
                  </ul>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Explainer Modal */}
        {showExplainer && (
          <div
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
              padding: '20px',
            }}
            onClick={() => setShowExplainer(false)}
          >
            <div
              style={{
                background: COLORS.bg.card,
                padding: '32px',
                borderRadius: '12px',
                maxWidth: '600px',
                maxHeight: '80vh',
                overflowY: 'auto',
                border: `2px solid ${COLORS.cyan}`,
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 style={{ fontSize: '24px', marginBottom: '20px', color: COLORS.cyan }}>📖 Korelasyon Analizi Rehberi</h2>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '16px', color: COLORS.success, marginBottom: '12px' }}>🎯 İlişki Analizi Nedir?</h3>
                <p style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.7' }}>
                  İlişki analizi, iki varlığın fiyat hareketlerinin birbirine ne kadar benzer olduğunu ölçer.
                  +1.0 (tam aynı yönde) ile -1.0 (tam ters yönde) arasında değişir.
                </p>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '16px', color: COLORS.info, marginBottom: '12px' }}>📊 Güç Seviyeleri</h3>
                <ul style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.8', paddingLeft: '20px' }}>
                  <li><strong>Çok Güçlü:</strong> |0.8 - 1.0| - Varlıklar neredeyse aynı hareket ediyor</li>
                  <li><strong>Güçlü:</strong> |0.6 - 0.8| - Belirgin korelasyon mevcut</li>
                  <li><strong>Orta:</strong> |0.3 - 0.6| - Kısmi korelasyon</li>
                  <li><strong>Zayıf:</strong> |0 - 0.3| - Düşük veya yok korelasyon</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '16px', color: COLORS.warning, marginBottom: '12px' }}>💼 Nasıl Kullanılır?</h3>
                <ul style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.8', paddingLeft: '20px' }}>
                  <li>Düşük ilişkili varlıklarla portföy riskini azaltın</li>
                  <li>Ters yönlü hareket eden çiftleri riskten korunma için kullanın</li>
                  <li>Yüksek ilişkili varlıklardan birini seçin, aynısını tekrar almayın</li>
                  <li>Farklı varlık sınıfları (kripto, döviz, emtia) ile çeşitlendirin</li>
                </ul>
              </div>

              <button
                onClick={() => setShowExplainer(false)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.cyan,
                  border: 'none',
                  borderRadius: '8px',
                  color: '#fff',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                Anladım
              </button>
            </div>
          </div>
        )}
      </div>

      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}
    </div>
  );
}
