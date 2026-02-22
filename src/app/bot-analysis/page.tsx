/**
 * 🤖 BOT ANALİZİ - KAPSAMLI LONG SİNYAL TESPİTİ
 *
 * FAZ 1 - TAM UYGULAMA:
 * - Emir Defteri Dengesizliği ve Balina Duvarları
 * - Fonlama Oranı Analizi
 * - Açık Pozisyon ve CVD (Kümülatif Hacim Deltası)
 * - Bileşik LONG Sinyal Puanlaması
 * - Likİdasyon Isı Haritası ve Kümeleri
 * - Pozisyon Boyut Hesaplayıcısı (Risk Yönetimi)
 *
 * FAZ 2 - PREMİUM 2 SÜTUNLU CAM MORFİZM DÜZENİ:
 * - Geri sayım ile otomatik yenileme
 * - 24 saatlik sinyal geçmişi grafiği
 * - Modern premium tasarım
 *
 * FAZ 6 - PREMİUM YENİDEN TASARIM:
 * - Mükemmel Long Fırsatları ÜSTTE (tam genişlik)
 * - Kalan widget'lar için 2 sütunlu ızgara düzeni
 * - Her widget için premium renk paleti
 * - Parlayan kenarlıklı gradyan arka planlar
 * - Duyarlı tasarım
 *
 * BEYAZ ŞAPKA İLKELERİ:
 * - Salt okunur piyasa analizi
 * - Sadece eğitim amaçlı
 * - Otomatik ticaret yürütmesi yok
 */

'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { SharedSidebar } from '@/components/SharedSidebar';
import { Icons } from '@/components/Icons';
import { CVDWidget } from '@/components/CVDWidget';
import { CZBinanceSentimentWidget } from '@/components/CZBinanceSentimentWidget';

// Dynamic imports for scanner components (with SSR disabled for client-side functionality)
const MultiTimeframeScanner = dynamic(
  () => import('@/components/Scanner/MultiTimeframeScanner').then((mod) => ({ default: mod.MultiTimeframeScanner })),
  {
    ssr: false,
    loading: () => (
      <div style={{ padding: '40px', textAlign: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
        Multi-Timeframe Scanner yükleniyor...
      </div>
    )
  }
);

const TopGainersTracker = dynamic(
  () => import('@/components/TopGainersTracker').then((mod) => ({ default: mod.TopGainersTracker })),
  {
    ssr: false,
    loading: () => (
      <div style={{ padding: '40px', textAlign: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
        Top Gainers Tracker yükleniyor...
      </div>
    )
  }
);

const WhaleTransactionWidget = dynamic(
  () => import('@/components/WhaleTransactionWidget').then((mod) => ({ default: mod.WhaleTransactionWidget })),
  {
    ssr: false,
    loading: () => (
      <div style={{ padding: '40px', textAlign: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
        🐋 Whale Tracker yükleniyor...
      </div>
    )
  }
);

// ============================================================================
// YARDIMCI FONKSİYONLAR (daha iyi performans için bileşen dışında tanımlanmıştır)
// ============================================================================

// Kalite rengini al
const getQualityColor = (quality: string) => {
  switch (quality) {
    case 'EXCELLENT': return '#10B981';
    case 'GOOD': return '#3B82F6';
    case 'MODERATE': return '#F59E0B';
    case 'POOR': return '#EF4444';
    case 'NONE': return '#6B7280';
    default: return '#6B7280';
  }
};

// Kalite Türkçe çevirisini al
const getQualityTurkish = (quality: string) => {
  switch (quality) {
    case 'EXCELLENT': return 'MÜKEMMEL';
    case 'GOOD': return 'İYİ';
    case 'MODERATE': return 'ORTA';
    case 'POOR': return 'ZAYIF';
    case 'NONE': return 'YOK';
    default: return quality;
  }
};

// Hızlı erişim için popüler semboller
const POPULAR_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'DOGEUSDT', 'DOTUSDT'];

export default function BotAnalysisPage() {
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Helper function to check if a coin is bull ready
  const isBullReady = (symbol: string, bullReadyData: any): boolean => {
    if (!bullReadyData || !bullReadyData.bullReadySignals) return false;
    return bullReadyData.bullReadySignals.some((signal: any) => signal.symbol === symbol && signal.isBullReady);
  };

  // Helper function to get peak hours for a coin
  const getPeakHours = (symbol: string): string | null => {
    if (!peakHoursData || !peakHoursData.analyses) return null;
    const analysis = peakHoursData.analyses.find((a: any) => a.symbol === symbol);
    if (!analysis || !analysis.bestHours || analysis.bestHours.length === 0) return null;

    // Return top 3 best hours in Turkey time
    return analysis.bestHours
      .slice(0, 3)
      .map((h: any) => `${h.hour.toString().padStart(2, '0')}:00`)
      .join(', ');
  };

  const [mounted, setMounted] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  // State for all data sources
  const [orderbookData, setOrderbookData] = useState<any>(null);
  const [fundingData, setFundingData] = useState<any>(null);
  const [oiData, setOiData] = useState<any>(null);
  const [longSignalData, setLongSignalData] = useState<any>(null);
  const [liquidationData, setLiquidationData] = useState<any>(null);
  const [positionCalcData, setPositionCalcData] = useState<any>(null);

  const [loading, setLoading] = useState(false);
  const [allSymbols, setAllSymbols] = useState<Array<{ symbol: string; baseAsset: string; displayName: string }>>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);

  // Position calculator inputs
  const [accountBalance, setAccountBalance] = useState(10000);
  const [riskPercent, setRiskPercent] = useState(1.5);

  // PHASE 2: Auto-refresh state
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState(300); // 5 minutes = 300 seconds

  // PHASE 2: Signal History state
  const [signalHistory, setSignalHistory] = useState<Array<{ timestamp: number; score: number; quality: string }>>([]);

  // PHASE 2: Whale Transaction state
  const [whaleData, setWhaleData] = useState<any>(null);

  // PHASE 3: Perfect Long Opportunities state
  const [perfectLongsData, setPerfectLongsData] = useState<any>(null);
  const [perfectLongsSearch, setPerfectLongsSearch] = useState('');

  // PHASE 4: Bull Ready Momentum state
  const [bullReadyData, setBullReadyData] = useState<any>(null);

  // PHASE 5: Peak Hours Analysis state
  const [peakHoursData, setPeakHoursData] = useState<any>(null);

  // PHASE 7: CVD (Cumulative Volume Delta) state
  const [cvdData, setCvdData] = useState<any>(null);

  // MANTIK Modal state
  const [showLogicModal, setShowLogicModal] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch all available symbols on mount
  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const response = await fetch('/api/bot-analysis/symbols');
        const data = await response.json();
        if (data.success) {
          setAllSymbols(data.data.symbols);
        }
      } catch (error) {
        console.error('Failed to fetch symbols:', error);
      }
    };

    if (mounted) {
      fetchSymbols();
    }
  }, [mounted]);

  // Fetch all bot analysis data when symbol changes
  useEffect(() => {
    if (mounted) {
      fetchAllAnalysis();
    }
  }, [mounted, selectedSymbol]);

  // PHASE 2: Auto-refresh every 5 minutes
  useEffect(() => {
    if (!mounted || !autoRefreshEnabled) return;

    const interval = setInterval(() => {
      console.log('[Auto-Refresh] Refreshing data for', selectedSymbol);
      fetchAllAnalysis();
    }, 300000); // 5 minutes = 300 seconds

    return () => clearInterval(interval);
  }, [mounted, selectedSymbol, autoRefreshEnabled]);

  // PHASE 2: Countdown timer (updates every second)
  useEffect(() => {
    if (!mounted || !autoRefreshEnabled) return;

    const timer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          return 300; // Reset to 300 (5 minutes)
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [mounted, autoRefreshEnabled]);

  // PHASE 2: Reset countdown when data is fetched
  useEffect(() => {
    if (lastUpdated) {
      setCountdown(300); // Reset to 5 minutes
    }
  }, [lastUpdated]);

  const fetchAllAnalysis = async () => {
    setLoading(true);
    try {
      // Fetch all data sources in parallel (PHASE 7: Added CVD analysis)
      const [obRes, fundRes, oiRes, longRes, liqRes, whaleRes, perfectLongsRes, bullReadyRes, cvdRes] = await Promise.all([
        fetch(`/api/bot-analysis/orderbook?symbol=${selectedSymbol}`),
        fetch(`/api/bot-analysis/funding?symbol=${selectedSymbol}`),
        fetch(`/api/bot-analysis/open-interest?symbol=${selectedSymbol}`),
        fetch(`/api/bot-analysis/long-signal?symbol=${selectedSymbol}`),
        fetch(`/api/bot-analysis/liquidations?symbol=${selectedSymbol}`),
        fetch(`/api/bot-analysis/whale-transactions?symbol=${selectedSymbol}&minSize=500000`),
        fetch('/api/bot-analysis/perfect-longs'),
        fetch('/api/bot-analysis/bull-ready?limit=100'),
        fetch(`/api/bot-analysis/cvd?symbol=${selectedSymbol}`)
      ]);

      const [obData, fundData, oiDataRes, longData, liqData, whaleDataRes, perfectLongsDataRes, bullReadyDataRes, cvdDataRes] = await Promise.all([
        obRes.json(),
        fundRes.json(),
        oiRes.json(),
        longRes.json(),
        liqRes.json(),
        whaleRes.json(),
        perfectLongsRes.json(),
        bullReadyRes.json(),
        cvdRes.json()
      ]);

      setOrderbookData(obData.success ? obData.data : null);
      setFundingData(fundData.success ? fundData.data : null);
      setOiData(oiDataRes.success ? oiDataRes.data : null);
      setLongSignalData(longData.success ? longData.data : null);
      setLiquidationData(liqData.success ? liqData.data : null);
      setWhaleData(whaleDataRes.success ? whaleDataRes.data : null);
      setPerfectLongsData(perfectLongsDataRes.success ? perfectLongsDataRes.data : null);
      setBullReadyData(bullReadyDataRes.success ? bullReadyDataRes.data : null);
      setCvdData(cvdDataRes.success ? cvdDataRes.data : null);

      // PHASE 5: Fetch peak hours for Perfect Long opportunities
      if (perfectLongsDataRes.success && perfectLongsDataRes.data.opportunities && perfectLongsDataRes.data.opportunities.length > 0) {
        try {
          const symbols = perfectLongsDataRes.data.opportunities.map((opp: any) => opp.symbol);
          console.log('[Peak Hours] Fetched data for', symbols.length, 'symbols');
          const peakHoursRes = await fetch(`/api/bot-analysis/peak-hours?symbols=${symbols.join(',')}`);
          const peakHoursDataRes = await peakHoursRes.json();
          setPeakHoursData(peakHoursDataRes.success ? peakHoursDataRes.data : null);
        } catch (error) {
          console.error('[Peak Hours] Failed to fetch peak hours data:', error);
          setPeakHoursData(null);
        }
      }

      // Fetch position calculator with signal score
      if (longData.success) {
        const posCalcRes = await fetch(
          `/api/bot-analysis/position-calculator?symbol=${selectedSymbol}&accountBalance=${accountBalance}&riskPercent=${riskPercent}&signalScore=${longData.data.overallScore}`
        );
        const posCalcData = await posCalcRes.json();
        setPositionCalcData(posCalcData.success ? posCalcData.data : null);
      }

      // PHASE 2: Update last updated timestamp
      setLastUpdated(new Date());

      // PHASE 2: Add signal to history (POST to signal-history API)
      if (longData.success) {
        try {
          await fetch('/api/bot-analysis/signal-history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              symbol: selectedSymbol,
              score: longData.data.overallScore,
              quality: longData.data.quality
            })
          });

          // Fetch updated history (GET from signal-history API)
          const historyRes = await fetch(`/api/bot-analysis/signal-history?symbol=${selectedSymbol}&limit=100`);
          const historyData = await historyRes.json();
          if (historyData.success && historyData.data.entries) {
            setSignalHistory(historyData.data.entries);
          }
        } catch (error) {
          console.error('Failed to update signal history:', error);
        }
      }
    } catch (error) {
      console.error('Failed to fetch bot analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  // BUG FIX: Recalculate position when inputs change - REAL-TIME UPDATE
  const recalculatePosition = async () => {
    if (!longSignalData) return;

    try {
      const res = await fetch(
        `/api/bot-analysis/position-calculator?symbol=${selectedSymbol}&accountBalance=${accountBalance}&riskPercent=${riskPercent}&signalScore=${longSignalData.overallScore}`
      );
      const data = await res.json();
      setPositionCalcData(data.success ? data.data : null);
    } catch (error) {
      console.error('Failed to recalculate position:', error);
    }
  };

  // BUG FIX: Auto-recalculate when account balance or risk % changes
  useEffect(() => {
    if (longSignalData && positionCalcData) {
      recalculatePosition();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accountBalance, riskPercent]);

  // Filter symbols based on search term (memoized to avoid recomputation)
  const filteredSymbols = useMemo(() => {
    return allSymbols.filter(s =>
      s.baseAsset.toLowerCase().includes(searchTerm.toLowerCase()) ||
      s.symbol.toLowerCase().includes(searchTerm.toLowerCase())
    ).slice(0, 50);
  }, [allSymbols, searchTerm]);

  if (!mounted) {
    return (
      <div style={{ minHeight: '100vh', background: '#0A0A0A', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'rgba(255,255,255,0.6)' }}>Yükleniyor...</div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0A0A0A', paddingTop: isLocalhost ? '116px' : '60px' }}>
      <SharedSidebar currentPage="bot-analysis" />

      {/* Main Content */}
      <div style={{ paddingTop: '60px', padding: '76px 16px 16px 16px', maxWidth: '1800px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', marginBottom: '8px', flexWrap: 'wrap' }}>
            <h1 style={{ fontSize: '32px', fontWeight: '700', color: '#FFFFFF', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Icons.Bot style={{ width: '36px', height: '36px', color: '#00D4FF' }} />
              Bot Analizi - Kapsamlı LONG Sinyal Tespiti
            </h1>
            <div>
              <style>{`
                @media (max-width: 768px) {
                  .mantik-button-bot {
                    padding: 10px 20px !important;
                    fontSize: 13px !important;
                    height: 42px !important;
                  }
                  .mantik-button-bot span {
                    fontSize: 18px !important;
                  }
                }
                @media (max-width: 480px) {
                  .mantik-button-bot {
                    padding: 8px 16px !important;
                    fontSize: 12px !important;
                    height: 40px !important;
                  }
                  .mantik-button-bot span {
                    fontSize: 16px !important;
                  }
                }
              `}</style>
              <button onClick={() => setShowLogicModal(true)} className="mantik-button-bot" style={{background: 'linear-gradient(135deg, #8B5CF6, #7C3AED)', border: '2px solid rgba(139, 92, 246, 0.5)', borderRadius: '10px', padding: '12px 24px', color: '#FFFFFF', fontSize: '14px', fontWeight: '700', cursor: 'pointer', transition: 'all 0.3s', boxShadow: '0 4px 16px rgba(139, 92, 246, 0.3)', display: 'flex', alignItems: 'center', gap: '8px', height: '44px'}} onMouseEnter={(e) => {e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 6px 24px rgba(139, 92, 246, 0.5)';}} onMouseLeave={(e) => {e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(139, 92, 246, 0.3)';}}>
                <span style={{ fontSize: '18px' }}>🧠</span>MANTIK
              </button>
            </div>
          </div>
          <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', lineHeight: '1.6' }}>
            FAZ 6 Premium Tasarım TAMAMLANDI: Tüm Widgetlar + Order Book + Funding + OI/CVD + Liquidations
          </p>
        </div>

        {/* PHASE 2: Auto-Refresh Controls */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(8, 145, 178, 0.05) 100%)',
          border: '1px solid rgba(6, 182, 212, 0.3)',
          borderRadius: '12px',
          padding: '16px 20px',
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
          boxShadow: '0 4px 20px rgba(6, 182, 212, 0.15)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: autoRefreshEnabled ? '#10B981' : '#6B7280',
                boxShadow: autoRefreshEnabled ? '0 0 10px #10B981' : 'none',
                animation: autoRefreshEnabled ? 'pulse 2s infinite' : 'none'
              }} />
              <span style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                Otomatik Yenileme
              </span>
            </div>

            <button
              onClick={() => setAutoRefreshEnabled(!autoRefreshEnabled)}
              style={{
                padding: '8px 16px',
                background: autoRefreshEnabled ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                border: `1px solid ${autoRefreshEnabled ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                borderRadius: '8px',
                color: autoRefreshEnabled ? '#EF4444' : '#10B981',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s'
              }}
            >
              {autoRefreshEnabled ? 'DURDUR' : 'BAŞLAT'}
            </button>

            {autoRefreshEnabled && (
              <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)' }}>
                Sonraki güncelleme: <span style={{ color: '#06B6D4', fontWeight: '600' }}>{countdown}s</span>
              </div>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            {lastUpdated && (
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Son güncelleme: <span style={{ color: '#FFFFFF', fontWeight: '500' }}>
                  {lastUpdated.toLocaleTimeString('tr-TR')}
                </span>
              </div>
            )}

            <button
              onClick={() => fetchAllAnalysis()}
              disabled={loading}
              style={{
                padding: '8px 16px',
                background: loading ? 'rgba(107, 114, 128, 0.1)' : 'rgba(6, 182, 212, 0.1)',
                border: `1px solid ${loading ? 'rgba(107, 114, 128, 0.3)' : 'rgba(6, 182, 212, 0.3)'}`,
                borderRadius: '8px',
                color: loading ? '#6B7280' : '#06B6D4',
                fontSize: '13px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.3s',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <Icons.Refresh style={{ width: '14px', height: '14px' }} />
              {loading ? 'Yükleniyor...' : 'Manuel Yenile'}
            </button>
          </div>
        </div>

        {/* Symbol Selector */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(10, 10, 10, 0.9) 100%)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '16px',
          padding: '20px',
          marginBottom: '24px'
        }}>
          <div style={{ fontSize: '14px', fontWeight: '600', color: 'rgba(255, 255, 255, 0.8)', marginBottom: '12px' }}>
            Coin Seçimi ({allSymbols.length} USDT-M Perpetual):
          </div>

          {/* Search Input */}
          <div style={{ position: 'relative', marginBottom: '16px' }}>
            <input
              type="text"
              placeholder="Coin ara... (örn: BTC, ETH, SOL)"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setShowDropdown(e.target.value.length > 0);
              }}
              onFocus={() => setShowDropdown(searchTerm.length > 0)}
              style={{
                width: '100%',
                padding: '12px 16px',
                background: 'rgba(0, 0, 0, 0.4)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '14px',
                outline: 'none'
              }}
            />

            {/* Search Dropdown */}
            {showDropdown && filteredSymbols.length > 0 && (
              <div style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                marginTop: '8px',
                background: 'rgba(20, 20, 20, 0.98)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                maxHeight: '300px',
                overflowY: 'auto',
                zIndex: 1000,
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
              }}>
                {filteredSymbols.map(s => (
                  <button
                    key={s.symbol}
                    onClick={() => {
                      setSelectedSymbol(s.symbol);
                      setSearchTerm('');
                      setShowDropdown(false);
                    }}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      background: 'transparent',
                      border: 'none',
                      color: '#FFFFFF',
                      fontSize: '14px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'background 0.2s',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(0, 212, 255, 0.1)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <span style={{ fontWeight: '600', color: '#00D4FF' }}>{s.baseAsset}</span>
                    <span style={{ color: 'rgba(255, 255, 255, 0.5)', marginLeft: '8px' }}>USDT</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Popular Coins Quick Select */}
          <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
            Popüler Coinler:
          </div>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {POPULAR_SYMBOLS.map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                style={{
                  padding: '10px 20px',
                  background: selectedSymbol === symbol
                    ? 'linear-gradient(135deg, #00D4FF 0%, #0EA5E9 100%)'
                    : 'rgba(255, 255, 255, 0.05)',
                  border: selectedSymbol === symbol ? 'none' : '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
              >
                {symbol.replace('USDT', '')}
              </button>
            ))}
          </div>

          {/* Currently Selected */}
          <div style={{
            marginTop: '16px',
            padding: '12px',
            background: 'rgba(0, 212, 255, 0.1)',
            borderRadius: '8px',
            border: '1px solid rgba(0, 212, 255, 0.2)'
          }}>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
              Seçili Coin:
            </div>
            <div style={{ fontSize: '18px', fontWeight: '700', color: '#00D4FF' }}>
              {selectedSymbol}
            </div>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '60px', color: 'rgba(255, 255, 255, 0.6)' }}>
            <div style={{ fontSize: '18px', marginBottom: '12px' }}>Tüm veri kaynakları analiz ediliyor...</div>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.4)' }}>
              Emir Defteri • Funding • Açık Pozisyon • CVD • Likidasyonlar • LONG Sinyali • Pozisyon Hesaplayıcı
            </div>
          </div>
        )}

        {/* ===== PHASE 6: PREMIUM REDESIGNED LAYOUT ===== */}
        {!loading && longSignalData && (
          <>
            {/* ===== PERFECT LONG OPPORTUNITIES - FULL WIDTH TOP ===== */}
            {perfectLongsData && perfectLongsData.opportunities && perfectLongsData.opportunities.length > 0 && (
              <div style={{
                marginBottom: '24px',
                background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.10) 100%)',
                backdropFilter: 'blur(20px)',
                border: '2px solid rgba(245, 158, 11, 0.4)',
                borderRadius: '12px',
                padding: '24px',
                boxShadow: '0 4px 24px rgba(245, 158, 11, 0.2)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div style={{
                      width: '48px',
                      height: '48px',
                      background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
                      borderRadius: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)'
                    }}>
                      <span style={{ fontSize: '28px' }}>🎯</span>
                    </div>
                    <div>
                      <h2 style={{ fontSize: '22px', fontWeight: '700', color: '#F59E0B', marginBottom: '4px' }}>
                        MÜKEMMEL LONG FIRSATLARI
                      </h2>
                      <div style={{ fontSize: '13px', color: 'rgba(245, 158, 11, 0.8)' }}>
                        Shortçular longlara funding fee ödüyor • {perfectLongsData.count} coin tespit edildi
                      </div>
                    </div>
                  </div>

                  {/* Total Count Badge */}
                  <div style={{
                    padding: '12px 20px',
                    background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
                    borderRadius: '12px',
                    boxShadow: '0 4px 12px rgba(245, 158, 11, 0.3)'
                  }}>
                    <div style={{ fontSize: '13px', color: 'rgba(0, 0, 0, 0.7)', fontWeight: '600', marginBottom: '2px' }}>
                      TOPLAM FIRSAT
                    </div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#000000' }}>
                      {perfectLongsData.count}
                    </div>
                  </div>
                </div>

                {/* Search Box */}
                <div style={{ marginBottom: '20px' }}>
                  <input
                    type="text"
                    placeholder="🔍 Coin ara (örn: BTC, ETH, DASH...)"
                    value={perfectLongsSearch}
                    onChange={(e) => setPerfectLongsSearch(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '14px 20px',
                      background: 'rgba(245, 158, 11, 0.05)',
                      border: '2px solid rgba(245, 158, 11, 0.3)',
                      borderRadius: '12px',
                      fontSize: '15px',
                      fontWeight: '500',
                      color: '#F59E0B',
                      outline: 'none',
                      transition: 'all 0.3s ease'
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.6)';
                      e.currentTarget.style.background = 'rgba(245, 158, 11, 0.1)';
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                      e.currentTarget.style.background = 'rgba(245, 158, 11, 0.05)';
                    }}
                  />
                  {perfectLongsSearch && (
                    <div style={{
                      marginTop: '8px',
                      fontSize: '13px',
                      color: 'rgba(245, 158, 11, 0.8)',
                      textAlign: 'center'
                    }}>
                      {perfectLongsData.opportunities.filter((opp: any) =>
                        opp.symbol.toLowerCase().includes(perfectLongsSearch.toLowerCase())
                      ).length} sonuç bulundu
                    </div>
                  )}
                </div>

                {/* Opportunities Grid */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                  gap: '16px',
                  maxHeight: '600px',
                  overflowY: 'auto',
                  padding: '4px'
                }}>
                  {perfectLongsData.opportunities
                    .filter((opp: any) =>
                      perfectLongsSearch === '' ||
                      opp.symbol.toLowerCase().includes(perfectLongsSearch.toLowerCase())
                    )
                    .map((opp: any, index: number) => {
                      const bullReady = isBullReady(opp.symbol, bullReadyData);
                      return (
                    <div
                      key={`${opp.symbol}-${index}`}
                      style={{
                        background: bullReady
                          ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(10, 10, 10, 0.9) 100%)'
                          : 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(10, 10, 10, 0.8) 100%)',
                        border: bullReady
                          ? '2px solid rgba(239, 68, 68, 0.6)'
                          : '1px solid rgba(245, 158, 11, 0.3)',
                        borderRadius: '12px',
                        padding: '16px',
                        transition: 'all 0.3s ease',
                        cursor: 'pointer',
                        boxShadow: bullReady ? '0 0 20px rgba(239, 68, 68, 0.3)' : 'none'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-4px)';
                        e.currentTarget.style.boxShadow = '0 8px 24px rgba(245, 158, 11, 0.3)';
                        e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.6)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = 'none';
                        e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.3)';
                      }}
                      onClick={() => {
                        setSelectedSymbol(opp.symbol);
                        fetchAllAnalysis();
                        // Smooth scroll to top
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                    >
                      {/* Symbol Header */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: bullReady ? '#EF4444' : '#F59E0B' }}>
                          {opp.symbol.replace('USDT', '')}
                        </div>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                          <div style={{
                            padding: '4px 8px',
                            background: 'rgba(245, 158, 11, 0.2)',
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '700',
                            color: '#F59E0B'
                          }}>
                            EXCELLENT
                          </div>
                          {bullReady && (
                            <div style={{
                              padding: '4px 8px',
                              background: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
                              borderRadius: '6px',
                              fontSize: '11px',
                              fontWeight: '700',
                              color: '#FFFFFF',
                              boxShadow: '0 0 10px rgba(239, 68, 68, 0.5)',
                              animation: 'pulse 2s ease-in-out infinite'
                            }}>
                              🐂 BOĞA HAZIR
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Funding Rate */}
                      <div style={{ marginBottom: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Fonlama Oranı
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '700', color: '#10B981' }}>
                          {opp.fundingRatePercent}
                        </div>
                      </div>

                      {/* Price */}
                      {opp.price && (
                        <div style={{ marginBottom: '12px' }}>
                          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                            Güncel Fiyat
                          </div>
                          <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                            ${opp.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 8 })}
                          </div>
                        </div>
                      )}

                      {/* Recommendation */}
                      <div style={{
                        marginTop: '12px',
                        padding: '12px',
                        background: 'rgba(16, 185, 129, 0.1)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: '8px'
                      }}>
                        <div style={{ fontSize: '11px', color: '#10B981', lineHeight: '1.6' }}>
                          {opp.recommendation}
                        </div>
                      </div>

                      {/* PHASE 5: Peak Hours Display - ENHANCED */}
                      {getPeakHours(opp.symbol) && (
                        <div style={{
                          marginTop: '10px',
                          padding: '10px 12px',
                          background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%)',
                          border: '1.5px solid rgba(16, 185, 129, 0.5)',
                          borderRadius: '8px',
                          boxShadow: '0 0 15px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
                        }}>
                          <div style={{
                            fontSize: '11px',
                            fontWeight: '600',
                            color: 'rgba(16, 185, 129, 0.9)',
                            marginBottom: '4px',
                            letterSpacing: '0.5px',
                            textTransform: 'uppercase'
                          }}>
                            🕐 Peak Hours (Turkey)
                          </div>
                          <div style={{
                            fontSize: '15px',
                            fontWeight: '700',
                            color: '#10b981',
                            fontFamily: 'Monaco, monospace',
                            textShadow: '0 0 10px rgba(16, 185, 129, 0.3)',
                            letterSpacing: '1px'
                          }}>
                            {getPeakHours(opp.symbol)}
                          </div>
                        </div>
                      )}

                      {/* Click to Analyze */}
                      <div style={{
                        marginTop: '12px',
                        textAlign: 'center',
                        fontSize: '11px',
                        color: 'rgba(245, 158, 11, 0.8)',
                        fontWeight: '600'
                      }}>
                        📊 Analiz için tıklayın
                      </div>
                    </div>
                  );
                    })}
                </div>

                {/* Auto-Refresh Notice */}
                <div style={{
                  marginTop: '16px',
                  padding: '12px',
                  background: 'rgba(245, 158, 11, 0.05)',
                  border: '1px solid rgba(245, 158, 11, 0.2)',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: 'rgba(245, 158, 11, 0.9)',
                  textAlign: 'center'
                }}>
                  ⏱️ Bu liste otomatik olarak her 5 dakikada güncellenir. Öneri değişen coinler otomatik olarak kaldırılır.
                </div>
              </div>
            )}

            {/* ===== 2-COLUMN GRID LAYOUT FOR REMAINING WIDGETS ===== */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px',
              marginBottom: '24px'
            }}>

              {/* ===== LONG SIGNAL ANALYSIS (Top Left) ===== */}
              <div style={{
                background: 'linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%)',
                backdropFilter: 'blur(20px)',
                border: `2px solid ${getQualityColor(longSignalData.quality)}`,
                borderRadius: '12px',
                padding: '24px',
                boxShadow: `0 4px 20px ${getQualityColor(longSignalData.quality)}30`,
                transition: 'all 0.3s ease'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                  <div>
                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px' }}>
                      BİRLEŞİK LONG SİNYAL SKORU
                    </div>
                    <div style={{ fontSize: '42px', fontWeight: '700', color: '#FFFFFF' }}>
                      {longSignalData.overallScore.toFixed(1)}
                      <span style={{ fontSize: '20px', color: 'rgba(255, 255, 255, 0.5)' }}>/100</span>
                    </div>
                  </div>
                  <div style={{
                    padding: '12px 24px',
                    background: getQualityColor(longSignalData.quality),
                    borderRadius: '12px',
                    fontSize: '16px',
                    fontWeight: '700',
                    color: '#FFFFFF'
                  }}>
                    {getQualityTurkish(longSignalData.quality)}
                  </div>
                </div>

                <div style={{ marginBottom: '16px' }}>
                  <div style={{ fontSize: '14px', color: '#FFFFFF', lineHeight: '1.6' }}>
                    {longSignalData.summary}
                  </div>
                </div>

                {/* Component Scores Breakdown */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '10px', marginBottom: '16px' }}>
                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Order Book</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                      {longSignalData.scores.orderbook.toFixed(0)}
                    </div>
                  </div>
                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Funding</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                      {longSignalData.scores.funding.toFixed(0)}
                    </div>
                  </div>
                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>OI & CVD</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                      {longSignalData.scores.openInterest.toFixed(0)}
                    </div>
                  </div>
                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Whale</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                      {longSignalData.scores.whale.toFixed(0)}
                    </div>
                  </div>
                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Technical</div>
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                      {longSignalData.scores.technical.toFixed(0)}
                    </div>
                  </div>
                </div>

                {/* Reasons */}
                {longSignalData.reasons && longSignalData.reasons.length > 0 && (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px' }}>
                      YÜKSELİŞ NEDENLERİ:
                    </div>
                    {longSignalData.reasons.map((reason: string, idx: number) => (
                      <div key={idx} style={{ fontSize: '12px', color: '#10B981', marginBottom: '3px' }}>
                        ✓ {reason}
                      </div>
                    ))}
                  </div>
                )}

                {/* Warnings */}
                {longSignalData.warnings && longSignalData.warnings.length > 0 && (
                  <div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px' }}>
                      UYARILAR:
                    </div>
                    {longSignalData.warnings.map((warning: string, idx: number) => (
                      <div key={idx} style={{ fontSize: '12px', color: '#F59E0B', marginBottom: '3px' }}>
                        ⚠ {warning}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* ===== POSITION CALCULATOR (Top Right) ===== */}
              {positionCalcData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(16, 185, 129, 0.3)',
                  borderRadius: '12px',
                  padding: '24px',
                  boxShadow: '0 4px 20px rgba(16, 185, 129, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                    <Icons.TrendingUp style={{ width: '20px', height: '20px', color: '#10B981' }} />
                    <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                      Pozisyon Büyüklüğü Hesaplayıcı
                    </h2>
                  </div>

                  {/* BUG FIX: onChange triggers real-time recalculation */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' }}>
                    <div>
                      <label style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px', display: 'block' }}>
                        Hesap Bakiyesi (USD):
                      </label>
                      <input
                        type="number"
                        value={accountBalance}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val) && val > 0) {
                            setAccountBalance(val);
                          }
                        }}
                        style={{
                          width: '100%',
                          padding: '10px',
                          background: 'rgba(0, 0, 0, 0.4)',
                          border: '1px solid rgba(255, 255, 255, 0.15)',
                          borderRadius: '8px',
                          color: '#FFFFFF',
                          fontSize: '13px'
                        }}
                      />
                    </div>
                    <div>
                      <label style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px', display: 'block' }}>
                        İşlem Başına Risk (%):
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={riskPercent}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val) && val > 0 && val <= 10) {
                            setRiskPercent(val);
                          }
                        }}
                        style={{
                          width: '100%',
                          padding: '10px',
                          background: 'rgba(0, 0, 0, 0.4)',
                          border: '1px solid rgba(255, 255, 255, 0.15)',
                          borderRadius: '8px',
                          color: '#FFFFFF',
                          fontSize: '13px'
                        }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                    <div style={{ padding: '12px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Optimum Pozisyon</div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#10B981' }}>
                        ${positionCalcData.optimalPositionSize?.toFixed(2) || '0.00'}
                      </div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '3px' }}>
                        {positionCalcData.optimalQuantity?.toFixed(6) || '0.000000'} {selectedSymbol.replace('USDT', '')}
                      </div>
                    </div>
                    <div style={{ padding: '12px', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Önerilen Kaldıraç</div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#3B82F6' }}>
                        {positionCalcData.recommendedLeverage || 0}x
                      </div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '3px' }}>
                        Max: {positionCalcData.maxLeverage || 0}x
                      </div>
                    </div>
                    <div style={{ padding: '12px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Stop Loss</div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#EF4444' }}>
                        ${positionCalcData.stopLossPrice?.toFixed(2) || '0.00'}
                      </div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '3px' }}>
                        -{positionCalcData.stopLossPercent?.toFixed(2) || '0.00'}%
                      </div>
                    </div>
                  </div>

                  <div style={{ marginBottom: '14px' }}>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                      Kar Al Seviyeleri:
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                      {positionCalcData.takeProfitLevels?.map((tp: any, idx: number) => (
                        <div key={idx} style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '8px' }}>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                            TP{tp.level || 0} ({tp.percentOfPosition || 0}%)
                          </div>
                          <div style={{ fontSize: '14px', fontWeight: '600', color: '#10B981' }}>
                            ${tp.price?.toFixed(2) || '0.00'}
                          </div>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>
                            +{tp.percentGain?.toFixed(2) || '0.00'}%
                          </div>
                        </div>
                      )) || []}
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div style={{ padding: '10px', background: 'rgba(0, 212, 255, 0.1)', borderRadius: '8px' }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Risk/Kazanç Oranı</div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#00D4FF' }}>
                        {positionCalcData.riskRewardRatio?.toFixed(2) || '0.00'}:1
                      </div>
                    </div>
                    <div style={{ padding: '10px', background: 'rgba(0, 212, 255, 0.1)', borderRadius: '8px' }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>Expected Value</div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#00D4FF' }}>
                        {(positionCalcData.expectedValue || 0) > 0 ? '+' : ''}{positionCalcData.expectedValue?.toFixed(2) || '0.00'}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ===== SIGNAL HISTORY (Bottom Left) ===== */}
              {signalHistory.length > 0 && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(124, 58, 237, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(139, 92, 246, 0.3)',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 4px 20px rgba(139, 92, 246, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '14px'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <Icons.TrendingUp style={{ width: '18px', height: '18px', color: '#8B5CF6' }} />
                      <span style={{ fontSize: '15px', fontWeight: '700', color: '#FFFFFF' }}>
                        Signal Score Geçmişi (Son 24 Saat)
                      </span>
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                      {signalHistory.length} veri noktası
                    </div>
                  </div>

                  {/* Mini Line Chart */}
                  <div style={{
                    background: 'rgba(0, 0, 0, 0.3)',
                    borderRadius: '8px',
                    padding: '14px',
                    height: '160px',
                    position: 'relative',
                    overflow: 'hidden'
                  }}>
                    <svg width="100%" height="100%" viewBox="0 0 800 140" preserveAspectRatio="none">
                      {/* Grid lines */}
                      {[0, 25, 50, 75, 100].map((y) => (
                        <line
                          key={y}
                          x1="0"
                          y1={140 - (y * 1.4)}
                          x2="800"
                          y2={140 - (y * 1.4)}
                          stroke="rgba(255, 255, 255, 0.05)"
                          strokeWidth="1"
                        />
                      ))}

                      {/* Score line */}
                      <polyline
                        points={signalHistory.map((entry, index) => {
                          const x = (index / (signalHistory.length - 1)) * 800;
                          const y = 140 - (entry.score * 1.4);
                          return `${x},${y}`;
                        }).join(' ')}
                        fill="none"
                        stroke="#8B5CF6"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />

                      {/* Gradient fill area */}
                      <polygon
                        points={[
                          '0,140',
                          ...signalHistory.map((entry, index) => {
                            const x = (index / (signalHistory.length - 1)) * 800;
                            const y = 140 - (entry.score * 1.4);
                            return `${x},${y}`;
                          }),
                          '800,140'
                        ].join(' ')}
                        fill="url(#gradient-violet)"
                        opacity="0.3"
                      />

                      {/* Gradient definition */}
                      <defs>
                        <linearGradient id="gradient-violet" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.5" />
                          <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0" />
                        </linearGradient>
                      </defs>

                      {/* Score threshold lines */}
                      <line x1="0" y1={140 - (85 * 1.4)} x2="800" y2={140 - (85 * 1.4)}
                            stroke="#10B981" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
                      <line x1="0" y1={140 - (70 * 1.4)} x2="800" y2={140 - (70 * 1.4)}
                            stroke="#3B82F6" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
                      <line x1="0" y1={140 - (55 * 1.4)} x2="800" y2={140 - (55 * 1.4)}
                            stroke="#F59E0B" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
                    </svg>
                  </div>

                  {/* Summary Stats */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    gap: '10px',
                    marginTop: '14px'
                  }}>
                    <div style={{
                      background: 'rgba(16, 185, 129, 0.1)',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                      borderRadius: '8px',
                      padding: '10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                        Güncel
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#10B981' }}>
                        {signalHistory[signalHistory.length - 1]?.score.toFixed(1)}
                      </div>
                    </div>

                    <div style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      border: '1px solid rgba(59, 130, 246, 0.2)',
                      borderRadius: '8px',
                      padding: '10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                        24h Ort.
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#3B82F6' }}>
                        {(signalHistory.reduce((sum, e) => sum + e.score, 0) / signalHistory.length).toFixed(1)}
                      </div>
                    </div>

                    <div style={{
                      background: 'rgba(245, 158, 11, 0.1)',
                      border: '1px solid rgba(245, 158, 11, 0.2)',
                      borderRadius: '8px',
                      padding: '10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                        En Yüksek
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#F59E0B' }}>
                        {Math.max(...signalHistory.map(e => e.score)).toFixed(1)}
                      </div>
                    </div>

                    <div style={{
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.2)',
                      borderRadius: '8px',
                      padding: '10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                        En Düşük
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#EF4444' }}>
                        {Math.min(...signalHistory.map(e => e.score)).toFixed(1)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ===== WHALE TRANSACTIONS (Bottom Right) ===== */}
              {whaleData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.12) 0%, rgba(219, 39, 119, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(236, 72, 153, 0.3)',
                  borderRadius: '12px',
                  padding: '20px',
                  boxShadow: '0 4px 20px rgba(236, 72, 153, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{
                        width: '36px',
                        height: '36px',
                        background: 'linear-gradient(135deg, #EC4899 0%, #DB2777 100%)',
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <span style={{ fontSize: '20px' }}>🐋</span>
                      </div>
                      <div>
                        <h2 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '2px' }}>
                          Balina İşlem Takibi
                        </h2>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                          &gt;$500K işlemler
                        </div>
                      </div>
                    </div>

                    {/* Whale Signal Badge */}
                    <div style={{
                      padding: '6px 12px',
                      background: whaleData.whaleSignal === 'STRONG_BUY' || whaleData.whaleSignal === 'BUY'
                        ? 'linear-gradient(135deg, #10B981 0%, #059669 100%)'
                        : whaleData.whaleSignal === 'STRONG_SELL' || whaleData.whaleSignal === 'SELL'
                        ? 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)'
                        : 'linear-gradient(135deg, #6B7280 0%, #4B5563 100%)',
                      borderRadius: '8px',
                      fontSize: '11px',
                      fontWeight: '700',
                      color: '#FFFFFF'
                    }}>
                      {whaleData.whaleSignal === 'STRONG_BUY' && '🔥 GÜÇLÜ ALIM'}
                      {whaleData.whaleSignal === 'BUY' && '✅ ALIM'}
                      {whaleData.whaleSignal === 'NEUTRAL' && '➖ NÖTR'}
                      {whaleData.whaleSignal === 'SELL' && '⚠️ SATIM'}
                      {whaleData.whaleSignal === 'STRONG_SELL' && '🚨 GÜÇLÜ SATIM'}
                    </div>
                  </div>

                  {/* Summary Stats */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '12px',
                    marginBottom: '16px'
                  }}>
                    {/* Total Buy Volume */}
                    <div style={{
                      background: 'rgba(16, 185, 129, 0.1)',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                      borderRadius: '10px',
                      padding: '12px'
                    }}>
                      <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '6px', fontWeight: '600' }}>TOPLAM ALIM</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#10B981' }}>
                        ${(whaleData.totalBuyVolume / 1000000).toFixed(2)}M
                      </div>
                    </div>

                    {/* Total Sell Volume */}
                    <div style={{
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.2)',
                      borderRadius: '10px',
                      padding: '12px'
                    }}>
                      <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '6px', fontWeight: '600' }}>TOPLAM SATIM</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444' }}>
                        ${(whaleData.totalSellVolume / 1000000).toFixed(2)}M
                      </div>
                    </div>

                    {/* Net Flow */}
                    <div style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      border: '1px solid rgba(59, 130, 246, 0.2)',
                      borderRadius: '10px',
                      padding: '12px'
                    }}>
                      <div style={{ fontSize: '10px', color: '#3B82F6', marginBottom: '6px', fontWeight: '600' }}>NET AKIŞ</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: whaleData.netWhaleFlow > 0 ? '#10B981' : '#EF4444' }}>
                        {whaleData.netWhaleFlow > 0 ? '+' : ''}${(whaleData.netWhaleFlow / 1000000).toFixed(2)}M
                      </div>
                    </div>

                    {/* Cluster Count */}
                    <div style={{
                      background: 'rgba(168, 85, 247, 0.1)',
                      border: '1px solid rgba(168, 85, 247, 0.2)',
                      borderRadius: '10px',
                      padding: '12px'
                    }}>
                      <div style={{ fontSize: '10px', color: '#A855F7', marginBottom: '6px', fontWeight: '600' }}>KÜME SAYISI</div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#A855F7' }}>
                        {whaleData.clusterCount} küme
                      </div>
                    </div>
                  </div>

                  {/* Interpretation */}
                  <div style={{
                    background: 'rgba(236, 72, 153, 0.05)',
                    border: '1px solid rgba(236, 72, 153, 0.15)',
                    borderRadius: '10px',
                    padding: '12px',
                    marginBottom: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '6px', fontWeight: '600' }}>
                      📊 ANALİZ YORUMU:
                    </div>
                    <div style={{ fontSize: '12px', color: '#FFFFFF', lineHeight: '1.5' }}>
                      {whaleData.interpretation}
                    </div>
                  </div>

                  {/* Largest Transaction Highlight */}
                  {whaleData.largestTransaction && (
                    <div style={{
                      background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(10, 10, 10, 0.8) 100%)',
                      border: '2px solid rgba(251, 191, 36, 0.3)',
                      borderRadius: '10px',
                      padding: '12px',
                      boxShadow: '0 0 15px rgba(251, 191, 36, 0.1)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                        <span style={{ fontSize: '16px' }}>👑</span>
                        <div style={{ fontSize: '11px', fontWeight: '700', color: '#FBBF24' }}>
                          EN BÜYÜK İŞLEM
                        </div>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
                        <div>
                          <div style={{ fontSize: '9px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '3px' }}>TARAF</div>
                          <div style={{
                            fontSize: '12px',
                            fontWeight: '700',
                            color: whaleData.largestTransaction.side === 'BUY' ? '#10B981' : '#EF4444'
                          }}>
                            {whaleData.largestTransaction.side === 'BUY' ? '📈 ALIM' : '📉 SATIM'}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '9px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '3px' }}>USD DEĞER</div>
                          <div style={{ fontSize: '12px', fontWeight: '700', color: '#FBBF24' }}>
                            ${(whaleData.largestTransaction.usdValue / 1000000).toFixed(2)}M
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: '9px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '3px' }}>FİYAT ETKİSİ</div>
                          <div style={{ fontSize: '12px', fontWeight: '600', color: '#FBBF24' }}>
                            ~{whaleData.largestTransaction.priceImpact.toFixed(3)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>

            {/* ===== ADDITIONAL 2x2 GRID: ORDER BOOK, FUNDING, OI/CVD, LIQUIDATIONS ===== */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px',
              marginBottom: '24px'
            }}>

              {/* ===== ORDER BOOK IMBALANCE (Orange Theme) ===== */}
              {orderbookData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.12) 0%, rgba(234, 88, 12, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(249, 115, 22, 0.35)',
                  borderRadius: '12px',
                  padding: '24px',
                  boxShadow: '0 4px 20px rgba(249, 115, 22, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                    <div style={{
                      width: '42px',
                      height: '42px',
                      background: 'linear-gradient(135deg, #F97316 0%, #EA580C 100%)',
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(249, 115, 22, 0.3)'
                    }}>
                      <Icons.BarChart style={{ width: '22px', height: '22px', color: '#FFFFFF' }} />
                    </div>
                    <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                      Emir Defteri Dengesizliği
                    </h2>
                  </div>

                  {orderbookData?.imbalance ? (
                    <>
                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Sinyal
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: orderbookData.imbalance.signal === 'BULLISH' ? '#10B981' :
                                 orderbookData.imbalance.signal === 'BEARISH' ? '#EF4444' : '#F59E0B'
                        }}>
                          {orderbookData.imbalance.signal === 'BULLISH' ? 'YUKARIŞ' :
                           orderbookData.imbalance.signal === 'BEARISH' ? 'AŞAĞIŞ' : 'NÖTR'}
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Güven
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '600', color: '#FFFFFF' }}>
                          {orderbookData.imbalance.confidence?.toFixed(1) || '0.0'}%
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                          Baskı Dağılımı
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                          <div style={{
                            padding: '12px',
                            background: 'rgba(16, 185, 129, 0.1)',
                            border: '1px solid rgba(16, 185, 129, 0.3)',
                            borderRadius: '8px'
                          }}>
                            <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '4px', fontWeight: '600' }}>
                              ALIŞ BASKISI
                            </div>
                            <div style={{ fontSize: '18px', fontWeight: '700', color: '#10B981' }}>
                              {orderbookData.imbalance.bidPressure?.toFixed(1) || '0.0'}%
                            </div>
                          </div>
                          <div style={{
                            padding: '12px',
                            background: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid rgba(239, 68, 68, 0.3)',
                            borderRadius: '8px'
                          }}>
                            <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '4px', fontWeight: '600' }}>
                              SATIŞ BASKISI
                            </div>
                            <div style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444' }}>
                              {orderbookData.imbalance.askPressure?.toFixed(1) || '0.0'}%
                            </div>
                          </div>
                        </div>
                      </div>

                      <div style={{
                        padding: '12px',
                        background: 'rgba(249, 115, 22, 0.1)',
                        border: '1px solid rgba(249, 115, 22, 0.3)',
                        borderRadius: '8px'
                      }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Balina Duvarları
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '700', color: '#F97316' }}>
                          {orderbookData.whaleLevels?.length || 0} tespit edildi
                        </div>
                      </div>
                    </>
                  ) : (
                    <div style={{ color: 'rgba(255, 255, 255, 0.4)', padding: '20px', textAlign: 'center' }}>
                      Veri yükleniyor...
                    </div>
                  )}
                </div>
              )}

              {/* ===== FUNDING RATE ANALYSIS (Teal Theme) ===== */}
              {fundingData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.12) 0%, rgba(13, 148, 136, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(20, 184, 166, 0.35)',
                  borderRadius: '12px',
                  padding: '24px',
                  boxShadow: '0 4px 20px rgba(20, 184, 166, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                    <div style={{
                      width: '42px',
                      height: '42px',
                      background: 'linear-gradient(135deg, #14B8A6 0%, #0D9488 100%)',
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(20, 184, 166, 0.3)'
                    }}>
                      <Icons.Percent style={{ width: '22px', height: '22px', color: '#FFFFFF' }} />
                    </div>
                    <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                      Fonlama Oranı Analizi
                    </h2>
                  </div>

                  {fundingData?.signal ? (
                    <>
                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Güncel Oran
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: fundingData.signal.currentRate > 0 ? '#EF4444' : '#10B981'
                        }}>
                          {((fundingData.signal.currentRate || 0) * 100).toFixed(4)}%
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Sinyal
                        </div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: '#FFFFFF' }}>
                          {fundingData.signal.signal?.includes('LONG') ? 'LONG FIRSATI' :
                           fundingData.signal.signal?.includes('SHORT') ? 'SHORT FIRSATI' : 'NÖTR'}
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Güven
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '600', color: '#FFFFFF' }}>
                          {fundingData.signal.confidence?.toFixed(1) || '0.0'}%
                        </div>
                      </div>

                      <div style={{
                        padding: '12px',
                        background: 'rgba(20, 184, 166, 0.1)',
                        border: '1px solid rgba(20, 184, 166, 0.3)',
                        borderRadius: '8px'
                      }}>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '5px', fontWeight: '600' }}>
                          ÖNERİ:
                        </div>
                        <div style={{ fontSize: '12px', color: '#14B8A6', lineHeight: '1.5' }}>
                          {fundingData.signal.recommendation || 'Analiz yapılıyor...'}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div style={{ color: 'rgba(255, 255, 255, 0.4)', padding: '20px', textAlign: 'center' }}>
                      Veri yükleniyor...
                    </div>
                  )}
                </div>
              )}

              {/* ===== OPEN INTEREST & CVD (Indigo Theme) ===== */}
              {oiData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(79, 70, 229, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(99, 102, 241, 0.35)',
                  borderRadius: '12px',
                  padding: '24px',
                  boxShadow: '0 4px 20px rgba(99, 102, 241, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                    <div style={{
                      width: '42px',
                      height: '42px',
                      background: 'linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)',
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
                    }}>
                      <Icons.TrendingUp style={{ width: '22px', height: '22px', color: '#FFFFFF' }} />
                    </div>
                    <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                      Açık Pozisyon & CVD
                    </h2>
                  </div>

                  {oiData?.interpretation ? (
                    <>
                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Birleşik Sinyal
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: oiData.interpretation.compositeSignal === 'BULLISH' ? '#10B981' :
                                 oiData.interpretation.compositeSignal === 'BEARISH' ? '#EF4444' : '#F59E0B'
                        }}>
                          {oiData.interpretation.compositeSignal === 'BULLISH' ? 'YUKARIŞ' :
                           oiData.interpretation.compositeSignal === 'BEARISH' ? 'AŞAĞIŞ' : 'NÖTR'}
                        </div>
                      </div>

                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Güven
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '600', color: '#FFFFFF' }}>
                          {oiData.interpretation.confidence || 0}%
                        </div>
                      </div>

                      <div style={{ marginBottom: '12px', padding: '10px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '3px', fontWeight: '600' }}>
                          OI SİNYALİ:
                        </div>
                        <div style={{ fontSize: '11px', color: '#6366F1', lineHeight: '1.5' }}>
                          {oiData.interpretation.oiSignal || 'Analiz yapılıyor...'}
                        </div>
                      </div>

                      <div style={{ padding: '10px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '3px', fontWeight: '600' }}>
                          CVD SİNYALİ:
                        </div>
                        <div style={{ fontSize: '11px', color: '#6366F1', lineHeight: '1.5' }}>
                          {oiData.interpretation.cvdSignal || 'Analiz yapılıyor...'}
                        </div>
                      </div>
                    </>
                  ) : (
                    <div style={{ color: 'rgba(255, 255, 255, 0.4)', padding: '20px', textAlign: 'center' }}>
                      Veri yükleniyor...
                    </div>
                  )}
                </div>
              )}

              {/* ===== LIQUIDATION HEATMAP (Rose Theme) ===== */}
              {liquidationData && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(225, 29, 72, 0.08) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: '2px solid rgba(244, 63, 94, 0.35)',
                  borderRadius: '12px',
                  padding: '24px',
                  boxShadow: '0 4px 20px rgba(244, 63, 94, 0.15)',
                  transition: 'all 0.3s ease'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                    <div style={{
                      width: '42px',
                      height: '42px',
                      background: 'linear-gradient(135deg, #F43F5E 0%, #E11D48 100%)',
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 4px 12px rgba(244, 63, 94, 0.3)'
                    }}>
                      <Icons.Zap style={{ width: '22px', height: '22px', color: '#FFFFFF' }} />
                    </div>
                    <h2 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                      Likidasyion Isı Haritası
                    </h2>
                  </div>

                  {liquidationData?.heatmap ? (
                    <>
                      <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Baskın Taraf
                        </div>
                        <div style={{
                          fontSize: '24px',
                          fontWeight: '700',
                          color: liquidationData.heatmap.dominantSide === 'SHORT' ? '#10B981' : '#EF4444'
                        }}>
                          {liquidationData.heatmap.dominantSide === 'SHORT' ? 'SHORT (YUKARIŞ)' :
                           liquidationData.heatmap.dominantSide === 'LONG' ? 'LONG (AŞAĞIŞ)' : 'DENGELİ'}
                        </div>
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                        <div style={{
                          padding: '12px',
                          background: 'rgba(16, 185, 129, 0.1)',
                          border: '1px solid rgba(16, 185, 129, 0.3)',
                          borderRadius: '8px'
                        }}>
                          <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '4px', fontWeight: '600' }}>
                            SHORT LIQ.
                          </div>
                          <div style={{ fontSize: '18px', fontWeight: '700', color: '#10B981' }}>
                            {liquidationData.heatmap.shortLiquidations || 0}
                          </div>
                        </div>
                        <div style={{
                          padding: '12px',
                          background: 'rgba(239, 68, 68, 0.1)',
                          border: '1px solid rgba(239, 68, 68, 0.3)',
                          borderRadius: '8px'
                        }}>
                          <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '4px', fontWeight: '600' }}>
                            LONG LIQ.
                          </div>
                          <div style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444' }}>
                            {liquidationData.heatmap.longLiquidations || 0}
                          </div>
                        </div>
                      </div>

                      <div style={{ marginBottom: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Toplam Küme
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: '600', color: '#FFFFFF' }}>
                          {liquidationData.heatmap.clusters?.length || 0} küme tespit edildi
                        </div>
                      </div>

                      {liquidationData.nearestCluster && (
                        <div style={{
                          padding: '12px',
                          background: 'rgba(244, 63, 94, 0.1)',
                          border: '1px solid rgba(244, 63, 94, 0.3)',
                          borderRadius: '8px'
                        }}>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '5px', fontWeight: '600' }}>
                            EN YAKIN KÜME:
                          </div>
                          <div style={{ fontSize: '16px', color: '#F43F5E', fontWeight: '700' }}>
                            ${liquidationData.heatmap.nearestCluster?.priceLevel?.toFixed(2) || '0.00'}
                          </div>
                          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                            Güncel fiyatın {liquidationData.nearestClusterDirection === 'ABOVE' ? 'üstünde' : 'altında'}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ color: 'rgba(255, 255, 255, 0.4)', padding: '20px', textAlign: 'center' }}>
                      Veri yükleniyor...
                    </div>
                  )}
                </div>
              )}

            </div>

            {/* ===== CVD WIDGET - FULL WIDTH ===== */}
            <CVDWidget cvdData={cvdData} />

            {/* ===== CZ BINANCE SENTIMENT WIDGET - FULL WIDTH ===== */}
            <div style={{ marginTop: '24px' }}>
              <CZBinanceSentimentWidget />
            </div>

            {/* ===== SCANNER WIDGETS - FULL WIDTH GRID ===== */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
              gap: '24px',
              marginTop: '24px'
            }}>
              {/* Multi-Timeframe Scanner */}
              <MultiTimeframeScanner />

              {/* Top Gainers Tracker */}
              <TopGainersTracker />
            </div>

            {/* ===== WHALE TRANSACTION TRACKER - FULL WIDTH ===== */}
            <div style={{ marginTop: '24px' }}>
              <WhaleTransactionWidget />
            </div>

          </>
        )}

        {/* CSS Animation for pulse */}
        <style jsx>{`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }
        `}</style>
      </div>

      {/* MANTIK MODAL */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.8)',
            backdropFilter: 'blur(10px)',
            zIndex: 3000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px',
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
              border: '2px solid rgba(139, 92, 246, 0.5)',
              borderRadius: '24px',
              padding: '40px',
              maxWidth: '600px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 60px rgba(139, 92, 246, 0.3)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setShowLogicModal(false)}
              style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                background: 'rgba(255, 255, 255, 0.1)',
                border: 'none',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                fontSize: '24px',
                color: '#FFFFFF',
              }}
            >
              ×
            </button>

            {/* Header */}
            <div style={{ marginBottom: '32px' }}>
              <div style={{ fontSize: '36px', marginBottom: '12px' }}>🧠</div>
              <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#8B5CF6', marginBottom: '8px' }}>
                MANTIK Rehberi
              </h2>
              <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
                Bot Analizi sayfasının detaylı açıklaması
              </p>
            </div>

            {/* Section 1: Sayfa Amacı */}
            <div style={{ marginBottom: '28px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>🎯</span>Sayfa Amacı
              </h3>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6', paddingLeft: '28px' }}>
                <p style={{ margin: '0 0 8px 0' }}>Whale hareketleri, order book analizi, funding rate, liquidation tracking</p>
              </div>
            </div>

            {/* Section 2: Nasıl Çalışır */}
            <div style={{ marginBottom: '28px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>⚙️</span>Nasıl Çalışır
              </h3>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6', paddingLeft: '28px' }}>
                <p style={{ margin: '0 0 8px 0' }}>/api/onchain/whale-alerts, /api/binance/orderbook kullanarak bot aktivitesi tespit</p>
              </div>
            </div>

            {/* Section 3: Özellikler */}
            <div style={{ marginBottom: '28px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>✨</span>Özellikler
              </h3>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6', paddingLeft: '28px' }}>
                <ul style={{ margin: 0, paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '6px' }}>Whale alert sistemi</li>
                  <li style={{ marginBottom: '6px' }}>Order book depth</li>
                  <li style={{ marginBottom: '6px' }}>Funding rates</li>
                  <li>Liquidation heatmap</li>
                </ul>
              </div>
            </div>

            {/* Section 4: Veri Kaynakları */}
            <div style={{ marginBottom: '28px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>📊</span>Veri Kaynakları
              </h3>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6', paddingLeft: '28px' }}>
                <ul style={{ margin: 0, paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '6px' }}>/api/onchain/whale-alerts</li>
                  <li style={{ marginBottom: '6px' }}>/api/binance/orderbook</li>
                  <li>Binance Futures</li>
                </ul>
              </div>
            </div>

            {/* Section 5: İpuçları */}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '20px' }}>💡</span>İpuçları
              </h3>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6', paddingLeft: '28px' }}>
                <ul style={{ margin: 0, paddingLeft: '20px' }}>
                  <li style={{ marginBottom: '6px' }}>Büyük hareketlere dikkat</li>
                  <li>Funding rate değişimlerini takip</li>
                </ul>
              </div>
            </div>

            {/* Footer Note */}
            <div style={{
              background: 'rgba(139, 92, 246, 0.1)',
              border: '1px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '12px',
              padding: '16px',
              fontSize: '12px',
              color: 'rgba(255, 255, 255, 0.6)',
              textAlign: 'center',
              fontStyle: 'italic'
            }}>
              Bu açıklamalar sayfanın özelliklerini ve kullanımını anlamanıza yardımcı olur.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
