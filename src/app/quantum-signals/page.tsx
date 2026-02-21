'use client';

/**
 * QUANTUM SIGNALS PAGE - Premium Modern Design
 * Quantum Engine Pro Portfolio Optimization + Risk Analysis
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { calculateTop10, isTop10 as checkTop10 } from '@/lib/top10-helper';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { COLORS, getSignalColor as getSignalColorHelper } from '@/lib/colors';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface QuantumSignal {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  price: number;
  confidence: number;
  strength: number;
  strategy: string;
  reasoning: string;
  targets?: string[];
  timestamp: string;
  quantumScore: number;
  quantumAdvantage: number;
  portfolioOptimization?: {
    optimalWeight: number;
    expectedReturn: number;
    risk: number;
    sharpeRatio: number;
  };
  riskAnalysis?: {
    valueAtRisk: number;
    conditionalVaR: number;
    expectedShortfall: number;
    quantumSpeedup: number;
  };
}

export default function QuantumSignalsPage() {
  const [signals, setSignals] = useState<QuantumSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [countdown, setCountdown] = useState(10);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [selectedSignal, setSelectedSignal] = useState<QuantumSignal | null>(null);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [top10List, setTop10List] = useState<string[]>([]);
  const [conservativeNotificationCount, setConservativeNotificationCount] = useState(0);
  const [notificationCount, setNotificationCount] = useState(0);
  const [previousSignalCount, setPreviousSignalCount] = useState(0);
  const [aiLearning, setAiLearning] = useState<any>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [timeRange, setTimeRange] = useState<'ALL' | '5m' | '15m' | '1h' | '4h'>('ALL');
  const [minConfidence, setMinConfidence] = useState(0);
  const [minQuantumAdvantage, setMinQuantumAdvantage] = useState(0);
  const [showLogicModal, setShowLogicModal] = useState(false);

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Global filters (synchronized across all pages)
  const { timeframe, sortBy } = useGlobalFilters();

  // Sayfa yüklendiğinde bildirim izni iste
  useEffect(() => {
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission();
      }
    }
    const savedCount = localStorage.getItem('quantum_notification_count');
    if (savedCount) setNotificationCount(parseInt(savedCount));
  }, []);

  // Kullanıcı bu sayfayı ziyaret ettiğinde bildirim sayısını otomatik temizle
  useEffect(() => {
    const timer = setTimeout(() => {
      localStorage.setItem('quantum_notification_count', '0');
      setNotificationCount(0);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  // Conservative notification badge sync
  useEffect(() => {
    const load = () => { if (typeof window !== 'undefined') { const c = localStorage.getItem('conservative_notification_count'); if (c) setConservativeNotificationCount(parseInt(c)); } };
    load();
    const h = (e: StorageEvent) => { if (e.key === 'conservative_notification_count' && e.newValue) setConservativeNotificationCount(parseInt(e.newValue)); };
    window.addEventListener('storage', h);
    const i = setInterval(load, 2000);
    return () => { window.removeEventListener('storage', h); clearInterval(i); };
  }, []);

  const fetchSignals = async () => {
    try {
      const response = await fetch('/api/quantum-signals');
      const result = await response.json();
      if (result.success) {
        const newSignals = result.data.signals;
        const newSignalCount = newSignals.length;

        // Detect new signals and trigger notification
        if (previousSignalCount > 0 && newSignalCount > previousSignalCount) {
          const newSignalsCount = newSignalCount - previousSignalCount;
          const currentCount = parseInt(localStorage.getItem('quantum_notification_count') || '0');
          const updatedCount = currentCount + newSignalsCount;
          localStorage.setItem('quantum_notification_count', updatedCount.toString());
          setNotificationCount(updatedCount);

          // Browser notification
          if (typeof window !== 'undefined' && 'Notification' in window && Notification.permission === 'granted') {
            new Notification('⚛️ Yeni Quantum Sinyali!', {
              body: `${newSignalsCount} yeni quantum sinyali tespit edildi. Toplam: ${newSignalCount}`,
              icon: '/icons/icon-192x192.png',
              badge: '/icons/icon-96x96.png',
              tag: 'quantum-signal',
              requireInteraction: true,
            });
          }
        }

        setPreviousSignalCount(newSignalCount);
        setSignals(newSignals);
        setAiLearning(result.data.aiLearning); // AI Learning verisini kaydet
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Quantum Signals fetch error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSignals();
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchSignals();
          return 10;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // TOP 10'u getir (arka planda)
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

  // Use unified color helper
  const getSignalColorClass = (type: string) => {
    if (type === 'BUY') return 'signal-buy';
    if (type === 'SELL') return 'signal-sell';
    return 'signal-wait';
  };

  const filteredSignals = signals.filter((signal) => {
    const matchesSearch = signal.symbol.toLowerCase().includes(searchTerm.toLowerCase());

    // Time range filter
    let matchesTimeRange = true;
    if (timeRange !== 'ALL') {
      const signalTime = new Date(signal.timestamp).getTime();
      const now = Date.now();
      const timeRanges: Record<string, number> = {
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
      };
      matchesTimeRange = (now - signalTime) <= timeRanges[timeRange];
    }

    // Confidence and quantum advantage filters
    const matchesConfidence = signal.confidence >= minConfidence;
    const matchesQuantumAdvantage = signal.quantumAdvantage >= minQuantumAdvantage;

    return matchesSearch && matchesTimeRange && matchesConfidence && matchesQuantumAdvantage;
  });

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div className="dashboard-container" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
      {/* Sidebar */}
      <SharedSidebar
        currentPage="quantum-signals"
        notificationCounts={{
          quantum: notificationCount,
          conservative: conservativeNotificationCount
        }}
      />

      {/* Main Content */}
      <div className="dashboard-main">
        {/* Page Header with MANTIK Button */}
        <div style={{ margin: '16px 24px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Icons.Zap style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                Quantum Sinyalleri
              </h1>
            </div>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
              Kuantum Inspirasyonlu Sinyal Algoritması - Çoklu Boyut Analizi
            </p>
          </div>

          {/* MANTIK Button - Responsive */}
          <div>
            <style>{`
              @media (max-width: 768px) {
                .mantik-button-quantum {
                  padding: 10px 20px !important;
                  fontSize: 13px !important;
                  height: 42px !important;
                }
                .mantik-button-quantum svg {
                  width: 18px !important;
                  height: 18px !important;
                }
              }
              @media (max-width: 480px) {
                .mantik-button-quantum {
                  padding: 8px 16px !important;
                  fontSize: 12px !important;
                  height: 40px !important;
                }
                .mantik-button-quantum svg {
                  width: 16px !important;
                  height: 16px !important;
                }
              }
            `}</style>
            <button
              onClick={() => setShowLogicModal(true)}
              className="mantik-button-quantum"
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

        <main className="dashboard-content" style={{ padding: '16px' }}>
          <table className="coin-table">
            <thead>
              <tr>
                <th>SİMGE</th>
                <th>SİNYAL</th>
                <th>FİYAT</th>
                <th>QUANTUM SKOR</th>
                <th>AVANTAJ</th>
                <th>SHARPE ORANI</th>
                <th>RİSK</th>
                <th>DETAYLAR</th>
              </tr>
            </thead>
            <tbody>
              {filteredSignals.map((signal) => (
                <tr key={signal.id}>
                  <td>
                    <div className="coin-symbol">
                      {checkTop10(signal.symbol, top10List) && (
                        <span style={{
                          background: '#FFD700',
                          color: '#000',
                          fontSize: '8px',
                          fontWeight: '700',
                          padding: '2px 4px',
                          borderRadius: '2px',
                          marginRight: '6px',
                          letterSpacing: '0.3px',
                        }}>
                          TOP10
                        </span>
                      )}
                      <span>{signal.symbol.replace('USDT', '')}</span>
                      <span className="coin-pair">/USDT</span>
                    </div>
                  </td>
                  <td className={getSignalColorClass(signal.type)} style={{ fontWeight: 'bold' }}>
                    {signal.type === 'BUY' ? 'AL' : signal.type === 'SELL' ? 'SAT' : 'BEKLE'}
                  </td>
                  <td className="coin-price">
                    ${(signal.price ?? 0).toFixed((signal.price ?? 0) < 1 ? 6 : 2)}
                  </td>
                  <td className="neon-text" style={{ fontFamily: 'monospace' }}>
                    {signal.quantumScore}/100
                  </td>
                  <td style={{ color: COLORS.text.primary, fontFamily: 'monospace' }}>
                    {(signal.quantumAdvantage ?? 0).toFixed(2)}x
                  </td>
                  <td style={{ color: COLORS.success, fontFamily: 'monospace' }}>
                    {(signal.portfolioOptimization?.sharpeRatio ?? 0).toFixed(2)}
                  </td>
                  <td style={{ fontFamily: 'monospace' }}>
                    <span style={{ color: signal.portfolioOptimization && (signal.portfolioOptimization.risk ?? 0) < 0.1 ? COLORS.success : COLORS.warning }}>
                      %{signal.portfolioOptimization ? ((signal.portfolioOptimization.risk ?? 0) * 100).toFixed(1) : 'N/A'}
                    </span>
                  </td>
                  <td>
                    <button
                      className="analyze-btn"
                      onClick={() => setSelectedSignal(signal)}
                    >
                      ANALİZ
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </main>
      </div>

      {/* Quantum Analysis Modal */}
      {selectedSignal && (
        <div
          className="modal-overlay"
          style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50, padding: '16px' }}
          onClick={() => setSelectedSignal(null)}
        >
          <div
            className="modal-content"
            style={{ maxWidth: '1000px', width: '100%', padding: '24px', maxHeight: '90vh', overflowY: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px' }}>
              <div>
                <h2 className="neon-text" style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '8px' }}>
                  {selectedSignal.symbol.replace('USDT', '')}/USDT
                </h2>
                <div style={{ display: 'flex', gap: '16px', fontSize: '14px' }}>
                  <span style={{ color: COLORS.text.primary }}>
                    ${(selectedSignal.price ?? 0).toFixed((selectedSignal.price ?? 0) < 1 ? 6 : 2)}
                  </span>
                  <span className={getSignalColorClass(selectedSignal.type)}>
                    {selectedSignal.type === 'BUY' ? 'AL' : selectedSignal.type === 'SELL' ? 'SAT' : 'BEKLE'}
                  </span>
                  <span style={{ color: COLORS.text.primary }}>
                    Quantum Skor: {selectedSignal.quantumScore}/100
                  </span>
                </div>
              </div>
              <button className="neon-button" onClick={() => setSelectedSignal(null)}>
                KAPAT
              </button>
            </div>

            <div className="neon-card" style={{ marginBottom: '24px' }}>
              <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>QUANTUM ANALİZİ</h3>
              <p style={{ color: COLORS.text.primary, lineHeight: '1.6' }}>{selectedSignal.reasoning}</p>
              <div style={{ marginTop: '12px', display: 'flex', gap: '16px' }}>
                <div>
                  <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Quantum Advantage:</span>
                  <span className="neon-text" style={{ marginLeft: '8px', fontSize: '16px' }}>
                    {(selectedSignal.quantumAdvantage ?? 0).toFixed(2)}x
                  </span>
                </div>
                <div>
                  <span style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Güvenilirlik:</span>
                  <span className="neon-text" style={{ marginLeft: '8px', fontSize: '16px' }}>
                    %{selectedSignal.confidence}
                  </span>
                </div>
              </div>
            </div>

            {selectedSignal.portfolioOptimization && (
              <div className="neon-card" style={{ marginBottom: '24px' }}>
                <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>PORTFÖY OPTİMİZASYONU</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Optimal Ağırlık</div>
                    <div className="neon-text" style={{ fontSize: '1.5rem' }}>
                      %{((selectedSignal.portfolioOptimization.optimalWeight ?? 0) * 100).toFixed(1)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Beklenen Getiri</div>
                    <div className="neon-text" style={{ fontSize: '1.5rem' }}>
                      %{((selectedSignal.portfolioOptimization.expectedReturn ?? 0) * 100).toFixed(2)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Risk Seviyesi</div>
                    <div style={{ color: selectedSignal.portfolioOptimization.risk < 0.1 ? COLORS.success : COLORS.warning, fontSize: '1.5rem' }}>
                      %{((selectedSignal.portfolioOptimization.risk ?? 0) * 100).toFixed(2)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Sharpe Ratio</div>
                    <div className="neon-text" style={{ fontSize: '1.5rem' }}>
                      {(selectedSignal.portfolioOptimization.sharpeRatio ?? 0).toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedSignal.riskAnalysis && (
              <div className="neon-card" style={{ marginBottom: '24px' }}>
                <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>RİSK ANALİZİ</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Value at Risk (VaR)</div>
                    <div style={{ color: COLORS.text.primary, fontSize: '1.25rem' }}>
                      %{((selectedSignal.riskAnalysis.valueAtRisk ?? 0) * 100).toFixed(2)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Conditional VaR</div>
                    <div style={{ color: COLORS.text.primary, fontSize: '1.25rem' }}>
                      %{((selectedSignal.riskAnalysis.conditionalVaR ?? 0) * 100).toFixed(2)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Expected Shortfall</div>
                    <div style={{ color: COLORS.text.primary, fontSize: '1.25rem' }}>
                      %{((selectedSignal.riskAnalysis.expectedShortfall ?? 0) * 100).toFixed(2)}
                    </div>
                  </div>
                  <div className="neon-card" style={{ padding: '12px' }}>
                    <div style={{ color: COLORS.text.secondary, fontSize: '12px' }}>Quantum Speedup</div>
                    <div className="neon-text" style={{ fontSize: '1.25rem' }}>
                      {(selectedSignal.riskAnalysis.quantumSpeedup ?? 0).toFixed(2)}x
                    </div>
                  </div>
                </div>
              </div>
            )}

            {selectedSignal.targets && selectedSignal.targets.length > 0 && (
              <div className="neon-card">
                <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>HEDEFLER</h3>
                <div style={{ display: 'flex', gap: '12px' }}>
                  {selectedSignal.targets.map((target, i) => (
                    <div key={i} className="neon-card" style={{ flex: 1, padding: '12px', textAlign: 'center' }}>
                      <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>Hedef {i + 1}</div>
                      <div className="neon-text" style={{ fontSize: '16px' }}>${target}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ marginTop: '24px', textAlign: 'center', color: COLORS.text.secondary, fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', flexDirection: 'column' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Icons.Clock className="w-3 h-3" />
                Zaman: {new Date(selectedSignal.timestamp).toLocaleString('tr-TR')}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Icons.AlertTriangle className="w-3 h-3" />
                Bu yatırım tavsiyesi değildir. Kendi riskinize karar verin.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Assistant Full Screen */}
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
                    Quantum Sinyalleri MANTIK
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
                  <Icons.Zap style={{ width: '24px', height: '24px' }} />
                  Genel Bakış
                </h3>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                  Quantum Sinyalleri sayfası, kuantum hesaplama prensiplerinden esinlenilmiş gelişmiş algoritmaları kullanır.
                  Çoklu boyut analizi ile piyasa verilerini daha derin bir perspektiften inceler.
                </p>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                  Kuantum inspirasyonlu yaklaşım, geleneksel analizlerin göremediği pattern ve korelasyonları ortaya çıkarır.
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
                    { name: 'Kuantum-İnspire Algoritmalar', desc: 'Süperpozisyon ve dolanıklık kavramlarından esinlenilmiş analiz yöntemleri.' },
                    { name: 'Çok Boyutlu Analiz', desc: 'Fiyat, hacim, momentum, volatilite ve zaman dilimlerini eş zamanlı değerlendirir.' },
                    { name: 'Olasılık Tabanlı Sinyaller', desc: 'Kesin tahmin yerine olasılık dağılımları ile daha gerçekçi öngörüler.' },
                    { name: 'Entanglement (Dolanıklık) Patternleri', desc: 'Farklı varlıklar arasındaki gizli korelasyonları tespit eder.' },
                    { name: 'Süperpozisyon Durumları', desc: 'Aynı anda birden fazla piyasa durumunu analiz eder ve en olası senaryoyu seçer.' },
                    { name: 'Dalga Fonksiyonu Çöküşü Tespiti', desc: 'Belirsiz piyasa durumlarından net trend geçişlerini yakalar.' }
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
                        Kuantum Kavramlarını Anlayın
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Quantum sinyalleri kesin tahmin yapmaz, olasılık dağılımları sunar. Bu belirsizlik aslında bir avantajdır.
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
                        Olasılık Skorlarını Gözden Geçirin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Her sinyal için quantum skor ve avantaj çarpanı gösterilir. Yüksek avantaj = güçlü quantum etkisi.
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
                        Entanglement Seviyelerini Kontrol Edin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Dolanıklık seviyeleri, bu varlığın diğer kripto varlıklarla ne kadar bağlantılı olduğunu gösterir.
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
                        Quantum Sinyallerini Takip Edin
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        Portföy optimizasyonu ve risk analizi bölümlerinde detaylı quantum metrikleri bulunur.
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
                    <strong style={{ color: COLORS.text.primary }}>Kuantum Metafor:</strong> Bu algoritmalar gerçek kuantum bilgisayarlar kullanmaz, ancak kuantum prensiplerinden esinlenir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Olasılıksal Yaklaşım:</strong> Kesin tahmin yerine olasılık dağılımları sunar, bu daha gerçekçi bir yaklaşımdır.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Çok Boyutlu Analiz:</strong> Birden fazla faktörü eş zamanlı değerlendirir, bu klasik yöntemlerden daha kapsamlıdır.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Deneysel Algoritma:</strong> Quantum-inspired yöntemler henüz geliştirilme aşamasındadır, dikkatli kullanın.
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
                Quantum Sinyalleri - Kuantum İnspire Algoritmalar ile Çok Boyutlu Analiz
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
