/**
 * 🔮 QUANTUM PRO KONTROL PANELİ
 *
 * Gelişmiş Kuantum-İlhamlı Yapay Zeka Ticaret Kontrol Paneli
 * Gerçek Binance Futures USDT-M verileri
 *
 * Özellikler:
 * - Gerçek zamanlı Kuantum sinyalleri
 * - Çoklu strateji YZ topluluğu (LSTM + Dönüştürücü + Gradyan Artırma)
 * - Risk yönetimi ve analiz
 * - Bot kontrol paneli
 * - Sinyal izleme ve geriye dönük test
 *
 * BEYAZ ŞAPKA: Sadece eğitim ve analiz amaçlı
 */

'use client';

import { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { getQuantumSettings, type QuantumProSettings } from '@/lib/quantumSettings';

interface QuantumSignal {
  symbol: string;
  signal: 'AL' | 'SAT' | 'BEKLE';
  confidence: number;
  price: number;
  priceChange24h: number;
  volume24h: number;
  aiScore: number;
  riskScore: number;
  triggers: string[];
  timeframeConfirmations: string[];
  strategies: {
    lstm: number;
    transformer: number;
    gradientBoosting: number;
  };
  timestamp: string;
}

interface QuantumData {
  signals: QuantumSignal[];
  totalSignals: number;
  buySignals: number;
  sellSignals: number;
  holdSignals: number;
  avgConfidence: number;
}

export default function QuantumProSayfasi() {
  // Check if running on localhost (using useState to avoid hydration mismatch)
  const [isLocalhost, setIsLocalhost] = useState(false);

  const [data, setData] = useState<QuantumData | null>(null);
  const [yukleniyor, setYukleniyor] = useState(true);
  const [hata, setHata] = useState<string | null>(null);
  const [seciliTab, setSeciliTab] = useState<'sinyaller' | 'backtest' | 'risk' | 'botlar' | 'monitoring'>('sinyaller');
  const [minGuven, setMinGuven] = useState(0.60);
  const [seciliCoin, setSeciliCoin] = useState<QuantumSignal | null>(null);

  const [backtestData, setBacktestData] = useState<any>(null);
  const [riskData, setRiskData] = useState<any>(null);
  const [botsData, setBotsData] = useState<any>(null);
  const [monitoringData, setMonitoringData] = useState<any>(null);

  const [settings, setSettings] = useState<QuantumProSettings>(getQuantumSettings());

  // Bot Modal State
  const [selectedBot, setSelectedBot] = useState<any>(null);
  const [showBotModal, setShowBotModal] = useState(false);

  // MANTIK Modal State
  const [showLogicModal, setShowLogicModal] = useState(false);

  // Detect localhost on client-side only (avoid hydration mismatch)
  useEffect(() => {
    setIsLocalhost(typeof window !== 'undefined' && window.location.hostname === 'localhost');
  }, []);

  // Load settings and listen for changes
  useEffect(() => {
    const loadSettings = () => {
      const newSettings = getQuantumSettings();
      setSettings(newSettings);
      setMinGuven(newSettings.signals.minConfidence);
    };

    loadSettings();

    const handleSettingsChange = () => {
      console.log('⚙️ Quantum Pro: Settings changed, reloading...');
      loadSettings();
    };

    window.addEventListener('quantumSettingsChanged', handleSettingsChange);
    return () => window.removeEventListener('quantumSettingsChanged', handleSettingsChange);
  }, []);

  useEffect(() => {
    quantumVerileriAl();
    const interval = setInterval(quantumVerileriAl, settings.signals.refreshInterval * 1000);
    return () => clearInterval(interval);
  }, [minGuven, settings.signals.refreshInterval, settings.signals.maxSignals]);

  useEffect(() => {
    if (seciliTab === 'backtest') backtestAl();
    else if (seciliTab === 'risk') riskAl();
    else if (seciliTab === 'botlar') botlarAl();
    else if (seciliTab === 'monitoring') {
      monitoringAl();
      const interval = setInterval(monitoringAl, settings.monitoring.refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [seciliTab, settings.monitoring.refreshInterval]);

  const quantumVerileriAl = async () => {
    try {
      const response = await fetch(`/api/quantum-pro/signals?minConfidence=${minGuven}&limit=${settings.signals.maxSignals}`);
      const result = await response.json();
      if (result.success && result.data) setData(result.data);
      setHata(null);
    } catch (err) {
      setHata('Quantum Pro API error');
      console.error('Quantum Pro fetch error:', err);
    } finally {
      setYukleniyor(false);
    }
  };

  const backtestAl = async () => {
    try {
      const response = await fetch('/api/quantum-pro/backtest');
      const result = await response.json();
      if (result.success) setBacktestData(result.data);
    } catch (err) {
      console.error('Backtest fetch error:', err);
    }
  };

  const riskAl = async () => {
    try {
      const response = await fetch('/api/quantum-pro/risk');
      const result = await response.json();
      if (result.success) setRiskData(result.data);
    } catch (err) {
      console.error('Risk fetch error:', err);
    }
  };

  const botlarAl = async () => {
    try {
      const response = await fetch('/api/quantum-pro/bots');
      const result = await response.json();
      if (result.success) setBotsData(result.data);
    } catch (err) {
      console.error('Bots fetch error:', err);
    }
  };

  const monitoringAl = async () => {
    try {
      const response = await fetch('/api/quantum-pro/monitoring');
      const result = await response.json();
      if (result.success) setMonitoringData(result.data);
    } catch (err) {
      console.error('Monitoring fetch error:', err);
    }
  };

  // Bot Control Functions
  const handleBotAction = async (botId: string, action: 'start' | 'pause') => {
    try {
      console.log(`🤖 ${action === 'start' ? 'Starting' : 'Pausing'} bot ${botId}...`);
      const response = await fetch('/api/quantum-pro/bots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ botId, action }),
      });
      const result = await response.json();
      if (result.success) {
        await botlarAl(); // Refresh bot list
      }
    } catch (err) {
      console.error('Bot action error:', err);
    }
  };

  const openBotModal = (bot: any) => {
    console.log('⚙️ Opening bot modal for:', bot.name);
    setSelectedBot(bot);
    setShowBotModal(true);
  };

  const closeBotModal = () => {
    setShowBotModal(false);
    setTimeout(() => setSelectedBot(null), 300);
  };

  const sinyalRengi = (signal: 'AL' | 'SAT' | 'BEKLE') => {
    switch (signal) {
      case 'AL':
        return { bg: 'rgba(16, 185, 129, 0.15)', border: '#10B981', text: '#10B981' };
      case 'SAT':
        return { bg: 'rgba(239, 68, 68, 0.15)', border: '#EF4444', text: '#EF4444' };
      default:
        return { bg: 'rgba(107, 114, 128, 0.15)', border: '#6B7280', text: '#6B7280' };
    }
  };

  const riskRengi = (risk: number) => {
    if (risk < 0.3) return { bg: 'rgba(16, 185, 129, 0.15)', border: '#10B981', text: '#10B981', label: 'DÜŞÜK' };
    if (risk < 0.6) return { bg: 'rgba(245, 158, 11, 0.15)', border: '#F59E0B', text: '#F59E0B', label: 'ORTA' };
    return { bg: 'rgba(239, 68, 68, 0.15)', border: '#EF4444', text: '#EF4444', label: 'YÜKSEK' };
  };

  const formatSayi = (num: number, decimal: number = 2) => {
    if (num >= 1000000000) return `${(num / 1000000000).toFixed(decimal)}B`;
    if (num >= 1000000) return `${(num / 1000000).toFixed(decimal)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(decimal)}K`;
    return num.toFixed(decimal);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
      <SharedSidebar currentPage="quantum-pro" />

      <main
        style={{
          padding: `${isLocalhost ? '116px' : '60px'} 24px 24px 24px`,
          maxWidth: '1920px',
          margin: '0 auto',
        }}
      >
        {/* Futuristik Header */}
        <div
          style={{
            background:
              'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.2) 50%, rgba(168, 85, 247, 0.15) 100%)',
            backdropFilter: 'blur(40px)',
            border: '2px solid rgba(139, 92, 246, 0.6)',
            borderRadius: '28px',
            padding: '48px',
            marginBottom: '32px',
            boxShadow:
              '0 25px 70px rgba(139, 92, 246, 0.4), inset 0 2px 0 rgba(255, 255, 255, 0.15)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Animated quantum particles */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background:
                'radial-gradient(circle at 20% 30%, rgba(139, 92, 246, 0.3) 0%, transparent 40%), radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.3) 0%, transparent 40%)',
              animation: 'quantum-pulse 5s ease-in-out infinite',
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '24px', position: 'relative', zIndex: 1 }}>
            <div
              style={{
                width: '80px',
                height: '80px',
                background: 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)',
                borderRadius: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '42px',
                boxShadow: '0 15px 40px rgba(139, 92, 246, 0.6), inset 0 2px 10px rgba(255, 255, 255, 0.3)',
                animation: 'quantum-float 3s ease-in-out infinite',
              }}
            >
              🔮
            </div>

            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '8px' }}>
                <h1
                  style={{
                    fontSize: '48px',
                    fontWeight: '900',
                    background: 'linear-gradient(135deg, #FFFFFF 0%, #A78BFA 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    margin: 0,
                    letterSpacing: '-1px',
                    textShadow: '0 0 40px rgba(167, 139, 250, 0.5)',
                  }}
                >
                  Quantum Pro Dashboard
                </h1>
                <div>
                  <style>{`
                    @media (max-width: 768px) {
                      .mantik-button-quantumpro {
                        padding: 10px 20px !important;
                        fontSize: 13px !important;
                        height: 42px !important;
                      }
                      .mantik-button-quantumpro span {
                        fontSize: 18px !important;
                      }
                    }
                    @media (max-width: 480px) {
                      .mantik-button-quantumpro {
                        padding: 8px 16px !important;
                        fontSize: 12px !important;
                        height: 40px !important;
                      }
                      .mantik-button-quantumpro span {
                        fontSize: 16px !important;
                      }
                    }
                  `}</style>
                  <button
                    onClick={() => setShowLogicModal(true)}
                    className="mantik-button-quantumpro"
                    style={{
                      background: 'linear-gradient(135deg, #8B5CF6, #7C3AED)',
                      border: '2px solid rgba(139, 92, 246, 0.5)',
                      borderRadius: '10px',
                      padding: '12px 24px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      boxShadow: '0 4px 16px rgba(139, 92, 246, 0.3)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      height: '44px'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 24px rgba(139, 92, 246, 0.5)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 4px 16px rgba(139, 92, 246, 0.3)';
                    }}
                  >
                    <span style={{ fontSize: '18px' }}>🧠</span>
                    MANTIK
                  </button>
                </div>
              </div>
              <p
                style={{
                  fontSize: '18px',
                  color: 'rgba(255,255,255,0.9)',
                  margin: '16px 0 0',
                  fontWeight: '600',
                  textShadow: '0 2px 4px rgba(0,0,0,0.3)',
                }}
              >
                Gelişmiş Quantum-Inspired AI Trading • Binance Futures USDT-M • Gerçek Zamanlı Analiz
              </p>
            </div>

            {data && (
              <div
                style={{
                  background: 'rgba(0, 0, 0, 0.5)',
                  backdropFilter: 'blur(15px)',
                  padding: '24px 32px',
                  borderRadius: '20px',
                  border: '2px solid rgba(139, 92, 246, 0.4)',
                  textAlign: 'center',
                  boxShadow: 'inset 0 2px 10px rgba(0, 0, 0, 0.3)',
                }}
              >
                <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)', marginBottom: '8px', fontWeight: '600' }}>
                  Toplam Sinyal
                </div>
                <div style={{ fontSize: '42px', fontWeight: '900', color: '#8B5CF6', textShadow: '0 0 20px rgba(139, 92, 246, 0.8)' }}>
                  {data.totalSignals}
                </div>
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)', marginTop: '4px' }}>
                  Ort. Güven: {(data.avgConfidence * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>

          {/* Stats Row */}
          {data && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '20px',
                marginTop: '32px',
                position: 'relative',
                zIndex: 1,
              }}
            >
              <div
                style={{
                  background: 'rgba(16, 185, 129, 0.15)',
                  backdropFilter: 'blur(10px)',
                  padding: '20px',
                  borderRadius: '16px',
                  border: '2px solid rgba(16, 185, 129, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                }}
              >
                <div style={{ fontSize: '32px' }}>📈</div>
                <div>
                  <div style={{ fontSize: '28px', fontWeight: '800', color: '#10B981' }}>
                    {data.buySignals}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)' }}>AL Sinyali</div>
                </div>
              </div>

              <div
                style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  backdropFilter: 'blur(10px)',
                  padding: '20px',
                  borderRadius: '16px',
                  border: '2px solid rgba(239, 68, 68, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                }}
              >
                <div style={{ fontSize: '32px' }}>📉</div>
                <div>
                  <div style={{ fontSize: '28px', fontWeight: '800', color: '#EF4444' }}>
                    {data.sellSignals}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)' }}>SAT Sinyali</div>
                </div>
              </div>

              <div
                style={{
                  background: 'rgba(107, 114, 128, 0.15)',
                  backdropFilter: 'blur(10px)',
                  padding: '20px',
                  borderRadius: '16px',
                  border: '2px solid rgba(107, 114, 128, 0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                }}
              >
                <div style={{ fontSize: '32px' }}>⏸️</div>
                <div>
                  <div style={{ fontSize: '28px', fontWeight: '800', color: '#6B7280' }}>
                    {data.holdSignals}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.7)' }}>BEKLE Sinyali</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Tab Navigation - Unique Design */}
        <div
          style={{
            display: 'flex',
            gap: '16px',
            marginBottom: '32px',
            background: 'rgba(26, 26, 26, 0.8)',
            backdropFilter: 'blur(20px)',
            padding: '12px',
            borderRadius: '20px',
            border: '2px solid rgba(139, 92, 246, 0.2)',
            boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)',
          }}
        >
          {[
            { id: 'sinyaller', label: '📡 Quantum Sinyaller', icon: '📡' },
            { id: 'backtest', label: '📊 Backtest Analizi', icon: '📊' },
            { id: 'risk', label: '⚠️ Risk Yönetimi', icon: '⚠️' },
            { id: 'botlar', label: '🤖 Bot Kontrolü', icon: '🤖' },
            { id: 'monitoring', label: '📈 Canlı İzleme', icon: '📈' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSeciliTab(tab.id as any)}
              style={{
                flex: 1,
                padding: '18px 24px',
                background:
                  seciliTab === tab.id
                    ? 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)'
                    : 'transparent',
                border:
                  seciliTab === tab.id
                    ? '2px solid #8B5CF6'
                    : '2px solid rgba(139, 92, 246, 0.2)',
                borderRadius: '14px',
                color: seciliTab === tab.id ? '#FFFFFF' : 'rgba(255,255,255,0.7)',
                fontSize: '15px',
                fontWeight: '800',
                cursor: 'pointer',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                boxShadow:
                  seciliTab === tab.id
                    ? '0 10px 30px rgba(139, 92, 246, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
                    : 'none',
                transform: seciliTab === tab.id ? 'translateY(-2px)' : 'none',
              }}
              onMouseEnter={(e) => {
                if (seciliTab !== tab.id) {
                  e.currentTarget.style.background = 'rgba(139, 92, 246, 0.1)';
                  e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.4)';
                }
              }}
              onMouseLeave={(e) => {
                if (seciliTab !== tab.id) {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.2)';
                }
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content Area */}
        {yukleniyor ? (
          <div
            style={{
              background: 'rgba(26, 26, 26, 0.95)',
              backdropFilter: 'blur(30px)',
              borderRadius: '28px',
              padding: '100px',
              textAlign: 'center',
              border: '2px solid rgba(139, 92, 246, 0.2)',
            }}
          >
            <div style={{ fontSize: '80px', marginBottom: '32px', animation: 'quantum-spin 2s linear infinite' }}>
              🔮
            </div>
            <div style={{ fontSize: '22px', color: 'rgba(255,255,255,0.7)', fontWeight: '600' }}>
              Quantum verileri yükleniyor...
            </div>
          </div>
        ) : hata ? (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '2px solid rgba(239, 68, 68, 0.5)',
              borderRadius: '20px',
              padding: '60px',
              textAlign: 'center',
              color: '#EF4444',
              fontSize: '18px',
              fontWeight: '700',
            }}
          >
            ❌ {hata}
          </div>
        ) : data ? (
          <>
            {/* Quantum Sinyaller Tab */}
            {seciliTab === 'sinyaller' && (
              <div>
                {/* Güven Seviyesi Filtresi */}
                <div
                  style={{
                    background: 'rgba(26, 26, 26, 0.95)',
                    backdropFilter: 'blur(30px)',
                    border: '2px solid rgba(139, 92, 246, 0.3)',
                    borderRadius: '20px',
                    padding: '24px',
                    marginBottom: '24px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', marginBottom: '12px', fontWeight: '600' }}>
                        Minimum Güven Seviyesi: {(minGuven * 100).toFixed(0)}%
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="0.9"
                        step="0.05"
                        value={minGuven}
                        onChange={(e) => setMinGuven(parseFloat(e.target.value))}
                        style={{
                          width: '100%',
                          height: '8px',
                          borderRadius: '4px',
                          background: `linear-gradient(to right, #8B5CF6 0%, #8B5CF6 ${((minGuven - 0.5) / 0.4) * 100}%, rgba(139, 92, 246, 0.2) ${((minGuven - 0.5) / 0.4) * 100}%, rgba(139, 92, 246, 0.2) 100%)`,
                          outline: 'none',
                          cursor: 'pointer',
                        }}
                      />
                    </div>
                    <button
                      onClick={quantumVerileriAl}
                      style={{
                        padding: '12px 24px',
                        background: 'linear-gradient(135deg, #8B5CF6, #6366F1)',
                        border: '2px solid #8B5CF6',
                        borderRadius: '12px',
                        color: '#FFFFFF',
                        fontWeight: '700',
                        cursor: 'pointer',
                        fontSize: '14px',
                        boxShadow: '0 8px 20px rgba(139, 92, 246, 0.4)',
                      }}
                    >
                      🔄 Yenile
                    </button>
                  </div>
                </div>

                {/* Sinyal Listesi */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))',
                    gap: '24px',
                  }}
                >
                  {data.signals.map((signal, index) => {
                    const sRenk = sinyalRengi(signal.signal);
                    const rRenk = riskRengi(signal.riskScore);

                    return (
                      <div
                        key={`${signal.symbol}-${index}`}
                        onClick={() => setSeciliCoin(signal)}
                        style={{
                          background: `linear-gradient(135deg, ${sRenk.bg} 0%, rgba(26, 26, 26, 0.95) 100%)`,
                          backdropFilter: 'blur(30px)',
                          border: `2px solid ${sRenk.border}`,
                          borderRadius: '24px',
                          padding: '28px',
                          cursor: 'pointer',
                          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                          boxShadow: `0 15px 40px ${sRenk.border}30`,
                          position: 'relative',
                          overflow: 'hidden',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.transform = 'translateY(-6px) scale(1.02)';
                          e.currentTarget.style.boxShadow = `0 25px 60px ${sRenk.border}50`;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.transform = 'translateY(0) scale(1)';
                          e.currentTarget.style.boxShadow = `0 15px 40px ${sRenk.border}30`;
                        }}
                      >
                        {/* Radial gradient background */}
                        <div
                          style={{
                            position: 'absolute',
                            top: 0,
                            right: 0,
                            width: '150px',
                            height: '150px',
                            background: `radial-gradient(circle, ${sRenk.border}20 0%, transparent 70%)`,
                          }}
                        />

                        {/* Header */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '20px', position: 'relative' }}>
                          <div>
                            <div style={{ fontSize: '28px', fontWeight: '900', color: '#FFFFFF', marginBottom: '4px' }}>
                              {signal.symbol}
                            </div>
                            <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)' }}>
                              ${signal.price.toFixed(signal.price < 1 ? 6 : 2)}
                            </div>
                          </div>

                          <div
                            style={{
                              background: sRenk.bg,
                              border: `2px solid ${sRenk.border}`,
                              borderRadius: '12px',
                              padding: '10px 20px',
                              fontWeight: '900',
                              fontSize: '16px',
                              color: sRenk.text,
                              boxShadow: `0 4px 12px ${sRenk.border}40`,
                            }}
                          >
                            {signal.signal}
                          </div>
                        </div>

                        {/* 24h Change */}
                        <div
                          style={{
                            fontSize: '22px',
                            fontWeight: '800',
                            color: signal.priceChange24h >= 0 ? '#10B981' : '#EF4444',
                            marginBottom: '20px',
                          }}
                        >
                          {signal.priceChange24h >= 0 ? '↗' : '↘'} {signal.priceChange24h.toFixed(2)}%
                          <span style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)', marginLeft: '8px' }}>
                            24h
                          </span>
                        </div>

                        {/* AI Scores */}
                        <div style={{ marginBottom: '20px' }}>
                          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '12px', fontWeight: '600' }}>
                            AI Ensemble Skorları
                          </div>
                          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                            <div style={{ background: 'rgba(99, 102, 241, 0.2)', padding: '6px 12px', borderRadius: '8px', border: '1px solid rgba(99, 102, 241, 0.4)' }}>
                              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)' }}>LSTM:</span>{' '}
                              <span style={{ fontSize: '12px', fontWeight: '800', color: '#6366F1' }}>
                                {(signal.strategies.lstm * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div style={{ background: 'rgba(139, 92, 246, 0.2)', padding: '6px 12px', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.4)' }}>
                              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)' }}>Trans:</span>{' '}
                              <span style={{ fontSize: '12px', fontWeight: '800', color: '#8B5CF6' }}>
                                {(signal.strategies.transformer * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div style={{ background: 'rgba(168, 85, 247, 0.2)', padding: '6px 12px', borderRadius: '8px', border: '1px solid rgba(168, 85, 247, 0.4)' }}>
                              <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.7)' }}>Boost:</span>{' '}
                              <span style={{ fontSize: '12px', fontWeight: '800', color: '#A855F7' }}>
                                {(signal.strategies.gradientBoosting * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Confidence & Risk */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                          <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '12px', borderRadius: '12px' }}>
                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>
                              Güven
                            </div>
                            <div style={{ fontSize: '20px', fontWeight: '900', color: '#8B5CF6' }}>
                              {(signal.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div style={{ background: rRenk.bg, padding: '12px', borderRadius: '12px', border: `1px solid ${rRenk.border}` }}>
                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>
                              Risk
                            </div>
                            <div style={{ fontSize: '16px', fontWeight: '900', color: rRenk.text }}>
                              {rRenk.label}
                            </div>
                          </div>
                        </div>

                        {/* Triggers */}
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.7)', lineHeight: '1.6' }}>
                          {signal.triggers.slice(0, 2).map((trigger, i) => (
                            <div key={i} style={{ marginBottom: '4px' }}>
                              • {trigger}
                            </div>
                          ))}
                        </div>

                        {/* Volume */}
                        <div style={{ marginTop: '12px', fontSize: '11px', color: 'rgba(255,255,255,0.5)' }}>
                          Hacim: ${formatSayi(signal.volume24h)}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {data.signals.length === 0 && (
                  <div
                    style={{
                      background: 'rgba(26, 26, 26, 0.95)',
                      borderRadius: '24px',
                      padding: '80px',
                      textAlign: 'center',
                      border: '2px solid rgba(139, 92, 246, 0.2)',
                    }}
                  >
                    <svg width="72" height="72" viewBox="0 0 24 24" fill="none" style={{ margin: '0 auto 24px' }}>
                      <circle cx="12" cy="12" r="10" stroke="#8B5CF6" strokeWidth="2" opacity="0.3"/>
                      <path d="M12 8v8M8 12h8" stroke="#8B5CF6" strokeWidth="2" strokeLinecap="round"/>
                    </svg>
                    <div style={{ fontSize: '20px', color: 'rgba(255,255,255,0.6)', fontWeight: '600' }}>
                      Bu güven seviyesinde sinyal bulunamadı
                    </div>
                    <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.4)', marginTop: '12px' }}>
                      Minimum güven seviyesini düşürmeyi deneyin
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Backtest Analizi Tab */}
            {seciliTab === 'backtest' && backtestData && (
              <div>
                {/* Backtest Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px' }}>
                  {[
                    { label: 'Toplam Test', value: backtestData.summary.totalTests, color: '#8B5CF6' },
                    { label: 'Başarılı', value: backtestData.summary.successfulTrades, color: '#10B981' },
                    { label: 'Başarısız', value: backtestData.summary.failedTrades, color: '#EF4444' },
                    { label: 'Başarı Oranı', value: `${backtestData.summary.successRate.toFixed(1)}%`, color: '#F59E0B' },
                  ].map((stat, i) => (
                    <div
                      key={i}
                      style={{
                        background: `linear-gradient(135deg, ${stat.color}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                        backdropFilter: 'blur(30px)',
                        border: `2px solid ${stat.color}40`,
                        borderRadius: '20px',
                        padding: '28px',
                        boxShadow: `0 15px 40px ${stat.color}20`,
                      }}
                    >
                      <div style={{ fontSize: '32px', fontWeight: '900', color: stat.color, marginBottom: '8px', marginTop: '12px' }}>
                        {stat.value}
                      </div>
                      <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)' }}>{stat.label}</div>
                    </div>
                  ))}
                </div>

                {/* Backtest Results */}
                <div style={{ background: 'rgba(26, 26, 26, 0.95)', backdropFilter: 'blur(30px)', border: '2px solid rgba(139, 92, 246, 0.3)', borderRadius: '24px', padding: '32px' }}>
                  <div style={{ fontSize: '22px', fontWeight: '800', color: '#FFFFFF', marginBottom: '24px' }}>
                    Real Binance Data Backtest Results
                  </div>

                  <div style={{ display: 'grid', gap: '16px' }}>
                    {backtestData.results.map((result: any, i: number) => (
                      <div
                        key={i}
                        style={{
                          background: `linear-gradient(135deg, ${result.color}10 0%, rgba(0, 0, 0, 0.3) 100%)`,
                          border: `2px solid ${result.color}40`,
                          borderRadius: '16px',
                          padding: '24px',
                          display: 'grid',
                          gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr',
                          gap: '24px',
                          alignItems: 'center',
                        }}
                      >
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '800', color: '#FFFFFF', marginBottom: '4px' }}>
                            {result.strategy}
                          </div>
                          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)' }}>
                            Test Dönemi: {result.period}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Kâr</div>
                          <div style={{ fontSize: '20px', fontWeight: '900', color: result.totalReturn > 0 ? '#10B981' : '#EF4444' }}>
                            {result.totalReturn > 0 ? '+' : ''}{result.totalReturn.toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>İşlem</div>
                          <div style={{ fontSize: '20px', fontWeight: '900', color: '#8B5CF6' }}>
                            {result.totalTrades}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Başarı</div>
                          <div style={{ fontSize: '20px', fontWeight: '900', color: '#10B981' }}>
                            {result.winRate.toFixed(0)}%
                          </div>
                        </div>

                        <div>
                          <button
                            style={{
                              background: `linear-gradient(135deg, ${result.color}, ${result.color}CC)`,
                              border: 'none',
                              borderRadius: '10px',
                              padding: '10px 20px',
                              color: '#FFFFFF',
                              fontWeight: '700',
                              fontSize: '13px',
                              cursor: 'pointer',
                              boxShadow: `0 6px 20px ${result.color}40`,
                            }}
                          >
                            Detay
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Risk Yönetimi Tab */}
            {seciliTab === 'risk' && (
              <div>
                {/* Risk Metrikleri */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
                  {[
                    { label: 'Toplam Risk Skoru', value: '42/100', status: 'ORTA', color: '#F59E0B', icon: '⚠️' },
                    { label: 'Pozisyon Riski', value: '28/100', status: 'DÜŞÜK', color: '#10B981', icon: '✓' },
                    { label: 'Volatilite Riski', value: '68/100', status: 'YÜKSEK', color: '#EF4444', icon: '⚡' },
                  ].map((metric, i) => (
                    <div
                      key={i}
                      style={{
                        background: `linear-gradient(135deg, ${metric.color}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                        backdropFilter: 'blur(30px)',
                        border: `2px solid ${metric.color}50`,
                        borderRadius: '24px',
                        padding: '32px',
                        boxShadow: `0 20px 50px ${metric.color}30`,
                      }}
                    >
                      <div style={{ fontSize: '48px', marginBottom: '16px' }}>{metric.icon}</div>
                      <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', marginBottom: '12px', fontWeight: '600' }}>
                        {metric.label}
                      </div>
                      <div style={{ fontSize: '36px', fontWeight: '900', color: metric.color, marginBottom: '12px' }}>
                        {metric.value}
                      </div>
                      <div
                        style={{
                          display: 'inline-block',
                          background: `${metric.color}30`,
                          border: `2px solid ${metric.color}`,
                          borderRadius: '8px',
                          padding: '6px 16px',
                          fontSize: '13px',
                          fontWeight: '800',
                          color: metric.color,
                        }}
                      >
                        {metric.status}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Risk Kontrol Kuralları */}
                <div style={{ background: 'rgba(26, 26, 26, 0.95)', backdropFilter: 'blur(30px)', border: '2px solid rgba(139, 92, 246, 0.3)', borderRadius: '24px', padding: '32px', marginBottom: '24px' }}>
                  <div style={{ fontSize: '22px', fontWeight: '800', color: '#FFFFFF', marginBottom: '24px' }}>
                    🛡️ Risk Kontrol Kuralları
                  </div>

                  <div style={{ display: 'grid', gap: '16px' }}>
                    {[
                      { rule: 'Maksimum Pozisyon Büyüklüğü', value: '2% / İşlem', status: 'AKTİF', icon: '📊' },
                      { rule: 'Stop Loss Mesafesi', value: '1.5%', status: 'AKTİF', icon: '🛑' },
                      { rule: 'Take Profit Hedefi', value: '3%', status: 'AKTİF', icon: '🎯' },
                      { rule: 'Günlük Maksimum Kayıp', value: '5%', status: 'AKTİF', icon: '📉' },
                      { rule: 'Eşzamanlı Maksimum İşlem', value: '5', status: 'AKTİF', icon: '🔢' },
                      { rule: 'Leverage Sınırı', value: '3x', status: 'AKTİF', icon: '⚡' },
                    ].map((rule, i) => (
                      <div
                        key={i}
                        style={{
                          background: 'rgba(16, 185, 129, 0.1)',
                          border: '2px solid rgba(16, 185, 129, 0.3)',
                          borderRadius: '16px',
                          padding: '20px 24px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '20px',
                        }}
                      >
                        <div style={{ fontSize: '28px' }}>{rule.icon}</div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '15px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
                            {rule.rule}
                          </div>
                          <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)' }}>
                            Değer: {rule.value}
                          </div>
                        </div>
                        <div
                          style={{
                            background: '#10B98120',
                            border: '2px solid #10B981',
                            borderRadius: '10px',
                            padding: '8px 16px',
                            fontSize: '12px',
                            fontWeight: '800',
                            color: '#10B981',
                          }}
                        >
                          ✓ {rule.status}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Risk Uyarıları */}
                <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '2px solid rgba(239, 68, 68, 0.4)', borderRadius: '20px', padding: '28px' }}>
                  <div style={{ fontSize: '20px', fontWeight: '800', color: '#EF4444', marginBottom: '16px' }}>
                    ⚠️ Aktif Risk Uyarıları
                  </div>
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {[
                      'Volatilite son 24 saatte %45 artış gösterdi',
                      'BTC/USDT çiftinde yüksek hacimli işlemler tespit edildi',
                      '3 pozisyon için stop-loss seviyesine yaklaşıldı',
                    ].map((warning, i) => (
                      <div
                        key={i}
                        style={{
                          background: 'rgba(0, 0, 0, 0.3)',
                          padding: '14px 18px',
                          borderRadius: '12px',
                          fontSize: '14px',
                          color: 'rgba(255,255,255,0.9)',
                        }}
                      >
                        • {warning}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Bot Kontrolü Tab */}
            {seciliTab === 'botlar' && botsData && (
              <div>
                {/* Bot İstatistikleri */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px' }}>
                  {[
                    { label: 'Toplam Bot', value: botsData.summary.totalBots, color: '#8B5CF6', icon: '🤖' },
                    { label: 'Aktif', value: botsData.summary.activeBots, color: '#10B981', icon: '✓' },
                    { label: 'Pasif', value: botsData.summary.inactiveBots, color: '#6B7280', icon: '⏸' },
                    { label: 'Hata', value: botsData.summary.errorBots, color: '#EF4444', icon: '❌' },
                  ].map((stat, i) => (
                    <div
                      key={i}
                      style={{
                        background: `linear-gradient(135deg, ${stat.color}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                        backdropFilter: 'blur(30px)',
                        border: `2px solid ${stat.color}40`,
                        borderRadius: '20px',
                        padding: '28px',
                        textAlign: 'center',
                        boxShadow: `0 15px 40px ${stat.color}20`,
                      }}
                    >
                      <div style={{ fontSize: '40px', marginBottom: '12px' }}>{stat.icon}</div>
                      <div style={{ fontSize: '36px', fontWeight: '900', color: stat.color, marginBottom: '8px' }}>
                        {stat.value}
                      </div>
                      <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)' }}>{stat.label}</div>
                    </div>
                  ))}
                </div>

                {/* Bot Listesi */}
                <div style={{ display: 'grid', gap: '20px' }}>
                  {botsData.bots.map((bot: any, i: number) => {
                    const statusColor = bot.status === 'ACTIVE' ? '#10B981' : bot.status === 'INACTIVE' ? '#6B7280' : '#EF4444';
                    return (
                    <div
                      key={i}
                      style={{
                        background: `linear-gradient(135deg, ${statusColor}10 0%, rgba(26, 26, 26, 0.95) 100%)`,
                        backdropFilter: 'blur(30px)',
                        border: `2px solid ${statusColor}40`,
                        borderRadius: '20px',
                        padding: '24px',
                        display: 'grid',
                        gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 120px',
                        gap: '24px',
                        alignItems: 'center',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '16px', fontWeight: '800', color: '#FFFFFF', marginBottom: '6px' }}>
                          🤖 {bot.name}
                        </div>
                        <div
                          style={{
                            display: 'inline-block',
                            background: `${statusColor}20`,
                            border: `2px solid ${statusColor}`,
                            borderRadius: '8px',
                            padding: '4px 12px',
                            fontSize: '11px',
                            fontWeight: '800',
                            color: statusColor,
                          }}
                        >
                          {bot.statusText}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>İşlem</div>
                        <div style={{ fontSize: '18px', fontWeight: '900', color: '#8B5CF6' }}>{bot.trades24h}</div>
                      </div>

                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Kâr</div>
                        <div style={{ fontSize: '18px', fontWeight: '900', color: bot.profitPercentage.startsWith('+') ? '#10B981' : bot.profitPercentage.startsWith('-') ? '#EF4444' : '#6B7280' }}>
                          {bot.profitPercentage}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Uptime</div>
                        <div style={{ fontSize: '18px', fontWeight: '900', color: statusColor }}>{bot.uptime}%</div>
                      </div>

                      <div>
                        <div style={{ background: 'rgba(255, 255, 255, 0.1)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${bot.uptime}%`,
                              height: '100%',
                              background: `linear-gradient(90deg, ${statusColor}, ${statusColor}CC)`,
                            }}
                          />
                        </div>
                      </div>

                      <div style={{ display: 'flex', gap: '8px' }}>
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            handleBotAction(bot.id, bot.status === 'ACTIVE' ? 'pause' : 'start');
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'scale(1.05)';
                            e.currentTarget.style.boxShadow = bot.status === 'ACTIVE' ? '0 4px 12px rgba(239, 68, 68, 0.4)' : '0 4px 12px rgba(16, 185, 129, 0.4)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = 'none';
                          }}
                          style={{
                            flex: 1,
                            background: bot.status === 'ACTIVE' ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(220, 38, 38, 0.2))' : 'linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.2))',
                            border: bot.status === 'ACTIVE' ? '2px solid #EF4444' : '2px solid #10B981',
                            borderRadius: '8px',
                            padding: '10px',
                            color: bot.status === 'ACTIVE' ? '#EF4444' : '#10B981',
                            fontWeight: '700',
                            fontSize: '14px',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                            position: 'relative',
                            zIndex: 10,
                          }}
                        >
                          {bot.status === 'ACTIVE' ? '⏸' : '▶'}
                        </button>
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            openBotModal(bot);
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'scale(1.05)';
                            e.currentTarget.style.boxShadow = '0 4px 12px rgba(139, 92, 246, 0.4)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = 'none';
                          }}
                          style={{
                            flex: 1,
                            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(99, 102, 241, 0.2))',
                            border: '2px solid #8B5CF6',
                            borderRadius: '8px',
                            padding: '10px',
                            color: '#8B5CF6',
                            fontWeight: '700',
                            fontSize: '14px',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                            position: 'relative',
                            zIndex: 10,
                          }}
                        >
                          ⚙️
                        </button>
                      </div>
                    </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Bot Detail Modal */}
            {showBotModal && selectedBot && (
              <div
                style={{
                  position: 'fixed',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'rgba(0, 0, 0, 0.85)',
                  backdropFilter: 'blur(10px)',
                  zIndex: 9999,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '20px',
                }}
                onClick={closeBotModal}
              >
                <div
                  style={{
                    background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(20, 20, 20, 0.98) 100%)',
                    border: '2px solid rgba(139, 92, 246, 0.5)',
                    borderRadius: '24px',
                    padding: '32px',
                    maxWidth: '900px',
                    width: '100%',
                    maxHeight: '90vh',
                    overflowY: 'auto',
                    boxShadow: '0 25px 50px rgba(139, 92, 246, 0.3)',
                  }}
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Modal Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '32px' }}>
                    <div>
                      <h2 style={{ fontSize: '28px', fontWeight: '900', color: '#FFFFFF', marginBottom: '8px' }}>
                        {selectedBot.name}
                      </h2>
                      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                        <div
                          style={{
                            padding: '6px 14px',
                            background: selectedBot.status === 'ACTIVE' ? '#10B98120' : selectedBot.status === 'INACTIVE' ? '#6B728020' : '#EF444420',
                            border: `2px solid ${selectedBot.status === 'ACTIVE' ? '#10B981' : selectedBot.status === 'INACTIVE' ? '#6B7280' : '#EF4444'}`,
                            borderRadius: '8px',
                            fontSize: '13px',
                            fontWeight: '700',
                            color: selectedBot.status === 'ACTIVE' ? '#10B981' : selectedBot.status === 'INACTIVE' ? '#6B7280' : '#EF4444',
                          }}
                        >
                          {selectedBot.statusText}
                        </div>
                        <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)' }}>
                          {selectedBot.strategy}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={closeBotModal}
                      style={{
                        background: 'rgba(255,255,255,0.1)',
                        border: '2px solid rgba(255,255,255,0.2)',
                        borderRadius: '12px',
                        width: '44px',
                        height: '44px',
                        cursor: 'pointer',
                        fontSize: '20px',
                        color: '#FFFFFF',
                        transition: 'all 0.3s',
                      }}
                    >
                      ✕
                    </button>
                  </div>

                  {/* Performance Stats */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '32px' }}>
                    {[
                      { label: '24h İşlem', value: selectedBot.trades24h, color: '#8B5CF6' },
                      { label: '24h Kâr', value: selectedBot.profitPercentage, color: parseFloat(selectedBot.profit24h) >= 0 ? '#10B981' : '#EF4444' },
                      { label: 'Kazanma Oranı', value: selectedBot.winRate + '%', color: '#F59E0B' },
                      { label: 'Çalışma Süresi', value: selectedBot.uptime + '%', color: '#6366F1' },
                    ].map((stat, i) => (
                      <div
                        key={i}
                        style={{
                          background: `linear-gradient(135deg, ${stat.color}15 0%, rgba(26, 26, 26, 0.5) 100%)`,
                          border: `2px solid ${stat.color}40`,
                          borderRadius: '16px',
                          padding: '20px',
                          textAlign: 'center',
                        }}
                      >
                        <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                          {stat.label}
                        </div>
                        <div style={{ fontSize: '24px', fontWeight: '900', color: stat.color }}>
                          {stat.value}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Market Data */}
                  <div style={{ background: 'rgba(139, 92, 246, 0.1)', border: '2px solid rgba(139, 92, 246, 0.3)', borderRadius: '16px', padding: '24px', marginBottom: '24px' }}>
                    <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                      Piyasa Verisi
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Son Sinyal</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#8B5CF6' }}>{selectedBot.lastSignal}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Mevcut Fiyat</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF' }}>${selectedBot.currentPrice.toLocaleString()}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>24h Değişim</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: selectedBot.priceChange24h >= 0 ? '#10B981' : '#EF4444' }}>
                          {selectedBot.priceChange24h >= 0 ? '+' : ''}{selectedBot.priceChange24h.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Bot Configuration */}
                  <div style={{ background: 'rgba(16, 185, 129, 0.1)', border: '2px solid rgba(16, 185, 129, 0.3)', borderRadius: '16px', padding: '24px', marginBottom: '24px' }}>
                    <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                      Bot Yapılandırması
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px' }}>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Kaldıraç</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#10B981' }}>{selectedBot.config.leverage}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Max Pozisyon</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#10B981' }}>{selectedBot.config.maxPosition}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Stop Loss</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#EF4444' }}>{selectedBot.config.stopLoss}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '6px' }}>Take Profit</div>
                        <div style={{ fontSize: '18px', fontWeight: '800', color: '#10B981' }}>{selectedBot.config.takeProfit}</div>
                      </div>
                    </div>
                  </div>

                  {/* Recent Trades */}
                  {selectedBot.recentTrades && selectedBot.recentTrades.length > 0 && (
                    <div style={{ background: 'rgba(245, 158, 11, 0.1)', border: '2px solid rgba(245, 158, 11, 0.3)', borderRadius: '16px', padding: '24px', marginBottom: '24px' }}>
                      <h3 style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                        Son İşlemler
                      </h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {selectedBot.recentTrades.map((trade: any, i: number) => (
                          <div
                            key={i}
                            style={{
                              background: 'rgba(26, 26, 26, 0.5)',
                              border: '2px solid rgba(255,255,255,0.1)',
                              borderRadius: '12px',
                              padding: '16px',
                              display: 'grid',
                              gridTemplateColumns: '1fr 1fr 1fr 1fr',
                              gap: '16px',
                              alignItems: 'center',
                            }}
                          >
                            <div>
                              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Zaman</div>
                              <div style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF' }}>
                                {new Date(trade.time).toLocaleTimeString('tr-TR')}
                              </div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>İşlem</div>
                              <div
                                style={{
                                  fontSize: '13px',
                                  fontWeight: '800',
                                  color: trade.action === 'BUY' ? '#10B981' : '#EF4444',
                                }}
                              >
                                {trade.action === 'BUY' ? 'ALIŞ' : 'SATIŞ'}
                              </div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Fiyat</div>
                              <div style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF' }}>
                                ${trade.price.toLocaleString()}
                              </div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>Kâr</div>
                              <div style={{ fontSize: '13px', fontWeight: '800', color: '#10B981' }}>
                                {trade.profit}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Bot Info */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '20px', borderTop: '2px solid rgba(255,255,255,0.1)' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>
                      Oluşturulma: {new Date(selectedBot.createdAt).toLocaleDateString('tr-TR')}
                    </div>
                    <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>
                      Son Güncelleme: {new Date(selectedBot.lastUpdate).toLocaleTimeString('tr-TR')}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
                    <button
                      onClick={() => {
                        handleBotAction(selectedBot.id, selectedBot.status === 'ACTIVE' ? 'pause' : 'start');
                        closeBotModal();
                      }}
                      style={{
                        flex: 1,
                        background: selectedBot.status === 'ACTIVE' ? 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)' : 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                        border: 'none',
                        borderRadius: '12px',
                        padding: '16px',
                        fontSize: '16px',
                        fontWeight: '800',
                        color: '#FFFFFF',
                        cursor: 'pointer',
                        transition: 'all 0.3s',
                        boxShadow: selectedBot.status === 'ACTIVE' ? '0 10px 30px rgba(239, 68, 68, 0.3)' : '0 10px 30px rgba(16, 185, 129, 0.3)',
                      }}
                    >
                      {selectedBot.status === 'ACTIVE' ? 'Botu Duraklat' : 'Botu Başlat'}
                    </button>
                    <button
                      onClick={closeBotModal}
                      style={{
                        flex: 1,
                        background: 'rgba(255,255,255,0.1)',
                        border: '2px solid rgba(255,255,255,0.2)',
                        borderRadius: '12px',
                        padding: '16px',
                        fontSize: '16px',
                        fontWeight: '800',
                        color: '#FFFFFF',
                        cursor: 'pointer',
                        transition: 'all 0.3s',
                      }}
                    >
                      Kapat
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Canlı İzleme Tab */}
            {seciliTab === 'monitoring' && (
              <div>
                {/* Live Stats */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px', marginBottom: '32px' }}>
                  {[
                    { label: 'Aktif İşlem', value: '23', change: '+3', color: '#10B981' },
                    { label: 'Kâr/Zarar', value: '+$1,247', change: '+$89', color: '#10B981' },
                    { label: 'Win Rate', value: '73%', change: '+2%', color: '#8B5CF6' },
                    { label: 'Hacim (24h)', value: '$45.2K', change: '+$8K', color: '#F59E0B' },
                    { label: 'API Calls', value: '1,892', change: '+127', color: '#6366F1' },
                  ].map((stat, i) => (
                    <div
                      key={i}
                      style={{
                        background: `linear-gradient(135deg, ${stat.color}15 0%, rgba(26, 26, 26, 0.95) 100%)`,
                        backdropFilter: 'blur(30px)',
                        border: `2px solid ${stat.color}40`,
                        borderRadius: '16px',
                        padding: '20px',
                        boxShadow: `0 10px 30px ${stat.color}20`,
                      }}
                    >
                      <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px', fontWeight: '600' }}>
                        {stat.label}
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: '900', color: stat.color, marginBottom: '6px' }}>
                        {stat.value}
                      </div>
                      <div style={{ fontSize: '12px', fontWeight: '700', color: stat.color }}>
                        {stat.change}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Aktif Pozisyonlar */}
                <div style={{ background: 'rgba(26, 26, 26, 0.95)', backdropFilter: 'blur(30px)', border: '2px solid rgba(139, 92, 246, 0.3)', borderRadius: '24px', padding: '32px', marginBottom: '24px' }}>
                  <div style={{ fontSize: '22px', fontWeight: '800', color: '#FFFFFF', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span>📈 Aktif Pozisyonlar</span>
                    <div
                      style={{
                        background: '#10B98120',
                        border: '2px solid #10B981',
                        borderRadius: '8px',
                        padding: '4px 12px',
                        fontSize: '13px',
                        fontWeight: '800',
                        color: '#10B981',
                      }}
                    >
                      23 Açık
                    </div>
                  </div>

                  <div style={{ display: 'grid', gap: '12px' }}>
                    {[
                      { symbol: 'BTC', side: 'LONG', entry: '$42,150', current: '$43,280', pnl: '+2.68%', size: '0.5', color: '#10B981' },
                      { symbol: 'ETH', side: 'LONG', entry: '$2,245', current: '$2,318', pnl: '+3.25%', size: '5', color: '#10B981' },
                      { symbol: 'SOL', side: 'SHORT', entry: '$98.50', current: '$96.20', pnl: '+2.34%', size: '20', color: '#10B981' },
                      { symbol: 'XRP', side: 'LONG', entry: '$0.62', current: '$0.64', pnl: '+3.23%', size: '1000', color: '#10B981' },
                      { symbol: 'ADA', side: 'LONG', entry: '$0.48', current: '$0.47', pnl: '-2.08%', size: '800', color: '#EF4444' },
                    ].map((pos, i) => (
                      <div
                        key={i}
                        style={{
                          background: `linear-gradient(135deg, ${pos.color}10 0%, rgba(0, 0, 0, 0.3) 100%)`,
                          border: `2px solid ${pos.color}30`,
                          borderRadius: '14px',
                          padding: '16px 20px',
                          display: 'grid',
                          gridTemplateColumns: '1fr 1fr 1.5fr 1.5fr 1fr 1fr 100px',
                          gap: '16px',
                          alignItems: 'center',
                        }}
                      >
                        <div style={{ fontSize: '16px', fontWeight: '900', color: '#FFFFFF' }}>{pos.symbol}</div>

                        <div>
                          <div
                            style={{
                              display: 'inline-block',
                              background: pos.side === 'LONG' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                              border: `2px solid ${pos.side === 'LONG' ? '#10B981' : '#EF4444'}`,
                              borderRadius: '6px',
                              padding: '4px 10px',
                              fontSize: '11px',
                              fontWeight: '800',
                              color: pos.side === 'LONG' ? '#10B981' : '#EF4444',
                            }}
                          >
                            {pos.side}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '2px' }}>Giriş</div>
                          <div style={{ fontSize: '13px', fontWeight: '700', color: 'rgba(255,255,255,0.8)' }}>{pos.entry}</div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '2px' }}>Güncel</div>
                          <div style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF' }}>{pos.current}</div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '2px' }}>Miktar</div>
                          <div style={{ fontSize: '13px', fontWeight: '700', color: 'rgba(255,255,255,0.8)' }}>{pos.size}</div>
                        </div>

                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '900', color: pos.color }}>{pos.pnl}</div>
                        </div>

                        <div>
                          <button
                            style={{
                              background: 'rgba(239, 68, 68, 0.2)',
                              border: '2px solid #EF4444',
                              borderRadius: '8px',
                              padding: '6px 12px',
                              color: '#EF4444',
                              fontWeight: '800',
                              fontSize: '11px',
                              cursor: 'pointer',
                            }}
                          >
                            Kapat
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Sistem Durumu */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                  {/* API Health */}
                  <div style={{ background: 'rgba(26, 26, 26, 0.95)', backdropFilter: 'blur(30px)', border: '2px solid rgba(16, 185, 129, 0.3)', borderRadius: '20px', padding: '28px' }}>
                    <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '20px' }}>
                      🟢 API Durumu
                    </div>
                    <div style={{ display: 'grid', gap: '12px' }}>
                      {[
                        { name: 'Binance API', status: 'Bağlı', latency: '12ms', color: '#10B981' },
                        { name: 'WebSocket', status: 'Bağlı', latency: '8ms', color: '#10B981' },
                        { name: 'Database', status: 'Bağlı', latency: '4ms', color: '#10B981' },
                        { name: 'AI Engine', status: 'Bağlı', latency: '28ms', color: '#10B981' },
                      ].map((api, i) => (
                        <div
                          key={i}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: '12px 16px',
                            background: 'rgba(0, 0, 0, 0.3)',
                            borderRadius: '10px',
                          }}
                        >
                          <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>{api.name}</div>
                          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                            <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)' }}>{api.latency}</div>
                            <div style={{ fontSize: '12px', fontWeight: '800', color: api.color }}>● {api.status}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Son Aktiviteler */}
                  <div style={{ background: 'rgba(26, 26, 26, 0.95)', backdropFilter: 'blur(30px)', border: '2px solid rgba(139, 92, 246, 0.3)', borderRadius: '20px', padding: '28px' }}>
                    <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '20px' }}>
                      📋 Son Aktiviteler
                    </div>
                    <div style={{ display: 'grid', gap: '10px' }}>
                      {[
                        { action: 'BTC LONG açıldı', time: '2 dk önce', icon: '📈', color: '#10B981' },
                        { action: 'ETH pozisyonu güncellendi', time: '5 dk önce', icon: '🔄', color: '#8B5CF6' },
                        { action: 'SOL SHORT kapatıldı', time: '8 dk önce', icon: '📉', color: '#EF4444' },
                        { action: 'Risk uyarısı: Volatilite artışı', time: '12 dk önce', icon: '⚠️', color: '#F59E0B' },
                        { action: 'XRP pozisyonu açıldı', time: '15 dk önce', icon: '📈', color: '#10B981' },
                      ].map((activity, i) => (
                        <div
                          key={i}
                          style={{
                            display: 'flex',
                            gap: '12px',
                            padding: '10px 14px',
                            background: 'rgba(0, 0, 0, 0.3)',
                            borderRadius: '10px',
                            alignItems: 'center',
                          }}
                        >
                          <div style={{ fontSize: '18px' }}>{activity.icon}</div>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: '13px', fontWeight: '600', color: '#FFFFFF' }}>{activity.action}</div>
                            <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.5)' }}>{activity.time}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}

        {/* Coin Detail Modal */}
        {seciliCoin && (
          <div
            onClick={() => setSeciliCoin(null)}
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.9)',
              backdropFilter: 'blur(10px)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 9999,
              padding: '24px',
            }}
          >
            <div
              onClick={(e) => e.stopPropagation()}
              style={{
                background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
                borderRadius: '32px',
                padding: '48px',
                maxWidth: '700px',
                width: '100%',
                border: `3px solid ${sinyalRengi(seciliCoin.signal).border}`,
                boxShadow: `0 30px 80px ${sinyalRengi(seciliCoin.signal).border}50`,
                maxHeight: '90vh',
                overflowY: 'auto',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '32px' }}>
                <div>
                  <div style={{ fontSize: '42px', fontWeight: '900', color: '#FFFFFF', marginBottom: '8px' }}>
                    {seciliCoin.symbol}
                  </div>
                  <div style={{ fontSize: '24px', color: 'rgba(255,255,255,0.7)' }}>
                    ${seciliCoin.price.toFixed(seciliCoin.price < 1 ? 6 : 2)}
                  </div>
                </div>

                <button
                  onClick={() => setSeciliCoin(null)}
                  style={{
                    background: 'rgba(239, 68, 68, 0.2)',
                    border: '2px solid rgba(239, 68, 68, 0.5)',
                    borderRadius: '12px',
                    padding: '12px 20px',
                    color: '#EF4444',
                    fontWeight: '800',
                    cursor: 'pointer',
                    fontSize: '14px',
                  }}
                >
                  ✕ Kapat
                </button>
              </div>

              {/* Detaylı Bilgiler */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '32px' }}>
                <div style={{ background: 'rgba(139, 92, 246, 0.1)', padding: '20px', borderRadius: '16px', border: '2px solid rgba(139, 92, 246, 0.3)' }}>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>Sinyal</div>
                  <div style={{ fontSize: '28px', fontWeight: '900', color: sinyalRengi(seciliCoin.signal).text }}>
                    {seciliCoin.signal}
                  </div>
                </div>

                <div style={{ background: 'rgba(139, 92, 246, 0.1)', padding: '20px', borderRadius: '16px', border: '2px solid rgba(139, 92, 246, 0.3)' }}>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>Güven</div>
                  <div style={{ fontSize: '28px', fontWeight: '900', color: '#8B5CF6' }}>
                    {(seciliCoin.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              {/* AI Stratejiler */}
              <div style={{ marginBottom: '32px' }}>
                <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                  🤖 AI Ensemble Stratejileri
                </div>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: 'LSTM Neural Network', score: seciliCoin.strategies.lstm, color: '#6366F1' },
                    { name: 'Transformer Model', score: seciliCoin.strategies.transformer, color: '#8B5CF6' },
                    { name: 'Gradient Boosting', score: seciliCoin.strategies.gradientBoosting, color: '#A855F7' },
                  ].map((strategy) => (
                    <div key={strategy.name} style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '16px', borderRadius: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', fontWeight: '600' }}>
                          {strategy.name}
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '900', color: strategy.color }}>
                          {(strategy.score * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div style={{ background: 'rgba(255, 255, 255, 0.1)', height: '8px', borderRadius: '4px', overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${strategy.score * 100}%`,
                            height: '100%',
                            background: `linear-gradient(90deg, ${strategy.color} 0%, ${strategy.color}CC 100%)`,
                            transition: 'width 1s ease-out',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Triggers */}
              <div style={{ marginBottom: '32px' }}>
                <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                  ⚡ Tetikleyiciler
                </div>
                <div style={{ display: 'grid', gap: '10px' }}>
                  {seciliCoin.triggers.map((trigger, i) => (
                    <div
                      key={i}
                      style={{
                        background: 'rgba(139, 92, 246, 0.1)',
                        border: '2px solid rgba(139, 92, 246, 0.3)',
                        padding: '14px 18px',
                        borderRadius: '12px',
                        fontSize: '14px',
                        color: 'rgba(255,255,255,0.9)',
                      }}
                    >
                      • {trigger}
                    </div>
                  ))}
                </div>
              </div>

              {/* Timeframe Confirmations */}
              <div>
                <div style={{ fontSize: '18px', fontWeight: '800', color: '#FFFFFF', marginBottom: '16px' }}>
                  ⏰ Zaman Dilimi Onayları
                </div>
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                  {seciliCoin.timeframeConfirmations.map((tf, i) => (
                    <div
                      key={i}
                      style={{
                        background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                        border: '2px solid #10B981',
                        padding: '12px 20px',
                        borderRadius: '12px',
                        fontSize: '14px',
                        fontWeight: '800',
                        color: '#FFFFFF',
                        boxShadow: '0 6px 20px rgba(16, 185, 129, 0.4)',
                      }}
                    >
                      ✓ {tf}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      <style>{`
        @keyframes quantum-pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }

        @keyframes quantum-float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }

        @keyframes quantum-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
          width: 12px;
        }

        ::-webkit-scrollbar-track {
          background: rgba(26, 26, 26, 0.5);
          border-radius: 6px;
        }

        ::-webkit-scrollbar-thumb {
          background: linear-gradient(135deg, #8B5CF6, #6366F1);
          border-radius: 6px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(135deg, #7C3AED, #4F46E5);
        }

        /* Range input styling */
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #8B5CF6, #6366F1);
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.6);
        }

        input[type="range"]::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: linear-gradient(135deg, #8B5CF6, #6366F1);
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(139, 92, 246, 0.6);
          border: none;
        }
      `}</style>

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
              border: '2px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '800px',
              maxHeight: '80vh',
              overflow: 'auto',
              boxShadow: '0 20px 60px rgba(139, 92, 246, 0.3)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#8B5CF6', margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span>🧠</span>
                Quantum Pro MANTIK
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

            {/* Content */}
            <div style={{ color: 'rgba(255, 255, 255, 0.9)', lineHeight: '1.8' }}>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#10B981', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  📌 Sayfa Amacı
                </h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0 }}>
                  Quantum Pro, gelişmiş AI ensemble (LSTM + Transformer + Gradient Boosting) kullanan profesyonel trading dashboard'udur.
                  Gerçek zamanlı Binance Futures verilerini 5 farklı modül ile analiz eder: Sinyaller, Backtest, Risk Yönetimi, Bot Kontrolü ve Canlı İzleme.
                </p>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#3B82F6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ⚙️ Nasıl Çalışır?
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li><strong>Sinyaller Tab:</strong> Real-time quantum-inspired AI sinyalleri (AL/SAT/BEKLE) confidence ve risk score ile</li>
                  <li><strong>Backtest Tab:</strong> Historical performance, win rate, total returns analizi</li>
                  <li><strong>Risk Yönetimi:</strong> Position risk, volatility risk, stop-loss/take-profit kuralları</li>
                  <li><strong>Bot Kontrolü:</strong> Aktif/Pasif bot yönetimi, performance metrikleri, başlat/durdur kontrolü</li>
                  <li><strong>Canlı İzleme:</strong> Aktif pozisyonlar, PnL tracking, API health status, activity feed</li>
                </ul>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#F59E0B', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ✨ Önemli Özellikler
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Multi-strategy AI ensemble: LSTM, Transformer, Gradient Boosting skorları ayrı ayrı</li>
                  <li>Confidence slider ile sinyal filtreleme (60%-90%)</li>
                  <li>Settings sayfasından dinamik konfigürasyon (refresh interval, max signals, min confidence)</li>
                  <li>Timeframe confirmations: Birden fazla timeframe'den doğrulama</li>
                  <li>Trigger sistemleri: Volume surge, momentum breakout, RSI oversold/overbought</li>
                </ul>
              </div>

              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#EC4899', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  🎯 Veri Kaynakları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Binance Futures USDT-M API (gerçek zamanlı fiyatlar, volume, change%)</li>
                  <li>/api/quantum-pro/signals - Quantum sinyal üretimi</li>
                  <li>/api/quantum-pro/backtest - Historical backtest sonuçları</li>
                  <li>/api/quantum-pro/risk - Risk metrikleri ve uyarılar</li>
                  <li>/api/quantum-pro/bots - Bot durum ve kontrol</li>
                  <li>/api/quantum-pro/monitoring - Canlı pozisyon ve API health</li>
                </ul>
              </div>

              <div>
                <h3 style={{ color: '#8B5CF6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  💡 Kullanım İpuçları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Yüksek confidence (&gt;80%) sinyalleri daha güvenilirdir</li>
                  <li>Risk Score düşük (&lt;30) coinleri tercih edin</li>
                  <li>Birden fazla timeframe'de doğrulanan sinyaller daha güçlüdür</li>
                  <li>Backtest sonuçlarına bakarak stratejilerin geçmiş performansını görün</li>
                  <li>Bot kontrolünde aktif bot sayısını ve performance'ı takip edin</li>
                  <li>Settings sayfasından refresh interval ve diğer parametreleri özelleştirin</li>
                </ul>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
