/**
 * ☁️ AZURE AI SERVICES DASHBOARD
 *
 * Microsoft Azure OpenAI ve SignalR entegrasyonu
 * Features:
 * - Azure OpenAI market analysis
 * - Sentiment analysis
 * - Real-time SignalR streaming
 * - AI-powered insights
 *
 * WHITE-HAT: Educational and analysis purposes only
 */

'use client';

import { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { Icons as _Icons } from '@/components/Icons';

interface AzureAnalysis {
  symbol: string;
  analysis: string;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number;
  timestamp: string;
}

export default function AzureAIPage() {
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
  const [analysis, setAnalysis] = useState<AzureAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedTab, setSelectedTab] = useState<'analysis' | 'sentiment' | 'realtime'>('analysis');
  const [showLogicModal, setShowLogicModal] = useState(false);

  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];

  const fetchAzureAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/azure/market-analysis?symbol=${selectedSymbol}`);
      const data = await response.json();

      if (data.success && data.data) {
        setAnalysis(data.data);
      } else {
        setError(data.error || 'Azure AI analiz alınamadı');
      }
    } catch (err) {
      setError('Azure AI API bağlantı hatası');
      console.error('Azure AI fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return '#10B981';
      case 'BEARISH': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getSentimentEmoji = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return '🐂';
      case 'BEARISH': return '🐻';
      default: return '🦘';
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
      <SharedSidebar currentPage="azure-ai" />

      <main style={{
        marginTop: '0px',
        padding: '24px',
        maxWidth: '1920px',
        margin: '120px auto 0',
        paddingTop: isLocalhost ? '116px' : '60px'
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(0, 120, 212, 0.15) 0%, rgba(0, 90, 158, 0.1) 100%)',
          backdropFilter: 'blur(20px)',
          border: '2px solid rgba(0, 120, 212, 0.4)',
          borderRadius: '16px',
          padding: '32px',
          marginBottom: '24px',
          boxShadow: '0 8px 32px rgba(0, 120, 212, 0.25)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: 1 }}>
              <div style={{
                width: '56px',
                height: '56px',
                background: 'linear-gradient(135deg, #0078D4, #005A9E)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '28px'
              }}>
                ☁️
              </div>
              <div>
                <h1 style={{ fontSize: '32px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                  Azure AI Services
                </h1>
                <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', margin: '8px 0 0' }}>
                  Microsoft Azure OpenAI destekli piyasa analizi ve sentiment tracking
                </p>
              </div>
            </div>
            <div>
              <style>{`
                @media (max-width: 768px) {
                  .mantik-button-azure {
                    padding: 10px 20px !important;
                    fontSize: 13px !important;
                    height: 42px !important;
                  }
                  .mantik-button-azure span {
                    fontSize: 18px !important;
                  }
                }
                @media (max-width: 480px) {
                  .mantik-button-azure {
                    padding: 8px 16px !important;
                    fontSize: 12px !important;
                    height: 40px !important;
                  }
                  .mantik-button-azure span {
                    fontSize: 16px !important;
                  }
                }
              `}</style>
              <button onClick={() => setShowLogicModal(true)} className="mantik-button-azure" style={{background: 'linear-gradient(135deg, #8B5CF6, #7C3AED)', border: '2px solid rgba(139, 92, 246, 0.5)', borderRadius: '10px', padding: '12px 24px', color: '#FFFFFF', fontSize: '14px', fontWeight: '700', cursor: 'pointer', transition: 'all 0.3s', boxShadow: '0 4px 16px rgba(139, 92, 246, 0.3)', display: 'flex', alignItems: 'center', gap: '8px', height: '44px'}} onMouseEnter={(e) => {e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 6px 24px rgba(139, 92, 246, 0.5)';}} onMouseLeave={(e) => {e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(139, 92, 246, 0.3)';}}>
                <span style={{ fontSize: '18px' }}>🧠</span>MANTIK
              </button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {[
              { key: 'analysis', label: 'Market Analysis', icon: '📊' },
              { key: 'sentiment', label: 'Sentiment', icon: '💭' },
              { key: 'realtime', label: 'Real-time (SignalR)', icon: '⚡' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setSelectedTab(tab.key as any)}
                style={{
                  padding: '12px 24px',
                  background: selectedTab === tab.key
                    ? 'linear-gradient(135deg, #0078D4, #005A9E)'
                    : 'rgba(255, 255, 255, 0.05)',
                  border: `2px solid ${selectedTab === tab.key ? '#0078D4' : 'rgba(255, 255, 255, 0.1)'}`,
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px'
                }}
              >
                <span>{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Symbol Selector */}
        {selectedTab === 'analysis' && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '2px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '12px',
            padding: '20px',
            marginBottom: '24px',
            display: 'flex',
            gap: '16px',
            alignItems: 'center',
            flexWrap: 'wrap'
          }}>
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', fontWeight: '600' }}>
              Symbol Seç:
            </div>
            {symbols.map((symbol) => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                style={{
                  padding: '10px 20px',
                  background: selectedSymbol === symbol ? '#0078D4' : 'rgba(255, 255, 255, 0.05)',
                  border: `2px solid ${selectedSymbol === symbol ? '#0078D4' : 'rgba(255, 255, 255, 0.1)'}`,
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                {symbol}
              </button>
            ))}
            <button
              onClick={fetchAzureAnalysis}
              disabled={loading}
              style={{
                padding: '10px 24px',
                background: 'linear-gradient(135deg, #0078D4, #005A9E)',
                border: '2px solid #0078D4',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '14px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer',
                opacity: loading ? 0.6 : 1,
                transition: 'all 0.3s ease',
                marginLeft: 'auto'
              }}
            >
              {loading ? 'Analiz Ediliyor...' : '🤖 AI Analiz Başlat'}
            </button>
          </div>
        )}

        {/* Content */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          border: '2px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '16px',
          padding: '32px',
          minHeight: '600px'
        }}>
          {selectedTab === 'analysis' && (
            <div>
              <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', marginBottom: '24px' }}>
                Azure OpenAI Market Analysis
              </h2>

              {error ? (
                <div style={{
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '2px solid rgba(239, 68, 68, 0.3)',
                  borderRadius: '12px',
                  padding: '24px',
                  textAlign: 'center',
                  color: '#EF4444'
                }}>
                  {error}
                </div>
              ) : analysis ? (
                <div>
                  {/* Sentiment Badge */}
                  <div style={{
                    background: `${getSentimentColor(analysis.sentiment)}20`,
                    border: `2px solid ${getSentimentColor(analysis.sentiment)}`,
                    borderRadius: '12px',
                    padding: '20px',
                    marginBottom: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '16px'
                  }}>
                    <div style={{ fontSize: '48px' }}>
                      {getSentimentEmoji(analysis.sentiment)}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                        Sentiment: {analysis.sentiment}
                      </div>
                      <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', marginTop: '4px' }}>
                        Confidence: {analysis.confidence}%
                      </div>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  <div style={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '2px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    padding: '24px'
                  }}>
                    <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                      AI Analizi
                    </h3>
                    <div style={{
                      fontSize: '15px',
                      lineHeight: '1.8',
                      color: 'rgba(255,255,255,0.9)',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {analysis.analysis}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '60px', color: 'rgba(255,255,255,0.6)' }}>
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>🤖</div>
                  <div>Bir symbol seçip "AI Analiz Başlat" butonuna tıklayın</div>
                </div>
              )}
            </div>
          )}

          {selectedTab === 'sentiment' && (
            <SentimentTab />
          )}

          {selectedTab === 'realtime' && (
            <RealtimeTab />
          )}
        </div>

        {/* MANTIK Modal */}
        {showLogicModal && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(10px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000
          }}>
            <div style={{
              background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
              border: '2px solid rgba(139, 92, 246, 0.4)',
              borderRadius: '20px',
              padding: '40px',
              maxWidth: '600px',
              width: '90%',
              maxHeight: '80vh',
              overflowY: 'auto',
              boxShadow: '0 20px 60px rgba(139, 92, 246, 0.3)'
            }}>
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <span style={{ fontSize: '28px' }}>🧠</span>
                  <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>MANTIK</h2>
                </div>
                <button
                  onClick={() => setShowLogicModal(false)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: '#FFFFFF',
                    fontSize: '28px',
                    cursor: 'pointer',
                    padding: 0
                  }}
                >
                  ×
                </button>
              </div>

              {/* Section 1: Sayfa Amacı */}
              <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '2px solid rgba(139, 92, 246, 0.2)' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>🎯</span> Sayfa Amacı
                </h3>
                <p style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', lineHeight: '1.6', margin: 0 }}>
                  Azure OpenAI sentiment analizi ve gerçek zamanlı piyasa korelasyon verisi
                </p>
              </div>

              {/* Section 2: Nasıl Çalışır */}
              <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '2px solid rgba(139, 92, 246, 0.2)' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>⚙️</span> Nasıl Çalışır
                </h3>
                <ul style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', margin: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>3 tab (Genel Bakış, Sentiment, Realtime)</li>
                  <li>/api/crypto-news ve /api/market-correlation kullanımı</li>
                </ul>
              </div>

              {/* Section 3: Özellikler */}
              <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '2px solid rgba(139, 92, 246, 0.2)' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>✨</span> Özellikler
                </h3>
                <ul style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', margin: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>Sentiment tracking</li>
                  <li>Real-time correlation</li>
                  <li>LIVE indicator</li>
                  <li>5-30s refresh</li>
                </ul>
              </div>

              {/* Section 4: Veri Kaynakları */}
              <div style={{ marginBottom: '24px', paddingBottom: '24px', borderBottom: '2px solid rgba(139, 92, 246, 0.2)' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>📡</span> Veri Kaynakları
                </h3>
                <ul style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', margin: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>/api/crypto-news</li>
                  <li>/api/market-correlation</li>
                </ul>
              </div>

              {/* Section 5: İpuçları */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>💡</span> İpuçları
                </h3>
                <ul style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', margin: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
                  <li>Realtime tab için 5s yenileme</li>
                  <li>Sentiment renklerine dikkat</li>
                </ul>
              </div>

              {/* Close Button */}
              <button
                onClick={() => setShowLogicModal(false)}
                style={{
                  width: '100%',
                  padding: '12px 24px',
                  background: 'linear-gradient(135deg, #8B5CF6, #7C3AED)',
                  border: '2px solid rgba(139, 92, 246, 0.5)',
                  borderRadius: '12px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 24px rgba(139, 92, 246, 0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                Kapat
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Sentiment Analysis Tab Component
function SentimentTab() {
  const [sentimentData, setSentimentData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSentiment();
    const interval = setInterval(fetchSentiment, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchSentiment = async () => {
    try {
      const response = await fetch('/api/crypto-news');
      const result = await response.json();

      if (result.success && result.data) {
        setSentimentData(result.data.slice(0, 10));
      }
    } catch (err) {
      console.error('Sentiment fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    if (sentiment === 'positive') return '#10B981';
    if (sentiment === 'negative') return '#EF4444';
    return '#6B7280';
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px', color: 'rgba(255,255,255,0.6)' }}>
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>💭</div>
        <div>Sentiment analizi yükleniyor...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{
        background: 'rgba(255, 255, 255, 0.03)',
        border: '2px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <h3 style={{ fontSize: '20px', color: '#FFFFFF', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          💭 Piyasa Sentiment Analizi
        </h3>
        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px', marginBottom: '24px' }}>
          Kripto haberlerinden çıkarılan gerçek zamanlı sentiment skorları
        </p>

        <div style={{ display: 'grid', gap: '16px' }}>
          {sentimentData.map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                padding: '16px',
                transition: 'all 0.3s'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF', marginBottom: '4px' }}>
                    {item.titleTR}
                  </div>
                  <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>
                    {item.source.name} • {new Date(item.publishedAt).toLocaleString('tr-TR')}
                  </div>
                </div>
                <div style={{
                  padding: '4px 12px',
                  background: getSentimentColor(item.sentiment) + '20',
                  border: `2px solid ${getSentimentColor(item.sentiment)}`,
                  borderRadius: '6px',
                  color: getSentimentColor(item.sentiment),
                  fontSize: '11px',
                  fontWeight: '700',
                  textTransform: 'uppercase'
                }}>
                  {item.sentiment}
                </div>
              </div>
              <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '8px' }}>
                {item.descriptionTR}
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {item.tags.slice(0, 3).map((tag: string, i: number) => (
                  <span
                    key={i}
                    style={{
                      background: 'rgba(59, 130, 246, 0.1)',
                      border: '1px solid rgba(59, 130, 246, 0.3)',
                      color: '#3B82F6',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '10px',
                      fontWeight: '600'
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Real-time SignalR Tab Component
function RealtimeTab() {
  const [realtimeData, setRealtimeData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    fetchRealtime();
    const interval = setInterval(() => {
      fetchRealtime();
      setLastUpdate(new Date());
    }, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchRealtime = async () => {
    try {
      const response = await fetch('/api/market-correlation?limit=10');
      const result = await response.json();

      if (result.success && result.data.correlations) {
        setRealtimeData(result.data.correlations);
      }
    } catch (err) {
      console.error('Realtime fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '60px', color: 'rgba(255,255,255,0.6)' }}>
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚡</div>
        <div>Real-time veriler yükleniyor...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{
        background: 'rgba(255, 255, 255, 0.03)',
        border: '2px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '16px',
        padding: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '20px', color: '#FFFFFF', margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
            ⚡ Real-time Market Stream
          </h3>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            background: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: '6px',
            fontSize: '11px',
            color: '#10B981'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: '#10B981',
              animation: 'pulse 2s infinite'
            }} />
            LIVE • {lastUpdate.toLocaleTimeString('tr-TR')}
          </div>
        </div>
        <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '14px', marginBottom: '24px' }}>
          Binance Futures gerçek zamanlı market korelasyon verileri
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
          {realtimeData.map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.05))',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: '12px',
                padding: '16px',
                transition: 'all 0.3s'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                  {item.symbol}
                </div>
                <div style={{
                  fontSize: '14px',
                  fontWeight: '700',
                  color: item.changePercent24h >= 0 ? '#10B981' : '#EF4444'
                }}>
                  {item.changePercent24h >= 0 ? '+' : ''}{item.changePercent24h.toFixed(2)}%
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '12px' }}>
                <div>
                  <div style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}>Price</div>
                  <div style={{ color: '#FFFFFF', fontWeight: '600' }}>${item.price.toFixed(4)}</div>
                </div>
                <div>
                  <div style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}>Volume 24h</div>
                  <div style={{ color: '#FFFFFF', fontWeight: '600' }}>
                    ${(item.volume24h / 1000000).toFixed(2)}M
                  </div>
                </div>
                <div>
                  <div style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}>Correlation</div>
                  <div style={{ color: '#3B82F6', fontWeight: '700' }}>
                    {item.correlation ? (item.correlation * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: 'rgba(255,255,255,0.5)', marginBottom: '4px' }}>Status</div>
                  <div style={{
                    color: item.changePercent24h >= 5 ? '#10B981' : item.changePercent24h <= -5 ? '#EF4444' : '#6B7280',
                    fontWeight: '700',
                    fontSize: '10px'
                  }}>
                    {item.changePercent24h >= 5 ? '🔥 HOT' : item.changePercent24h <= -5 ? '❄️ COLD' : '➖ STABLE'}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
