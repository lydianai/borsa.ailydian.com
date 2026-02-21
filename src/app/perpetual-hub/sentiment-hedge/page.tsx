'use client';

/**
 * 📊 FUTURES SENTIMENT HEDGER
 *
 * Whale vs Retail positioning analysis with contrarian signals
 *
 * Features:
 * - Long/Short ratio tracking
 * - Whale wallet activity
 * - Retail vs Pro trader positioning
 * - AI contrarian signal generation
 * - Funding rate sentiment
 * - Open interest trends
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface SentimentData {
  symbol: string;
  longShortRatio: number;
  whaleLongPercent: number;
  retailLongPercent: number;
  fundingRate: number;
  openInterest: number;
  oiChange24h: number;
  signal: 'LONG' | 'SHORT' | 'NEUTRAL';
  confidence: number;
  contrarian: boolean;
}

export default function SentimentHedger() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<string>('BTCUSDT');

  useEffect(() => {
    setMounted(true);
  }, []);

  const sentiments: SentimentData[] = [
    {
      symbol: 'BTCUSDT',
      longShortRatio: 2.34,
      whaleLongPercent: 67,
      retailLongPercent: 78,
      fundingRate: 0.0125,
      openInterest: 42800000000,
      oiChange24h: 8.2,
      signal: 'LONG',
      confidence: 82,
      contrarian: false,
    },
    {
      symbol: 'ETHUSDT',
      longShortRatio: 0.68,
      whaleLongPercent: 34,
      retailLongPercent: 82,
      fundingRate: -0.0032,
      openInterest: 12400000000,
      oiChange24h: -4.5,
      signal: 'SHORT',
      confidence: 76,
      contrarian: true,
    },
    {
      symbol: 'SOLUSDT',
      longShortRatio: 3.12,
      whaleLongPercent: 89,
      retailLongPercent: 91,
      fundingRate: 0.0285,
      openInterest: 890000000,
      oiChange24h: 15.7,
      signal: 'LONG',
      confidence: 91,
      contrarian: false,
    },
  ];

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'LONG': return '#10B981';
      case 'SHORT': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const selected = sentiments.find(s => s.symbol === selectedAsset) || sentiments[0];

  if (!mounted) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.6)' }}>Yükleniyor...</div>
      </div>
    );
  }

  return (
    <PWAProvider>
      <div
        suppressHydrationWarning
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
          paddingTop: '60px',
        }}
      >
        <SharedSidebar
          currentPage="perpetual-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1800px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
          <div style={{ marginBottom: '32px' }}>
            <Link
              href="/perpetual-hub"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '14px',
                marginBottom: '12px',
              }}
            >
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              Perpetual Hub'a Dön
            </Link>

            <h1
              style={{
                fontSize: '40px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Futures Sentiment Hedger
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Whale vs Retail pozisyon analizi ve AI kontrarian sinyal üretimi
            </p>
          </div>

          {/* Asset Selector */}
          <div style={{ display: 'flex', gap: '12px', marginBottom: '32px' }}>
            {sentiments.map((s) => (
              <button
                key={s.symbol}
                onClick={() => setSelectedAsset(s.symbol)}
                style={{
                  padding: '12px 24px',
                  background: selectedAsset === s.symbol ? getSignalColor(s.signal) : 'rgba(255, 255, 255, 0.05)',
                  color: '#FFFFFF',
                  border: selectedAsset === s.symbol ? 'none' : '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
              >
                {s.symbol.replace('USDT', '')}
              </button>
            ))}
          </div>

          {/* Main Signal Card */}
          <div
            style={{
              background: `${getSignalColor(selected.signal)}10`,
              border: `2px solid ${getSignalColor(selected.signal)}`,
              borderRadius: '20px',
              padding: '32px',
              marginBottom: '32px',
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '32px', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  AI SINYAL {selected.contrarian && '(CONTRARIAN)'}
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '16px' }}>
                  <h2 style={{ fontSize: '56px', fontWeight: '900', color: getSignalColor(selected.signal), margin: 0 }}>
                    {selected.signal}
                  </h2>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: 'rgba(255, 255, 255, 0.6)' }}>
                    {selected.confidence}% güven
                  </div>
                </div>
                {selected.contrarian && (
                  <div
                    style={{
                      marginTop: '12px',
                      padding: '8px 16px',
                      background: 'rgba(245, 158, 11, 0.2)',
                      border: '1px solid rgba(245, 158, 11, 0.4)',
                      borderRadius: '8px',
                      display: 'inline-block',
                    }}
                  >
                    <span style={{ fontSize: '13px', fontWeight: '600', color: '#F59E0B' }}>
                      ⚠️ Kontrarian pozisyon - Çoğunluğa karşı
                    </span>
                  </div>
                )}
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  LONG/SHORT RATIO
                </div>
                <div style={{ fontSize: '48px', fontWeight: '900', color: '#FFFFFF' }}>
                  {selected.longShortRatio.toFixed(2)}
                </div>
                <div style={{ fontSize: '14px', color: selected.longShortRatio > 1 ? '#10B981' : '#EF4444' }}>
                  {selected.longShortRatio > 1 ? 'Long baskın' : 'Short baskın'}
                </div>
              </div>
            </div>
          </div>

          {/* Whale vs Retail */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
            {/* Whale Positioning */}
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icons.TrendingUp style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                    Whale Pozisyonları
                  </h3>
                  <p style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', margin: 0 }}>
                    Büyük cüzdanlar (&gt;$1M)
                  </p>
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>Long</span>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#10B981' }}>
                    {selected.whaleLongPercent}%
                  </span>
                </div>
                <div
                  style={{
                    height: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${selected.whaleLongPercent}%`,
                      height: '100%',
                      background: '#10B981',
                      transition: 'width 0.5s',
                    }}
                  />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>Short</span>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#EF4444' }}>
                    {100 - selected.whaleLongPercent}%
                  </span>
                </div>
                <div
                  style={{
                    height: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${100 - selected.whaleLongPercent}%`,
                      height: '100%',
                      background: '#EF4444',
                      transition: 'width 0.5s',
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Retail Positioning */}
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icons.Users style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                    Retail Pozisyonları
                  </h3>
                  <p style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', margin: 0 }}>
                    Küçük yatırımcılar (&lt;$10K)
                  </p>
                </div>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>Long</span>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#10B981' }}>
                    {selected.retailLongPercent}%
                  </span>
                </div>
                <div
                  style={{
                    height: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${selected.retailLongPercent}%`,
                      height: '100%',
                      background: '#10B981',
                      transition: 'width 0.5s',
                    }}
                  />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>Short</span>
                  <span style={{ fontSize: '16px', fontWeight: '700', color: '#EF4444' }}>
                    {100 - selected.retailLongPercent}%
                  </span>
                </div>
                <div
                  style={{
                    height: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '6px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${100 - selected.retailLongPercent}%`,
                      height: '100%',
                      background: '#EF4444',
                      transition: 'width 0.5s',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Additional Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '20px',
              }}
            >
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                FUNDING RATE
              </div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: selected.fundingRate > 0 ? '#10B981' : '#EF4444' }}>
                {selected.fundingRate > 0 ? '+' : ''}{(selected.fundingRate * 100).toFixed(3)}%
              </div>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                {selected.fundingRate > 0 ? 'Long ödüyor' : 'Short ödüyor'}
              </div>
            </div>

            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '20px',
              }}
            >
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                OPEN INTEREST
              </div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: '#FFFFFF' }}>
                ${(selected.openInterest / 1000000000).toFixed(2)}B
              </div>
              <div style={{ fontSize: '14px', fontWeight: '600', color: selected.oiChange24h > 0 ? '#10B981' : '#EF4444' }}>
                {selected.oiChange24h > 0 ? '+' : ''}{selected.oiChange24h.toFixed(2)}% (24h)
              </div>
            </div>

            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '20px',
              }}
            >
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                WHALE/RETAIL DIVERGENCE
              </div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: '#F59E0B' }}>
                {Math.abs(selected.whaleLongPercent - selected.retailLongPercent)}%
              </div>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                {Math.abs(selected.whaleLongPercent - selected.retailLongPercent) > 15 ? '⚠️ Yüksek ayrışma' : '✅ Uyumlu'}
              </div>
            </div>
          </div>
        </main>

        {aiAssistantOpen && (
          <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
        )}
      </div>
    </PWAProvider>
  );
}
