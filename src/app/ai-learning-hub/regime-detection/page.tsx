'use client';

/**
 * 📈 ADAPTIVE REGIME DETECTION
 *
 * Market rejimlerini otomatik tespit eder - Bull, Bear, Range, Volatile
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

export default function RegimeDetectionPage() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Mock regime data
  const currentRegime = {
    regime: 'Bull',
    confidence: 92.3,
    duration: '14 days',
    next_transition_probability: 0.23,
    recommended_strategy: 'Momentum Trading',
    probabilities: {
      'Bull': 0.65,
      'Bear': 0.10,
      'Range': 0.15,
      'Volatile': 0.10,
    },
  };

  const regimes = [
    {
      name: 'Bull',
      icon: '📈',
      color: '#10B981',
      description: 'Yükselen piyasa - Sürekli yeni zirveler',
      strategy: 'Momentum Trading, Trend Following',
      indicators: 'RSI > 50, MACD pozitif, Price > MA',
    },
    {
      name: 'Bear',
      icon: '📉',
      color: '#EF4444',
      description: 'Düşen piyasa - Sürekli yeni dipler',
      strategy: 'Short Selling, Mean Reversion',
      indicators: 'RSI < 50, MACD negatif, Price < MA',
    },
    {
      name: 'Range',
      icon: '↔️',
      color: '#F59E0B',
      description: 'Yatay piyasa - Belirli aralıkta hareket',
      strategy: 'Range Trading, Support/Resistance',
      indicators: 'ADX < 25, Bollinger Bands dar',
    },
    {
      name: 'Volatile',
      icon: '⚡',
      color: '#8B5CF6',
      description: 'Yüksek volatilite - Hızlı ve keskin hareketler',
      strategy: 'Scalping, Breakout Trading',
      indicators: 'ATR yüksek, Bollinger Bands geniş',
    },
  ];

  const history = [
    { date: '2025-01-15', regime: 'Bull', duration: '14 days', performance: '+12.5%' },
    { date: '2025-01-01', regime: 'Range', duration: '7 days', performance: '+2.3%' },
    { date: '2024-12-25', regime: 'Bear', duration: '5 days', performance: '-8.2%' },
    { date: '2024-12-20', regime: 'Bull', duration: '21 days', performance: '+18.7%' },
  ];

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ minHeight: '100vh', background: '#0a0a0a' }} />;
  }

  return (
    <PWAProvider>
      <div suppressHydrationWarning style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)', paddingTop: '80px' }}>
        <SharedSidebar currentPage="ai-learning-hub" onAiAssistantOpen={() => setAiAssistantOpen(true)} />

        <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '40px 24px' }}>
          <div style={{ marginBottom: '32px' }}>
            <Link href="/ai-learning-hub" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', color: 'rgba(255, 255, 255, 0.6)', textDecoration: 'none', fontSize: '14px', marginBottom: '16px' }}>
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              AI/ML Learning Hub
            </Link>

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              📈 Adaptive Regime Detection
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              Market rejimlerini otomatik tespit eder. Bull, Bear, Range, Volatile - her birine farklı strateji uygular.
            </p>
          </div>

          {/* Current Regime */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '2px solid #10B981', borderRadius: '20px', padding: '40px', marginBottom: '32px' }}>
            <div style={{ textAlign: 'center', marginBottom: '32px' }}>
              <div style={{ fontSize: '64px', marginBottom: '16px' }}>📈</div>
              <h2 style={{ fontSize: '48px', fontWeight: '900', color: '#10B981', marginBottom: '8px' }}>
                {currentRegime.regime} Market
              </h2>
              <div style={{ fontSize: '18px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Confidence: {currentRegime.confidence}% | Duration: {currentRegime.duration}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginBottom: '24px' }}>
              {Object.entries(currentRegime.probabilities).map(([regime, prob]) => (
                <div key={regime} style={{ textAlign: 'center', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                  <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>{regime}</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>{(prob * 100).toFixed(0)}%</div>
                  <div style={{ width: '100%', height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '2px', marginTop: '8px', overflow: 'hidden' }}>
                    <div style={{ width: `${prob * 100}%`, height: '100%', background: '#10B981' }} />
                  </div>
                </div>
              ))}
            </div>

            <div style={{ padding: '20px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '12px' }}>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '8px' }}>
                🎯 Recommended Strategy
              </div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#10B981' }}>
                {currentRegime.recommended_strategy}
              </div>
            </div>
          </div>

          {/* All Regimes */}
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              🎭 Market Regimes
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
              {regimes.map((regime) => (
                <div key={regime.name} style={{ background: 'rgba(255, 255, 255, 0.03)', border: `1px solid ${regime.name === currentRegime.regime ? regime.color : 'rgba(255, 255, 255, 0.1)'}`, borderRadius: '16px', padding: '24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                    <div style={{ fontSize: '32px' }}>{regime.icon}</div>
                    <div>
                      <div style={{ fontSize: '20px', fontWeight: '700', color: regime.color }}>{regime.name}</div>
                      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>{regime.description}</div>
                    </div>
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '8px' }}>
                    <strong>Strategy:</strong> {regime.strategy}
                  </div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                    <strong>Indicators:</strong> {regime.indicators}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* History */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              📊 Regime History
            </h2>
            <div style={{ display: 'grid', gap: '12px' }}>
              {history.map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', width: '100px' }}>{item.date}</div>
                    <div style={{ padding: '6px 16px', background: item.regime === 'Bull' ? 'rgba(16, 185, 129, 0.2)' : item.regime === 'Bear' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)', borderRadius: '8px', fontSize: '14px', fontWeight: '700', color: item.regime === 'Bull' ? '#10B981' : item.regime === 'Bear' ? '#EF4444' : '#F59E0B' }}>
                      {item.regime}
                    </div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>{item.duration}</div>
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: item.performance.startsWith('+') ? '#10B981' : '#EF4444' }}>
                    {item.performance}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Info */}
          <div style={{ marginTop: '24px', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444', marginBottom: '12px' }}>
              🧠 Regime Detection Nasıl Çalışır?
            </h3>
            <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              Hidden Markov Models (HMM) ve Gaussian Mixture Models (GMM) kullanarak market rejimlerini tespit eder.
              Volatilite, trend strength, trading volume gibi çok sayıda indikatör analiz edilir.
              Her rejime özel en uygun trading stratejisi otomatik olarak seçilir.
            </p>
          </div>
        </main>

        {aiAssistantOpen && <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />}
      </div>
    </PWAProvider>
  );
}
