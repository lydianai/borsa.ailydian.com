'use client';

/**
 * ⚙️ LEVERAGE EFFICIENCY OPTIMIZER
 *
 * Risk-adjusted return calculator and optimal leverage recommendation
 *
 * Features:
 * - Optimal leverage calculator
 * - Expected return simulation
 * - Funding cost analysis
 * - Kelly Criterion implementation
 * - Risk/Reward visualization
 * - Position size optimizer
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

export default function LeverageOptimizer() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Calculator inputs
  const [capital, setCapital] = useState('10000');
  const [winRate, setWinRate] = useState('60');
  const [avgWin, setAvgWin] = useState('5');
  const [avgLoss, setAvgLoss] = useState('3');
  const [fundingRate, setFundingRate] = useState('0.01');

  useEffect(() => {
    setMounted(true);
  }, []);

  // Kelly Criterion calculation
  const calculateKelly = () => {
    const w = parseFloat(winRate) / 100;
    const r = parseFloat(avgWin) / parseFloat(avgLoss);
    return ((w * r) - (1 - w)) / r;
  };

  // Optimal leverage
  const calculateOptimalLeverage = () => {
    const kelly = calculateKelly();
    const maxLeverage = 125;
    const optimal = Math.min(Math.max(kelly * 20, 1), maxLeverage);
    return optimal;
  };

  // Expected return
  const calculateExpectedReturn = (leverage: number) => {
    const w = parseFloat(winRate) / 100;
    const avgWinPct = parseFloat(avgWin) / 100;
    const avgLossPct = parseFloat(avgLoss) / 100;
    const funding = parseFloat(fundingRate) / 100;

    const expectedReturn = (w * avgWinPct * leverage) - ((1 - w) * avgLossPct * leverage) - (funding * leverage * 0.333); // 0.333 for 8h funding
    return expectedReturn * 100;
  };

  // Risk scenarios
  const scenarios = [
    { leverage: 5, label: 'Muhafazakâr' },
    { leverage: 10, label: 'Dengeli' },
    { leverage: calculateOptimalLeverage(), label: 'Optimal (Kelly)' },
    { leverage: 25, label: 'Agresif' },
    { leverage: 50, label: 'Aşırı Risk' },
  ];

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

        <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
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
                background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Leverage Efficiency Optimizer
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Risk ayarlı optimal kaldıraç hesaplama ve beklenen getiri simülasyonu
            </p>
          </div>

          {/* Input Parameters */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Parametreler
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              <div>
                <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>
                  Başlangıç Sermayesi (USDT)
                </label>
                <input
                  type="number"
                  value={capital}
                  onChange={(e) => setCapital(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>
                  Win Rate (%)
                </label>
                <input
                  type="number"
                  value={winRate}
                  onChange={(e) => setWinRate(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>
                  Ortalama Kazanç (%)
                </label>
                <input
                  type="number"
                  value={avgWin}
                  onChange={(e) => setAvgWin(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>
                  Ortalama Kayıp (%)
                </label>
                <input
                  type="number"
                  value={avgLoss}
                  onChange={(e) => setAvgLoss(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                  }}
                />
              </div>

              <div>
                <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>
                  Funding Rate (%)
                </label>
                <input
                  type="number"
                  step="0.001"
                  value={fundingRate}
                  onChange={(e) => setFundingRate(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    color: '#FFFFFF',
                    fontSize: '14px',
                  }}
                />
              </div>
            </div>
          </div>

          {/* Optimal Leverage Result */}
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%)',
              border: '2px solid #F59E0B',
              borderRadius: '20px',
              padding: '32px',
              marginBottom: '32px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px' }}>
              OPTIMAL KALDIRAÇ (KELLY CRITERION)
            </div>
            <div style={{ fontSize: '72px', fontWeight: '900', color: '#F59E0B', marginBottom: '8px' }}>
              {calculateOptimalLeverage().toFixed(1)}x
            </div>
            <div style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)' }}>
              Beklenen günlük getiri: <span style={{ fontWeight: '700', color: '#10B981' }}>
                +{calculateExpectedReturn(calculateOptimalLeverage()).toFixed(2)}%
              </span>
            </div>
          </div>

          {/* Leverage Scenarios */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Kaldıraç Senaryoları
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {scenarios.map((scenario, index) => {
                const expectedReturn = calculateExpectedReturn(scenario.leverage);
                const isOptimal = scenario.label.includes('Optimal');

                return (
                  <div
                    key={index}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '150px 100px 1fr 150px 150px',
                      alignItems: 'center',
                      gap: '16px',
                      padding: '20px',
                      background: isOptimal ? 'rgba(245, 158, 11, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                      border: isOptimal ? '1px solid #F59E0B' : '1px solid rgba(255, 255, 255, 0.05)',
                      borderRadius: '12px',
                    }}
                  >
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                        {scenario.label}
                      </div>
                      {isOptimal && (
                        <div style={{ fontSize: '11px', color: '#F59E0B', marginTop: '4px' }}>
                          ✨ Önerilen
                        </div>
                      )}
                    </div>

                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>
                      {scenario.leverage.toFixed(1)}x
                    </div>

                    <div
                      style={{
                        height: '32px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '16px',
                        overflow: 'hidden',
                        position: 'relative',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.min((scenario.leverage / 50) * 100, 100)}%`,
                          background: scenario.leverage <= 10 ? '#10B981' : scenario.leverage <= 25 ? '#F59E0B' : '#EF4444',
                          transition: 'width 0.5s',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'flex-end',
                          paddingRight: '12px',
                        }}
                      >
                        <span style={{ fontSize: '12px', fontWeight: '600', color: '#FFFFFF' }}>
                          {((scenario.leverage / 50) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <div
                        style={{
                          fontSize: '20px',
                          fontWeight: '700',
                          color: expectedReturn > 0 ? '#10B981' : '#EF4444',
                        }}
                      >
                        {expectedReturn > 0 ? '+' : ''}{expectedReturn.toFixed(2)}%
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        Günlük beklenti
                      </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: 'rgba(255, 255, 255, 0.7)' }}>
                        ${(parseFloat(capital) * (expectedReturn / 100) * 30).toFixed(2)}
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        Aylık tahmini
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Warning */}
          <div
            style={{
              marginTop: '32px',
              padding: '20px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '12px',
              display: 'flex',
              gap: '12px',
            }}
          >
            <Icons.AlertTriangle style={{ width: '24px', height: '24px', color: '#EF4444', flexShrink: 0 }} />
            <div>
              <h4 style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444', marginBottom: '8px' }}>
                Önemli Uyarı
              </h4>
              <p style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.5' }}>
                Kelly Criterion hesaplaması, geçmiş performansa ve olasılık teorisine dayalıdır. Gerçek piyasa koşulları değişkendir.
                Yüksek kaldıraç kullanımı hızlı kayıplara yol açabilir. Risk yönetimi her zaman öncelik olmalıdır.
              </p>
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
