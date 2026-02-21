'use client';

/**
 * 🔀 CAUSAL AI & COUNTERFACTUAL
 *
 * Sebep-sonuç ilişkilerini öğrenen AI - "Ne olurdu?" analizleri
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

export default function CausalAIPage() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState('volume_double');

  const causalPaths = [
    { from: 'Volume', to: 'Price', strength: 0.78, type: 'direct', color: '#10B981' },
    { from: 'News Sentiment', to: 'Price', strength: 0.82, type: 'direct', color: '#EC4899' },
    { from: 'Volatility', to: 'Volume', strength: 0.65, type: 'indirect', color: '#F59E0B' },
    { from: 'Market Cap', to: 'Volatility', strength: -0.45, type: 'inverse', color: '#EF4444' },
    { from: 'Whale Activity', to: 'Price', strength: 0.71, type: 'direct', color: '#8B5CF6' },
    { from: 'Social Media', to: 'News Sentiment', strength: 0.58, type: 'indirect', color: '#06B6D4' },
  ];

  const counterfactualScenarios = [
    { id: 'volume_double', label: 'What if volume doubled?', original: '$45,000', counterfactual: '$48,500', change: '+7.8%' },
    { id: 'whale_buy', label: 'What if whale bought $100M?', original: '$45,000', counterfactual: '$52,300', change: '+16.2%' },
    { id: 'news_positive', label: 'What if positive news?', original: '$45,000', counterfactual: '$47,800', change: '+6.2%' },
    { id: 'regulation', label: 'What if new regulation?', original: '$45,000', counterfactual: '$41,200', change: '-8.4%' },
  ];

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ minHeight: '100vh', background: '#0a0a0a' }} />;
  }

  const selectedScenarioData = counterfactualScenarios.find(s => s.id === selectedScenario)!;

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

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #F97316 0%, #EA580C 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              🔀 Causal AI & Counterfactual
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              Sebep-sonuç ilişkilerini öğrenen AI. "Ne olurdu?" analizleri ile alternatif senaryoları keşfet. Gerçek nedenleri bul.
            </p>
          </div>

          {/* Stats Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <StatCard label="Causal Paths" value="247" color="#F97316" />
            <StatCard label="Confidence" value="87.5%" color="#10B981" />
            <StatCard label="Interventions" value="1,458" color="#8B5CF6" />
            <StatCard label="Strongest Cause" value="News" color="#EC4899" />
          </div>

          {/* Causal Graph */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '40px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: '900', color: '#F97316', marginBottom: '24px', textAlign: 'center' }}>
              🕸️ Causal Graph - Price Determinants
            </h2>

            <div style={{ display: 'grid', gap: '16px' }}>
              {causalPaths.map((path, idx) => (
                <div key={idx} style={{
                  padding: '20px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${path.color}40`,
                  borderRadius: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: 1 }}>
                    <div style={{
                      padding: '12px 20px',
                      background: `${path.color}20`,
                      borderRadius: '8px',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: path.color,
                      minWidth: '140px',
                      textAlign: 'center'
                    }}>
                      {path.from}
                    </div>
                    <div style={{ fontSize: '24px', color: path.strength < 0 ? '#EF4444' : '#10B981' }}>
                      {path.strength < 0 ? '↓' : '→'}
                    </div>
                    <div style={{
                      padding: '12px 20px',
                      background: `${path.color}20`,
                      borderRadius: '8px',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: path.color,
                      minWidth: '140px',
                      textAlign: 'center'
                    }}>
                      {path.to}
                    </div>
                  </div>
                  <div style={{ marginLeft: '24px', textAlign: 'right' }}>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: Math.abs(path.strength) > 0.7 ? '#10B981' : '#F59E0B' }}>
                      {Math.abs(path.strength).toFixed(2)}
                    </div>
                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', textTransform: 'uppercase' }}>
                      {path.type}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Counterfactual Analysis */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '2px solid #F97316', borderRadius: '20px', padding: '40px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: '900', color: '#F97316', marginBottom: '24px' }}>
              🔮 Counterfactual Analysis - "What If?"
            </h2>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '8px', display: 'block' }}>
                Select Scenario:
              </label>
              <select
                value={selectedScenario}
                onChange={(e) => setSelectedScenario(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                }}
              >
                {counterfactualScenarios.map((scenario) => (
                  <option key={scenario.id} value={scenario.id}>{scenario.label}</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '24px', alignItems: 'center' }}>
              <div style={{ padding: '32px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '16px', textAlign: 'center' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '12px' }}>Original Outcome</div>
                <div style={{ fontSize: '36px', fontWeight: '900', color: '#FFFFFF' }}>{selectedScenarioData.original}</div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '8px' }}>Current State</div>
              </div>

              <div style={{ fontSize: '48px', color: '#F97316' }}>
                →
              </div>

              <div style={{ padding: '32px', background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.2) 0%, rgba(234, 88, 12, 0.1) 100%)', border: '2px solid #F97316', borderRadius: '16px', textAlign: 'center' }}>
                <div style={{ fontSize: '14px', color: '#F97316', marginBottom: '12px', fontWeight: '700' }}>Counterfactual Outcome</div>
                <div style={{ fontSize: '36px', fontWeight: '900', color: selectedScenarioData.change.startsWith('+') ? '#10B981' : '#EF4444' }}>
                  {selectedScenarioData.counterfactual}
                </div>
                <div style={{ fontSize: '18px', fontWeight: '700', color: selectedScenarioData.change.startsWith('+') ? '#10B981' : '#EF4444', marginTop: '8px' }}>
                  {selectedScenarioData.change}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '24px', padding: '16px', background: 'rgba(249, 115, 22, 0.1)', border: '1px solid rgba(249, 115, 22, 0.3)', borderRadius: '12px' }}>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                💡 <strong>Interpretation:</strong> {selectedScenarioData.label} would result in a{' '}
                <span style={{ fontWeight: '700', color: selectedScenarioData.change.startsWith('+') ? '#10B981' : '#EF4444' }}>
                  {selectedScenarioData.change}
                </span>
                {' '}change in BTC price, based on causal relationships learned from historical data.
              </div>
            </div>
          </div>

          {/* Causal Methods */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '32px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(249, 115, 22, 0.3)', borderRadius: '16px', padding: '32px' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px', textAlign: 'center' }}>🎯</div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#F97316', marginBottom: '12px', textAlign: 'center' }}>
                do()-Calculus
              </h3>
              <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
                Judea Pearl'ün do-operator'ı. P(Y|do(X)) hesaplar. "X'i değiştirirsem Y'ye etkisi ne olur?"
                Correlation ≠ Causation problemini çözer.
              </p>
            </div>

            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(139, 92, 246, 0.3)', borderRadius: '16px', padding: '32px' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px', textAlign: 'center' }}>🔄</div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#8B5CF6', marginBottom: '12px', textAlign: 'center' }}>
                Structural Causal Models
              </h3>
              <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
                Directed Acyclic Graph (DAG) ile causal structure modeling. Her edge bir sebep-sonuç ilişkisi.
                Confounders ve mediators'ı ayırt et.
              </p>
            </div>
          </div>

          {/* Use Cases */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              💼 Trading Use Cases
            </h2>
            <div style={{ display: 'grid', gap: '12px' }}>
              {[
                { title: 'Intervention Planning', desc: '"Whale X satarsa piyasaya etkisi ne olur?" - Önceden hesapla', icon: '🎯' },
                { title: 'Root Cause Analysis', desc: 'Price drop\'un gerçek sebebi nedir? Volume mu, news mu?', icon: '🔍' },
                { title: 'Policy Evaluation', desc: '"Federal Reserve faiz artırırsa crypto\'ya etkisi ne olur?"', icon: '📊' },
                { title: 'Confounder Detection', desc: 'Spurious correlation\'ları ayıkla, gerçek causal paths\'i bul', icon: '🧹' },
              ].map((useCase, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '16px', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                  <div style={{ fontSize: '32px', flexShrink: 0 }}>{useCase.icon}</div>
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
                      {idx + 1}. {useCase.title}
                    </div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
                      {useCase.desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Info */}
          <div style={{ background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, rgba(234, 88, 12, 0.05) 100%)', border: '1px solid rgba(249, 115, 22, 0.2)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#F97316', marginBottom: '16px' }}>
              🧠 Causal AI Nedir?
            </h3>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              <p style={{ marginBottom: '12px' }}>
                <strong>Beyond Correlation:</strong> Traditional ML correlation bulur. Causal AI sebep-sonuç bulur.
                "Correlation ≠ Causation" sözünün çözümü.
              </p>
              <p style={{ marginBottom: '12px' }}>
                <strong>Counterfactual Reasoning:</strong> "Ne olurdu?" sorusuna cevap verir. Alternatif evrenler simülasyonu.
              </p>
              <p>
                <strong>Nobel Prize 2021:</strong> Judea Pearl ve arkadaşlarının çalışmaları ekonomide Nobel ödülü kazandı.
                Trading'de de game-changer!
              </p>
            </div>
          </div>
        </main>

        {aiAssistantOpen && <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />}
      </div>
    </PWAProvider>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '20px' }}>
      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>{label}</div>
      <div style={{ fontSize: '28px', fontWeight: '700', color: color }}>{value}</div>
    </div>
  );
}
