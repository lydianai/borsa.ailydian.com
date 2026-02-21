'use client';

/**
 * 🔍 EXPLAINABLE AI DASHBOARD
 *
 * AI neden bu kararı verdi? SHAP, attention, counterfactual açıklamalar
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';
import { useResponsive } from '@/hooks/useResponsive';

export default function ExplainableAIPage() {
  const { isMobile, isTablet } = useResponsive();
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [explanation, setExplanation] = useState<any>(null);

  useEffect(() => {
    setMounted(true);
    // Get a sample explanation
    fetchExplanation();
  }, []);

  const fetchExplanation = async () => {
    try {
      // Mock explanation for demo
      const _mockPrediction = {
        action: 'BUY',
        confidence: 85.5,
      };

      const mockExplanation = {
        success: true,
        prediction: 'BUY',
        confidence: 85.5,
        shap_values: {
          'Volume': 0.35,
          'RSI': 0.28,
          'MACD': 0.18,
          'BB_Width': 0.12,
          'News_Sentiment': 0.07,
        },
        top_features: [
          ['Volume', 0.35],
          ['RSI', 0.28],
          ['MACD', 0.18],
        ],
        attention_weights: {
          'timeframe_1h': 0.45,
          'timeframe_4h': 0.30,
          'timeframe_1d': 0.25,
        },
        explanation: 'Bu BUY sinyali ağırlıklı olarak Volume ve RSI verilerine dayanıyor.',
        explainability_score: 96.8,
      };

      setExplanation(mockExplanation);
    } catch (error) {
      console.error('Failed to fetch explanation:', error);
    }
  };

  if (!mounted) {
    return <div style={{ minHeight: '100vh', background: '#0a0a0a' }} />;
  }

  return (
    <PWAProvider>
      <div suppressHydrationWarning style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)', paddingTop: isMobile ? '70px' : '80px' }}>
        <SharedSidebar currentPage="ai-learning-hub" onAiAssistantOpen={() => setAiAssistantOpen(true)} />

        <main style={{ maxWidth: '1400px', margin: '0 auto', padding: isMobile ? '20px 16px' : '40px 24px' }}>
          <div style={{ marginBottom: '32px' }}>
            <Link href="/ai-learning-hub" style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', color: 'rgba(255, 255, 255, 0.6)', textDecoration: 'none', fontSize: '14px', marginBottom: '16px' }}>
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              AI/ML Learning Hub
            </Link>

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              🔍 Explainable AI Dashboard
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              AI neden bu kararı verdi? SHAP values, attention weights ve counterfactual açıklamalar ile tam şeffaflık.
            </p>
          </div>

          {explanation && (
            <>
              {/* Prediction Summary */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  🎯 AI Prediction
                </h2>
                <div style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '16px' }}>
                  <div style={{ padding: '16px 32px', background: explanation.prediction === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)', border: `2px solid ${explanation.prediction === 'BUY' ? '#10B981' : '#EF4444'}`, borderRadius: '12px' }}>
                    <div style={{ fontSize: '32px', fontWeight: '900', color: explanation.prediction === 'BUY' ? '#10B981' : '#EF4444' }}>
                      {explanation.prediction}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Confidence</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>{explanation.confidence}%</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Explainability Score</div>
                    <div style={{ fontSize: '24px', fontWeight: '700', color: '#3B82F6' }}>{explanation.explainability_score}%</div>
                  </div>
                </div>
                <div style={{ padding: '16px', background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '12px' }}>
                  <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.9)' }}>
                    💡 {explanation.explanation}
                  </div>
                </div>
              </div>

              {/* SHAP Values */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  📊 SHAP Values - Feature Importance
                </h2>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '16px' }}>
                  Her özelliğin prediction üzerindeki etkisi
                </div>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {Object.entries(explanation.shap_values).sort((a: any, b: any) => b[1] - a[1]).map(([feature, value]: [string, any], _idx: number) => {
                    const percentage = value * 100;
                    return (
                      <div key={feature}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                          <span style={{ fontSize: '14px', color: '#FFFFFF' }}>{feature}</span>
                          <span style={{ fontSize: '14px', fontWeight: '700', color: '#3B82F6' }}>{percentage.toFixed(1)}%</span>
                        </div>
                        <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{ width: `${percentage}%`, height: '100%', background: 'linear-gradient(90deg, #3B82F6 0%, #1D4ED8 100%)', transition: 'width 0.5s' }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Attention Weights */}
              <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
                <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  🎯 Attention Weights - Timeframe Focus
                </h2>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '16px' }}>
                  Model hangi timeframe'e daha çok odaklandı?
                </div>
                <div style={{ display: 'flex', gap: '16px', justifyContent: 'space-around' }}>
                  {Object.entries(explanation.attention_weights).map(([timeframe, weight]: [string, any]) => (
                    <div key={timeframe} style={{ flex: 1, textAlign: 'center' }}>
                      <div style={{ width: '120px', height: '120px', margin: '0 auto 12px', position: 'relative' }}>
                        <svg viewBox="0 0 120 120">
                          <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
                          <circle
                            cx="60"
                            cy="60"
                            r="54"
                            fill="none"
                            stroke="#3B82F6"
                            strokeWidth="8"
                            strokeDasharray={`${weight * 339.292} 339.292`}
                            strokeLinecap="round"
                            transform="rotate(-90 60 60)"
                          />
                          <text x="60" y="60" textAnchor="middle" dy="7" fill="#FFFFFF" fontSize="24" fontWeight="700">
                            {(weight * 100).toFixed(0)}%
                          </text>
                        </svg>
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        {timeframe.replace('timeframe_', '').toUpperCase()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Contributing Features */}
              <div style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(29, 78, 216, 0.05) 100%)', border: '1px solid rgba(59, 130, 246, 0.2)', borderRadius: '20px', padding: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#3B82F6', marginBottom: '16px' }}>
                  🏆 Top 3 Contributing Features
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  {explanation.top_features.map(([feature, value]: [string, number], idx: number) => (
                    <div key={feature} style={{ textAlign: 'center', padding: '20px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                      <div style={{ fontSize: '36px', marginBottom: '8px' }}>
                        {idx === 0 ? '🥇' : idx === 1 ? '🥈' : '🥉'}
                      </div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
                        {feature}
                      </div>
                      <div style={{ fontSize: '14px', color: '#3B82F6' }}>
                        {(value * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Info */}
          <div style={{ marginTop: '24px', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '12px' }}>
              🧠 Explainable AI Nedir?
            </h3>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              <p style={{ marginBottom: '8px' }}>
                <strong>SHAP (SHapley Additive exPlanations):</strong> Her özelliğin prediction üzerindeki katkısını hesaplar.
              </p>
              <p style={{ marginBottom: '8px' }}>
                <strong>Attention Weights:</strong> Transformer modelleri hangi timeframe'e odaklandığını gösterir.
              </p>
              <p>
                <strong>Transparency = Trust:</strong> Black-box AI yerine şeffaf ve açıklanabilir AI kullanıyoruz.
              </p>
            </div>
          </div>
        </main>

        {aiAssistantOpen && <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />}
      </div>
    </PWAProvider>
  );
}
