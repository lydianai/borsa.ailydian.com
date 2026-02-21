'use client';

/**
 * 🤖 REINFORCEMENT LEARNING TRADING AGENT
 *
 * Kendi trading stratejisini keşfeden ve optimize eden yapay zeka
 * Q-Learning algoritması ile sürekli öğrenen agent
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface RLStats {
  episodes: number;
  win_rate: number;
  learning_rate: number;
  total_reward: number;
  q_table_size: number;
  epsilon: number;
}

interface TrainingResult {
  episode: number;
  reward: number;
  total_reward: number;
  q_table_size: number;
}

export default function RLAgentPage() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [stats, setStats] = useState<RLStats | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResults, setTrainingResults] = useState<TrainingResult[]>([]);
  const [prediction, setPrediction] = useState<any>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  useEffect(() => {
    setMounted(true);
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/ai-learning/rl-agent');
      const data = await res.json();
      if (data.success) {
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch RL stats:', error);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      const res = await fetch('/api/ai-learning/rl-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'train', episodes: 10 }),
      });
      const data = await res.json();
      if (data.success) {
        setTrainingResults(data.results);
        fetchStats();
      }
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async () => {
    try {
      const res = await fetch('/api/ai-learning/rl-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'predict', symbol: selectedSymbol }),
      });
      const data = await res.json();
      if (data.success) {
        setPrediction(data);
      }
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

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
          paddingTop: '80px',
        }}
      >
        <SharedSidebar
          currentPage="ai-learning-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '40px 24px' }}>
          {/* Header */}
          <div style={{ marginBottom: '32px' }}>
            <Link
              href="/ai-learning-hub"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '14px',
                marginBottom: '16px',
              }}
            >
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              AI/ML Learning Hub
            </Link>

            <h1 style={{
              fontSize: '36px',
              fontWeight: '900',
              background: 'linear-gradient(135deg, #8B5CF6 0%, #A78BFA 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: '12px',
            }}>
              ⚡ Reinforcement Learning Agent
            </h1>
            <p style={{
              fontSize: '16px',
              color: 'rgba(255, 255, 255, 0.7)',
              lineHeight: '1.6',
            }}>
              Kendi trading stratejisini keşfeden ve optimize eden yapay zeka. Q-Learning ile her işlemden öğrenir.
            </p>
          </div>

          {/* Stats Grid */}
          {stats && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
              marginBottom: '32px',
            }}>
              <StatCard label="Total Episodes" value={stats.episodes.toLocaleString()} color="#8B5CF6" />
              <StatCard label="Win Rate" value={`${stats.win_rate}%`} color="#10B981" />
              <StatCard label="Learning Rate" value={`${stats.learning_rate}%`} color="#06B6D4" />
              <StatCard label="Q-Table Size" value={stats.q_table_size.toLocaleString()} color="#F59E0B" />
              <StatCard label="Epsilon" value={stats.epsilon.toFixed(2)} color="#EC4899" />
              <StatCard label="Total Reward" value={stats.total_reward.toFixed(2)} color="#14B8A6" />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
            {/* Training Panel */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '20px',
              padding: '32px',
            }}>
              <h2 style={{
                fontSize: '20px',
                fontWeight: '700',
                color: '#FFFFFF',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <Icons.Zap style={{ width: '24px', height: '24px', color: '#8B5CF6' }} />
                Training
              </h2>

              <button
                onClick={handleTrain}
                disabled={isTraining}
                style={{
                  width: '100%',
                  padding: '16px 24px',
                  background: isTraining
                    ? 'rgba(139, 92, 246, 0.3)'
                    : 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                  border: 'none',
                  borderRadius: '12px',
                  color: '#FFFFFF',
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: isTraining ? 'not-allowed' : 'pointer',
                  transition: 'all 0.3s',
                  marginBottom: '16px',
                }}
              >
                {isTraining ? '⏳ Training...' : '🚀 Train 10 Episodes'}
              </button>

              {trainingResults.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                  <div style={{
                    fontSize: '14px',
                    color: 'rgba(255, 255, 255, 0.7)',
                    marginBottom: '12px',
                  }}>
                    Recent Training Results:
                  </div>
                  {trainingResults.slice(-5).map((result, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '12px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                        marginBottom: '8px',
                        fontSize: '12px',
                        color: 'rgba(255, 255, 255, 0.8)',
                      }}
                    >
                      Episode {result.episode} - Reward: {result.reward.toFixed(3)} - Q-Table: {result.q_table_size}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Prediction Panel */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '20px',
              padding: '32px',
            }}>
              <h2 style={{
                fontSize: '20px',
                fontWeight: '700',
                color: '#FFFFFF',
                marginBottom: '16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}>
                <Icons.TrendingUp style={{ width: '24px', height: '24px', color: '#10B981' }} />
                Live Prediction
              </h2>

              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  marginBottom: '12px',
                }}
              >
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="BNBUSDT">BNB/USDT</option>
              </select>

              <button
                onClick={handlePredict}
                style={{
                  width: '100%',
                  padding: '16px 24px',
                  background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                  border: 'none',
                  borderRadius: '12px',
                  color: '#FFFFFF',
                  fontSize: '16px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  marginBottom: '16px',
                }}
              >
                🎯 Get Prediction
              </button>

              {prediction && (
                <div style={{
                  padding: '20px',
                  background: 'rgba(16, 185, 129, 0.1)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  borderRadius: '12px',
                  marginTop: '16px',
                }}>
                  <div style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: prediction.action === 'BUY' ? '#10B981' : prediction.action === 'SELL' ? '#EF4444' : '#F59E0B',
                    marginBottom: '8px',
                  }}>
                    {prediction.action}
                  </div>
                  <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    Confidence: {prediction.confidence}%
                  </div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '8px' }}>
                    State: {prediction.state?.trend} / {prediction.state?.volatility}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Info Panel */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(109, 40, 217, 0.05) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.2)',
            borderRadius: '20px',
            padding: '32px',
          }}>
            <h3 style={{
              fontSize: '20px',
              fontWeight: '700',
              color: '#8B5CF6',
              marginBottom: '16px',
            }}>
              🧠 Nasıl Çalışır?
            </h3>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              <p style={{ marginBottom: '12px' }}>
                <strong>Q-Learning Algoritması:</strong> Agent, farklı piyasa durumlarında (state) hangi aksiyonun (BUY/SELL/HOLD)
                en yüksek getiriyi sağladığını öğrenir.
              </p>
              <p style={{ marginBottom: '12px' }}>
                <strong>Epsilon-Greedy Stratejisi:</strong> %10 rastgele aksiyon ile keşif yapar, %90 en iyi bilinen aksiyonu seçer.
              </p>
              <p>
                <strong>Sürekli Öğrenme:</strong> Her trade sonucuna göre Q-table'ı günceller. Zamanla optimal stratejiyi keşfeder.
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

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      background: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '16px',
      padding: '20px',
    }}>
      <div style={{
        fontSize: '12px',
        color: 'rgba(255, 255, 255, 0.5)',
        marginBottom: '8px',
        textTransform: 'uppercase',
        letterSpacing: '1px',
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '28px',
        fontWeight: '700',
        color: color,
      }}>
        {value}
      </div>
    </div>
  );
}
