'use client';

/**
 * 👥 MULTI-AGENT SYSTEM
 *
 * Birbirleriyle yarışan 5 farklı AI agent - En iyisi kazanır
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

export default function MultiAgentPage() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  useEffect(() => {
    setMounted(true);
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/ai-learning/multi-agent');
      const data = await res.json();
      if (data.success) {
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch multi-agent stats:', error);
    }
  };

  const getPrediction = async () => {
    try {
      const res = await fetch('/api/ai-learning/multi-agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol: selectedSymbol, timeframe: '1h' }),
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
    return <div style={{ minHeight: '100vh', background: '#0a0a0a' }} />;
  }

  const agents = stats?.agents ? Object.entries(stats.agents).map(([name, data]: [string, any]) => ({
    name,
    ...data,
  })) : [];

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

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              👥 Multi-Agent System
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              Birbirleriyle yarışan 5 farklı AI agent. En iyisi kazanır, diğerleri ondan öğrenir. Toplu zeka.
            </p>
          </div>

          {/* Agent Leaderboard */}
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              🏆 Agent Leaderboard
            </h2>
            <div style={{ display: 'grid', gap: '12px' }}>
              {agents.sort((a, b) => b.win_rate - a.win_rate).map((agent, idx) => (
                <div key={agent.name} style={{
                  background: idx === 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255, 255, 255, 0.03)',
                  border: `1px solid ${idx === 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
                  borderRadius: '16px',
                  padding: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '50%',
                      background: idx === 0 ? 'linear-gradient(135deg, #10B981 0%, #059669 100%)' : 'rgba(255, 255, 255, 0.1)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px',
                      fontWeight: '700',
                      color: '#FFFFFF',
                    }}>
                      {idx + 1}
                    </div>
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', textTransform: 'capitalize' }}>
                        {agent.name.replace('_', ' ')}
                      </div>
                      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        {agent.total_trades.toLocaleString()} trades
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: idx === 0 ? '#10B981' : '#FFFFFF' }}>
                    {agent.win_rate}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Ensemble Stats */}
          {stats && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '32px' }}>
              <StatCard label="Active Agents" value="5" color="#10B981" />
              <StatCard label="Best Agent" value={stats.best_agent?.replace('_', ' ') || 'N/A'} color="#8B5CF6" />
              <StatCard label="Ensemble Accuracy" value={`${stats.ensemble_accuracy || 0}%`} color="#F59E0B" />
            </div>
          )}

          {/* Prediction Panel */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              🎯 Ensemble Prediction
            </h2>

            <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
              <select value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)} style={{ flex: 1, padding: '12px', background: 'rgba(255, 255, 255, 0.05)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', color: '#FFFFFF', fontSize: '14px' }}>
                <option value="BTCUSDT">BTC/USDT</option>
                <option value="ETHUSDT">ETH/USDT</option>
                <option value="SOLUSDT">SOL/USDT</option>
              </select>
              <button onClick={getPrediction} style={{ padding: '12px 32px', background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)', border: 'none', borderRadius: '8px', color: '#FFFFFF', fontSize: '14px', fontWeight: '700', cursor: 'pointer' }}>
                Get Prediction
              </button>
            </div>

            {prediction && (
              <div>
                <div style={{ padding: '24px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '12px', marginBottom: '16px' }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: prediction.ensemble_action === 'BUY' ? '#10B981' : prediction.ensemble_action === 'SELL' ? '#EF4444' : '#F59E0B', marginBottom: '8px' }}>
                    {prediction.ensemble_action}
                  </div>
                  <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    Ensemble Confidence: {prediction.confidence}%
                  </div>
                </div>

                {prediction.individual_predictions && (
                  <div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '12px' }}>
                      Individual Agent Predictions:
                    </div>
                    <div style={{ display: 'grid', gap: '8px' }}>
                      {prediction.individual_predictions.map((pred: any, idx: number) => (
                        <div key={idx} style={{ padding: '12px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '14px', color: '#FFFFFF', textTransform: 'capitalize' }}>
                            {pred.agent.replace('_', ' ')}
                          </span>
                          <span style={{ fontSize: '14px', fontWeight: '700', color: pred.action === 'BUY' ? '#10B981' : pred.action === 'SELL' ? '#EF4444' : '#F59E0B' }}>
                            {pred.action} ({pred.confidence}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Info */}
          <div style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#10B981', marginBottom: '16px' }}>
              🧠 Multi-Agent Learning
            </h3>
            <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              Her agent farklı bir strateji kullanır: Momentum, Mean Reversion, Trend Following, Breakout, Scalping.
              Ensemble voting ile en güvenilir sinyali üretir. Diversity = Better Performance!
            </p>
          </div>
        </main>

        {aiAssistantOpen && <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />}
      </div>
    </PWAProvider>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '20px', textAlign: 'center' }}>
      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: '24px', fontWeight: '700', color: color, textTransform: 'capitalize' }}>{value}</div>
    </div>
  );
}
