'use client';

/**
 * 📊 ONLINE LEARNING PIPELINE
 *
 * Hiç durmadan öğrenen sistem - Her yeni veri ile kendini günceller
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

export default function OnlineLearningPage() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [driftResult, setDriftResult] = useState<any>(null);

  useEffect(() => {
    setMounted(true);
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/ai-learning/online-learning');
      const data = await res.json();
      if (data.success) {
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch online learning stats:', error);
    }
  };

  const handleUpdate = async () => {
    setIsUpdating(true);
    try {
      const res = await fetch('/api/ai-learning/online-learning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'update' }),
      });
      const data = await res.json();
      if (data.success) {
        fetchStats();
      }
    } catch (error) {
      console.error('Update failed:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  const checkDrift = async () => {
    try {
      const res = await fetch('/api/ai-learning/online-learning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'drift' }),
      });
      const data = await res.json();
      if (data.success) {
        setDriftResult(data);
      }
    } catch (error) {
      console.error('Drift check failed:', error);
    }
  };

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

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              🔄 Online Learning Pipeline
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              Hiç durmadan öğrenen sistem. Her yeni veri ile kendini günceller, piyasa değişimlerine anında adapte olur.
            </p>
          </div>

          {stats && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
              <StatCard label="Total Updates" value={stats.updates?.toLocaleString() || '0'} color="#06B6D4" />
              <StatCard label="Model Accuracy" value={`${stats.accuracy || 0}%`} color="#10B981" />
              <StatCard label="Drift Score" value={(stats.drift_score || 0).toFixed(3)} color="#F59E0B" />
              <StatCard label="Model Version" value={`v${stats.model_version || 0}`} color="#8B5CF6" />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                🚀 Model Update
              </h2>
              <button onClick={handleUpdate} disabled={isUpdating} style={{ width: '100%', padding: '16px 24px', background: isUpdating ? 'rgba(6, 182, 212, 0.3)' : 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)', border: 'none', borderRadius: '12px', color: '#FFFFFF', fontSize: '16px', fontWeight: '700', cursor: isUpdating ? 'not-allowed' : 'pointer' }}>
                {isUpdating ? '⏳ Updating...' : '📈 Update Model'}
              </button>
            </div>

            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                🔍 Drift Detection
              </h2>
              <button onClick={checkDrift} style={{ width: '100%', padding: '16px 24px', background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)', border: 'none', borderRadius: '12px', color: '#FFFFFF', fontSize: '16px', fontWeight: '700', cursor: 'pointer' }}>
                🎯 Check Drift
              </button>

              {driftResult && (
                <div style={{ marginTop: '16px', padding: '16px', background: driftResult.drift_detected ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)', border: `1px solid ${driftResult.drift_detected ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`, borderRadius: '12px' }}>
                  <div style={{ fontSize: '14px', fontWeight: '700', color: driftResult.drift_detected ? '#EF4444' : '#10B981', marginBottom: '8px' }}>
                    {driftResult.drift_detected ? '⚠️ Drift Detected!' : '✅ No Drift'}
                  </div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    Score: {driftResult.drift_score} | Action: {driftResult.action}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div style={{ background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(8, 145, 178, 0.05) 100%)', border: '1px solid rgba(6, 182, 212, 0.2)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#06B6D4', marginBottom: '16px' }}>
              🧠 Online Learning Nedir?
            </h3>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              <p style={{ marginBottom: '12px' }}>
                <strong>Sürekli Öğrenme:</strong> Geleneksel ML modelleri statik verilerle eğitilir ve donuklaşır. Online learning ise her yeni veri ile modeli günceller.
              </p>
              <p style={{ marginBottom: '12px' }}>
                <strong>Concept Drift Detection:</strong> Piyasa davranışları değiştiğinde (drift) sistem bunu tespit eder ve modeli yeniden eğitir.
              </p>
              <p>
                <strong>Adaptif Sistem:</strong> Kripto piyasaları gibi hızlı değişen ortamlarda kritik öneme sahiptir. Model her zaman güncel kalır.
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
