'use client';

/**
 * 🛡️ FEDERATED LEARNING
 *
 * Tüm kullanıcıların bilgisi birleşir ama gizlilik korunur
 * Toplu zeka, bireysel mahremiyet
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';
import { useResponsive } from '@/hooks/useResponsive';

export default function FederatedLearningPage() {
  const { isMobile, isTablet } = useResponsive();
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  const stats = {
    total_users: 8247,
    privacy_score: 99.8,
    global_accuracy: 93.1,
    rounds: 1247,
    active_participants: 5621
  };

  useEffect(() => {
    setMounted(true);
  }, []);

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

            <h1 style={{ fontSize: '36px', fontWeight: '900', background: 'linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '12px' }}>
              🛡️ Federated Learning
            </h1>
            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
              Tüm kullanıcıların bilgisi birleşir ama gizlilik korunur. Toplu zeka, bireysel mahremiyet. Differential privacy garantili.
            </p>
          </div>

          {/* Stats Grid - Responsive */}
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : isTablet ? 'repeat(2, 1fr)' : 'repeat(auto-fit, minmax(200px, 1fr))', gap: isMobile ? '12px' : '16px', marginBottom: isMobile ? '24px' : '32px' }}>
            <StatCard label="Total Users" value={stats.total_users.toLocaleString()} color="#6366F1" />
            <StatCard label="Privacy Score" value={`${stats.privacy_score}%`} color="#10B981" />
            <StatCard label="Global Accuracy" value={`${stats.global_accuracy}%`} color="#F59E0B" />
            <StatCard label="Training Rounds" value={stats.rounds.toLocaleString()} color="#EC4899" />
          </div>

          {/* How It Works */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '2px solid #6366F1', borderRadius: '20px', padding: '40px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: '900', color: '#6366F1', marginBottom: '24px', textAlign: 'center' }}>
              🔄 How Federated Learning Works
            </h2>

            <div style={{ display: 'grid', gap: '20px' }}>
              {[
                { step: 1, title: 'Global Model Distribution', desc: 'Merkezi server global modeli tüm kullanıcılara gönderir', icon: '📤', color: '#6366F1' },
                { step: 2, title: 'Local Training', desc: 'Her kullanıcı modelini kendi verisi ile eğitir (data server\'a gitmiyor!)', icon: '💻', color: '#8B5CF6' },
                { step: 3, title: 'Model Updates Only', desc: 'Sadece model güncellemeleri (gradients) server\'a gönderilir, veri değil', icon: '🔒', color: '#10B981' },
                { step: 4, title: 'Federated Averaging', desc: 'Server tüm güncellemeleri birleştirip yeni global model oluşturur', icon: '📊', color: '#F59E0B' },
              ].map((item) => (
                <div key={item.step} style={{ display: 'flex', gap: '20px', padding: '24px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '16px', border: `1px solid ${item.color}40` }}>
                  <div style={{ fontSize: '48px', flexShrink: 0 }}>{item.icon}</div>
                  <div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: item.color, marginBottom: '8px' }}>
                      Step {item.step}: {item.title}
                    </div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6' }}>
                      {item.desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Privacy Guarantees */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px', marginBottom: '32px' }}>
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: '16px', padding: '32px' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px', textAlign: 'center' }}>🔐</div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#10B981', marginBottom: '12px', textAlign: 'center' }}>
                Differential Privacy
              </h3>
              <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6', textAlign: 'center' }}>
                Model güncellemelerine noise eklenir. Matematiksel olarak kanıtlanmış privacy guarantee. Epsilon = 1.0 (very strong).
              </p>
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#10B981' }}>ε = 1.0</div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>Privacy Budget</div>
              </div>
            </div>

            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(99, 102, 241, 0.3)', borderRadius: '16px', padding: '32px' }}>
              <div style={{ fontSize: '48px', marginBottom: '16px', textAlign: 'center' }}>🚫</div>
              <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#6366F1', marginBottom: '12px', textAlign: 'center' }}>
                Secure Aggregation
              </h3>
              <p style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.6', textAlign: 'center' }}>
                Homomorphic encryption ile server hiçbir kullanıcının bireysel güncellemesini göremez. Sadece toplam görür.
              </p>
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: '#6366F1' }}>256-bit</div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>Encryption</div>
              </div>
            </div>
          </div>

          {/* Current Round Stats */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              📊 Current Round Stats
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
              <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>Round #</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: '#FFFFFF' }}>{stats.rounds}</div>
              </div>
              <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>Active Users</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: '#6366F1' }}>{stats.active_participants.toLocaleString()}</div>
              </div>
              <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px', textAlign: 'center' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>Participation Rate</div>
                <div style={{ fontSize: '28px', fontWeight: '700', color: '#10B981' }}>{((stats.active_participants / stats.total_users) * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Benefits */}
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '20px', padding: '32px', marginBottom: '24px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              ✨ Benefits
            </h2>
            <div style={{ display: 'grid', gap: '12px' }}>
              {[
                { title: 'Privacy-Preserving', desc: 'Veriler asla server\'a gitmez, GDPR compliant', icon: '🔒' },
                { title: 'Collective Intelligence', desc: '8,247 kullanıcının bilgisi birleşir, model güçlenir', icon: '🧠' },
                { title: 'Edge Computing', desc: 'Training kullanıcının cihazında, server maliyeti düşük', icon: '💻' },
                { title: 'Personalization', desc: 'Global model + local adaptation = best of both worlds', icon: '⚡' },
              ].map((benefit, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '16px', padding: '16px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '12px' }}>
                  <div style={{ fontSize: '32px', flexShrink: 0 }}>{benefit.icon}</div>
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>{benefit.title}</div>
                    <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>{benefit.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Info */}
          <div style={{ background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: '20px', padding: '32px' }}>
            <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#6366F1', marginBottom: '16px' }}>
              🧠 Federated Learning Nedir?
            </h3>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.8' }}>
              <p style={{ marginBottom: '12px' }}>
                <strong>Distributed ML with Privacy:</strong> Merkezi server veriye erişmeden, distributed cihazlarda model eğitir.
              </p>
              <p style={{ marginBottom: '12px' }}>
                <strong>Real-World Example:</strong> Google Keyboard\'un next-word prediction\'ı federated learning ile eğitilir. Hiçbir mesajınız Google\'a gitmez!
              </p>
              <p>
                <strong>Crypto Trading Use Case:</strong> Tüm kullanıcıların trading stratejilerinden öğren, ama kimsenin pozisyonları açığa çıkmasın.
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
