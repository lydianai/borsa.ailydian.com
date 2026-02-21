'use client';

/**
 * 🤖 AI/ML LEARNING HUB - MODERN DASHBOARD
 *
 * Kendi kendine öğrenen ve gelişen yapay zeka sistemi
 * - Real-time WebSocket updates
 * - Modern glassmorphic design
 * - 538 Binance Futures USDT-M coins continuous scanning
 * - Live prediction stream
 */

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { io, Socket } from 'socket.io-client';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

// ============================================================================
// TYPES
// ============================================================================

interface AISystemStats {
  rl_agent: {
    episodes: number;
    win_rate: number;
    learning_rate: number;
  };
  online_learning: {
    updates: number;
    accuracy: number;
    drift_score: number;
  };
  multi_agent: {
    ensemble_acc: number;
    best_agent: string;
  };
  automl: {
    trials: number;
    best_sharpe: number;
  };
  nas: {
    generations: number;
    best_arch: string;
  };
  meta_learning: {
    adaptation: number;
  };
  federated: {
    users: number;
    global_acc: number;
  };
  causal: {
    paths: number;
    confidence: number;
  };
  regime: {
    current: string;
    confidence: number;
  };
  explainable: {
    explainability: number;
  };
}

interface Prediction {
  symbol: string;
  timestamp: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  source: string;
  price: number;
  isPinned?: boolean; // Kalıcı LONG sinyali
  timeframe?: string; // 1h, 4h, etc.
  signalChangeInfo?: string; // Sinyal değişim bilgisi
  lastChecked?: string; // Son kontrol zamanı
  longReason?: string; // LONG sinyali nedeni (Türkçe açıklama)
}

interface Notification {
  id: string;
  type: string;
  priority: 'YÜKSEK' | 'ORTA' | 'DÜŞÜK';
  symbol: string;
  price: number;
  confidence: number;
  consensus_count: number;
  ai_sources: string;
  title: string;
  message: string;
  details: string;
  action: string;
  timestamp: string;
  expires_at: string;
  status: string;
}

interface AIFeature {
  id: string;
  title: string;
  shortTitle: string;
  description: string;
  icon: any;
  color: string;
  gradient: string;
  href: string;
  statsKey: keyof AISystemStats;
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function AILearningHub() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [connected, setConnected] = useState(false);
  const [systemStats, setSystemStats] = useState<AISystemStats | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [selectedAI, setSelectedAI] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [showNotifications, setShowNotifications] = useState(false);

  const socketRef = useRef<Socket | null>(null);

  // ============================================================================
  // AI SYSTEMS CONFIGURATION
  // ============================================================================

  const aiSystems: AIFeature[] = [
    {
      id: 'rl-agent',
      title: 'Pekiştirmeli Öğrenme Ajanı',
      shortTitle: 'RL Ajanı',
      description: 'Kendi trading stratejisini keşfeden ve optimize eden yapay zeka',
      icon: Icons.Zap,
      color: '#8B5CF6',
      gradient: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
      href: '/ai-learning-hub/rl-agent',
      statsKey: 'rl_agent',
    },
    {
      id: 'online-learning',
      title: 'Çevrimiçi Öğrenme Hattı',
      shortTitle: 'Çevrimiçi Öğrenme',
      description: 'Hiç durmadan öğrenen sistem',
      icon: Icons.RefreshCw,
      color: '#06B6D4',
      gradient: 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)',
      href: '/ai-learning-hub/online-learning',
      statsKey: 'online_learning',
    },
    {
      id: 'multi-agent',
      title: 'Çoklu Ajan Sistemi',
      shortTitle: 'Çoklu Ajan',
      description: 'Birbirleriyle yarışan 5 farklı yapay zeka ajanı',
      icon: Icons.Users,
      color: '#10B981',
      gradient: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
      href: '/ai-learning-hub/multi-agent',
      statsKey: 'multi_agent',
    },
    {
      id: 'automl',
      title: 'Otomatik ML Optimize Edici',
      shortTitle: 'Otomatik ML',
      description: 'Kendi hiperparametrelerini optimize eden sistem',
      icon: Icons.Settings,
      color: '#F59E0B',
      gradient: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
      href: '/ai-learning-hub/automl',
      statsKey: 'automl',
    },
    {
      id: 'nas',
      title: 'Sinir Ağı Mimari Arama',
      shortTitle: 'Mimari Arama',
      description: 'Kendi sinir ağı mimarisini tasarlayan yapay zeka',
      icon: Icons.Layers,
      color: '#EC4899',
      gradient: 'linear-gradient(135deg, #EC4899 0%, #BE185D 100%)',
      href: '/ai-learning-hub/nas',
      statsKey: 'nas',
    },
    {
      id: 'meta-learning',
      title: 'Meta Öğrenme Sistemi',
      shortTitle: 'Meta Öğrenme',
      description: 'Öğrenmeyi öğrenen yapay zeka',
      icon: Icons.Sparkles,
      color: '#14B8A6',
      gradient: 'linear-gradient(135deg, #14B8A6 0%, #0D9488 100%)',
      href: '/ai-learning-hub/meta-learning',
      statsKey: 'meta_learning',
    },
    {
      id: 'federated',
      title: 'Federatif Öğrenme',
      shortTitle: 'Federatif Öğrenme',
      description: 'Gizlilik korumalı toplu zeka',
      icon: Icons.Shield,
      color: '#6366F1',
      gradient: 'linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)',
      href: '/ai-learning-hub/federated',
      statsKey: 'federated',
    },
    {
      id: 'causal-ai',
      title: 'Nedensel Yapay Zeka',
      shortTitle: 'Nedensel YZ',
      description: 'Sebep-sonuç ilişkilerini öğrenen yapay zeka',
      icon: Icons.GitBranch,
      color: '#F97316',
      gradient: 'linear-gradient(135deg, #F97316 0%, #EA580C 100%)',
      href: '/ai-learning-hub/causal-ai',
      statsKey: 'causal',
    },
    {
      id: 'regime-detection',
      title: 'Uyarlanabilir Rejim Tespiti',
      shortTitle: 'Rejim Tespiti',
      description: 'Market rejimlerini otomatik tespit eden sistem',
      icon: Icons.TrendingUp,
      color: '#EF4444',
      gradient: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
      href: '/ai-learning-hub/regime-detection',
      statsKey: 'regime',
    },
    {
      id: 'explainable-ai',
      title: 'Açıklanabilir Yapay Zeka',
      shortTitle: 'Açıklanabilir YZ',
      description: 'Yapay zeka kararlarının açıklanması',
      icon: Icons.Info,
      color: '#3B82F6',
      gradient: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
      href: '/ai-learning-hub/explainable-ai',
      statsKey: 'explainable',
    },
  ];

  // ============================================================================
  // WEBSOCKET CONNECTION
  // ============================================================================

  useEffect(() => {
    setMounted(true);

    // Connect to Flask WebSocket server (PORT 5020)
    const socket = io('http://localhost:5020', {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('✅ WebSocket connected');
      setConnected(true);
    });

    socket.on('disconnect', () => {
      console.log('❌ WebSocket disconnected');
      setConnected(false);
    });

    socket.on('connection_status', (data) => {
      console.log('📡 Connection status:', data);
    });

    // Real-time system stats updates (every 2 seconds)
    socket.on('system_stats_update', (data: AISystemStats) => {
      setSystemStats(data);
    });

    // Real-time prediction updates with pinned LONG logic
    socket.on('new_prediction', (prediction: Prediction) => {
      setPredictions((prev) => {
        // Check if this coin already has a pinned LONG signal
        const existingPinned = prev.find(
          (p) => p.symbol === prediction.symbol && p.isPinned && p.action === 'BUY'
        );

        // If new prediction is BUY with high confidence, pin it
        if (prediction.action === 'BUY' && prediction.confidence >= 75) {
          prediction.isPinned = true;
          prediction.lastChecked = new Date().toISOString();

          // Check for signal change
          if (existingPinned) {
            const timeDiff = Math.abs(
              new Date(prediction.timestamp).getTime() -
              new Date(existingPinned.timestamp).getTime()
            );
            if (timeDiff < 3600000) { // Within 1 hour
              prediction.signalChangeInfo = '🔄 Sinyal güncellendi';
            }
          }
        } else if (existingPinned && prediction.action !== 'BUY') {
          // Signal changed from BUY to something else
          prediction.signalChangeInfo = '⚠️ LONG sinyali kayboldu!';
          // Remove pinned status
          const filtered = prev.filter((p) => !(p.symbol === prediction.symbol && p.isPinned));
          return [prediction, ...filtered].slice(0, 50);
        }

        // Keep pinned LONGs at top, others below
        const pinnedLongs = prev.filter((p) => p.isPinned && p.action === 'BUY' && p.symbol !== prediction.symbol);
        const others = prev.filter((p) => !p.isPinned || p.symbol === prediction.symbol);

        const updated = [...pinnedLongs, prediction, ...others].slice(0, 50);
        return updated;
      });
    });

    // Real-time notification updates (LONG signals)
    socket.on('new_notification', (notification: Notification) => {
      console.log('🔔 Yeni LONG bildirimi:', notification);
      setNotifications((prev) => {
        const updated = [notification, ...prev].slice(0, 20); // Keep last 20
        return updated;
      });

      // Browser notification (if permission granted)
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(notification.title, {
          body: notification.message,
          icon: '/icon-192x192.png',
          badge: '/icon-192x192.png',
        });
      }
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
    };
  }, []);

  // ============================================================================
  // RENDER
  // ============================================================================

  if (!mounted) {
    return (
      <div
        style={{
          minHeight: '100vh',
          background: '#0a0a0a',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: '18px' }}>
          🤖 AI Learning Hub yükleniyor...
        </div>
      </div>
    );
  }

  return (
    <PWAProvider>
      <div
        suppressHydrationWarning
        style={{
          minHeight: '100vh',
          background: '#0a0a0a',
          paddingTop: '80px',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Animated gradient background */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(139, 92, 246, 0.15) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(236, 72, 153, 0.15) 0%, transparent 50%)',
            pointerEvents: 'none',
            zIndex: 0,
          }}
        />

        <SharedSidebar
          currentPage="ai-learning-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1800px', margin: '0 auto', padding: '24px', position: 'relative', zIndex: 1 }}>
          {/* Hero Section with Connection Status */}
          <div style={{ marginBottom: '32px' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '16px',
                marginBottom: '24px',
              }}
            >
              <div>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '12px',
                    marginBottom: '12px',
                    padding: '8px 20px',
                    background: 'rgba(139, 92, 246, 0.1)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: '50px',
                    border: '1px solid rgba(139, 92, 246, 0.3)',
                  }}
                >
                  <span style={{ fontSize: '20px' }}>🤖</span>
                  <span
                    style={{
                      fontSize: '14px',
                      fontWeight: '700',
                      color: '#8B5CF6',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    AI/ML Learning Hub
                  </span>
                </div>

                <h1
                  style={{
                    fontSize: 'clamp(32px, 5vw, 48px)',
                    fontWeight: '900',
                    background: 'linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    marginBottom: '8px',
                    letterSpacing: '-1px',
                  }}
                >
                  Kendi Kendine Öğrenen AI Sistemi
                </h1>

                <p
                  style={{
                    fontSize: '16px',
                    color: 'rgba(255, 255, 255, 0.6)',
                    maxWidth: '700px',
                  }}
                >
                  538 Binance Futures USDT-M coin için anlık tahmin ve sürekli öğrenme
                </p>
              </div>

              {/* Connection Status & Notifications */}
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {/* Active AI Systems Counter */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 24px',
                    background: 'rgba(16, 185, 129, 0.1)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                  }}
                >
                  <span style={{ fontSize: '20px' }}>🤖</span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <span
                      style={{
                        fontSize: '14px',
                        fontWeight: '700',
                        color: '#10B981',
                      }}
                    >
                      {aiSystems.length}/10 AI Aktif
                    </span>
                    <span
                      style={{
                        fontSize: '10px',
                        color: 'rgba(16, 185, 129, 0.7)',
                      }}
                    >
                      Tüm Sistemler Çalışıyor
                    </span>
                  </div>
                </div>

                {/* Notification Button */}
                <button
                  onClick={() => setShowNotifications(!showNotifications)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 24px',
                    background: notifications.length > 0
                      ? 'rgba(236, 72, 153, 0.1)'
                      : 'rgba(139, 92, 246, 0.1)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: '12px',
                    border: `1px solid ${
                      notifications.length > 0 ? 'rgba(236, 72, 153, 0.3)' : 'rgba(139, 92, 246, 0.3)'
                    }`,
                    cursor: 'pointer',
                    transition: 'all 0.3s',
                  }}
                >
                  <span style={{ fontSize: '20px' }}>
                    {notifications.length > 0 ? '🔔' : '🔕'}
                  </span>
                  <span
                    style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      color: notifications.length > 0 ? '#EC4899' : '#8B5CF6',
                    }}
                  >
                    {notifications.length > 0
                      ? `${notifications.length} LONG Fırsatı`
                      : 'Bildirim Yok'}
                  </span>
                </button>

                {/* Connection Status */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: '12px 24px',
                    background: connected
                      ? 'rgba(16, 185, 129, 0.1)'
                      : 'rgba(239, 68, 68, 0.1)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: '12px',
                    border: `1px solid ${
                      connected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
                    }`,
                  }}
                >
                  <div
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: connected ? '#10B981' : '#EF4444',
                      boxShadow: connected
                        ? '0 0 12px rgba(16, 185, 129, 0.6)'
                        : '0 0 12px rgba(239, 68, 68, 0.6)',
                      animation: connected ? 'pulse 2s infinite' : 'none',
                    }}
                  />
                  <span
                    style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      color: connected ? '#10B981' : '#EF4444',
                    }}
                  >
                    {connected ? 'Canlı Bağlantı Aktif' : 'Bağlantı Kesildi'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Grid Layout */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '20px',
              gridAutoRows: 'minmax(200px, auto)',
            }}
          >
            {/* AI System Cards - 2x5 grid */}
            {aiSystems.map((system) => {
              const IconComponent = system.icon;
              const stats = systemStats?.[system.statsKey];
              const isSelected = selectedAI === system.id;

              return (
                <Link
                  key={system.id}
                  href={system.href}
                  style={{ textDecoration: 'none' }}
                  onMouseEnter={() => setSelectedAI(system.id)}
                  onMouseLeave={() => setSelectedAI(null)}
                >
                  <div
                    style={{
                      background: 'rgba(255, 255, 255, 0.02)',
                      backdropFilter: 'blur(20px)',
                      border: `1px solid ${
                        isSelected ? system.color : 'rgba(255, 255, 255, 0.1)'
                      }`,
                      borderRadius: '20px',
                      padding: '24px',
                      height: '100%',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: isSelected ? 'translateY(-4px) scale(1.02)' : 'translateY(0)',
                      boxShadow: isSelected
                        ? `0 20px 40px rgba(0, 0, 0, 0.4), 0 0 40px ${system.color}30`
                        : '0 4px 12px rgba(0, 0, 0, 0.2)',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    {/* Header */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
                      <div
                        style={{
                          width: '48px',
                          height: '48px',
                          borderRadius: '12px',
                          background: system.gradient,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          boxShadow: `0 4px 12px ${system.color}40`,
                          flexShrink: 0,
                          position: 'relative',
                        }}
                      >
                        <IconComponent style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
                        {/* Active Status Indicator */}
                        <div
                          style={{
                            position: 'absolute',
                            top: '-4px',
                            right: '-4px',
                            width: '12px',
                            height: '12px',
                            borderRadius: '50%',
                            background: '#10B981',
                            border: '2px solid rgba(0, 0, 0, 0.3)',
                            boxShadow: '0 0 8px rgba(16, 185, 129, 0.6)',
                          }}
                          title="Aktif"
                        />
                      </div>

                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                          <h3
                            style={{
                              fontSize: '16px',
                              fontWeight: '700',
                              color: '#FFFFFF',
                              whiteSpace: 'nowrap',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                          >
                            {system.shortTitle}
                          </h3>
                          <span
                            style={{
                              fontSize: '10px',
                              fontWeight: '600',
                              color: '#10B981',
                              background: 'rgba(16, 185, 129, 0.15)',
                              padding: '2px 8px',
                              borderRadius: '6px',
                              border: '1px solid rgba(16, 185, 129, 0.3)',
                            }}
                          >
                            AKTIF
                          </span>
                        </div>
                        <p
                          style={{
                            fontSize: '12px',
                            color: 'rgba(255, 255, 255, 0.5)',
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {system.description}
                        </p>
                      </div>
                    </div>

                    {/* Real-time Stats */}
                    {stats && (
                      <div
                        style={{
                          flex: 1,
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '12px',
                          paddingTop: '12px',
                          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        {Object.entries(stats).map(([key, value], idx) => {
                          if (idx > 2) return null; // Max 3 stats

                          // Turkish label mapping
                          const labelMap: Record<string, string> = {
                            'episodes': 'bölümler',
                            'win_rate': 'kazanma oranı',
                            'learning_rate': 'öğrenme oranı',
                            'updates': 'güncellemeler',
                            'accuracy': 'doğruluk',
                            'drift_score': 'sapma skoru',
                            'ensemble_acc': 'topluluk doğruluğu',
                            'best_agent': 'en iyi ajan',
                            'trials': 'denemeler',
                            'best_sharpe': 'en iyi sharpe',
                            'generations': 'nesiller',
                            'best_arch': 'en iyi mimari',
                            'adaptation': 'uyarlama',
                            'users': 'kullanıcılar',
                            'global_acc': 'global doğruluk',
                            'paths': 'yollar',
                            'confidence': 'güven',
                            'current': 'mevcut',
                            'explainability': 'açıklanabilirlik'
                          };

                          const label = labelMap[key] || key.replace('_', ' ');

                          return (
                            <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.4)', textTransform: 'capitalize' }}>
                                {label}
                              </span>
                              <span style={{ fontSize: '14px', fontWeight: '700', color: system.color }}>
                                {typeof value === 'number' ? value.toLocaleString() : value}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Live Prediction Stream */}
          <div
            style={{
              marginTop: '32px',
              background: 'rgba(255, 255, 255, 0.02)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '20px',
              padding: '24px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
              <div
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: '#10B981',
                  boxShadow: '0 0 12px rgba(16, 185, 129, 0.6)',
                  animation: 'pulse 2s infinite',
                }}
              />
              <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                Canlı Tahmin Akışı
              </h2>
              <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.4)' }}>
                ({predictions.length} tahmin)
              </span>
            </div>

            {/* Predictions List */}
            <div style={{ maxHeight: '400px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {predictions.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255, 255, 255, 0.4)' }}>
                  Tahminler yükleniyor...
                </div>
              ) : (
                predictions.map((pred, idx) => {
                  const isPinnedLong = pred.isPinned && pred.action === 'BUY';

                  return (
                    <div
                      key={`${pred.symbol}-${pred.timestamp}-${idx}`}
                      style={{
                        background: isPinnedLong
                          ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)'
                          : 'rgba(255, 255, 255, 0.03)',
                        border: isPinnedLong
                          ? '2px solid rgba(16, 185, 129, 0.4)'
                          : '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '12px',
                        padding: '16px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '12px',
                        animation: idx === 0 ? 'slideIn 0.3s ease-out' : 'none',
                        boxShadow: isPinnedLong ? '0 4px 20px rgba(16, 185, 129, 0.2)' : 'none',
                      }}
                    >
                      {/* Main Info Row */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flex: 1 }}>
                          {/* Pinned Icon */}
                          {isPinnedLong && (
                            <div
                              style={{
                                fontSize: '18px',
                                animation: 'pulse 2s infinite',
                              }}
                              title="Kalıcı LONG Sinyali"
                            >
                              📌
                            </div>
                          )}

                          <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF', minWidth: '100px' }}>
                            {pred.symbol}
                          </div>

                          <div
                            style={{
                              padding: '4px 12px',
                              borderRadius: '8px',
                              fontSize: '12px',
                              fontWeight: '700',
                              background:
                                pred.action === 'BUY'
                                  ? 'rgba(16, 185, 129, 0.2)'
                                  : pred.action === 'SELL'
                                  ? 'rgba(239, 68, 68, 0.2)'
                                  : 'rgba(156, 163, 175, 0.2)',
                              color:
                                pred.action === 'BUY'
                                  ? '#10B981'
                                  : pred.action === 'SELL'
                                  ? '#EF4444'
                                  : '#9CA3AF',
                            }}
                          >
                            {pred.action === 'BUY' ? 'AL' : pred.action === 'SELL' ? 'SAT' : 'BEKLE'}
                          </div>

                          <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)' }}>
                            ${pred.price.toLocaleString()}
                          </div>

                          <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.4)' }}>
                            {pred.source}
                          </div>

                          {/* Timeframe Badge */}
                          {pred.timeframe && (
                            <div
                              style={{
                                padding: '2px 8px',
                                borderRadius: '6px',
                                fontSize: '10px',
                                fontWeight: '600',
                                background: 'rgba(139, 92, 246, 0.2)',
                                color: '#8B5CF6',
                                border: '1px solid rgba(139, 92, 246, 0.3)',
                              }}
                            >
                              {pred.timeframe}
                            </div>
                          )}
                        </div>

                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#8B5CF6' }}>
                          {pred.confidence.toFixed(1)}%
                        </div>
                      </div>

                      {/* Signal Change Info */}
                      {pred.signalChangeInfo && (
                        <div
                          style={{
                            padding: '8px 12px',
                            borderRadius: '8px',
                            background: pred.signalChangeInfo.includes('kayboldu')
                              ? 'rgba(239, 68, 68, 0.1)'
                              : 'rgba(139, 92, 246, 0.1)',
                            border: `1px solid ${
                              pred.signalChangeInfo.includes('kayboldu')
                                ? 'rgba(239, 68, 68, 0.3)'
                                : 'rgba(139, 92, 246, 0.3)'
                            }`,
                            fontSize: '12px',
                            color: pred.signalChangeInfo.includes('kayboldu') ? '#EF4444' : '#8B5CF6',
                            fontWeight: '600',
                          }}
                        >
                          {pred.signalChangeInfo}
                          {pred.lastChecked && (
                            <span style={{ marginLeft: '8px', opacity: 0.7, fontSize: '10px' }}>
                              ({new Date(pred.lastChecked).toLocaleTimeString('tr-TR')})
                            </span>
                          )}
                        </div>
                      )}

                      {/* LONG Reason Explanation */}
                      {isPinnedLong && pred.longReason && (
                        <div
                          style={{
                            padding: '10px 14px',
                            borderRadius: '8px',
                            background: 'rgba(16, 185, 129, 0.1)',
                            border: '1px solid rgba(16, 185, 129, 0.3)',
                            fontSize: '12px',
                            color: '#10B981',
                            fontWeight: '600',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                          }}
                        >
                          <span style={{ fontSize: '14px' }}>💡</span>
                          <div>
                            <span style={{ opacity: 0.8, marginRight: '6px' }}>Neden LONG:</span>
                            {pred.longReason}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </main>

        {/* Notification Panel (Bildirim Merkezi) */}
        {showNotifications && (
          <div
            style={{
              position: 'fixed',
              top: '80px',
              right: '24px',
              width: '400px',
              maxHeight: '600px',
              background: 'rgba(10, 10, 10, 0.95)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(236, 72, 153, 0.3)',
              borderRadius: '20px',
              padding: '24px',
              zIndex: 1000,
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5), 0 0 40px rgba(236, 72, 153, 0.2)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
              <h3 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                🚀 LONG Fırsatları
              </h3>
              <button
                onClick={() => setShowNotifications(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '8px 12px',
                  color: '#FFFFFF',
                  cursor: 'pointer',
                  fontSize: '18px',
                }}
              >
                ✕
              </button>
            </div>

            {notifications.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: 'rgba(255, 255, 255, 0.4)' }}>
                <div style={{ fontSize: '48px', marginBottom: '12px' }}>🔕</div>
                <div style={{ fontSize: '14px' }}>Henüz LONG bildirimi yok</div>
                <div style={{ fontSize: '12px', marginTop: '8px' }}>
                  3+ AI sistemi aynı yönde sinyal verdiğinde bildirilecek
                </div>
              </div>
            ) : (
              <div style={{ maxHeight: '500px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {notifications.map((notif) => (
                  <div
                    key={notif.id}
                    style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${notif.priority === 'YÜKSEK' ? 'rgba(236, 72, 153, 0.3)' : 'rgba(139, 92, 246, 0.3)'}`,
                      borderRadius: '16px',
                      padding: '16px',
                      animation: 'slideIn 0.3s ease-out',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontSize: '16px',
                            fontWeight: '700',
                            color: '#FFFFFF',
                            marginBottom: '4px',
                          }}
                        >
                          {notif.symbol}
                        </div>
                        <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                          {new Date(notif.timestamp).toLocaleTimeString('tr-TR')}
                        </div>
                      </div>
                      <div
                        style={{
                          padding: '4px 12px',
                          borderRadius: '8px',
                          fontSize: '11px',
                          fontWeight: '700',
                          background: notif.priority === 'YÜKSEK' ? 'rgba(236, 72, 153, 0.2)' : 'rgba(139, 92, 246, 0.2)',
                          color: notif.priority === 'YÜKSEK' ? '#EC4899' : '#8B5CF6',
                        }}
                      >
                        {notif.priority}
                      </div>
                    </div>

                    <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.8)', marginBottom: '12px' }}>
                      {notif.message}
                    </div>

                    <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginBottom: '4px' }}>
                          Güven Skoru
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#10B981' }}>
                          %{notif.confidence.toFixed(1)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginBottom: '4px' }}>
                          Konsensüs
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#EC4899' }}>
                          {notif.consensus_count} AI
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginBottom: '4px' }}>
                          Fiyat
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                          ${notif.price.toFixed(2)}
                        </div>
                      </div>
                    </div>

                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.4)', lineHeight: '1.5' }}>
                      {notif.details}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {aiAssistantOpen && <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />}

        {/* Animation Keyframes */}
        <style jsx global>{`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }

          @keyframes slideIn {
            from {
              opacity: 0;
              transform: translateY(-10px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}</style>
      </div>
    </PWAProvider>
  );
}
