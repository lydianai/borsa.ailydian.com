/**
 * CRITICAL NEWS ALERT BANNER
 *
 * Kritik haber uyarılarını tüm sayfaların üstünde gösteren banner
 *
 * Özellikler:
 * - Auto-refresh (30 saniye)
 * - Dismiss fonksiyonu
 * - Pause durumu gösterimi
 * - Otomatik aksiyonlar listesi
 * - Severity-based renklendirme
 */

'use client';

import { useState, useEffect } from 'react';
import type { CriticalNewsAlert } from '@/types/news-risk';

interface NewsRiskAlertsResponse {
  success: boolean;
  data: {
    activeAlerts: CriticalNewsAlert[];
    pauseState: {
      globalPause: boolean;
      pausedSymbols: Array<{ symbol: string; reason: string; endsAt: string }>;
      pauseEndsAt: string | null;
      reason: string | null;
    };
    systemEnabled: boolean;
  };
}

export default function CriticalNewsAlertBanner() {
  const [alerts, setAlerts] = useState<CriticalNewsAlert[]>([]);
  const [pauseState, setPauseState] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Fetch alerts
  const fetchAlerts = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch('/api/news-risk-alerts', {
        cache: 'no-store',
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      // Eğer endpoint bulunamadıysa (404), sessizce başarısız ol
      if (response.status === 404) {
        console.warn('[CriticalAlertBanner] API endpoint bulunamadı');
        return;
      }

      // Diğer HTTP hataları için de devam et
      if (!response.ok) {
        console.warn(`[CriticalAlertBanner] API error: ${response.status}`);
        return;
      }

      const data: NewsRiskAlertsResponse = await response.json();

      if (data.success) {
        setAlerts(data.data.activeAlerts || []);
        setPauseState(data.data.pauseState);
      }
    } catch (error: any) {
      // Timeout veya network hatalarını logla ama UI'ı bozma
      if (error.name === 'AbortError') {
        console.warn('[CriticalAlertBanner] Request timeout');
      } else {
        console.warn('[CriticalAlertBanner] Fetch error:', error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  // Dismiss alert
  const dismissAlert = async (alertId: string) => {
    try {
      await fetch('/api/news-risk-alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'dismiss', alertId }),
      });

      // Local state'ten kaldır
      setAlerts(prev => prev.filter(a => a.id !== alertId));
    } catch (error) {
      console.error('[CriticalAlertBanner] Dismiss error:', error);
    }
  };

  // Auto-refresh (30 saniye)
  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  // Eğer alert yoksa hiçbir şey gösterme
  if (loading || alerts.length === 0) {
    return null;
  }

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999 }}>
      {/* Aktif Alertler */}
      {alerts.map(alert => {
        const severityColors = {
          critical: { bg: '#dc2626', border: '#ef4444', icon: '🔴' },
          high: { bg: '#ea580c', border: '#f97316', icon: '🟠' },
          medium: { bg: '#ca8a04', border: '#eab308', icon: '🟡' },
        };

        const colors = severityColors[alert.severity];

        return (
          <div
            key={alert.id}
            style={{
              background: colors.bg,
              borderBottom: `2px solid ${colors.border}`,
              padding: '12px 24px',
              color: 'white',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '16px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            }}
          >
            {/* Left: Icon + Message */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
              <span style={{ fontSize: '24px' }}>{colors.icon}</span>

              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '14px', fontWeight: '700', marginBottom: '4px' }}>
                  KRİTİK HABER UYARISI - {alert.severity.toUpperCase()}
                </div>
                <div style={{ fontSize: '13px', opacity: 0.95 }}>
                  {alert.news.titleTR}
                </div>

                {/* Otomatik Aksiyonlar */}
                {(alert.actionsExecuted.pausedEntries ||
                  alert.actionsExecuted.reducedPositions) && (
                  <div
                    style={{
                      marginTop: '8px',
                      display: 'flex',
                      gap: '12px',
                      fontSize: '12px',
                      opacity: 0.9,
                    }}
                  >
                    {alert.actionsExecuted.pausedEntries && (
                      <span>⏸️ Yeni girişler duraklatıldı</span>
                    )}
                    {alert.actionsExecuted.reducedPositions && (
                      <span>📉 Pozisyonlar azaltıldı</span>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Time + Close */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ textAlign: 'right', fontSize: '11px', opacity: 0.8 }}>
                <div>
                  {new Date(alert.expiresAt).toLocaleTimeString('tr-TR', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}{' '}
                  'e kadar
                </div>
                <div style={{ marginTop: '2px' }}>
                  {alert.affectedSymbols.length > 0
                    ? alert.affectedSymbols.join(', ')
                    : 'Genel'}
                </div>
              </div>

              <button
                onClick={() => dismissAlert(alert.id)}
                style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '4px',
                  padding: '4px 8px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '18px',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.3)';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.2)';
                }}
              >
                ✕
              </button>
            </div>
          </div>
        );
      })}

      {/* Global Pause Banner */}
      {pauseState?.globalPause && (
        <div
          style={{
            background: 'linear-gradient(135deg, #7c3aed 0%, #6366f1 100%)',
            borderBottom: '2px solid #8b5cf6',
            padding: '10px 24px',
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            fontSize: '13px',
            fontWeight: '600',
          }}
        >
          <span>⏸️</span>
          <span>GLOBAL PAUSE AKTİF - Tüm yeni girişler duraklatıldı</span>
          {pauseState.pauseEndsAt && (
            <span style={{ opacity: 0.9 }}>
              ({new Date(pauseState.pauseEndsAt).toLocaleTimeString('tr-TR')} 'e kadar)
            </span>
          )}
        </div>
      )}
    </div>
  );
}
