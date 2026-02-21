/**
 * 🤖 LIVE OTONOM SİSTEM PANELİ
 *
 * Real-time olarak otonom sistemin durumunu gösterir.
 * SSE stream ile sürekli güncellenir.
 *
 * Kullanım:
 * ```tsx
 * <LiveAutonomousPanel />
 * ```
 */

'use client';

import { useState } from 'react';
import { useAutonomousSystem } from '@/hooks/useAutonomousSystem';

export function LiveAutonomousPanel() {
  const { isConnected, health, queueStatus, heartbeat, error, reconnect } = useAutonomousSystem();
  const [isExpanded, setIsExpanded] = useState(false);

  // Uptime formatter
  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  // Status color helper
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return '#00ff00';
      case 'unhealthy':
        return '#ff0000';
      default:
        return '#666';
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 9999,
        background: '#111',
        border: `2px solid ${isConnected ? '#00ff00' : '#ff0000'}`,
        borderRadius: '12px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
        minWidth: '280px',
        maxWidth: isExpanded ? '500px' : '280px',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Header - Always Visible */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          padding: '12px 16px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: isExpanded ? '1px solid #333' : 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Connection indicator */}
          <div
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: isConnected ? '#00ff00' : '#ff0000',
              boxShadow: `0 0 10px ${isConnected ? '#00ff00' : '#ff0000'}`,
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }}
          />
          <span style={{ fontSize: '14px', fontWeight: '600', color: '#fff' }}>🤖 Otonom Sistem</span>
        </div>
        <span style={{ fontSize: '18px', color: '#666' }}>{isExpanded ? '▼' : '▲'}</span>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div style={{ padding: '16px' }}>
          {/* Connection Status */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '6px', fontWeight: '600' }}>BAĞLANTI DURUMU</div>
            <div
              style={{
                fontSize: '13px',
                color: isConnected ? '#00ff00' : '#ff0000',
                fontWeight: '600',
              }}
            >
              {isConnected ? '✅ Bağlı' : '❌ Bağlantı Kesildi'}
            </div>
            {error && (
              <div style={{ fontSize: '11px', color: '#ff6600', marginTop: '4px' }}>⚠️ {error}</div>
            )}
            {!isConnected && (
              <button
                onClick={reconnect}
                style={{
                  marginTop: '8px',
                  padding: '6px 12px',
                  background: '#00ff00',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#000',
                  fontSize: '11px',
                  fontWeight: '600',
                  cursor: 'pointer',
                }}
              >
                🔄 Yeniden Bağlan
              </button>
            )}
          </div>

          {/* Health Status */}
          {health && (
            <>
              <div style={{ marginBottom: '12px' }}>
                <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', fontWeight: '600' }}>SERVİS SAĞLIĞI</div>

                {/* Redis */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontSize: '12px', color: '#ccc' }}>🔴 Redis</span>
                  <span
                    style={{
                      fontSize: '12px',
                      color: getStatusColor(health.redis.status),
                      fontWeight: '600',
                    }}
                  >
                    {health.redis.status === 'healthy' ? `✓ ${health.redis.latency}ms` : '✗ Unhealthy'}
                  </span>
                </div>

                {/* Queue */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontSize: '12px', color: '#ccc' }}>📦 Queue</span>
                  <span
                    style={{
                      fontSize: '12px',
                      color: getStatusColor(health.queue.status),
                      fontWeight: '600',
                    }}
                  >
                    {health.queue.status === 'healthy' ? '✓ Active' : '✗ Unhealthy'}
                  </span>
                </div>

                {/* Cron */}
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: '#ccc' }}>⏰ Cron Jobs</span>
                  <span
                    style={{
                      fontSize: '12px',
                      color: getStatusColor(health.cron.status),
                      fontWeight: '600',
                    }}
                  >
                    {health.cron.totalJobs ? `✓ ${health.cron.totalJobs} jobs` : '✗ Unhealthy'}
                  </span>
                </div>
              </div>
            </>
          )}

          {/* Queue Stats */}
          {queueStatus && (
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '8px', fontWeight: '600' }}>QUEUE İSTATİSTİKLERİ</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: '#666' }}>Aktif</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#00ff00' }}>{queueStatus.active}</div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: '#666' }}>Bekliyor</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#ffff00' }}>{queueStatus.waiting}</div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: '#666' }}>Tamamlandı</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#00bfff' }}>{queueStatus.completed}</div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: '#666' }}>Başarısız</div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#ff0000' }}>{queueStatus.failed}</div>
                </div>
              </div>
            </div>
          )}

          {/* Uptime */}
          {heartbeat && (
            <div>
              <div style={{ fontSize: '11px', color: '#666', marginBottom: '4px' }}>UPTIME</div>
              <div style={{ fontSize: '13px', color: '#00bfff', fontWeight: '600' }}>{formatUptime(heartbeat.uptime)}</div>
            </div>
          )}

          {/* Last Update */}
          {health && (
            <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #333' }}>
              <div style={{ fontSize: '10px', color: '#666' }}>
                Son güncelleme: {new Date(health.timestamp).toLocaleTimeString('tr-TR')}
              </div>
            </div>
          )}
        </div>
      )}

      {/* CSS Animation */}
      <style jsx>{`
        @keyframes pulse {
          0%,
          100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}
