'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

interface PipelineStage {
  status: 'pending' | 'running' | 'completed' | 'error';
  duration: number;
  last_error: string | null;
}

interface PipelineStatus {
  status: 'idle' | 'running' | 'completed' | 'error';
  current_stage: string | null;
  last_run: string | null;
  run_count: number;
  success_count: number;
  error_count: number;
  avg_duration: number;
  stages: {
    [key: string]: PipelineStage;
  };
}

const STAGE_NAMES: { [key: string]: string } = {
  data_fetch: 'Veri Toplama',
  technical_analysis: 'Teknik Analiz',
  feature_extraction: 'Özellik Çıkarımı',
  ai_prediction: 'AI Tahmin',
  risk_assessment: 'Risk Değerlendirme',
  signal_generation: 'Sinyal Üretimi',
  storage: 'Veri Saklama'
};

const STAGE_ICONS: { [key: string]: any } = {
  data_fetch: Icons.Database,
  technical_analysis: Icons.LineChart,
  feature_extraction: Icons.Boxes,
  ai_prediction: Icons.Brain,
  risk_assessment: Icons.Shield,
  signal_generation: Icons.Zap,
  storage: Icons.Save
};

const STAGE_COLORS: { [key: string]: string } = {
  data_fetch: '#3B82F6',
  technical_analysis: '#8B5CF6',
  feature_extraction: '#F59E0B',
  ai_prediction: '#EC4899',
  risk_assessment: '#EF4444',
  signal_generation: '#10B981',
  storage: '#06B6D4'
};

export default function PipelineMonitor({ onSave: _onSave }: any) {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000); // Refresh every 2s
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/pipeline/status');
      const data = await response.json();

      if (data.success) {
        setStatus(data.data);
        setError(null);
      } else {
        setError(data.error || 'Failed to fetch status');
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const startPipeline = async () => {
    setStarting(true);
    try {
      const response = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: ['BTCUSDT', 'ETHUSDT'] })
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || 'Failed to start pipeline');
      } else {
        setError(null);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setStarting(false);
    }
  };

  if (loading) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-loading">
          {(Icons.Loader2 as any)({ className: "animate-spin", size: 32, style: { color: '#fff' } })}
          <p style={{ color: '#fff', marginTop: '16px' }}>Pipeline durumu yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">
          Pipeline servisi kullanılamıyor: {error}
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#F59E0B';
      case 'completed': return '#10B981';
      case 'error': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return Icons.Play;
      case 'completed': return Icons.CheckCircle2;
      case 'error': return Icons.XCircle;
      default: return Icons.Clock;
    }
  };

  const StatusIcon = getStatusIcon(status.status);

  return (
    <div className="settings-content-wrapper">
      {/* Control Panel */}
      <div className="settings-premium-card" style={{ marginBottom: '24px' }}>
        <div className="settings-card-header">
          {(Icons.Activity as any)({ style: { color: '#8B5CF6' }, size: 24 })}
          <h3>Pipeline Kontrol Paneli</h3>
        </div>
        <div className="settings-card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '16px' }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                {(StatusIcon as any)({ style: { color: getStatusColor(status.status) }, size: 32 })}
                <div>
                  <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600', textTransform: 'capitalize' }}>
                    {status.status === 'idle' ? 'Hazır' :
                     status.status === 'running' ? 'Çalışıyor' :
                     status.status === 'completed' ? 'Tamamlandı' : 'Hata'}
                  </div>
                  {status.current_stage && (
                    <div style={{ color: '#9CA3AF', fontSize: '14px' }}>
                      {STAGE_NAMES[status.current_stage]}
                    </div>
                  )}
                </div>
              </div>
            </div>
            <button
              onClick={startPipeline}
              disabled={status.status === 'running' || starting}
              style={{
                padding: '12px 24px',
                backgroundColor: status.status === 'running' ? '#374151' : '#10B981',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: status.status === 'running' ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                opacity: status.status === 'running' ? 0.5 : 1
              }}
            >
              {starting || status.status === 'running' ? (
                <>
                  {(Icons.Loader2 as any)({ className: "animate-spin", size: 18 })}
                  Çalışıyor...
                </>
              ) : (
                <>
                  {(Icons.Play as any)({ size: 18 })}
                  Pipeline Başlat
                </>
              )}
            </button>
          </div>

          {/* Stats Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '16px' }}>
            <div style={{ backgroundColor: '#000', border: '1px solid #222', borderRadius: '8px', padding: '12px' }}>
              <div style={{ color: '#9CA3AF', fontSize: '12px', marginBottom: '4px' }}>Toplam Çalıştırma</div>
              <div style={{ color: '#fff', fontSize: '20px', fontWeight: '700' }}>{status.run_count}</div>
            </div>
            <div style={{ backgroundColor: '#000', border: '1px solid #222', borderRadius: '8px', padding: '12px' }}>
              <div style={{ color: '#9CA3AF', fontSize: '12px', marginBottom: '4px' }}>Başarılı</div>
              <div style={{ color: '#10B981', fontSize: '20px', fontWeight: '700' }}>{status.success_count}</div>
            </div>
            <div style={{ backgroundColor: '#000', border: '1px solid #222', borderRadius: '8px', padding: '12px' }}>
              <div style={{ color: '#9CA3AF', fontSize: '12px', marginBottom: '4px' }}>Hatalı</div>
              <div style={{ color: '#EF4444', fontSize: '20px', fontWeight: '700' }}>{status.error_count}</div>
            </div>
            <div style={{ backgroundColor: '#000', border: '1px solid #222', borderRadius: '8px', padding: '12px' }}>
              <div style={{ color: '#9CA3AF', fontSize: '12px', marginBottom: '4px' }}>Ort. Süre</div>
              <div style={{ color: '#fff', fontSize: '20px', fontWeight: '700' }}>
                {status.avg_duration ? `${status.avg_duration.toFixed(1)}s` : '-'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          {(Icons.GitBranch as any)({ style: { color: '#06B6D4' }, size: 24 })}
          <h3>Pipeline Aşamaları</h3>
        </div>
        <div className="settings-card-body">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Object.entries(status.stages).map(([key, stage]) => {
              const StageIcon = STAGE_ICONS[key];
              const isActive = status.current_stage === key;
              const stageColor = STAGE_COLORS[key];

              return (
                <div
                  key={key}
                  style={{
                    backgroundColor: isActive ? '#1F2937' : '#000',
                    border: `2px solid ${isActive ? stageColor : '#222'}`,
                    borderRadius: '12px',
                    padding: '16px',
                    transition: 'all 0.3s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    {/* Icon */}
                    <div style={{
                      width: '48px',
                      height: '48px',
                      backgroundColor: isActive ? stageColor : '#1F2937',
                      borderRadius: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}>
                      <StageIcon style={{ color: '#fff' }} size={24} />
                    </div>

                    {/* Info */}
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <span style={{ color: '#fff', fontSize: '16px', fontWeight: '600' }}>
                          {STAGE_NAMES[key]}
                        </span>
                        {stage.status === 'running' && (Icons.Loader2 as any)({ className: "animate-spin", size: 16, style: { color: stageColor } })}
                        {stage.status === 'completed' && (Icons.CheckCircle2 as any)({ size: 16, style: { color: '#10B981' } })}
                        {stage.status === 'error' && (Icons.XCircle as any)({ size: 16, style: { color: '#EF4444' } })}
                      </div>
                      {stage.duration > 0 && (
                        <div style={{ color: '#9CA3AF', fontSize: '13px' }}>
                          Süre: {stage.duration.toFixed(2)}s
                        </div>
                      )}
                      {stage.last_error && (
                        <div style={{ color: '#EF4444', fontSize: '12px', marginTop: '4px' }}>
                          Hata: {stage.last_error}
                        </div>
                      )}
                    </div>

                    {/* Status Badge */}
                    <div style={{
                      padding: '6px 12px',
                      backgroundColor: getStatusColor(stage.status),
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: '600',
                      color: '#fff',
                      textTransform: 'capitalize'
                    }}>
                      {stage.status === 'pending' ? 'Bekliyor' :
                       stage.status === 'running' ? 'Çalışıyor' :
                       stage.status === 'completed' ? 'Tamamlandı' : 'Hata'}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  {isActive && (
                    <div style={{
                      marginTop: '12px',
                      height: '4px',
                      backgroundColor: '#1F2937',
                      borderRadius: '2px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        height: '100%',
                        backgroundColor: stageColor,
                        animation: 'progress 2s ease-in-out infinite',
                        width: '50%'
                      }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {error && (
        <div className="settings-alert-error" style={{ marginTop: '16px' }}>
          {error}
        </div>
      )}

      <style jsx>{`
        @keyframes progress {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(300%); }
        }
      `}</style>
    </div>
  );
}
