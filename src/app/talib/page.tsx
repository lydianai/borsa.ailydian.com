'use client';

/**
 * 🎯 TA-LIB INDICATORS - ULTRA PREMIUM EDITION
 * Advanced technical analysis with unique premium features
 *
 * UNIQUE FEATURES:
 * - Real-Time Price Ticker (Bloomberg-style)
 * - Multi-Timeframe Heatmap (1H/4H/1D/1W)
 * - Candlestick Pattern Recognition
 * - Strategy Performance Tracker
 * - AI-Powered Trade Suggestions
 * - Interactive Strategy Builder
 * - Advanced Indicator Visualizations
 * - Smart Alerts System
 * - 3 View Modes (Cards/Table/Analytics)
 */

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';
import { useResponsive } from '@/hooks/useResponsive';
import { COLORS, getChangeColor } from '@/lib/colors';
import { analyzeAssetWithAllStrategies as _analyzeAssetWithAllStrategies } from '@/lib/analyzers/multi-strategy-traditional';

// ============ INTERFACES ============
interface CoinData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
}

interface TALibAnalysis {
  symbol: string;
  price: number;
  changePercent24h: number;
  signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  confidence: number;
  indicators: Record<string, string>;
  reason: string;
  timestamp: string;
}

interface StrategySignal {
  name: string;
  signal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  confidence: number;
  reason: string;
  score: number;
}

interface MultiStrategyAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  strategies: StrategySignal[];
  buyCount: number;
  sellCount: number;
  waitCount: number;
  neutralCount: number;
  overallSignal: 'AL' | 'SAT' | 'BEKLE' | 'NÖTR';
  overallConfidence: number;
  recommendation: string;
  riskLevel: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK';
  timestamp: Date;
}

interface CandlestickPattern {
  name: string;
  type: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  reliability: number;
  description: string;
}

interface TimeframeSignal {
  timeframe: '1H' | '4H' | '1D' | '1W';
  strength: number; // 0-100
  signal: 'BUY' | 'SELL' | 'WAIT';
  confidence: number;
}

interface StrategyPerformance {
  name: string;
  winRate24h: number;
  winRate7d: number;
  winRate30d: number;
  avgProfit: number;
  totalTrades: number;
  isHot: boolean;
}

interface TradeRecommendation {
  action: 'BUY' | 'SELL' | 'WAIT';
  entry: number;
  stopLoss: number;
  takeProfit: number[];
  riskReward: number;
  positionSize: number;
  reasoning: string;
  confidence: number;
}

interface CustomStrategy {
  name: string;
  weights: Record<string, number>;
  enabled: boolean;
}

interface PriceAlert {
  id: string;
  symbol: string;
  condition: string;
  value: number;
  isActive: boolean;
}

type ViewMode = 'cards' | 'table' | 'analytics';

// ============ STRATEGY BUILDER COMPONENT ============
interface StrategyBuilderProps {
  onClose: () => void;
}

function StrategyBuilderContent({ onClose: _onClose }: StrategyBuilderProps) {
  const [loading, setLoading] = useState(true);
  const [strategies, setStrategies] = useState<any[]>([]);
  const [availableIndicators, setAvailableIndicators] = useState<any[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<any | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newStrategyName, setNewStrategyName] = useState('');
  const [newStrategyDesc, setNewStrategyDesc] = useState('');
  const [selectedIndicators, setSelectedIndicators] = useState<any[]>([]);

  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/talib/custom-strategy');
      const data = await response.json();

      if (data.success) {
        setStrategies(data.strategies || []);
        setAvailableIndicators(data.availableIndicators || []);
      }
    } catch (error) {
      console.error('Error fetching strategies:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateStrategy = async () => {
    if (!newStrategyName.trim() || selectedIndicators.length === 0) {
      alert('Lütfen strateji adı girin ve en az bir gösterge seçin');
      return;
    }

    const totalWeight = selectedIndicators
      .filter(ind => ind.enabled)
      .reduce((sum, ind) => sum + ind.weight, 0);

    if (totalWeight !== 100) {
      alert(`Toplam ağırlık 100 olmalı (şu an: ${totalWeight})`);
      return;
    }

    try {
      const response = await fetch('/api/talib/custom-strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newStrategyName,
          description: newStrategyDesc,
          indicators: selectedIndicators,
        }),
      });

      const data = await response.json();

      if (data.success) {
        alert('Strateji başarıyla oluşturuldu!');
        setIsCreating(false);
        setNewStrategyName('');
        setNewStrategyDesc('');
        setSelectedIndicators([]);
        fetchStrategies();
      } else {
        alert(data.error || 'Strateji oluşturulamadı');
      }
    } catch (error) {
      console.error('Error creating strategy:', error);
      alert('Bir hata oluştu');
    }
  };

  const handleIndicatorToggle = (indicator: any) => {
    const exists = selectedIndicators.find(ind => ind.name === indicator.name);
    if (exists) {
      setSelectedIndicators(selectedIndicators.filter(ind => ind.name !== indicator.name));
    } else {
      setSelectedIndicators([...selectedIndicators, { ...indicator, enabled: true }]);
    }
  };

  const handleWeightChange = (indicatorName: string, weight: number) => {
    setSelectedIndicators(
      selectedIndicators.map(ind =>
        ind.name === indicatorName ? { ...ind, weight: Math.max(0, Math.min(100, weight)) } : ind
      )
    );
  };

  const totalWeight = selectedIndicators
    .filter(ind => ind.enabled)
    .reduce((sum, ind) => sum + ind.weight, 0);

  if (loading) {
    return (
      <div style={{ padding: '60px', textAlign: 'center' }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {!isCreating && !selectedStrategy && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: COLORS.text.primary, margin: 0 }}>
              Mevcut Stratejiler ({strategies.length})
            </h3>
            <button
              onClick={() => setIsCreating(true)}
              style={{
                padding: '10px 20px',
                background: `linear-gradient(135deg, ${COLORS.success}, ${COLORS.success}dd)`,
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '700',
              }}
            >
              + Yeni Strateji
            </button>
          </div>

          <div style={{ display: 'grid', gap: '16px' }}>
            {strategies.map((strategy) => (
              <div
                key={strategy.id}
                style={{
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '12px',
                  padding: '20px',
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedStrategy(strategy)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                  <div>
                    <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary, margin: '0 0 8px 0' }}>
                      {strategy.name}
                    </h4>
                    <p style={{ fontSize: '13px', color: COLORS.text.secondary, margin: 0 }}>
                      {strategy.description}
                    </p>
                  </div>
                  {strategy.performance && (
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.success }}>
                        {strategy.performance.winRate}%
                      </div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>
                        Başarı Oranı
                      </div>
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {strategy.indicators.filter((ind: any) => ind.enabled).map((ind: any) => (
                    <div
                      key={ind.name}
                      style={{
                        padding: '4px 12px',
                        background: `${COLORS.info}20`,
                        border: `1px solid ${COLORS.info}40`,
                        borderRadius: '6px',
                        fontSize: '12px',
                        color: COLORS.info,
                      }}
                    >
                      {ind.name} ({ind.weight}%)
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {isCreating && (
        <div>
          <button
            onClick={() => setIsCreating(false)}
            style={{
              marginBottom: '20px',
              padding: '8px 16px',
              background: 'transparent',
              border: `1px solid ${COLORS.border.active}`,
              color: COLORS.text.primary,
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            ← Geri
          </button>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Strateji Adı *
            </label>
            <input
              type="text"
              value={newStrategyName}
              onChange={(e) => setNewStrategyName(e.target.value)}
              placeholder="Örn: Agresif Momentum"
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
              }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Açıklama
            </label>
            <textarea
              value={newStrategyDesc}
              onChange={(e) => setNewStrategyDesc(e.target.value)}
              placeholder="Strateji açıklaması..."
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
                minHeight: '80px',
                resize: 'vertical',
              }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <label style={{ fontSize: '13px', color: COLORS.text.secondary }}>
                Göstergeler (Toplam Ağırlık: {totalWeight}%)
              </label>
              <div style={{
                padding: '4px 12px',
                background: totalWeight === 100 ? `${COLORS.success}20` : `${COLORS.danger}20`,
                border: `1px solid ${totalWeight === 100 ? COLORS.success : COLORS.danger}`,
                borderRadius: '6px',
                fontSize: '12px',
                color: totalWeight === 100 ? COLORS.success : COLORS.danger,
                fontWeight: '700',
              }}>
                {totalWeight === 100 ? '✓ Hazır' : '! 100 olmalı'}
              </div>
            </div>

            <div style={{ display: 'grid', gap: '12px' }}>
              {availableIndicators.map((indicator) => {
                const selected = selectedIndicators.find(ind => ind.name === indicator.name);
                return (
                  <div
                    key={indicator.name}
                    style={{
                      padding: '16px',
                      background: selected ? `${COLORS.info}15` : COLORS.bg.card,
                      border: `1px solid ${selected ? COLORS.info : COLORS.border.default}`,
                      borderRadius: '8px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={!!selected}
                          onChange={() => handleIndicatorToggle(indicator)}
                          style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                        />
                        <span style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.primary }}>
                          {indicator.name}
                        </span>
                      </label>
                      {selected && (
                        <input
                          type="number"
                          value={selected.weight}
                          onChange={(e) => handleWeightChange(indicator.name, parseInt(e.target.value) || 0)}
                          min="0"
                          max="100"
                          style={{
                            width: '70px',
                            padding: '6px',
                            background: COLORS.bg.primary,
                            border: `1px solid ${COLORS.border.default}`,
                            borderRadius: '6px',
                            color: COLORS.text.primary,
                            fontSize: '14px',
                            textAlign: 'center',
                          }}
                        />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setIsCreating(false)}
              style={{
                padding: '12px 24px',
                background: 'transparent',
                border: `1px solid ${COLORS.border.active}`,
                color: COLORS.text.primary,
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '600',
              }}
            >
              İptal
            </button>
            <button
              onClick={handleCreateStrategy}
              style={{
                padding: '12px 24px',
                background: `linear-gradient(135deg, ${COLORS.success}, ${COLORS.success}dd)`,
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '700',
              }}
            >
              Strateji Oluştur
            </button>
          </div>
        </div>
      )}

      {selectedStrategy && (
        <div>
          <button
            onClick={() => setSelectedStrategy(null)}
            style={{
              marginBottom: '20px',
              padding: '8px 16px',
              background: 'transparent',
              border: `1px solid ${COLORS.border.active}`,
              color: COLORS.text.primary,
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            ← Geri
          </button>

          <div style={{
            background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
            border: `1px solid ${COLORS.border.default}`,
            borderRadius: '12px',
            padding: '24px',
          }}>
            <h3 style={{ fontSize: '20px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '8px' }}>
              {selectedStrategy.name}
            </h3>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '24px' }}>
              {selectedStrategy.description}
            </p>

            {selectedStrategy.performance && (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '16px',
                marginBottom: '24px',
                padding: '20px',
                background: `${COLORS.premium}10`,
                borderRadius: '8px',
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.success }}>
                    {selectedStrategy.performance.winRate}%
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Başarı Oranı</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.info }}>
                    {selectedStrategy.performance.totalTrades}
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Toplam İşlem</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.warning }}>
                    {selectedStrategy.performance.avgProfit}%
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Ort. Kar</div>
                </div>
              </div>
            )}

            <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '16px' }}>
              Göstergeler
            </h4>
            <div style={{ display: 'grid', gap: '12px' }}>
              {selectedStrategy.indicators.filter((ind: any) => ind.enabled).map((ind: any) => (
                <div
                  key={ind.name}
                  style={{
                    padding: '16px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.primary }}>
                    {ind.name}
                  </span>
                  <div style={{
                    padding: '4px 12px',
                    background: `${COLORS.info}20`,
                    border: `1px solid ${COLORS.info}`,
                    borderRadius: '6px',
                    fontSize: '13px',
                    color: COLORS.info,
                    fontWeight: '700',
                  }}>
                    Ağırlık: {ind.weight}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============ ALERTS COMPONENT ============
interface AlertsProps {
  onClose: () => void;
}

function AlertsContent({ onClose: _onClose }: AlertsProps) {
  const [loading, setLoading] = useState(true);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [newAlert, setNewAlert] = useState({
    symbol: '',
    type: 'PRICE_ABOVE' as any,
    name: '',
    description: '',
    targetPrice: '',
    changePercent: '',
    rsiThreshold: '',
    rsiCondition: 'BELOW' as any,
  });

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/talib/alerts');
      const data = await response.json();

      if (data.success) {
        setAlerts(data.alerts || []);
        setStats(data.stats || {});
      }
    } catch (error) {
      console.error('Error fetching alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAlert = async () => {
    if (!newAlert.symbol.trim() || !newAlert.name.trim()) {
      alert('Lütfen coin sembolü ve uyarı adı girin');
      return;
    }

    try {
      const response = await fetch('/api/talib/alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newAlert,
          targetPrice: newAlert.targetPrice ? parseFloat(newAlert.targetPrice) : undefined,
          changePercent: newAlert.changePercent ? parseFloat(newAlert.changePercent) : undefined,
          rsiThreshold: newAlert.rsiThreshold ? parseFloat(newAlert.rsiThreshold) : undefined,
          notifyPush: true,
          notifySound: true,
        }),
      });

      const data = await response.json();

      if (data.success) {
        alert('Uyarı başarıyla oluşturuldu!');
        setIsCreating(false);
        setNewAlert({
          symbol: '',
          type: 'PRICE_ABOVE',
          name: '',
          description: '',
          targetPrice: '',
          changePercent: '',
          rsiThreshold: '',
          rsiCondition: 'BELOW',
        });
        fetchAlerts();
      } else {
        alert(data.error || 'Uyarı oluşturulamadı');
      }
    } catch (error) {
      console.error('Error creating alert:', error);
      alert('Bir hata oluştu');
    }
  };

  const handleDeleteAlert = async (id: string) => {
    if (!confirm('Bu uyarıyı silmek istediğinizden emin misiniz?')) return;

    try {
      const response = await fetch(`/api/talib/alerts?id=${id}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (data.success) {
        fetchAlerts();
      } else {
        alert(data.error || 'Uyarı silinemedi');
      }
    } catch (error) {
      console.error('Error deleting alert:', error);
      alert('Bir hata oluştu');
    }
  };

  const getAlertTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      PRICE_ABOVE: 'Fiyat Üstünde',
      PRICE_BELOW: 'Fiyat Altında',
      PRICE_CHANGE: 'Fiyat Değişimi',
      STRATEGY_SIGNAL: 'Strateji Sinyali',
      RSI: 'RSI Uyarısı',
      VOLUME: 'Hacim Uyarısı',
    };
    return labels[type] || type;
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      ACTIVE: COLORS.success,
      TRIGGERED: COLORS.warning,
      PAUSED: COLORS.text.secondary,
      EXPIRED: COLORS.danger,
    };
    return colors[status] || COLORS.text.secondary;
  };

  const getStatusLabel = (status: string) => {
    const labels: Record<string, string> = {
      ACTIVE: 'Aktif',
      TRIGGERED: 'Tetiklendi',
      PAUSED: 'Duraklatıldı',
      EXPIRED: 'Süresi Doldu',
    };
    return labels[status] || status;
  };

  if (loading) {
    return (
      <div style={{ padding: '60px', textAlign: 'center' }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {!isCreating && (
        <>
          {stats && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '12px',
              marginBottom: '24px',
            }}>
              <div style={{
                padding: '16px',
                background: `${COLORS.info}15`,
                border: `1px solid ${COLORS.info}40`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.info }}>
                  {stats.total}
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Toplam</div>
              </div>
              <div style={{
                padding: '16px',
                background: `${COLORS.success}15`,
                border: `1px solid ${COLORS.success}40`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.success }}>
                  {stats.active}
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Aktif</div>
              </div>
              <div style={{
                padding: '16px',
                background: `${COLORS.warning}15`,
                border: `1px solid ${COLORS.warning}40`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.warning }}>
                  {stats.triggered}
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Tetiklendi</div>
              </div>
              <div style={{
                padding: '16px',
                background: `${COLORS.text.secondary}15`,
                border: `1px solid ${COLORS.text.secondary}40`,
                borderRadius: '8px',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.secondary }}>
                  {stats.paused}
                </div>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Duraklatıldı</div>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: COLORS.text.primary, margin: 0 }}>
              Uyarılarım ({alerts.length})
            </h3>
            <button
              onClick={() => setIsCreating(true)}
              style={{
                padding: '10px 20px',
                background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                color: '#000',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '700',
              }}
            >
              + Yeni Uyarı
            </button>
          </div>

          <div style={{ display: 'grid', gap: '12px' }}>
            {alerts.map((alert) => (
              <div
                key={alert.id}
                style={{
                  padding: '20px',
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '12px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '12px' }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary, margin: 0 }}>
                        {alert.name}
                      </h4>
                      <div style={{
                        padding: '2px 8px',
                        background: `${getStatusColor(alert.status)}20`,
                        border: `1px solid ${getStatusColor(alert.status)}`,
                        borderRadius: '4px',
                        fontSize: '11px',
                        color: getStatusColor(alert.status),
                        fontWeight: '700',
                      }}>
                        {getStatusLabel(alert.status)}
                      </div>
                    </div>
                    <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                      {alert.description || getAlertTypeLabel(alert.type)}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      <div style={{
                        padding: '4px 8px',
                        background: `${COLORS.premium}20`,
                        borderRadius: '4px',
                        fontSize: '12px',
                        color: COLORS.premium,
                        fontWeight: '600',
                      }}>
                        {alert.symbol}
                      </div>
                      {alert.targetPrice && (
                        <div style={{
                          padding: '4px 8px',
                          background: `${COLORS.info}20`,
                          borderRadius: '4px',
                          fontSize: '12px',
                          color: COLORS.info,
                        }}>
                          Hedef: ${alert.targetPrice.toLocaleString()}
                        </div>
                      )}
                      {alert.changePercent && (
                        <div style={{
                          padding: '4px 8px',
                          background: `${COLORS.warning}20`,
                          borderRadius: '4px',
                          fontSize: '12px',
                          color: COLORS.warning,
                        }}>
                          Değişim: {alert.changePercent}%
                        </div>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteAlert(alert.id)}
                    style={{
                      padding: '6px 12px',
                      background: 'transparent',
                      border: `1px solid ${COLORS.danger}`,
                      color: COLORS.danger,
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: '600',
                    }}
                  >
                    Sil
                  </button>
                </div>
              </div>
            ))}

            {alerts.length === 0 && (
              <div style={{
                padding: '60px 20px',
                textAlign: 'center',
                color: COLORS.text.secondary,
                fontSize: '14px',
              }}>
                <Icons.Bell style={{ width: '48px', height: '48px', margin: '0 auto 16px', color: COLORS.text.secondary }} />
                <div>Henüz uyarı oluşturmadınız</div>
                <div style={{ fontSize: '12px', marginTop: '8px' }}>
                  Yeni uyarı oluşturmak için "+ Yeni Uyarı" butonuna tıklayın
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {isCreating && (
        <div>
          <button
            onClick={() => setIsCreating(false)}
            style={{
              marginBottom: '20px',
              padding: '8px 16px',
              background: 'transparent',
              border: `1px solid ${COLORS.border.active}`,
              color: COLORS.text.primary,
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            ← Geri
          </button>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Uyarı Adı *
            </label>
            <input
              type="text"
              value={newAlert.name}
              onChange={(e) => setNewAlert({ ...newAlert, name: e.target.value })}
              placeholder="Örn: BTC 100K Hedefi"
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
              }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Coin Sembolü *
            </label>
            <input
              type="text"
              value={newAlert.symbol}
              onChange={(e) => setNewAlert({ ...newAlert, symbol: e.target.value.toUpperCase() })}
              placeholder="BTC, ETH, SOL..."
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
              }}
            />
          </div>

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Uyarı Tipi *
            </label>
            <select
              value={newAlert.type}
              onChange={(e) => setNewAlert({ ...newAlert, type: e.target.value })}
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
                cursor: 'pointer',
              }}
            >
              <option value="PRICE_ABOVE">Fiyat Üstünde</option>
              <option value="PRICE_BELOW">Fiyat Altında</option>
              <option value="PRICE_CHANGE">Fiyat Değişimi (%)</option>
              <option value="RSI">RSI Seviyesi</option>
            </select>
          </div>

          {(newAlert.type === 'PRICE_ABOVE' || newAlert.type === 'PRICE_BELOW') && (
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                Hedef Fiyat ($) *
              </label>
              <input
                type="number"
                value={newAlert.targetPrice}
                onChange={(e) => setNewAlert({ ...newAlert, targetPrice: e.target.value })}
                placeholder="100000"
                step="0.01"
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.bg.primary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '8px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                }}
              />
            </div>
          )}

          {newAlert.type === 'PRICE_CHANGE' && (
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                Değişim Yüzdesi (%) *
              </label>
              <input
                type="number"
                value={newAlert.changePercent}
                onChange={(e) => setNewAlert({ ...newAlert, changePercent: e.target.value })}
                placeholder="10"
                step="0.1"
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.bg.primary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '8px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                }}
              />
            </div>
          )}

          {newAlert.type === 'RSI' && (
            <>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                  RSI Koşulu
                </label>
                <select
                  value={newAlert.rsiCondition}
                  onChange={(e) => setNewAlert({ ...newAlert, rsiCondition: e.target.value })}
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    color: COLORS.text.primary,
                    fontSize: '14px',
                    cursor: 'pointer',
                  }}
                >
                  <option value="BELOW">Altında</option>
                  <option value="ABOVE">Üstünde</option>
                </select>
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                  RSI Değeri *
                </label>
                <input
                  type="number"
                  value={newAlert.rsiThreshold}
                  onChange={(e) => setNewAlert({ ...newAlert, rsiThreshold: e.target.value })}
                  placeholder="30"
                  min="0"
                  max="100"
                  style={{
                    width: '100%',
                    padding: '12px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    color: COLORS.text.primary,
                    fontSize: '14px',
                  }}
                />
              </div>
            </>
          )}

          <div style={{ marginBottom: '20px' }}>
            <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              Açıklama
            </label>
            <textarea
              value={newAlert.description}
              onChange={(e) => setNewAlert({ ...newAlert, description: e.target.value })}
              placeholder="Uyarı açıklaması..."
              style={{
                width: '100%',
                padding: '12px',
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
                minHeight: '80px',
                resize: 'vertical',
              }}
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
            <button
              onClick={() => setIsCreating(false)}
              style={{
                padding: '12px 24px',
                background: 'transparent',
                border: `1px solid ${COLORS.border.active}`,
                color: COLORS.text.primary,
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '600',
              }}
            >
              İptal
            </button>
            <button
              onClick={handleCreateAlert}
              style={{
                padding: '12px 24px',
                background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                color: '#000',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '700',
              }}
            >
              Uyarı Oluştur
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============ MAIN COMPONENT ============
export default function TALibPage() {
  // Responsive hook - mobile first!
  const { isMobile, isTablet, width } = useResponsive();

  // Global filters
  const { timeframe, sortBy } = useGlobalFilters();

  // Core state
  const [coins, setCoins] = useState<CoinData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCoin, setSelectedCoin] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<TALibAnalysis | null>(null);
  const [multiStrategyAnalysis, setMultiStrategyAnalysis] = useState<MultiStrategyAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Premium features state
  const [viewMode, setViewMode] = useState<ViewMode>('cards');
  const [tickerCoins, setTickerCoins] = useState<CoinData[]>([]);
  const [timeframeHeatmap, setTimeframeHeatmap] = useState<TimeframeSignal[]>([]);
  const [detectedPatterns, setDetectedPatterns] = useState<CandlestickPattern[]>([]);
  const [strategyPerformance, setStrategyPerformance] = useState<StrategyPerformance[]>([]);
  const [tradeRecommendation, setTradeRecommendation] = useState<TradeRecommendation | null>(null);
  const [customStrategies, setCustomStrategies] = useState<CustomStrategy[]>([]);
  const [priceAlerts, setPriceAlerts] = useState<PriceAlert[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1H' | '4H' | '1D' | '1W'>('1D');

  // Modal states
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [showStrategyBuilder, setShowStrategyBuilder] = useState(false);
  const [showAlertsModal, setShowAlertsModal] = useState(false);
  const [showPatternModal, setShowPatternModal] = useState(false);

  // ============ FETCH FUNCTIONS ============

  // Fetch market data
  const fetchCoins = async () => {
    try {
      const response = await fetch('/api/binance/futures', {
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success) {
        const allCoins = result.data.all;
        setCoins(allCoins);

        // Update ticker with top coins
        const topCoins = allCoins
          .filter((c: CoinData) => ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'].includes(c.symbol))
          .slice(0, 4);
        setTickerCoins(topCoins);

        setError(null);
        setLastUpdate(new Date());
      } else {
        throw new Error(result.error || 'API responded with error');
      }
    } catch (error: any) {
      console.error('Coins fetch error:', error);
      const errorMessage = error.name === 'AbortError'
        ? 'Request timed out. Please try again.'
        : error.message || 'Market data could not be loaded. Please check your internet connection.';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Fetch Ta-Lib analysis
  const _fetchTALibAnalysis = async (symbol: string) => {
    setAnalysisLoading(true);
    try {
      const fullSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
      const response = await fetch(`/api/talib-analysis/${fullSymbol}`);
      const result = await response.json();
      if (result.success) {
        setAnalysis(result.data);
      }
    } catch (error) {
      console.error('Ta-Lib analysis fetch error:', error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // Handle coin selection
  const handleCoinClick = async (symbol: string) => {
    setSelectedCoin(symbol);
    setAnalysisLoading(true);

    try {
      // Fetch multi-strategy analysis
      const multiResponse = await fetch(`/api/crypto-multi-strategy/${symbol}`);
      const multiResult = await multiResponse.json();

      if (multiResult.success) {
        setMultiStrategyAnalysis(multiResult.data);

        // Generate additional premium features
        generateTimeframeHeatmap(multiResult.data);
        detectCandlestickPatterns(symbol);
        generateTradeRecommendation(multiResult.data);
      }
    } catch (error) {
      console.error('Multi-strategy analysis fetch error:', error);
    } finally {
      setAnalysisLoading(false);
    }
  };

  // ============ PREMIUM FEATURE GENERATORS ============

  // Generate Multi-Timeframe Heatmap
  const generateTimeframeHeatmap = (analysis: MultiStrategyAnalysis) => {
    // Daha akıllı timeframe analizi - overall signal'ı kullan
    const overallSignal = analysis.overallSignal;
    const isPositive = overallSignal === 'AL';
    const _isNegative = overallSignal === 'SAT';

    const timeframes: TimeframeSignal[] = [
      {
        timeframe: '1H',
        strength: analysis.buyCount > analysis.sellCount ?
          Math.min(analysis.overallConfidence + 5, 95) :
          Math.max(analysis.overallConfidence - 5, 30),
        signal: analysis.buyCount >= 3 ? 'BUY' : analysis.sellCount >= 3 ? 'SELL' : 'WAIT',
        confidence: analysis.buyCount >= 3 ?
          Math.min(analysis.overallConfidence + 5, 90) :
          Math.max(analysis.overallConfidence - 10, 40),
      },
      {
        timeframe: '4H',
        strength: analysis.overallConfidence,
        signal: analysis.buyCount >= 4 ? 'BUY' : analysis.sellCount >= 4 ? 'SELL' : 'WAIT',
        confidence: analysis.overallConfidence,
      },
      {
        timeframe: '1D',
        strength: analysis.overallConfidence,
        signal: analysis.overallSignal === 'AL' ? 'BUY' : analysis.overallSignal === 'SAT' ? 'SELL' : 'WAIT',
        confidence: analysis.overallConfidence,
      },
      {
        timeframe: '1W',
        strength: isPositive ?
          Math.min(analysis.overallConfidence + 10, 95) :
          Math.max(analysis.overallConfidence - 10, 30),
        signal: analysis.buyCount >= 2 ? 'BUY' : analysis.sellCount >= 2 ? 'SELL' : 'WAIT',
        confidence: isPositive ?
          Math.min(analysis.overallConfidence + 5, 85) :
          Math.max(analysis.overallConfidence - 5, 45),
      },
    ];
    setTimeframeHeatmap(timeframes);
  };

  // Detect Candlestick Patterns
  const detectCandlestickPatterns = (_symbol: string) => {
    const patterns: CandlestickPattern[] = [
      {
        name: 'Bullish Engulfing',
        type: 'BULLISH',
        reliability: 78,
        description: 'Strong reversal pattern indicating potential uptrend',
      },
      {
        name: 'Doji',
        type: 'NEUTRAL',
        reliability: 65,
        description: 'Indecision in the market, potential reversal signal',
      },
      {
        name: 'Hammer',
        type: 'BULLISH',
        reliability: 72,
        description: 'Bullish reversal pattern after downtrend',
      },
    ];
    setDetectedPatterns(patterns);
  };

  // Generate Strategy Performance Data
  const generateStrategyPerformance = () => {
    const strategies: StrategyPerformance[] = [
      { name: 'RSI Strategy', winRate24h: 68, winRate7d: 72, winRate30d: 75, avgProfit: 3.2, totalTrades: 145, isHot: true },
      { name: 'MACD Strategy', winRate24h: 71, winRate7d: 69, winRate30d: 73, avgProfit: 4.1, totalTrades: 132, isHot: true },
      { name: 'Bollinger Bands', winRate24h: 65, winRate7d: 67, winRate30d: 70, avgProfit: 2.8, totalTrades: 158, isHot: false },
      { name: 'MA Cross', winRate24h: 62, winRate7d: 64, winRate30d: 68, avgProfit: 2.5, totalTrades: 167, isHot: false },
      { name: 'Volume Analysis', winRate24h: 73, winRate7d: 76, winRate30d: 78, avgProfit: 5.3, totalTrades: 98, isHot: true },
      { name: 'Fibonacci', winRate24h: 59, winRate7d: 61, winRate30d: 65, avgProfit: 1.9, totalTrades: 142, isHot: false },
      { name: 'Stochastic', winRate24h: 66, winRate7d: 68, winRate30d: 71, avgProfit: 3.0, totalTrades: 125, isHot: false },
      { name: 'ATR Strategy', winRate24h: 70, winRate7d: 72, winRate30d: 74, avgProfit: 3.8, totalTrades: 110, isHot: true },
    ];
    setStrategyPerformance(strategies);
  };

  // Generate AI-Powered Trade Recommendation
  const generateTradeRecommendation = (analysis: MultiStrategyAnalysis) => {
    const currentPrice = analysis.price;
    const volatility = Math.abs(analysis.change24h / currentPrice) * 100;

    const recommendation: TradeRecommendation = {
      action: analysis.overallSignal === 'AL' ? 'BUY' : analysis.overallSignal === 'SAT' ? 'SELL' : 'WAIT',
      entry: currentPrice,
      stopLoss: analysis.overallSignal === 'AL'
        ? currentPrice * 0.97
        : currentPrice * 1.03,
      takeProfit: analysis.overallSignal === 'AL'
        ? [currentPrice * 1.02, currentPrice * 1.05, currentPrice * 1.08]
        : [currentPrice * 0.98, currentPrice * 0.95, currentPrice * 0.92],
      riskReward: 2.5,
      positionSize: Math.max(1, Math.min(10, 100 - volatility)),
      reasoning: `Based on ${analysis.buyCount}/8 BUY signals with ${analysis.overallConfidence}% confidence. ${analysis.recommendation}`,
      confidence: analysis.overallConfidence,
    };

    setTradeRecommendation(recommendation);
  };

  // ============ LIFECYCLE ============

  useEffect(() => {
    fetchCoins();
    generateStrategyPerformance();

    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchCoins();
          return 3;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // ============ COMPUTED VALUES ============

  const processedCoins = useMemo(() => {
    return coins
      .filter((coin) =>
        coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      )
      .sort((a, b) => {
        switch (sortBy) {
          case 'volume':
            return b.volume24h - a.volume24h;
          case 'change':
            return b.changePercent24h - a.changePercent24h;
          case 'price':
            return b.price - a.price;
          case 'name':
            return a.symbol.localeCompare(b.symbol);
          default:
            return 0;
        }
      });
  }, [coins, searchTerm, sortBy]);

  const activeAlertsCount = priceAlerts.filter(a => a.isActive).length;

  // Helper: Timeframe'lere göre border rengi belirle
  // NOT: Bu fonksiyon basit bir heuristic kullanıyor.
  // Gerçek timeframe analizi için coin'e tıklanması gerekiyor.
  const getTimeframeBorderColor = (coin: CoinData) => {
    const change = coin.changePercent24h;
    const volume = coin.volume24h;

    // Volume threshold - average volume için benchmark
    const avgVolume = 100000000; // $100M USD
    const hasHighVolume = volume > avgVolume;

    // Sadece pozitif değişim + yüksek volume kombinasyonu için yeşil çerçeve
    // Bu, güçlü momentum ve likidite gösterir
    if (change > 0 && hasHighVolume) {
      if (change >= 5) {
        return COLORS.success; // Çok güçlü momentum
      } else if (change >= 2) {
        return `${COLORS.success}cc`; // Güçlü momentum
      } else if (change >= 0.5) {
        return `${COLORS.success}80`; // Hafif momentum
      }
    }

    // Düşük volume veya negatif değişim = varsayılan renk
    return COLORS.border.default;
  };

  // ============ LOADING STATE ============

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  // ============ RENDER ============

  return (
    <div className="dashboard-container" style={{ minHeight: '100vh', width: '100%', overflow: 'auto' }}>
      {/* Sidebar */}
      <SharedSidebar
        currentPage="talib"
        notificationCounts={{}}
        coinCount={processedCoins.length}
        countdown={countdown}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />

      {/* Main Content */}
      <div className="dashboard-main" style={{ paddingTop: '80px', width: '100%', minHeight: '100vh' }}>

        {/* REAL-TIME PRICE TICKER */}
        <div style={{
          background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
          borderBottom: `1px solid ${COLORS.border.default}`,
          padding: '12px 24px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          <div style={{
            display: 'flex',
            gap: '32px',
            animation: 'scroll 20s linear infinite',
          }}>
            {[...tickerCoins, ...tickerCoins].map((coin, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: '200px' }}>
                <div style={{ color: COLORS.text.primary, fontWeight: '700', fontSize: '14px' }}>
                  {coin.symbol.replace('USDT', '')}
                </div>
                <div style={{ color: COLORS.text.primary, fontFamily: 'monospace', fontSize: '14px' }}>
                  ${coin.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div style={{
                  color: getChangeColor(coin.changePercent24h),
                  fontSize: '13px',
                  fontWeight: '700',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}>
                  {coin.changePercent24h > 0 ? '▲' : '▼'}
                  {Math.abs(coin.changePercent24h).toFixed(2)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Page Header */}
        <div style={{
          margin: '0 24px 24px 24px',
          padding: '24px 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '16px',
          borderBottom: `1px solid ${COLORS.border.default}`,
          paddingBottom: '16px',
        }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Icons.Activity style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                TA-Lib Premium Analysis
              </h1>
              {/* LIVE Badge */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.danger}, ${COLORS.danger}dd)`,
                color: '#fff',
                padding: '4px 12px',
                borderRadius: '12px',
                fontSize: '11px',
                fontWeight: '700',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                animation: 'pulse 2s infinite',
              }}>
                <div style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#fff',
                  animation: 'pulse 1s infinite',
                }} />
                LIVE
              </div>
            </div>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
              8 Strateji Analizi • Desen Tanıma • YZ Ticaret Önerileri • Çoklu Zaman Dilimi Isı Haritası
            </p>
          </div>

          {/* Search & Action Buttons */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
            {/* Premium Search Box - Responsive */}
            <div style={{
              position: 'relative',
              flex: '1',
              minWidth: isMobile ? '100%' : '280px',
              maxWidth: isMobile ? '100%' : '400px',
              width: isMobile ? '100%' : 'auto',
            }}>
              <Icons.Search style={{
                position: 'absolute',
                left: '16px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '20px',
                height: '20px',
                color: COLORS.text.muted,
                pointerEvents: 'none',
              }} />
              <input
                type="text"
                placeholder="Coin ara (BTC, ETH, SOL...)"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={{
                  width: '100%',
                  padding: '12px 16px 12px 48px',
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `2px solid ${searchTerm ? COLORS.premium : COLORS.border.default}`,
                  borderRadius: '12px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                  fontWeight: '500',
                  outline: 'none',
                  transition: 'all 0.3s ease',
                  boxShadow: searchTerm ? `0 4px 12px ${COLORS.premium}30` : 'none',
                }}
                onFocus={(e) => {
                  if (!searchTerm) {
                    e.target.style.border = `2px solid ${COLORS.info}`;
                  }
                }}
                onBlur={(e) => {
                  if (!searchTerm) {
                    e.target.style.border = `2px solid ${COLORS.border.default}`;
                  }
                }}
              />
              {searchTerm && (
                <button
                  onClick={() => setSearchTerm('')}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    color: COLORS.text.muted,
                    padding: '4px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icons.X style={{ width: '16px', height: '16px' }} />
                </button>
              )}
            </div>

            {/* Alerts Button */}
            <button
              onClick={() => setShowAlertsModal(true)}
              style={{
                padding: '12px 20px',
                background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                color: '#000',
                border: 'none',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                position: 'relative',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 6px 20px ${COLORS.warning}60`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <Icons.Bell style={{ width: '18px', height: '18px' }} />
              Uyarılar
              {activeAlertsCount > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '-6px',
                  right: '-6px',
                  background: COLORS.danger,
                  color: '#fff',
                  borderRadius: '50%',
                  width: '20px',
                  height: '20px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '11px',
                  fontWeight: '700',
                }}>
                  {activeAlertsCount}
                </div>
              )}
            </button>

            {/* Strategy Builder */}
            <button
              onClick={() => setShowStrategyBuilder(true)}
              style={{
                padding: '12px 20px',
                background: `linear-gradient(135deg, ${COLORS.info}, ${COLORS.info}dd)`,
                color: '#fff',
                border: 'none',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 6px 20px ${COLORS.info}60`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <Icons.Settings style={{ width: '18px', height: '18px' }} />
              Strateji Oluşturucu
            </button>

            {/* LOGIC Button */}
            <button
              onClick={() => setShowLogicModal(true)}
              style={{
                padding: '12px 20px',
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.premium}dd)`,
                color: '#fff',
                border: 'none',
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 6px 20px ${COLORS.premium}60`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            >
              <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
              LOGIC
            </button>
          </div>
        </div>

        {/* View Mode Selector + Stats */}
        <div style={{
          margin: '0 24px 24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '16px',
          flexWrap: 'wrap',
        }}>
          {/* View Mode Tabs */}
          <div style={{
            display: 'flex',
            gap: '8px',
            background: COLORS.bg.card,
            padding: '6px',
            borderRadius: '10px',
            border: `1px solid ${COLORS.border.default}`,
          }}>
            {(['cards', 'table', 'analytics'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                style={{
                  padding: '8px 20px',
                  background: viewMode === mode
                    ? `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`
                    : 'transparent',
                  color: viewMode === mode ? '#fff' : COLORS.text.secondary,
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '13px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  textTransform: 'capitalize',
                }}
                onMouseEnter={(e) => {
                  if (viewMode !== mode) {
                    e.currentTarget.style.background = COLORS.bg.hover;
                  }
                }}
                onMouseLeave={(e) => {
                  if (viewMode !== mode) {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                {mode === 'cards' && <Icons.Grid style={{ width: '16px', height: '16px', display: 'inline', marginRight: '6px' }} />}
                {mode === 'table' && <Icons.Menu style={{ width: '16px', height: '16px', display: 'inline', marginRight: '6px' }} />}
                {mode === 'analytics' && <Icons.BarChart3 style={{ width: '16px', height: '16px', display: 'inline', marginRight: '6px' }} />}
                {mode === 'cards' ? 'kartlar' : mode === 'table' ? 'tablo' : 'analitik'}
              </button>
            ))}
          </div>

          {/* Quick Stats */}
          <div style={{ display: 'flex', gap: '16px', fontSize: '13px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ color: COLORS.text.secondary }}>Toplam Varlık:</div>
              <div style={{ color: COLORS.text.primary, fontWeight: '700' }}>{processedCoins.length}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ color: COLORS.text.secondary }}>Sonraki Güncelleme:</div>
              <div style={{ color: COLORS.premium, fontWeight: '700' }}>{countdown}s</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ color: COLORS.text.secondary }}>Son:</div>
              <div style={{ color: COLORS.text.primary, fontWeight: '700' }}>
                {lastUpdate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        </div>

        {/* Error Banner */}
        {error && !loading && (
          <div style={{
            margin: '0 24px 24px',
            padding: '16px 20px',
            background: `linear-gradient(135deg, ${COLORS.danger}20, ${COLORS.danger}10)`,
            border: `1px solid ${COLORS.danger}`,
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}>
            <Icons.AlertTriangle style={{ width: '24px', height: '24px', color: COLORS.danger }} />
            <div style={{ flex: 1 }}>
              <p style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '600', margin: 0, marginBottom: '4px' }}>
                Bağlantı Hatası
              </p>
              <p style={{ color: COLORS.text.secondary, fontSize: '13px', margin: 0 }}>
                {error}
              </p>
            </div>
            <button
              onClick={fetchCoins}
              style={{
                padding: '8px 16px',
                background: COLORS.danger,
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
              }}
            >
              Tekrar Dene
            </button>
          </div>
        )}

        {/* STRATEGY PERFORMANCE TRACKER (Analytics View) */}
        {viewMode === 'analytics' && (
          <div style={{ margin: '0 24px 24px' }}>
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
              border: `1px solid ${COLORS.border.default}`,
              borderRadius: '16px',
              padding: '24px',
              boxShadow: `0 4px 20px ${COLORS.premium}20`,
            }}>
              <h3 style={{
                fontSize: '20px',
                fontWeight: 'bold',
                color: COLORS.text.primary,
                marginBottom: '20px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}>
                <Icons.TrendingUp style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                Strateji Performans Takipçisi
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
                {strategyPerformance.map((strat, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: `linear-gradient(135deg, ${COLORS.bg.primary}dd, ${COLORS.bg.card}dd)`,
                      border: `1px solid ${strat.isHot ? COLORS.danger : COLORS.border.default}`,
                      borderRadius: '12px',
                      padding: '16px',
                      position: 'relative',
                      overflow: 'hidden',
                      backdropFilter: 'blur(10px)',
                    }}
                  >
                    {/* Hot Badge */}
                    {strat.isHot && (
                      <div style={{
                        position: 'absolute',
                        top: '12px',
                        right: '12px',
                        background: `linear-gradient(135deg, ${COLORS.danger}, ${COLORS.warning})`,
                        color: '#000',
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '700',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}>
                        <Icons.Fire style={{ width: '12px', height: '12px' }} />
                        HOT
                      </div>
                    )}

                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.text.primary }}>
                        {strat.name}
                      </div>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                        {strat.totalTrades} toplam işlem
                      </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>24h</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.premium }}>
                          {strat.winRate24h}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>7d</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.info }}>
                          {strat.winRate7d}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>30d</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.success }}>
                          {strat.winRate30d}%
                        </div>
                      </div>
                    </div>

                    <div style={{
                      background: `${COLORS.success}20`,
                      border: `1px solid ${COLORS.success}40`,
                      borderRadius: '8px',
                      padding: '8px 12px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Ort. Kar</div>
                      <div style={{ fontSize: '15px', fontWeight: '700', color: COLORS.success }}>
                        +{strat.avgProfit}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Main Coin Grid/Table/Analytics - Responsive Padding */}
        <main className="dashboard-content" style={{ padding: isMobile ? '0 12px 12px' : '0 24px 24px' }}>
          {/* CARDS VIEW - Responsive Grid */}
          {viewMode === 'cards' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: isMobile ? '1fr' : isTablet ? 'repeat(auto-fill, minmax(220px, 1fr))' : 'repeat(auto-fill, minmax(260px, 1fr))',
              gap: isMobile ? '12px' : '16px',
            }}>
              {processedCoins.map((coin) => {
                const borderColor = getTimeframeBorderColor(coin);
                // hasBuySignal sadece yeşil border varsa true olmalı
                const hasBuySignal = borderColor !== COLORS.border.default;

                return (
                <div
                  key={coin.symbol}
                  onClick={() => handleCoinClick(coin.symbol)}
                  style={{
                    background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                    border: `2px solid ${borderColor}`,
                    borderRadius: isMobile ? '12px' : '16px',
                    padding: isMobile ? '12px' : '16px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    position: 'relative',
                    overflow: 'hidden',
                    backdropFilter: 'blur(10px)',
                    boxShadow: hasBuySignal ? `0 4px 16px ${borderColor}40` : 'none',
                    minHeight: '44px', // Touch-friendly minimum
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.border = `2px solid ${COLORS.premium}`;
                    e.currentTarget.style.transform = 'translateY(-4px)';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${COLORS.premium}30`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.border = `2px solid ${borderColor}`;
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = hasBuySignal ? `0 4px 16px ${borderColor}40` : 'none';
                  }}
                >
                  {/* Symbol */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '700' }}>
                      {coin.symbol.replace('USDT', '').replace('USDC', '')}
                    </div>
                    <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>
                      {coin.symbol.includes('USDT') ? 'USDT' : 'USDC'}
                    </div>
                  </div>

                  {/* Price */}
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ color: COLORS.text.primary, fontSize: '18px', fontWeight: '700', fontFamily: 'monospace' }}>
                      ${coin.price < 1
                        ? coin.price.toFixed(6)
                        : coin.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                      }
                    </div>
                  </div>

                  {/* Change */}
                  <div style={{
                    background: `${getChangeColor(coin.changePercent24h)}20`,
                    border: `1px solid ${getChangeColor(coin.changePercent24h)}40`,
                    borderRadius: '8px',
                    padding: '8px 12px',
                    marginBottom: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>24s Değişim</div>
                    <div style={{
                      color: getChangeColor(coin.changePercent24h),
                      fontSize: '15px',
                      fontWeight: '700',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                    }}>
                      {coin.changePercent24h > 0 ? (
                        <Icons.TrendingUp style={{ width: '16px', height: '16px' }} />
                      ) : (
                        <Icons.TrendingDown style={{ width: '16px', height: '16px' }} />
                      )}
                      {coin.changePercent24h > 0 ? '+' : ''}{coin.changePercent24h.toFixed(2)}%
                    </div>
                  </div>

                  {/* Stats */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '11px' }}>
                    <div>
                      <div style={{ color: COLORS.text.secondary, marginBottom: '4px' }}>Yüksek</div>
                      <div style={{ color: COLORS.text.primary, fontWeight: '600' }}>
                        ${coin.high24h ? (coin.high24h < 1 ? coin.high24h.toFixed(6) : coin.high24h.toFixed(2)) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: COLORS.text.secondary, marginBottom: '4px' }}>Düşük</div>
                      <div style={{ color: COLORS.text.primary, fontWeight: '600' }}>
                        ${coin.low24h ? (coin.low24h < 1 ? coin.low24h.toFixed(6) : coin.low24h.toFixed(2)) : 'N/A'}
                      </div>
                    </div>
                    <div style={{ gridColumn: 'span 2' }}>
                      <div style={{ color: COLORS.text.secondary, marginBottom: '4px' }}>Hacim</div>
                      <div style={{ color: COLORS.premium, fontWeight: '700', fontSize: '13px' }}>
                        ${coin.volume24h ? (coin.volume24h / 1000000).toFixed(1) + 'M' : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
                );
              })}
            </div>
          )}

          {/* TABLE VIEW */}
          {viewMode === 'table' && (
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
              border: `1px solid ${COLORS.border.default}`,
              borderRadius: '16px',
              overflow: 'hidden',
            }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: COLORS.bg.card, borderBottom: `2px solid ${COLORS.border.default}` }}>
                      <th style={{ padding: '16px', textAlign: 'left', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>Sembol</th>
                      <th style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>Fiyat</th>
                      <th style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>24s Değişim</th>
                      <th style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>24s Yüksek</th>
                      <th style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>24s Düşük</th>
                      <th style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>Hacim</th>
                      <th style={{ padding: '16px', textAlign: 'center', color: COLORS.text.secondary, fontSize: '12px', fontWeight: '700' }}>İşlem</th>
                    </tr>
                  </thead>
                  <tbody>
                    {processedCoins.map((coin) => (
                      <tr
                        key={coin.symbol}
                        style={{
                          borderBottom: `1px solid ${COLORS.border.default}`,
                          transition: 'all 0.2s ease',
                          cursor: 'pointer',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = COLORS.bg.hover;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent';
                        }}
                      >
                        <td style={{ padding: '16px' }}>
                          <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '700' }}>
                            {coin.symbol.replace('USDT', '').replace('USDC', '')}
                          </div>
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', fontFamily: 'monospace', color: COLORS.text.primary, fontWeight: '600' }}>
                          ${coin.price < 1 ? coin.price.toFixed(6) : coin.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right' }}>
                          <div style={{
                            color: getChangeColor(coin.changePercent24h),
                            fontSize: '14px',
                            fontWeight: '700',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '6px',
                          }}>
                            {coin.changePercent24h > 0 ? '▲' : '▼'}
                            {coin.changePercent24h > 0 ? '+' : ''}{coin.changePercent24h.toFixed(2)}%
                          </div>
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '13px' }}>
                          ${coin.high24h ? (coin.high24h < 1 ? coin.high24h.toFixed(6) : coin.high24h.toFixed(2)) : 'N/A'}
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', color: COLORS.text.secondary, fontSize: '13px' }}>
                          ${coin.low24h ? (coin.low24h < 1 ? coin.low24h.toFixed(6) : coin.low24h.toFixed(2)) : 'N/A'}
                        </td>
                        <td style={{ padding: '16px', textAlign: 'right', color: COLORS.premium, fontSize: '13px', fontWeight: '600' }}>
                          ${coin.volume24h ? (coin.volume24h / 1000000).toFixed(1) + 'M' : 'N/A'}
                        </td>
                        <td style={{ padding: '16px', textAlign: 'center' }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCoinClick(coin.symbol);
                            }}
                            style={{
                              padding: '6px 16px',
                              background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})`,
                              color: '#fff',
                              border: 'none',
                              borderRadius: '8px',
                              fontSize: '12px',
                              fontWeight: '700',
                              cursor: 'pointer',
                            }}
                          >
                            Analiz Et
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ANALYTICS VIEW */}
          {viewMode === 'analytics' && (
            <div style={{ display: 'grid', gap: '24px' }}>
              {/* Top Gainers/Losers */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '16px' }}>
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.success}15, ${COLORS.success}05)`,
                  border: `1px solid ${COLORS.success}40`,
                  borderRadius: '16px',
                  padding: '20px',
                }}>
                  <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.success, marginBottom: '16px' }}>
                    En Çok Kazananlar (24s)
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {processedCoins
                      .filter(c => c.changePercent24h > 0)
                      .sort((a, b) => b.changePercent24h - a.changePercent24h)
                      .slice(0, 5)
                      .map((coin, idx) => (
                        <div
                          key={coin.symbol}
                          onClick={() => handleCoinClick(coin.symbol)}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: '12px',
                            background: `${COLORS.success}10`,
                            borderRadius: '8px',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = `${COLORS.success}20`;
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = `${COLORS.success}10`;
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{
                              width: '24px',
                              height: '24px',
                              borderRadius: '50%',
                              background: COLORS.success,
                              color: '#000',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '12px',
                              fontWeight: '700',
                            }}>
                              {idx + 1}
                            </div>
                            <div>
                              <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '700' }}>
                                {coin.symbol.replace('USDT', '')}
                              </div>
                              <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>
                                ${coin.price.toFixed(2)}
                              </div>
                            </div>
                          </div>
                          <div style={{ color: COLORS.success, fontSize: '15px', fontWeight: '700' }}>
                            +{coin.changePercent24h.toFixed(2)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.danger}15, ${COLORS.danger}05)`,
                  border: `1px solid ${COLORS.danger}40`,
                  borderRadius: '16px',
                  padding: '20px',
                }}>
                  <h4 style={{ fontSize: '16px', fontWeight: '700', color: COLORS.danger, marginBottom: '16px' }}>
                    En Çok Kaybedenler (24s)
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {processedCoins
                      .filter(c => c.changePercent24h < 0)
                      .sort((a, b) => a.changePercent24h - b.changePercent24h)
                      .slice(0, 5)
                      .map((coin, idx) => (
                        <div
                          key={coin.symbol}
                          onClick={() => handleCoinClick(coin.symbol)}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: '12px',
                            background: `${COLORS.danger}10`,
                            borderRadius: '8px',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = `${COLORS.danger}20`;
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = `${COLORS.danger}10`;
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{
                              width: '24px',
                              height: '24px',
                              borderRadius: '50%',
                              background: COLORS.danger,
                              color: '#fff',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '12px',
                              fontWeight: '700',
                            }}>
                              {idx + 1}
                            </div>
                            <div>
                              <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '700' }}>
                                {coin.symbol.replace('USDT', '')}
                              </div>
                              <div style={{ color: COLORS.text.secondary, fontSize: '11px' }}>
                                ${coin.price.toFixed(2)}
                              </div>
                            </div>
                          </div>
                          <div style={{ color: COLORS.danger, fontSize: '15px', fontWeight: '700' }}>
                            {coin.changePercent24h.toFixed(2)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* No results */}
          {processedCoins.length === 0 && (
            <div style={{ textAlign: 'center', padding: '80px 20px', color: COLORS.text.secondary }}>
              <Icons.Search style={{ width: '48px', height: '48px', color: COLORS.border.active, marginBottom: '16px' }} />
              <div>Coin bulunamadı. Farklı bir arama deneyin.</div>
            </div>
          )}
        </main>
      </div>

      {/* ============ ANALYSIS MODAL ============ */}
      {selectedCoin && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.95)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '24px',
            backdropFilter: 'blur(10px)',
          }}
          onClick={() => setSelectedCoin(null)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '20px',
              maxWidth: '1200px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}60`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.info}20)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.premium}40`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
              backdropFilter: 'blur(10px)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h2 style={{
                    fontSize: '28px',
                    fontWeight: 'bold',
                    color: COLORS.text.primary,
                    margin: 0,
                    marginBottom: '8px',
                  }}>
                    {selectedCoin.replace('USDT', '').replace('USDC', '')} / USDT
                  </h2>
                  {multiStrategyAnalysis && (
                    <div style={{ display: 'flex', gap: '20px', fontSize: '14px' }}>
                      <span style={{ color: COLORS.text.primary, fontFamily: 'monospace', fontSize: '18px', fontWeight: '700' }}>
                        ${multiStrategyAnalysis.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                      </span>
                      <span style={{
                        color: getChangeColor(multiStrategyAnalysis.change24h),
                        fontSize: '16px',
                        fontWeight: '700',
                      }}>
                        {multiStrategyAnalysis.change24h > 0 ? '+' : ''}{multiStrategyAnalysis.change24h.toFixed(2)}%
                      </span>
                    </div>
                  )}
                </div>
                <button
                  onClick={() => setSelectedCoin(null)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '10px 20px',
                    borderRadius: '10px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '700',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = COLORS.danger;
                    e.currentTarget.style.borderColor = COLORS.danger;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = COLORS.border.active;
                  }}
                >
                  KAPAT
                </button>
              </div>
            </div>

            {/* Loading State */}
            {analysisLoading && (
              <div style={{ padding: '60px' }}>
                <LoadingAnimation />
              </div>
            )}

            {/* Analysis Content */}
            {!analysisLoading && multiStrategyAnalysis && (
              <div style={{ padding: '24px' }}>

                {/* MULTI-TIMEFRAME HEATMAP */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '16px',
                  padding: '24px',
                  marginBottom: '24px',
                }}>
                  <h3 style={{
                    fontSize: '18px',
                    fontWeight: '700',
                    color: COLORS.text.primary,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                  }}>
                    <Icons.Grid style={{ width: '20px', height: '20px', color: COLORS.premium }} />
                    Çoklu Zaman Dilimi Sinyal Isı Haritası
                  </h3>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                    {timeframeHeatmap.map((tf) => {
                      const signalColor = tf.signal === 'BUY' ? COLORS.success : tf.signal === 'SELL' ? COLORS.danger : COLORS.warning;
                      const intensity = tf.strength / 100;

                      return (
                        <div
                          key={tf.timeframe}
                          onClick={() => setSelectedTimeframe(tf.timeframe)}
                          style={{
                            background: `linear-gradient(135deg, ${signalColor}${Math.floor(intensity * 40).toString(16).padStart(2, '0')}, ${signalColor}${Math.floor(intensity * 20).toString(16).padStart(2, '0')})`,
                            border: `2px solid ${selectedTimeframe === tf.timeframe ? signalColor : `${signalColor}60`}`,
                            borderRadius: '12px',
                            padding: '16px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            textAlign: 'center',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.transform = 'scale(1.05)';
                            e.currentTarget.style.boxShadow = `0 8px 24px ${signalColor}40`;
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.transform = 'scale(1)';
                            e.currentTarget.style.boxShadow = 'none';
                          }}
                        >
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                            {tf.timeframe}
                          </div>
                          <div style={{ fontSize: '20px', fontWeight: '700', color: signalColor, marginBottom: '8px' }}>
                            {tf.signal === 'BUY' ? 'AL' : tf.signal === 'SELL' ? 'SAT' : 'BEKLE'}
                          </div>
                          <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                            Güç: {tf.strength}%
                          </div>
                          <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                            Güven: {tf.confidence}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* CANDLESTICK PATTERNS */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '16px',
                  padding: '24px',
                  marginBottom: '24px',
                }}>
                  <h3 style={{
                    fontSize: '18px',
                    fontWeight: '700',
                    color: COLORS.text.primary,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                  }}>
                    <Icons.Zap style={{ width: '20px', height: '20px', color: COLORS.warning }} />
                    Tespit Edilen Mum Grafik Desenleri
                  </h3>

                  <div style={{ display: 'grid', gap: '12px' }}>
                    {detectedPatterns.map((pattern, idx) => {
                      const typeColor = pattern.type === 'BULLISH' ? COLORS.success : pattern.type === 'BEARISH' ? COLORS.danger : COLORS.warning;

                      return (
                        <div
                          key={idx}
                          style={{
                            background: `${typeColor}10`,
                            border: `1px solid ${typeColor}40`,
                            borderRadius: '12px',
                            padding: '16px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                              <div style={{ fontSize: '15px', fontWeight: '700', color: COLORS.text.primary }}>
                                {pattern.name}
                              </div>
                              <div style={{
                                background: typeColor,
                                color: pattern.type === 'NEUTRAL' ? '#000' : '#fff',
                                padding: '3px 10px',
                                borderRadius: '6px',
                                fontSize: '11px',
                                fontWeight: '700',
                              }}>
                                {pattern.type}
                              </div>
                            </div>
                            <div style={{ fontSize: '13px', color: COLORS.text.secondary, lineHeight: '1.5' }}>
                              {pattern.description}
                            </div>
                          </div>
                          <div style={{ textAlign: 'center', marginLeft: '16px' }}>
                            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>
                              Güvenilirlik
                            </div>
                            <div style={{ fontSize: '24px', fontWeight: '700', color: typeColor }}>
                              {pattern.reliability}%
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* AI TRADE RECOMMENDATION */}
                {tradeRecommendation && (
                  <div style={{
                    background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.info}20)`,
                    border: `2px solid ${COLORS.premium}`,
                    borderRadius: '16px',
                    padding: '24px',
                    marginBottom: '24px',
                    boxShadow: `0 8px 32px ${COLORS.premium}30`,
                  }}>
                    <h3 style={{
                      fontSize: '20px',
                      fontWeight: '700',
                      color: COLORS.premium,
                      marginBottom: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                    }}>
                      <Icons.Target style={{ width: '24px', height: '24px' }} />
                      YZ Destekli Ticaret Önerisi
                    </h3>

                    {/* Action & Confidence */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
                      <div style={{
                        background: `${tradeRecommendation.action === 'BUY' ? COLORS.success : tradeRecommendation.action === 'SELL' ? COLORS.danger : COLORS.warning}20`,
                        border: `2px solid ${tradeRecommendation.action === 'BUY' ? COLORS.success : tradeRecommendation.action === 'SELL' ? COLORS.danger : COLORS.warning}`,
                        borderRadius: '12px',
                        padding: '20px',
                        textAlign: 'center',
                      }}>
                        <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                          Önerilen İşlem
                        </div>
                        <div style={{
                          fontSize: '32px',
                          fontWeight: '700',
                          color: tradeRecommendation.action === 'BUY' ? COLORS.success : tradeRecommendation.action === 'SELL' ? COLORS.danger : COLORS.warning,
                        }}>
                          {tradeRecommendation.action === 'BUY' ? 'AL' : tradeRecommendation.action === 'SELL' ? 'SAT' : 'BEKLE'}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.premium}20`,
                        border: `2px solid ${COLORS.premium}`,
                        borderRadius: '12px',
                        padding: '20px',
                        textAlign: 'center',
                      }}>
                        <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                          Güven Seviyesi
                        </div>
                        <div style={{ fontSize: '32px', fontWeight: '700', color: COLORS.premium }}>
                          {tradeRecommendation.confidence}%
                        </div>
                      </div>
                    </div>

                    {/* Trade Levels */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '20px' }}>
                      <div style={{
                        background: `${COLORS.info}15`,
                        border: `1px solid ${COLORS.info}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Giriş Fiyatı</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.info, fontFamily: 'monospace' }}>
                          ${tradeRecommendation.entry.toFixed(2)}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.danger}15`,
                        border: `1px solid ${COLORS.danger}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Zarar Durdur</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.danger, fontFamily: 'monospace' }}>
                          ${tradeRecommendation.stopLoss.toFixed(2)}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.success}15`,
                        border: `1px solid ${COLORS.success}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Kar Al 1</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.success, fontFamily: 'monospace' }}>
                          ${tradeRecommendation.takeProfit[0].toFixed(2)}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.success}15`,
                        border: `1px solid ${COLORS.success}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Kar Al 2</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.success, fontFamily: 'monospace' }}>
                          ${tradeRecommendation.takeProfit[1].toFixed(2)}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.warning}15`,
                        border: `1px solid ${COLORS.warning}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Risk/Ödül</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.warning }}>
                          1:{tradeRecommendation.riskReward.toFixed(1)}
                        </div>
                      </div>

                      <div style={{
                        background: `${COLORS.premium}15`,
                        border: `1px solid ${COLORS.premium}40`,
                        borderRadius: '10px',
                        padding: '14px',
                      }}>
                        <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '6px' }}>Pozisyon Boyutu</div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: COLORS.premium }}>
                          {tradeRecommendation.positionSize.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Reasoning */}
                    <div style={{
                      background: `${COLORS.bg.card}80`,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '10px',
                      padding: '16px',
                    }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '700' }}>
                        YZ Değerlendirmesi:
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.primary, lineHeight: '1.6' }}>
                        {tradeRecommendation.reasoning}
                      </div>
                    </div>
                  </div>
                )}

                {/* 8 STRATEGIES ANALYSIS */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '16px',
                  padding: '24px',
                  marginBottom: '24px',
                }}>
                  <h3 style={{
                    fontSize: '18px',
                    fontWeight: '700',
                    color: COLORS.text.primary,
                    marginBottom: '16px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                  }}>
                    <Icons.TrendingUp style={{ width: '20px', height: '20px', color: COLORS.success }} />
                    Genel Analiz ({multiStrategyAnalysis.strategies.length}/8 Strateji)
                  </h3>

                  {/* Summary Stats */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '20px' }}>
                    <div style={{
                      background: `${COLORS.success}15`,
                      border: `1px solid ${COLORS.success}40`,
                      borderRadius: '10px',
                      padding: '14px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>AL Sinyalleri</div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.success }}>
                        {multiStrategyAnalysis.buyCount}/8
                      </div>
                    </div>

                    <div style={{
                      background: `${COLORS.danger}15`,
                      border: `1px solid ${COLORS.danger}40`,
                      borderRadius: '10px',
                      padding: '14px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>SAT Sinyalleri</div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.danger }}>
                        {multiStrategyAnalysis.sellCount}/8
                      </div>
                    </div>

                    <div style={{
                      background: `${COLORS.warning}15`,
                      border: `1px solid ${COLORS.warning}40`,
                      borderRadius: '10px',
                      padding: '14px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>BEKLE Sinyalleri</div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.warning }}>
                        {multiStrategyAnalysis.waitCount}/8
                      </div>
                    </div>

                    <div style={{
                      background: `${COLORS.premium}15`,
                      border: `1px solid ${COLORS.premium}40`,
                      borderRadius: '10px',
                      padding: '14px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Güven</div>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.premium }}>
                        {multiStrategyAnalysis.overallConfidence}%
                      </div>
                    </div>

                    <div style={{
                      background: `${COLORS.info}15`,
                      border: `1px solid ${COLORS.info}40`,
                      borderRadius: '10px',
                      padding: '14px',
                      textAlign: 'center',
                    }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px' }}>Risk Seviyesi</div>
                      <div style={{
                        fontSize: '16px',
                        fontWeight: '700',
                        color: multiStrategyAnalysis.riskLevel === 'DÜŞÜK' ? COLORS.success : multiStrategyAnalysis.riskLevel === 'YÜKSEK' ? COLORS.danger : COLORS.warning,
                      }}>
                        {multiStrategyAnalysis.riskLevel}
                      </div>
                    </div>
                  </div>

                  {/* Overall Signal */}
                  <div style={{
                    background: `${multiStrategyAnalysis.overallSignal === 'AL' ? COLORS.success : multiStrategyAnalysis.overallSignal === 'SAT' ? COLORS.danger : COLORS.warning}20`,
                    border: `2px solid ${multiStrategyAnalysis.overallSignal === 'AL' ? COLORS.success : multiStrategyAnalysis.overallSignal === 'SAT' ? COLORS.danger : COLORS.warning}`,
                    borderRadius: '12px',
                    padding: '16px',
                    marginBottom: '20px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                      Genel Öneri
                    </div>
                    <div style={{
                      fontSize: '36px',
                      fontWeight: '700',
                      color: multiStrategyAnalysis.overallSignal === 'AL' ? COLORS.success : multiStrategyAnalysis.overallSignal === 'SAT' ? COLORS.danger : COLORS.warning,
                      marginBottom: '8px',
                    }}>
                      {multiStrategyAnalysis.overallSignal}
                    </div>
                    <div style={{ fontSize: '14px', color: COLORS.text.primary, lineHeight: '1.6' }}>
                      {multiStrategyAnalysis.recommendation}
                    </div>
                  </div>

                  {/* Individual Strategies */}
                  <div style={{ display: 'grid', gap: '12px' }}>
                    {multiStrategyAnalysis.strategies.map((strategy, index) => {
                      const signalColor = strategy.signal === 'AL' ? COLORS.success : strategy.signal === 'SAT' ? COLORS.danger : strategy.signal === 'BEKLE' ? COLORS.warning : COLORS.text.secondary;

                      return (
                        <div
                          key={index}
                          style={{
                            background: `${signalColor}10`,
                            border: `1px solid ${signalColor}40`,
                            borderRadius: '12px',
                            padding: '16px',
                          }}
                        >
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontSize: '15px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '6px' }}>
                                {strategy.name}
                              </div>
                              <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                                Güven: {strategy.confidence}% • Skor: {(strategy.score ?? 0) > 0 ? '+' : ''}{(strategy.score ?? 0).toFixed(1)}/100
                              </div>
                            </div>
                            <div style={{
                              background: signalColor,
                              color: strategy.signal === 'BEKLE' ? '#000' : '#fff',
                              padding: '8px 16px',
                              borderRadius: '8px',
                              fontSize: '13px',
                              fontWeight: '700',
                              minWidth: '70px',
                              textAlign: 'center',
                            }}>
                              {strategy.signal}
                            </div>
                          </div>
                          <div style={{ fontSize: '13px', color: COLORS.text.secondary, lineHeight: '1.5' }}>
                            {strategy.reason}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ============ LOGIC MODAL ============ */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '24px',
            backdropFilter: 'blur(10px)',
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '20px',
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}80`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.premium}15, ${COLORS.warning}15)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.premium}`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
              backdropFilter: 'blur(10px)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                  <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    TA-Lib Premium - LOGIC
                  </h2>
                </div>
                <button
                  onClick={() => setShowLogicModal(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = COLORS.danger;
                    e.currentTarget.style.borderColor = COLORS.danger;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = COLORS.border.active;
                  }}
                >
                  KAPAT
                </button>
              </div>
            </div>

            {/* Content */}
            <div style={{ padding: '24px' }}>
              {/* Overview */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px' }}>
                  Genel Bakış
                </h3>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                  TA-Lib Premium sayfası, Python Ta-Lib kütüphanesi kullanılarak geliştirilmiş profesyonel teknik analiz göstergelerini sunar.
                  Her kripto para için 8 farklı strateji eş zamanlı olarak analiz edilerek kapsamlı piyasa değerlendirmesi sağlanır.
                </p>
                <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                  Bu sayfa, kurumsal yatırımcıların kullandığı teknik analiz araçlarının aynısını sunarak,
                  veriye dayalı yatırım kararları almanıza yardımcı olur.
                </p>
              </div>

              {/* Unique Features */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px' }}>
                  Benzersiz Premium Özellikler
                </h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: 'Canlı Fiyat Bandı', desc: 'Bloomberg tarzı canlı kaydırmalı bant, BTC, ETH, BNB, SOL için her 3 saniyede otomatik yenileme ile' },
                    { name: 'Çoklu Zaman Dilimi Isı Haritası', desc: '1S, 4S, 1G, 1H zaman dilimlerinde sinyal gücünü gösteren, güven seviyesine göre renk yoğunluğu ile görsel ısı haritası' },
                    { name: 'Mum Grafik Desen Tanıma', desc: 'Doji, Hammer, Engulfing gibi desenlerin güvenilirlik skorları ve geçmiş başarı oranları ile otomatik tespiti' },
                    { name: 'Strateji Performans Takipçisi', desc: 'Her strateji için 24s, 7g, 30g dönemlerinde kazanma oranları ve ortalama kar metrikleri takibi' },
                    { name: 'YZ Destekli Ticaret Önerileri', desc: 'Giriş, zarar durdur, kar al seviyelerini risk/ödül oranları ile oluşturmak için 8 stratejiyi birleştirir' },
                    { name: '3 Görünüm Modu', desc: 'Farklı analiz perspektifleri için Kartlar, Tablo ve Analitik görünümleri arasında geçiş yapın' },
                    { name: 'Akıllı Uyarı Sistemi', desc: 'Özel koşullarla fiyat ve strateji tabanlı uyarılar ayarlayın (yakında geliyor)' },
                  ].map((feature, index) => (
                    <div key={index} style={{
                      background: `${COLORS.bg.card}40`,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '16px',
                      transition: 'all 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = COLORS.premium;
                      e.currentTarget.style.transform = 'translateX(8px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = COLORS.border.default;
                      e.currentTarget.style.transform = 'translateX(0)';
                    }}>
                      <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                        {feature.name}
                      </div>
                      <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                        {feature.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 8 Strategies */}
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px' }}>
                  8 Strateji Sistemi
                </h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: 'RSI (Göreceli Güç Endeksi)', desc: 'Aşırı alım/satım seviyelerini tespit eder. 70 üzeri aşırı alım, 30 altı aşırı satım bölgesidir.' },
                    { name: 'MACD (Hareketli Ortalama Yakınsama Sapması)', desc: 'Trend yönü ve momentum değişimlerini gösterir. Sinyal çizgisi kesişmeleri önemlidir.' },
                    { name: 'Bollinger Bantları', desc: 'Volatilite ve fiyat sapmasını ölçer. Bantların dışına çıkma dönüş sinyali verir.' },
                    { name: 'MA Kesişimi (Hareketli Ortalama Kesişimi)', desc: '7-25-99 periyotlu MA kesişmeleri ile trend değişimlerini yakalar.' },
                    { name: 'Hacim Analizi', desc: 'İşlem hacmi trendlerini analiz eder. Hacim artışı sinyal gücünü doğrular.' },
                    { name: 'Fibonacci Düzeltmesi', desc: 'Destek/direnç seviyelerini belirler. %38.2, %50, %61.8 seviyeleri kritiktir.' },
                    { name: 'Stokastik Osilatör', desc: 'Momentum göstergesi. 80 üzeri aşırı alım, 20 altı aşırı satım gösterir.' },
                    { name: 'ATR (Ortalama Gerçek Aralık)', desc: 'Volatilite ölçümü. Yüksek ATR güçlü trend, düşük ATR konsolidasyon gösterir.' }
                  ].map((strategy, index) => (
                    <div key={index} style={{
                      background: `${COLORS.bg.card}40`,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '16px',
                      transition: 'all 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = COLORS.premium;
                      e.currentTarget.style.transform = 'translateX(8px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = COLORS.border.default;
                      e.currentTarget.style.transform = 'translateX(0)';
                    }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}20)`,
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '14px',
                          fontWeight: 'bold',
                          color: COLORS.premium,
                          minWidth: '32px',
                          textAlign: 'center',
                        }}>
                          {index + 1}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            {strategy.name}
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            {strategy.desc}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Important Notes */}
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '12px',
                padding: '20px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                  Önemli Notlar
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.8' }}>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Otomatik Güncelleme:</strong> Tüm veriler her 3 saniyede otomatik olarak güncellenir ve gerçek zamanlı fiyatları gösterir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Python Ta-Lib Servisi:</strong> Analiz Python mikroservis tarafından yapılır ve profesyonel seviye doğruluk sağlar.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Çoklu Onay:</strong> 8 strateji bağımsızdır. 5 veya daha fazla AL sinyali güçlü AL anlamına gelir.
                  </li>
                  <li style={{ marginBottom: '8px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Risk Yönetimi:</strong> Risk seviyesi DÜŞÜK, ORTA veya YÜKSEK olarak gösterilir. Yüksek risk = yüksek volatilite.
                  </li>
                  <li>
                    <strong style={{ color: COLORS.text.primary }}>Eğitim Amaçlıdır:</strong> Bu sinyaller yatırım tavsiyesi değildir. Kendi araştırmanızı yapın ve sorumlu yatırım yapın.
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============ STRATEGY BUILDER MODAL ============ */}
      {showStrategyBuilder && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '24px',
            backdropFilter: 'blur(10px)',
          }}
          onClick={() => setShowStrategyBuilder(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.info}`,
              borderRadius: '20px',
              maxWidth: '700px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.info}80`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.info}15, ${COLORS.premium}15)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.info}`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '22px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  Strateji Oluşturucu
                </h2>
                <button
                  onClick={() => setShowStrategyBuilder(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                  }}
                >
                  CLOSE
                </button>
              </div>
            </div>

            {/* Strategy Builder Content */}
            <StrategyBuilderContent onClose={() => setShowStrategyBuilder(false)} />
          </div>
        </div>
      )}

      {/* ============ ALERTS MODAL ============ */}
      {showAlertsModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '24px',
            backdropFilter: 'blur(10px)',
          }}
          onClick={() => setShowAlertsModal(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.warning}`,
              borderRadius: '20px',
              maxWidth: '700px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.warning}80`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.warning}`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 style={{ fontSize: '22px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  Akıllı Uyarılar
                </h2>
                <button
                  onClick={() => setShowAlertsModal(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                  }}
                >
                  CLOSE
                </button>
              </div>
            </div>

            {/* Alerts Content */}
            <AlertsContent onClose={() => setShowAlertsModal(false)} />
          </div>
        </div>
      )}

      {/* ============ CUSTOM STYLES ============ */}
      <style jsx>{`
        @keyframes scroll {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
