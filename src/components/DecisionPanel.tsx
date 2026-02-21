'use client';

/**
 * KARAR PANELİ - Decision Engine Component
 *
 * Tüm stratejileri analiz edip en doğru kararı gösterir:
 * - Alım/Satım/Bekle Kararı
 * - Giriş Fiyatı
 * - Stop Loss ve Hedefler
 * - Risk/Reward Oranı
 * - Gerekçeler
 */

import { useState, useEffect } from 'react';
import { COLORS } from '@/lib/colors';
import { Icons } from '@/components/Icons';
import { LoadingAnimation } from '@/components/LoadingAnimation';

interface DecisionData {
  symbol: string;
  currentPrice: number;
  decision: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  entryPrice: number;
  stopLoss: number;
  targets: {
    tp1: number;
    tp2: number;
    tp3: number;
  };
  riskRewardRatio: number;
  potentialGain: number;
  potentialLoss: number;
  buySignalsCount: number;
  sellSignalsCount: number;
  totalStrategies: number;
  strongestSignals: Array<{
    name: string;
    signal: string;
    confidence: number;
    reason: string;
  }>;
  reasons: string[];
  summary: string;
  timestamp: number;
}

interface DecisionPanelProps {
  symbol: string;
}

export function DecisionPanel({ symbol }: DecisionPanelProps) {
  const [data, setData] = useState<DecisionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let controller: AbortController | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    const fetchDecision = async () => {
      // Skip if previous fetch is still running
      if (!isMounted) return;

      try {
        setLoading(true);
        setError(null);

        // Clean up previous controller if exists
        if (controller) {
          try {
            controller.abort();
          } catch (err) {
            // Suppress AbortError during cleanup (expected behavior)
          }
        }

        controller = new AbortController();
        timeoutId = setTimeout(() => {
          if (controller) {
            try {
              controller.abort();
            } catch (err) {
              // Suppress AbortError during timeout (expected behavior)
            }
          }
        }, 45000); // 45 second timeout (increased from 30s)

        console.log(`[Decision Panel] Fetching decision for ${symbol}...`);

        const response = await fetch(`/api/decision-engine?symbol=${symbol}`, {
          cache: 'no-store',
          signal: controller.signal,
          next: { revalidate: 0 }
        });

        if (timeoutId) clearTimeout(timeoutId);

        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Karar motoru servisi bulunamadı');
          } else if (response.status === 500) {
            throw new Error('Sunucu hatası - Analiz yapılamadı');
          } else if (response.status === 429) {
            throw new Error('Çok fazla istek - Lütfen bekleyin');
          } else {
            throw new Error(`Bağlantı hatası (${response.status})`);
          }
        }

        const result = await response.json();

        if (!result.success || !result.data) {
          const errorMsg = result.error || 'Karar verisi alınamadı';
          console.log('[Decision Panel] API returned error:', errorMsg);
          throw new Error(errorMsg);
        }

        console.log(`[Decision Panel] Decision received: ${result.data.decision} (${result.data.confidence})`);

        // Only update state if component is still mounted
        if (isMounted) {
          setData(result.data);
        }
      } catch (err: any) {
        // Log only - don't use console.error to avoid showing in UI
        console.log('[Decision Panel] Fetch error:', err.message || err);

        if (!isMounted) return; // Skip error handling if unmounted

        if (err.name === 'AbortError') {
          // Don't show error for intentional aborts (component unmount or new fetch)
          console.log('[Decision Panel] Request aborted (normal behavior)');
        } else {
          // Show user-friendly error messages
          const errorMsg = err.message || '';

          if (errorMsg.includes('429') || errorMsg.includes('Too Many Requests') || errorMsg.includes('Rate limit')) {
            // Rate limit error - suggest waiting
            console.log('[Decision Panel] Rate limit hit - retrying automatically...');
            // Don't show error - will retry automatically on next interval
          } else if (errorMsg.includes('404') || errorMsg.includes('Coin bulunamadı')) {
            setError(`${symbol} için analiz şu anda kullanılamıyor`);
          } else if (errorMsg.includes('500') || errorMsg.includes('Sunucu hatası')) {
            setError('Analiz servisi geçici olarak kullanılamıyor - otomatik olarak yenilenecek');
          } else if (errorMsg.includes('Strategy analysis')) {
            // Generic strategy analysis error - show minimal message
            console.log('[Decision Panel] Strategy analysis error - will retry');
            // Don't show error to user, will retry on next interval
          } else {
            // Unknown error - show generic message
            setError('Analiz yükleniyor - lütfen bekleyin');
          }
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchDecision();

    // Refresh every 60 seconds (increased from 30s to avoid conflicts)
    const interval = setInterval(fetchDecision, 60000);

    return () => {
      isMounted = false;
      clearInterval(interval);
      if (controller) {
        try {
          controller.abort();
        } catch (err) {
          // Suppress AbortError during cleanup (expected behavior)
        }
      }
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [symbol]);

  const getDecisionColor = (decision: string): string => {
    switch (decision) {
      case 'STRONG_BUY':
        return COLORS.success;
      case 'BUY':
        return '#00D084';
      case 'HOLD':
        return COLORS.warning;
      case 'SELL':
        return COLORS.danger;
      case 'STRONG_SELL':
        return '#DC143C';
      default:
        return COLORS.text.secondary;
    }
  };

  const getDecisionGradient = (decision: string): string => {
    switch (decision) {
      case 'STRONG_BUY':
        return `linear-gradient(135deg, ${COLORS.success}, ${COLORS.premium})`;
      case 'BUY':
        return `linear-gradient(135deg, #00D084, ${COLORS.success})`;
      case 'HOLD':
        return `linear-gradient(135deg, ${COLORS.warning}, #FFA500)`;
      case 'SELL':
        return `linear-gradient(135deg, ${COLORS.danger}, #FF6347)`;
      case 'STRONG_SELL':
        return 'linear-gradient(135deg, #DC143C, #8B0000)';
      default:
        return `linear-gradient(135deg, ${COLORS.bg.secondary}, ${COLORS.border.default})`;
    }
  };

  const getDecisionEmoji = (decision: string): string => {
    switch (decision) {
      case 'STRONG_BUY':
        return '🚀';
      case 'BUY':
        return '📈';
      case 'HOLD':
        return '⏸️';
      case 'SELL':
        return '📉';
      case 'STRONG_SELL':
        return '🔻';
      default:
        return '❓';
    }
  };

  const formatPrice = (price: number): string => {
    return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercent = (value: number): string => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="neon-card" style={{
        padding: '40px',
        marginBottom: '24px',
        background: `linear-gradient(135deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
        border: `2px solid ${COLORS.border.default}`
      }}>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <LoadingAnimation />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="neon-card" style={{
        padding: '30px',
        marginBottom: '24px',
        background: `${COLORS.danger}10`,
        border: `2px solid ${COLORS.danger}40`
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Icons.AlertTriangle style={{ width: '24px', height: '24px', color: COLORS.danger }} />
          <div>
            <div style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.danger }}>
              Karar Analizi Hatası
            </div>
            <div style={{ fontSize: '13px', color: COLORS.text.secondary, marginTop: '4px' }}>
              {error || 'Veri yüklenemedi'}
            </div>
          </div>
        </div>
      </div>
    );
  }

  const decisionColor = getDecisionColor(data.decision);
  const decisionGradient = getDecisionGradient(data.decision);
  const decisionEmoji = getDecisionEmoji(data.decision);

  return (
    <div className="neon-card" style={{
      padding: '0',
      marginBottom: '24px',
      background: COLORS.bg.primary,
      border: `3px solid ${decisionColor}`,
      boxShadow: `0 0 30px ${decisionColor}40`,
      overflow: 'hidden'
    }}>
      {/* Header - Large Decision Badge */}
      <div style={{
        background: decisionGradient,
        padding: '32px 24px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 60%)',
          pointerEvents: 'none'
        }}></div>

        <div style={{ position: 'relative', zIndex: 1 }}>
          <div style={{ fontSize: '14px', fontWeight: '600', color: 'rgba(255,255,255,0.9)', marginBottom: '8px', letterSpacing: '2px' }}>
            {symbol} KARAR ANALİZİ
          </div>
          <div style={{ fontSize: '56px', marginBottom: '8px' }}>
            {decisionEmoji}
          </div>
          <div style={{ fontSize: '42px', fontWeight: 'bold', color: 'white', marginBottom: '12px', letterSpacing: '1px' }}>
            {data.decision.replace('_', ' ')}
          </div>
          <div style={{ fontSize: '18px', color: 'rgba(255,255,255,0.95)', fontWeight: '500' }}>
            Güven: {(data.confidence * 100).toFixed(0)}% | R/R: {data.riskRewardRatio.toFixed(2)}
          </div>
        </div>
      </div>

      <div style={{ padding: '24px' }}>
        {/* Summary */}
        <div style={{
          padding: '20px',
          borderRadius: '12px',
          background: `${decisionColor}15`,
          border: `1px solid ${decisionColor}30`,
          marginBottom: '24px',
          fontSize: '15px',
          color: COLORS.text.primary,
          fontWeight: '500',
          lineHeight: '1.6'
        }}>
          {data.summary}
        </div>

        {/* Price Levels Grid */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Icons.Target style={{ width: '20px', height: '20px', color: COLORS.premium }} />
            <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
              Fiyat Seviyeleri
            </h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
            {/* Current Price */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.info}15`,
              border: `1px solid ${COLORS.info}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Güncel Fiyat
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.info }}>
                {formatPrice(data.currentPrice)}
              </div>
            </div>

            {/* Entry Price */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.premium}15`,
              border: `1px solid ${COLORS.premium}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Giriş Fiyatı
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium }}>
                {formatPrice(data.entryPrice)}
              </div>
            </div>

            {/* Stop Loss */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.danger}15`,
              border: `1px solid ${COLORS.danger}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Stop Loss
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.danger }}>
                {formatPrice(data.stopLoss)}
              </div>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                {formatPercent(((data.stopLoss - data.currentPrice) / data.currentPrice) * 100)}
              </div>
            </div>

            {/* TP1 */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.success}15`,
              border: `1px solid ${COLORS.success}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Hedef 1 (TP1)
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                {formatPrice(data.targets.tp1)}
              </div>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                {formatPercent(((data.targets.tp1 - data.currentPrice) / data.currentPrice) * 100)}
              </div>
            </div>

            {/* TP2 */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.success}15`,
              border: `1px solid ${COLORS.success}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Hedef 2 (TP2)
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                {formatPrice(data.targets.tp2)}
              </div>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                {formatPercent(((data.targets.tp2 - data.currentPrice) / data.currentPrice) * 100)}
              </div>
            </div>

            {/* TP3 */}
            <div style={{
              padding: '16px',
              borderRadius: '10px',
              background: `${COLORS.success}15`,
              border: `1px solid ${COLORS.success}40`
            }}>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
                Hedef 3 (TP3)
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                {formatPrice(data.targets.tp3)}
              </div>
              <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                {formatPercent(((data.targets.tp3 - data.currentPrice) / data.currentPrice) * 100)}
              </div>
            </div>
          </div>
        </div>

        {/* Risk/Reward & Signal Counts */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '24px' }}>
          <div style={{
            padding: '18px',
            borderRadius: '10px',
            background: `${COLORS.premium}15`,
            border: `1px solid ${COLORS.premium}40`
          }}>
            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
              Risk/Reward Oranı
            </div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.premium }}>
              1:{data.riskRewardRatio.toFixed(2)}
            </div>
            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '6px' }}>
              Kar: {formatPrice(data.potentialGain)} | Zarar: {formatPrice(data.potentialLoss)}
            </div>
          </div>

          <div style={{
            padding: '18px',
            borderRadius: '10px',
            background: `${COLORS.success}15`,
            border: `1px solid ${COLORS.success}40`
          }}>
            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
              Alış Sinyalleri
            </div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.success }}>
              {data.buySignalsCount}/{data.totalStrategies}
            </div>
            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '6px' }}>
              {((data.buySignalsCount / data.totalStrategies) * 100).toFixed(0)}% Strateji
            </div>
          </div>

          <div style={{
            padding: '18px',
            borderRadius: '10px',
            background: `${COLORS.danger}15`,
            border: `1px solid ${COLORS.danger}40`
          }}>
            <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '6px', fontWeight: '600' }}>
              Satış Sinyalleri
            </div>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.danger }}>
              {data.sellSignalsCount}/{data.totalStrategies}
            </div>
            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '6px' }}>
              {((data.sellSignalsCount / data.totalStrategies) * 100).toFixed(0)}% Strateji
            </div>
          </div>
        </div>

        {/* Strongest Signals */}
        {data.strongestSignals && data.strongestSignals.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
              <Icons.TrendingUp style={{ width: '20px', height: '20px', color: COLORS.premium }} />
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                En Güçlü Sinyaller
              </h3>
            </div>

            <div style={{ display: 'grid', gap: '10px' }}>
              {data.strongestSignals.map((signal, index) => (
                <div key={index} style={{
                  padding: '14px 16px',
                  borderRadius: '8px',
                  background: COLORS.bg.secondary,
                  border: `1px solid ${COLORS.border.default}`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '14px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '4px' }}>
                      {index + 1}. {signal.name}
                    </div>
                    <div style={{ fontSize: '12px', color: COLORS.text.secondary, fontStyle: 'italic' }}>
                      {signal.reason}
                    </div>
                  </div>
                  <div style={{
                    background: signal.signal.toLowerCase().includes('buy') ? `${COLORS.success}25` :
                               signal.signal.toLowerCase().includes('sell') ? `${COLORS.danger}25` : `${COLORS.warning}25`,
                    color: signal.signal.toLowerCase().includes('buy') ? COLORS.success :
                           signal.signal.toLowerCase().includes('sell') ? COLORS.danger : COLORS.warning,
                    fontSize: '11px',
                    fontWeight: '700',
                    padding: '6px 12px',
                    borderRadius: '6px',
                    marginLeft: '12px',
                    whiteSpace: 'nowrap'
                  }}>
                    {(signal.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Reasons */}
        {data.reasons && data.reasons.length > 0 && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
              <Icons.Newspaper style={{ width: '20px', height: '20px', color: COLORS.info }} />
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                Gerekçeler
              </h3>
            </div>

            <div style={{ display: 'grid', gap: '8px' }}>
              {data.reasons.map((reason, index) => (
                <div key={index} style={{
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: `${COLORS.info}10`,
                  border: `1px solid ${COLORS.info}30`,
                  fontSize: '13px',
                  color: COLORS.text.primary,
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px'
                }}>
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    background: `${COLORS.info}30`,
                    color: COLORS.info,
                    fontSize: '11px',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0
                  }}>
                    {index + 1}
                  </div>
                  <div style={{ flex: 1, paddingTop: '1px' }}>{reason}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div style={{ marginTop: '20px', textAlign: 'center', fontSize: '11px', color: COLORS.text.secondary }}>
          Son güncelleme: {new Date(data.timestamp).toLocaleString('tr-TR')}
        </div>
      </div>
    </div>
  );
}
