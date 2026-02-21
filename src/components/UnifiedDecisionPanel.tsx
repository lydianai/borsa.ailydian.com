/**
 * UNIFIED DECISION PANEL v1.0
 *
 * Strategy synchronization visualization component
 *
 * Features:
 * - Real-time unified decision display
 * - Weighted votes chart
 * - Top 5 contributing strategies
 * - Conflict warnings
 * - Market condition badge
 * - Auto-refresh support
 *
 * Premium icon-only (no emojis!)
 */

'use client';

import { useState, useEffect, useRef } from 'react';
import { useStrategySync } from '@/hooks/useStrategySync';
import { useBinanceWebSocket } from '@/hooks/useBinanceWebSocket';
import { Icons } from '@/components/Icons';
import { audioNotificationService } from '@/lib/audio-notification-service';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface UnifiedDecisionPanelProps {
  symbol: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getRecommendationColor(recommendation: string): string {
  switch (recommendation) {
    case 'STRONG_BUY':
      return 'bg-green-500/20 border-green-500 text-green-400';
    case 'BUY':
      return 'bg-green-500/10 border-green-600 text-green-300';
    case 'NEUTRAL':
      return 'bg-gray-500/20 border-gray-500 text-gray-400';
    case 'WAIT':
      return 'bg-yellow-500/20 border-yellow-500 text-yellow-400';
    case 'SELL':
      return 'bg-red-500/10 border-red-600 text-red-300';
    case 'STRONG_SELL':
      return 'bg-red-500/20 border-red-500 text-red-400';
    default:
      return 'bg-gray-500/20 border-gray-500 text-gray-400';
  }
}

function getSignalIcon(signal: string) {
  switch (signal) {
    case 'BUY':
      return <Icons.TrendingUp className="w-4 h-4" />;
    case 'SELL':
      return <Icons.TrendingDown className="w-4 h-4" />;
    case 'WAIT':
      return <Icons.Clock className="w-4 h-4" />;
    case 'NEUTRAL':
      return <Icons.Minus className="w-4 h-4" />;
    default:
      return <Icons.CircleAlert className="w-4 h-4" />;
  }
}

function getMarketConditionBadge(condition: string) {
  switch (condition) {
    case 'volatile':
      return (
        <div className="inline-flex items-center gap-1 px-2 py-1 bg-orange-500/20 border border-orange-500 text-orange-400 rounded text-xs">
          <Icons.Zap className="w-3 h-3" />
          <span>Volatile</span>
        </div>
      );
    case 'trending':
      return (
        <div className="inline-flex items-center gap-1 px-2 py-1 bg-blue-500/20 border border-blue-500 text-blue-400 rounded text-xs">
          <Icons.TrendingUp className="w-3 h-3" />
          <span>Trending</span>
        </div>
      );
    case 'ranging':
      return (
        <div className="inline-flex items-center gap-1 px-2 py-1 bg-purple-500/20 border border-purple-500 text-purple-400 rounded text-xs">
          <Icons.Activity className="w-3 h-3" />
          <span>Ranging</span>
        </div>
      );
    case 'uncertain':
      return (
        <div className="inline-flex items-center gap-1 px-2 py-1 bg-gray-500/20 border border-gray-500 text-gray-400 rounded text-xs">
          <Icons.CircleAlert className="w-3 h-3" />
          <span>Uncertain</span>
        </div>
      );
    default:
      return null;
  }
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export function UnifiedDecisionPanel({
  symbol,
  autoRefresh = true,
  refreshInterval = 30000,
}: UnifiedDecisionPanelProps) {
  const { data, loading, error, refresh, lastUpdate } = useStrategySync(symbol, {
    autoRefresh,
    interval: refreshInterval,
  });

  // WebSocket real-time price
  const { price: livePrice, priceChangePercent: livePriceChange, connected: wsConnected } = useBinanceWebSocket(symbol);

  // Audio notification state
  const [audioEnabled, setAudioEnabled] = useState(false);
  const previousRecommendationRef = useRef<string | null>(null);

  // Initialize audio state from service
  useEffect(() => {
    setAudioEnabled(audioNotificationService.isEnabled());
  }, []);

  // Play sound when recommendation changes to BUY or SELL
  useEffect(() => {
    if (!data || !audioEnabled) return;

    const currentRecommendation = data.unifiedDecision.recommendation;
    const previousRecommendation = previousRecommendationRef.current;

    // First load - don't play sound
    if (previousRecommendation === null) {
      previousRecommendationRef.current = currentRecommendation;
      return;
    }

    // Recommendation changed
    if (currentRecommendation !== previousRecommendation) {
      // Play sound for actionable signals
      if (currentRecommendation === 'STRONG_BUY' || currentRecommendation === 'BUY') {
        audioNotificationService.playSuccess();
      } else if (currentRecommendation === 'STRONG_SELL' || currentRecommendation === 'SELL') {
        audioNotificationService.playWarning();
      } else if (currentRecommendation === 'WAIT') {
        audioNotificationService.playInfo();
      }

      previousRecommendationRef.current = currentRecommendation;
    }
  }, [data, audioEnabled]);

  // ============================================================================
  // LOADING STATE
  // ============================================================================

  if (loading && !data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-center gap-3">
          <Icons.Refresh className="w-5 h-5 text-purple-400 animate-spin" />
          <span className="text-gray-400">Stratejiler analiz ediliyor...</span>
        </div>
      </div>
    );
  }

  // ============================================================================
  // ERROR STATE
  // ============================================================================

  if (error) {
    return (
      <div className="bg-gray-900 border border-red-500/50 rounded-lg p-6">
        <div className="flex items-start gap-3">
          <Icons.CircleAlert className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="font-bold text-red-400 mb-1">Hata</div>
            <div className="text-sm text-gray-400">{error}</div>
            <button
              onClick={refresh}
              className="mt-3 px-4 py-2 bg-red-500/20 border border-red-500 text-red-400 rounded-lg hover:bg-red-500/30 transition-all text-sm"
            >
              Tekrar Dene
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // NO DATA STATE
  // ============================================================================

  if (!data) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="text-center text-gray-500">Veri yok</div>
      </div>
    );
  }

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  const { unifiedDecision, weightedVotes, topContributors, conflicts, meta } = data;

  return (
    <div className="space-y-4">
      {/* UNIFIED DECISION HEADER */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-lg font-bold flex items-center gap-2">
                <Icons.Layers className="w-5 h-5 text-purple-400" />
                Birleşik Strateji Kararı
              </h2>
              {livePrice !== null && (
                <div className="flex items-center gap-2">
                  <div className="h-4 w-px bg-gray-700" />
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-gray-400">{symbol.replace('USDT', '')}</span>
                    <span className="font-mono font-bold text-white">
                      ${livePrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                    {livePriceChange !== null && (
                      <span className={`text-xs font-medium ${livePriceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {livePriceChange >= 0 ? '+' : ''}{livePriceChange.toFixed(2)}%
                      </span>
                    )}
                    {wsConnected && (
                      <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" title="Canlı" />
                    )}
                  </div>
                </div>
              )}
            </div>
            <p className="text-xs text-gray-500">
              {meta.strategiesCount} strateji analiz edildi • {meta.processingTimeMs}ms
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                const newState = audioNotificationService.toggle();
                setAudioEnabled(newState);
              }}
              className={`p-2 rounded-lg transition-all ${
                audioEnabled
                  ? 'bg-blue-500/20 text-blue-400 hover:bg-blue-500/30'
                  : 'hover:bg-gray-800 text-gray-400'
              }`}
              title={audioEnabled ? 'Ses bildirimlerini kapat' : 'Ses bildirimlerini aç'}
            >
              {audioEnabled ? (
                <Icons.Volume2 className="w-4 h-4" />
              ) : (
                <Icons.VolumeX className="w-4 h-4" />
              )}
            </button>
            <button
              onClick={refresh}
              className="p-2 hover:bg-gray-800 rounded-lg transition-all"
              disabled={loading}
              title="Yenile"
            >
              <Icons.Refresh className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* RECOMMENDATION */}
        <div className="mb-6">
          <div className={`inline-flex items-center gap-2 px-6 py-3 rounded-lg border ${getRecommendationColor(unifiedDecision.recommendation)}`}>
            {getSignalIcon(unifiedDecision.signal)}
            <span className="text-xl font-bold">{unifiedDecision.recommendation.replace('_', ' ')}</span>
          </div>
        </div>

        {/* METRICS GRID */}
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <div className="text-xs text-gray-500 mb-1">Güven</div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold text-white">{unifiedDecision.confidence}</div>
              <div className="text-sm text-gray-400">%</div>
            </div>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  unifiedDecision.confidence >= 80
                    ? 'bg-green-500'
                    : unifiedDecision.confidence >= 60
                    ? 'bg-yellow-500'
                    : 'bg-orange-500'
                }`}
                style={{ width: `${unifiedDecision.confidence}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <div className="text-xs text-gray-500 mb-1">Konsensüs</div>
            <div className="flex items-baseline gap-2">
              <div className="text-3xl font-bold text-white">{unifiedDecision.consensus}</div>
              <div className="text-sm text-gray-400">%</div>
            </div>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  unifiedDecision.consensus >= 70
                    ? 'bg-blue-500'
                    : unifiedDecision.consensus >= 50
                    ? 'bg-purple-500'
                    : 'bg-gray-500'
                }`}
                style={{ width: `${unifiedDecision.consensus}%` }}
              />
            </div>
          </div>

          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <div className="text-xs text-gray-500 mb-1">Piyasa Durumu</div>
            <div className="mt-2">{getMarketConditionBadge(unifiedDecision.marketCondition)}</div>
          </div>
        </div>

        {/* LAST UPDATE */}
        {lastUpdate && (
          <div className="text-xs text-gray-500 flex items-center gap-2">
            <Icons.Clock className="w-3 h-3" />
            <span>Son güncelleme: {lastUpdate.toLocaleTimeString('tr-TR')}</span>
          </div>
        )}
      </div>

      {/* WEIGHTED VOTES */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
          <Icons.BarChart className="w-4 h-4 text-blue-400" />
          Ağırlıklı Oylar
        </h3>

        <div className="space-y-3">
          {Object.entries(weightedVotes)
            .sort(([, a], [, b]) => b - a)
            .map(([signal, percentage]) => (
              <div key={signal}>
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    {getSignalIcon(signal)}
                    <span className="text-sm font-medium text-gray-300">{signal}</span>
                  </div>
                  <span className="text-sm font-bold text-gray-400">{percentage.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      signal === 'BUY'
                        ? 'bg-green-500'
                        : signal === 'SELL'
                        ? 'bg-red-500'
                        : signal === 'WAIT'
                        ? 'bg-yellow-500'
                        : 'bg-gray-500'
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* TOP CONTRIBUTORS */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h3 className="text-sm font-bold mb-4 flex items-center gap-2">
          <Icons.Award className="w-4 h-4 text-yellow-400" />
          En Etkili 5 Strateji
        </h3>

        <div className="space-y-2">
          {topContributors.map((contributor, idx) => (
            <div
              key={idx}
              className="bg-gray-800/50 border border-gray-700 rounded-lg p-3 flex items-center justify-between"
            >
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-200">{contributor.name}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Güven: {contributor.confidence}% • Etki: {contributor.impact}%
                </div>
              </div>
              <div className={`px-3 py-1 rounded text-xs flex items-center gap-1 ${
                contributor.signal === 'BUY'
                  ? 'bg-green-500/20 text-green-400'
                  : contributor.signal === 'SELL'
                  ? 'bg-red-500/20 text-red-400'
                  : contributor.signal === 'WAIT'
                  ? 'bg-yellow-500/20 text-yellow-400'
                  : 'bg-gray-500/20 text-gray-400'
              }`}>
                {getSignalIcon(contributor.signal)}
                <span>{contributor.signal}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* CONFLICT WARNINGS */}
      {conflicts && conflicts.length > 0 && (
        <div className="bg-orange-900/20 border border-orange-500/50 rounded-lg p-6">
          <h3 className="text-sm font-bold mb-3 flex items-center gap-2 text-orange-400">
            <Icons.AlertTriangle className="w-4 h-4" />
            Çelişki Uyarıları
          </h3>

          <div className="space-y-2">
            {conflicts.map((warning, idx) => (
              <div key={idx} className="text-sm text-orange-300 flex items-start gap-2">
                <Icons.CircleAlert className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{warning}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
