'use client';

/**
 * WHALE ACTIVITY BADGE COMPONENT
 * Displays on-chain whale activity as a small badge
 *
 * Usage:
 * <WhaleActivityBadge symbol="BTCUSDT" />
 *
 * Features:
 * - Auto-fetches whale data
 * - Shows accumulation/distribution/neutral
 * - Color-coded by risk
 * - Tooltip with details
 * - Non-intrusive design
 */

import { useEffect, useState } from 'react';

interface WhaleActivityData {
  activity: 'accumulation' | 'distribution' | 'neutral';
  confidence: number;
  riskScore: number;
  summary: string;
}

interface WhaleActivityBadgeProps {
  symbol: string;
  size?: 'sm' | 'md' | 'lg';
  showTooltip?: boolean;
}

export function WhaleActivityBadge({
  symbol,
  size = 'sm',
  showTooltip = true,
}: WhaleActivityBadgeProps) {
  const [whaleData, setWhaleData] = useState<WhaleActivityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    async function fetchWhaleActivity() {
      try {
        const response = await fetch(`/api/onchain/whale-alerts?symbol=${symbol}`);
        const data = await response.json();

        if (data.success && data.data) {
          setWhaleData({
            activity: data.data.activity,
            confidence: data.data.confidence,
            riskScore: data.data.riskScore,
            summary: data.data.summary,
          });
        }
      } catch (error) {
        console.warn(`[WhaleActivityBadge] Failed to fetch data for ${symbol}:`, error);
      } finally {
        setLoading(false);
      }
    }

    fetchWhaleActivity();
  }, [symbol]);

  if (loading || !whaleData) {
    return null; // Don't show anything while loading or if no data
  }

  // Only show badge for significant whale activity (confidence >= 30)
  if (whaleData.confidence < 30) {
    return null;
  }

  // Determine badge appearance based on activity and risk
  let icon = '🐋';
  let bgColor = 'bg-blue-500/10';
  let textColor = 'text-blue-400';
  let borderColor = 'border-blue-500/30';

  if (whaleData.activity === 'accumulation') {
    icon = '🟢';
    bgColor = 'bg-green-500/10';
    textColor = 'text-green-400';
    borderColor = 'border-green-500/30';
  } else if (whaleData.activity === 'distribution') {
    icon = '🔴';
    bgColor = 'bg-red-500/10';
    textColor = 'text-red-400';
    borderColor = 'border-red-500/30';
  }

  // Size variants
  const sizeClasses = {
    sm: 'px-1.5 py-0.5 text-[10px]',
    md: 'px-2 py-1 text-xs',
    lg: 'px-3 py-1.5 text-sm',
  };

  return (
    <div className="relative inline-block">
      <div
        className={`
          inline-flex items-center gap-1 rounded-full border
          ${bgColor} ${textColor} ${borderColor} ${sizeClasses[size]}
          font-medium cursor-pointer transition-all duration-200
          hover:scale-105 hover:shadow-sm
        `}
        onMouseEnter={() => showTooltip && setShowDetails(true)}
        onMouseLeave={() => setShowDetails(false)}
        onClick={() => setShowDetails(!showDetails)}
      >
        <span>{icon}</span>
        {size !== 'sm' && (
          <span className="capitalize">
            {whaleData.activity === 'accumulation'
              ? 'Acc'
              : whaleData.activity === 'distribution'
              ? 'Dist'
              : 'Neutral'}
          </span>
        )}
      </div>

      {/* Tooltip */}
      {showTooltip && showDetails && (
        <div className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64">
          <div className="bg-gray-900 text-white rounded-lg shadow-xl border border-gray-700 p-3 text-xs">
            <div className="font-semibold mb-2 text-sm">{whaleData.summary}</div>
            <div className="space-y-1 text-gray-300">
              <div className="flex justify-between">
                <span>Confidence:</span>
                <span className="font-medium">{whaleData.confidence}%</span>
              </div>
              <div className="flex justify-between">
                <span>Risk Score:</span>
                <span
                  className={`font-medium ${
                    whaleData.riskScore >= 70
                      ? 'text-red-400'
                      : whaleData.riskScore >= 50
                      ? 'text-yellow-400'
                      : 'text-green-400'
                  }`}
                >
                  {whaleData.riskScore}/100
                </span>
              </div>
            </div>
            <div className="mt-2 pt-2 border-t border-gray-700 text-[10px] text-gray-400">
              On-chain whale movement detected
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
