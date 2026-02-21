'use client';

/**
 * VOLUME PROFILE OVERLAY COMPONENT
 * Professional volume distribution visualization for charts
 *
 * Features:
 * - Real-time volume profile fetching
 * - POC, VAH, VAL visualization
 * - HVN/LVN zone highlighting
 * - Responsive histogram display
 * - Auto-refresh capability
 */

import React, { useState, useEffect } from 'react';

interface VolumeProfileData {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  percentage: number;
}

interface KeyLevels {
  poc: number;
  vah: number;
  val: number;
  valueAreaVolume: number;
}

interface VolumeProfileResponse {
  success: boolean;
  data: {
    symbol: string;
    interval: string;
    volumeProfile: VolumeProfileData[];
    keyLevels: KeyLevels;
    hvnLevels: number[];
    lvnLevels: number[];
    statistics: {
      totalVolume: number;
      avgVolume: number;
      priceRange: {
        high: number;
        low: number;
      };
    };
  };
  processingTime?: number;
  timestamp?: string;
}

interface VolumeProfileOverlayProps {
  symbol: string;
  interval: string;
  currentPrice: number;
  visible?: boolean;
  onClose?: () => void;
}

export default function VolumeProfileOverlay({
  symbol,
  interval,
  currentPrice,
  visible = true,
  onClose
}: VolumeProfileOverlayProps) {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<VolumeProfileResponse['data'] | null>(null);
  const [maxVolume, setMaxVolume] = useState<number>(0);

  // Fetch volume profile data
  useEffect(() => {
    if (!visible) return;

    const fetchVolumeProfile = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `/api/charts/volume-profile?symbol=${symbol}&interval=${interval}&limit=500&priceLevels=50`
        );

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const result: VolumeProfileResponse = await response.json();

        if (!result.success) {
          throw new Error('Failed to fetch volume profile');
        }

        setData(result.data);

        // Calculate max volume for scaling
        const max = Math.max(...result.data.volumeProfile.map(v => v.volume));
        setMaxVolume(max);
      } catch (err) {
        console.error('Error fetching volume profile:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchVolumeProfile();

    // Auto-refresh based on interval
    const refreshInterval = interval === '1m' || interval === '3m'
      ? 30000 // 30 seconds for short timeframes
      : interval === '5m' || interval === '15m'
      ? 60000 // 1 minute for medium timeframes
      : 120000; // 2 minutes for longer timeframes

    const intervalId = setInterval(fetchVolumeProfile, refreshInterval);

    return () => clearInterval(intervalId);
  }, [symbol, interval, visible]);

  // Don't render if not visible
  if (!visible) return null;

  // Loading state
  if (loading && !data) {
    return (
      <div className="absolute right-0 top-0 w-64 h-full bg-gray-900/95 border-l border-gray-700 p-4 overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold text-sm">Volume Profile</h3>
          {onClose && (
            <button onClick={onClose} className="text-gray-400 hover:text-white">
              ✕
            </button>
          )}
        </div>
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        </div>
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className="absolute right-0 top-0 w-64 h-full bg-gray-900/95 border-l border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold text-sm">Volume Profile</h3>
          {onClose && (
            <button onClick={onClose} className="text-gray-400 hover:text-white">
              ✕
            </button>
          )}
        </div>
        <div className="text-red-400 text-xs">
          {error || 'Failed to load volume profile'}
        </div>
      </div>
    );
  }

  const { volumeProfile, keyLevels, hvnLevels, lvnLevels, statistics } = data;

  // Helper: Check if price is in HVN or LVN
  const isHVN = (price: number) => hvnLevels.some(lvl => Math.abs(lvl - price) < 1);
  const isLVN = (price: number) => lvnLevels.some(lvl => Math.abs(lvl - price) < 1);
  const isPOC = (price: number) => Math.abs(price - keyLevels.poc) < 1;
  const isVAH = (price: number) => Math.abs(price - keyLevels.vah) < 1;
  const isVAL = (price: number) => Math.abs(price - keyLevels.val) < 1;

  return (
    <div className="absolute right-0 top-0 w-64 h-full bg-gray-900/95 border-l border-gray-700 overflow-hidden flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div>
          <h3 className="text-white font-semibold text-sm">Volume Profile</h3>
          <div className="text-gray-400 text-xs mt-0.5">{symbol} • {interval}</div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      {/* Key Levels Summary */}
      <div className="p-3 bg-gray-800/50 border-b border-gray-700 space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">POC (Point of Control):</span>
          <span className="text-yellow-400 font-semibold">${keyLevels.poc.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">VAH (Value Area High):</span>
          <span className="text-green-400 font-semibold">${keyLevels.vah.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">VAL (Value Area Low):</span>
          <span className="text-red-400 font-semibold">${keyLevels.val.toLocaleString()}</span>
        </div>
      </div>

      {/* Volume Profile Histogram */}
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {volumeProfile.map((profile, index) => {
          const barWidth = (profile.volume / maxVolume) * 100;
          const buyWidth = (profile.buyVolume / profile.volume) * 100;
          const isCurrentPrice = Math.abs(profile.price - currentPrice) < (statistics.priceRange.high - statistics.priceRange.low) * 0.02;

          return (
            <div
              key={index}
              className={`relative group ${isCurrentPrice ? 'ring-2 ring-blue-500' : ''}`}
            >
              {/* Price Label */}
              <div className="flex items-center justify-between text-xs mb-0.5">
                <span
                  className={`font-mono ${
                    isPOC(profile.price) ? 'text-yellow-400 font-bold' :
                    isVAH(profile.price) ? 'text-green-400 font-semibold' :
                    isVAL(profile.price) ? 'text-red-400 font-semibold' :
                    isHVN(profile.price) ? 'text-emerald-300' :
                    isLVN(profile.price) ? 'text-orange-300' :
                    'text-gray-400'
                  }`}
                >
                  ${profile.price.toLocaleString()}
                </span>
                <span className="text-gray-500 text-[10px]">
                  {profile.percentage.toFixed(1)}%
                </span>
              </div>

              {/* Volume Bar */}
              <div className="relative h-4 bg-gray-800 rounded overflow-hidden">
                {/* Total volume bar */}
                <div
                  className={`absolute left-0 top-0 h-full transition-all ${
                    isPOC(profile.price) ? 'bg-yellow-500/40' :
                    isHVN(profile.price) ? 'bg-emerald-500/30' :
                    isLVN(profile.price) ? 'bg-orange-500/20' :
                    'bg-blue-500/20'
                  }`}
                  style={{ width: `${barWidth}%` }}
                />

                {/* Buy volume overlay */}
                <div
                  className="absolute left-0 top-0 h-full bg-green-500/30"
                  style={{ width: `${(barWidth * buyWidth) / 100}%` }}
                />

                {/* POC marker */}
                {isPOC(profile.price) && (
                  <div className="absolute left-0 top-0 w-1 h-full bg-yellow-400" />
                )}

                {/* Current price marker */}
                {isCurrentPrice && (
                  <div className="absolute right-0 top-0 w-1 h-full bg-blue-400 animate-pulse" />
                )}
              </div>

              {/* Tooltip on hover */}
              <div className="hidden group-hover:block absolute left-full ml-2 top-0 bg-gray-800 border border-gray-600 rounded p-2 text-xs whitespace-nowrap z-50 shadow-lg">
                <div className="text-white font-semibold mb-1">
                  ${profile.price.toLocaleString()}
                </div>
                <div className="text-gray-300">
                  Volume: {profile.volume.toLocaleString()}
                </div>
                <div className="text-green-400">
                  Buy: {profile.buyVolume.toLocaleString()}
                </div>
                <div className="text-red-400">
                  Sell: {profile.sellVolume.toLocaleString()}
                </div>
                <div className="text-gray-400 mt-1">
                  {profile.percentage.toFixed(2)}% of total
                </div>
                {isPOC(profile.price) && (
                  <div className="text-yellow-400 mt-1 font-semibold">
                    🎯 Point of Control
                  </div>
                )}
                {isHVN(profile.price) && (
                  <div className="text-emerald-400 mt-1">
                    💪 High Volume Node
                  </div>
                )}
                {isLVN(profile.price) && (
                  <div className="text-orange-400 mt-1">
                    ⚠️ Low Volume Node
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Statistics Footer */}
      <div className="p-3 bg-gray-800/50 border-t border-gray-700 space-y-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">HVN Levels:</span>
          <span className="text-emerald-400 font-semibold">{hvnLevels.length}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">LVN Levels:</span>
          <span className="text-orange-400 font-semibold">{lvnLevels.length}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Total Volume:</span>
          <span className="text-blue-400 font-semibold">
            {(statistics.totalVolume / 1000000).toFixed(2)}M
          </span>
        </div>
      </div>

      {/* Legend */}
      <div className="p-2 bg-gray-800/30 border-t border-gray-700">
        <div className="text-[10px] text-gray-500 space-y-0.5">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-yellow-500 rounded" />
            <span>POC - Highest volume</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-500 rounded" />
            <span>HVN - Strong S/R</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-orange-500 rounded" />
            <span>LVN - Weak S/R</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded" />
            <span className="mr-2">Buy Volume</span>
            <div className="w-2 h-2 bg-red-500 rounded" />
            <span>Sell Volume</span>
          </div>
        </div>
      </div>
    </div>
  );
}
