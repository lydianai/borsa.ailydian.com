'use client';

/**
 * INDICATOR PANEL COMPONENT
 * Toggleable panel for enabling/disabling advanced chart indicators
 *
 * Features:
 * - Volume Profile toggle
 * - VWAP (Volume Weighted Average Price)
 * - Order Flow / Heatmap
 * - Liquidation Levels
 * - Funding Rate visualization
 * - Custom indicator settings
 */

import React, { useState as _useState } from 'react';

export interface IndicatorSettings {
  volumeProfile: boolean;
  vwap: boolean;
  orderFlow: boolean;
  liquidationLevels: boolean;
  fundingRate: boolean;
  supportResistance: boolean;
}

interface IndicatorPanelProps {
  visible: boolean;
  onClose: () => void;
  settings: IndicatorSettings;
  onSettingsChange: (settings: IndicatorSettings) => void;
}

export default function IndicatorPanel({
  visible,
  onClose,
  settings,
  onSettingsChange
}: IndicatorPanelProps) {
  if (!visible) return null;

  const toggleIndicator = (key: keyof IndicatorSettings) => {
    onSettingsChange({
      ...settings,
      [key]: !settings[key]
    });
  };

  return (
    <div className="absolute left-0 top-0 w-72 bg-gray-900/95 border-r border-gray-700 overflow-y-auto max-h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-800/50">
        <h3 className="text-white font-semibold">📊 Advanced Indicators</h3>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ✕
        </button>
      </div>

      {/* Indicators List */}
      <div className="p-4 space-y-3">
        {/* Volume Profile */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.volumeProfile
              ? 'bg-blue-500/20 border-blue-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('volumeProfile')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">📊</span>
                <span className="text-white font-semibold text-sm">
                  Volume Profile
                </span>
                {settings.volumeProfile && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                POC, VAH, VAL with HVN/LVN zones
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.volumeProfile ? 'bg-blue-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.volumeProfile ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>

        {/* VWAP */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.vwap
              ? 'bg-purple-500/20 border-purple-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('vwap')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">📈</span>
                <span className="text-white font-semibold text-sm">VWAP</span>
                {settings.vwap && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                Volume Weighted Average Price
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.vwap ? 'bg-purple-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.vwap ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>

        {/* Order Flow / Heatmap */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.orderFlow
              ? 'bg-orange-500/20 border-orange-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('orderFlow')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">🔥</span>
                <span className="text-white font-semibold text-sm">
                  Order Flow Heatmap
                </span>
                {settings.orderFlow && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                Buy/Sell pressure visualization
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.orderFlow ? 'bg-orange-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.orderFlow ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>

        {/* Liquidation Levels */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.liquidationLevels
              ? 'bg-red-500/20 border-red-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('liquidationLevels')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">⚡</span>
                <span className="text-white font-semibold text-sm">
                  Liquidation Levels
                </span>
                {settings.liquidationLevels && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                Long/Short liquidation clusters
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.liquidationLevels ? 'bg-red-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.liquidationLevels ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>

        {/* Funding Rate */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.fundingRate
              ? 'bg-cyan-500/20 border-cyan-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('fundingRate')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">💰</span>
                <span className="text-white font-semibold text-sm">
                  Funding Rate
                </span>
                {settings.fundingRate && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                Perpetual futures funding rate
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.fundingRate ? 'bg-cyan-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.fundingRate ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>

        {/* Support & Resistance */}
        <div
          className={`p-3 rounded-lg border transition-all cursor-pointer ${
            settings.supportResistance
              ? 'bg-emerald-500/20 border-emerald-500'
              : 'bg-gray-800 border-gray-700 hover:border-gray-600'
          }`}
          onClick={() => toggleIndicator('supportResistance')}
        >
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-lg">🎯</span>
                <span className="text-white font-semibold text-sm">
                  Support & Resistance
                </span>
                {settings.supportResistance && (
                  <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-semibold rounded">
                    ACTIVE
                  </span>
                )}
              </div>
              <p className="text-gray-400 text-xs mt-1 ml-7">
                Automatic S/R level detection
              </p>
            </div>
            <div
              className={`w-10 h-5 rounded-full transition-colors ${
                settings.supportResistance ? 'bg-emerald-500' : 'bg-gray-600'
              }`}
            >
              <div
                className={`w-4 h-4 bg-white rounded-full m-0.5 transition-transform ${
                  settings.supportResistance ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Information Panel */}
      <div className="p-4 border-t border-gray-700 bg-gray-800/30">
        <h4 className="text-white font-semibold text-xs mb-2">
          ℹ️ Indicator Information
        </h4>
        <div className="space-y-2 text-xs text-gray-400">
          <div>
            <span className="text-blue-400 font-semibold">Volume Profile:</span>
            <span className="ml-1">
              Visualizes price levels with highest trading volume. POC (Point of Control) shows
              the price with most traded volume.
            </span>
          </div>
          <div>
            <span className="text-purple-400 font-semibold">VWAP:</span>
            <span className="ml-1">
              Average price weighted by volume. Used to identify trend direction and fair value.
            </span>
          </div>
          <div>
            <span className="text-orange-400 font-semibold">Order Flow:</span>
            <span className="ml-1">
              Shows buy/sell pressure at each price level. Green = buying, Red = selling.
            </span>
          </div>
          <div>
            <span className="text-red-400 font-semibold">Liquidations:</span>
            <span className="ml-1">
              Clusters of liquidated positions. High concentration often leads to volatile moves.
            </span>
          </div>
          <div>
            <span className="text-cyan-400 font-semibold">Funding Rate:</span>
            <span className="ml-1">
              Periodic payment for perp futures. Positive = longs pay shorts. Negative = opposite.
            </span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="p-4 border-t border-gray-700 bg-gray-800/50">
        <div className="flex gap-2">
          <button
            onClick={() =>
              onSettingsChange({
                volumeProfile: true,
                vwap: true,
                orderFlow: true,
                liquidationLevels: true,
                fundingRate: true,
                supportResistance: true
              })
            }
            className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-xs font-semibold rounded transition-colors"
          >
            Enable All
          </button>
          <button
            onClick={() =>
              onSettingsChange({
                volumeProfile: false,
                vwap: false,
                orderFlow: false,
                liquidationLevels: false,
                fundingRate: false,
                supportResistance: false
              })
            }
            className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded transition-colors"
          >
            Disable All
          </button>
        </div>
      </div>

      {/* Active Indicators Summary */}
      <div className="p-3 bg-blue-500/10 border-t border-blue-500/30">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400">Active Indicators:</span>
          <span className="text-blue-400 font-semibold">
            {Object.values(settings).filter(Boolean).length} / 6
          </span>
        </div>
      </div>
    </div>
  );
}
