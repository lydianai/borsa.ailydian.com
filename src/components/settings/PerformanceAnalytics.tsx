'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

export default function PerformanceAnalytics({ onSave: _onSave }: any) {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/performance-analytics');
      const data = await response.json();
      if (data.success) setMetrics(data.data);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-loading">
          {(Icons.Loader2 as any)({ className: "animate-spin", size: 32 })}
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">Analiz verileri yüklenemedi</div>
      </div>
    );
  }

  return (
    <div className="settings-content-wrapper">
      {/* Stats Grid */}
      <div className="settings-grid-3 mb-6">
        <div className="settings-stat-card">
          {(Icons.BarChart3 as any)({ style: { color: '#3B82F6' }, size: 32 })}
          <div className="stat-value">{metrics.overview.totalSignals}</div>
          <div className="stat-label">Toplam Sinyal</div>
        </div>
        <div className="settings-stat-card">
          {(Icons.CheckCircle2 as any)({ style: { color: '#10B981' }, size: 32 })}
          <div className="stat-value">{metrics.overview.winRate.toFixed(1)}%</div>
          <div className="stat-label">Kazanma Oranı</div>
        </div>
        <div className="settings-stat-card">
          {(Icons.Wallet as any)({ style: { color: '#10B981' }, size: 32 })}
          <div className="stat-value">${metrics.overview.totalProfit.toLocaleString()}</div>
          <div className="stat-label">Toplam Kar</div>
        </div>
      </div>

      <div className="settings-grid-2 mb-6">
        {/* Strategy Performance */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            {(Icons.TrendingUp as any)({ style: { color: '#10B981' }, size: 24 })}
            <h3>Strateji Performansı</h3>
          </div>
          <div className="settings-card-body">
            <div className="space-y-3">
              {Object.entries(metrics.strategyPerformance).map(([name, stats]: [string, any]) => (
                <div key={name} className="bg-black border border-white/30 rounded-lg p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-white">{name}</span>
                    <span className="text-sm text-white">
                      {stats.winRate.toFixed(1)}% KO
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs text-gray-400">
                    <div>{stats.signals} sinyal</div>
                    <div>${stats.profit.toFixed(0)} kar</div>
                    <div>{stats.avgResponseTime}ms ort</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Signals */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            {(Icons.Zap as any)({ style: { color: '#F59E0B' }, size: 24 })}
            <h3>Son Sinyaller</h3>
          </div>
          <div className="settings-card-body">
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {metrics.recentSignals.map((signal: any, i: number) => (
                <div key={i} className="bg-black border border-white/30 rounded-lg p-3 flex justify-between items-center">
                  <div>
                    <div className="font-semibold text-sm text-white">{signal.symbol}</div>
                    <div className="text-xs text-gray-400">{new Date(signal.timestamp).toLocaleString()}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-bold text-white">
                      {signal.type}
                    </div>
                    {signal.profit !== null && (
                      <div className="text-xs text-white">
                        {signal.profit > 0 ? '+' : ''}{signal.profit.toFixed(2)} USDT
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Daily Performance Chart */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          {(Icons.BarChart3 as any)({ style: { color: '#8B5CF6' }, size: 24 })}
          <h3>Günlük Performans (Son 30 Gün)</h3>
        </div>
        <div className="settings-card-body">
          <div className="h-48 flex items-end gap-1">
            {metrics.timeBasedMetrics.daily.slice(-30).map((day: any, i: number) => {
              const maxSignals = Math.max(...metrics.timeBasedMetrics.daily.map((d: any) => d.signals));
              const height = (day.signals / maxSignals) * 100;
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <div
                    className="w-full rounded-t bg-white/80 border border-white"
                    style={{ height: `${height}%` }}
                    title={`${day.date}: ${day.signals} sinyal, $${day.profit.toFixed(0)}`}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
