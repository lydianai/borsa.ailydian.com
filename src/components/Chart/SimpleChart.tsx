'use client';

import { useEffect, useState } from 'react';
import { COLORS } from '@/lib/colors';

interface SimpleChartProps {
  symbol: string;
  interval: string;
  marketType?: 'crypto' | 'traditional';
}

export default function SimpleChart({ symbol, interval, marketType = 'crypto' }: SimpleChartProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Use appropriate API endpoint based on market type
        const apiEndpoint = marketType === 'traditional'
          ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=100`
          : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=100`;

        const response = await fetch(apiEndpoint);
        const result = await response.json();

        if (result.success) {
          setData(result.data);
        } else {
          setError(result.error || 'Failed to load data');
        }
      } catch (err) {
        setError('Failed to fetch chart data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol, interval, marketType]);

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '600px',
        background: 'rgba(0, 0, 0, 0.3)',
        borderRadius: '12px'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '4px solid rgba(0, 212, 255, 0.3)',
            borderTop: '4px solid #00D4FF',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <div style={{ color: '#00D4FF', fontSize: '14px' }}>
            Loading {symbol} {interval} chart...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '600px',
        background: 'rgba(239, 68, 68, 0.1)',
        borderRadius: '12px',
        border: '1px solid rgba(239, 68, 68, 0.3)'
      }}>
        <div style={{
          textAlign: 'center',
          color: COLORS.danger,
          padding: '20px'
        }}>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
            ⚠️ Error Loading Chart
          </div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>
            {error}
          </div>
        </div>
      </div>
    );
  }

  if (!data || !data.klines || data.klines.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '600px',
        background: 'rgba(255, 255, 255, 0.03)',
        borderRadius: '12px'
      }}>
        <div style={{
          textAlign: 'center',
          color: 'rgba(255, 255, 255, 0.7)',
          padding: '20px'
        }}>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
            📊 No Data Available
          </div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>
            No chart data for {symbol} ({interval})
          </div>
        </div>
      </div>
    );
  }

  // Calculate stats
  const latestCandle = data.klines[data.klines.length - 1];
  const firstCandle = data.klines[0];
  const priceChange = latestCandle.close - firstCandle.open;
  const priceChangePercent = ((priceChange / firstCandle.open) * 100).toFixed(2);
  const isPositive = priceChange >= 0;

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.3)',
      borderRadius: '12px',
      padding: '20px',
      minHeight: '600px'
    }}>
      {/* Chart Header Stats */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        padding: '16px',
        background: 'rgba(255, 255, 255, 0.03)',
        borderRadius: '8px'
      }}>
        <div>
          <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
            Symbol
          </div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
            {symbol}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
            Latest Close
          </div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
            ${latestCandle.close.toLocaleString('en-US', { maximumFractionDigits: 2 })}
          </div>
        </div>

        <div>
          <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
            Change
          </div>
          <div style={{
            fontSize: '18px',
            fontWeight: '700',
            color: isPositive ? COLORS.success : COLORS.danger
          }}>
            {isPositive ? '+' : ''}{priceChangePercent}%
          </div>
        </div>

        <div>
          <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
            Candles
          </div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
            {data.klines.length}
          </div>
        </div>
      </div>

      {/* Simple ASCII-style Chart Visualization */}
      <div style={{
        background: 'rgba(0, 0, 0, 0.5)',
        borderRadius: '8px',
        padding: '20px',
        fontFamily: 'monospace',
        fontSize: '12px',
        color: '#00D4FF',
        height: '400px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '12px'
      }}>
        <div style={{ fontSize: '24px', marginBottom: '8px' }}>📊</div>
        <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
          Chart Display
        </div>
        <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)', textAlign: 'center', maxWidth: '400px' }}>
          {symbol} • {interval} • {data.klines.length} candles loaded
        </div>
        <div style={{
          marginTop: '12px',
          padding: '8px 16px',
          background: isPositive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          borderRadius: '6px',
          color: isPositive ? COLORS.success : COLORS.danger,
          fontSize: '14px',
          fontWeight: '600'
        }}>
          {isPositive ? '↑' : '↓'} {priceChangePercent}% ({data.klines.length} periods)
        </div>
      </div>

      {/* Support/Resistance Info */}
      {data.support && data.resistance && (
        <div style={{
          display: 'flex',
          gap: '16px',
          marginTop: '16px',
          padding: '16px',
          background: 'rgba(255, 255, 255, 0.02)',
          borderRadius: '8px'
        }}>
          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: '12px',
              fontWeight: '600',
              color: COLORS.success,
              marginBottom: '8px'
            }}>
              Support Levels ({data.support.length})
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
              {data.support.slice(0, 3).map((level: number, i: number) => (
                <div key={i}>S{i + 1}: ${level.toFixed(2)}</div>
              ))}
            </div>
          </div>

          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: '12px',
              fontWeight: '600',
              color: COLORS.danger,
              marginBottom: '8px'
            }}>
              Resistance Levels ({data.resistance.length})
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
              {data.resistance.slice(0, 3).map((level: number, i: number) => (
                <div key={i}>R{i + 1}: ${level.toFixed(2)}</div>
              ))}
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
