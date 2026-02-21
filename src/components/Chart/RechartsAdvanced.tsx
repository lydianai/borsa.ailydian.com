'use client';

import { useEffect, useState } from 'react';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { calculateRSI, calculateMFI } from '@/lib/indicators';
import { COLORS } from '@/lib/colors';

interface RechartsAdvancedProps {
  symbol: string;
  interval: string;
  marketType?: 'crypto' | 'traditional';
}

interface KlineData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface ChartDataPoint {
  time: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  candleColor: string;
  rsi?: number;
  mfi?: number;
}

export default function RechartsAdvanced({ symbol, interval, marketType = 'crypto' }: RechartsAdvancedProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [stats, setStats] = useState<{ rsiAvg: number; mfiAvg: number; latestRSI: number; latestMFI: number } | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const apiEndpoint = marketType === 'traditional'
          ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=200`
          : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=200`;

        console.log(`[RechartsAdvanced] Fetching: ${apiEndpoint}`);

        const response = await fetch(apiEndpoint, { cache: 'no-store' });
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to fetch data');
        }

        const candles = result.data.klines as KlineData[];
        console.log(`[RechartsAdvanced] Received ${candles.length} candles`);

        // Calculate indicators
        const rsiData = calculateRSI(candles, 14);
        const mfiData = calculateMFI(candles, 14);

        // Create map for quick lookup
        const rsiMap = new Map(rsiData.map(r => [r.time, r.value]));
        const mfiMap = new Map(mfiData.map(m => [m.time, m.value]));

        // Transform data for Recharts
        const transformed: ChartDataPoint[] = candles.map((candle) => {
          const date = new Date(candle.time * 1000);
          const timeStr = date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });

          return {
            time: timeStr,
            timestamp: candle.time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
            candleColor: candle.close >= candle.open ? '#10b981' : '#ef4444',
            rsi: rsiMap.get(candle.time),
            mfi: mfiMap.get(candle.time),
          };
        });

        setChartData(transformed);

        // Calculate stats
        const rsiValues = rsiData.map(r => r.value);
        const mfiValues = mfiData.map(m => m.value);
        const rsiAvg = rsiValues.reduce((a, b) => a + b, 0) / rsiValues.length;
        const mfiAvg = mfiValues.reduce((a, b) => a + b, 0) / mfiValues.length;
        const latestRSI = rsiValues[rsiValues.length - 1] || 0;
        const latestMFI = mfiValues[mfiValues.length - 1] || 0;

        setStats({ rsiAvg, mfiAvg, latestRSI, latestMFI });
        setLoading(false);
      } catch (err) {
        console.error('[RechartsAdvanced] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval, marketType]);

  const _CandlestickShape = (props: any) => {
    const { x, y, width, height, payload } = props;
    if (!payload) return null;

    const { open, close, high, low, candleColor } = payload;
    const _isGreen = close >= open;

    const wickX = x + width / 2;
    const bodyTop = Math.min(open, close);
    const bodyBottom = Math.max(open, close);
    const _bodyHeight = Math.abs(close - open);

    // Scale factors (need to calculate based on chart domain)
    const priceRange = high - low;
    const pixelsPerPrice = height / priceRange;

    const highY = y;
    const lowY = y + height;
    const bodyTopY = y + (high - bodyBottom) * pixelsPerPrice;
    const bodyBottomY = y + (high - bodyTop) * pixelsPerPrice;

    return (
      <g>
        {/* Wick (tail) */}
        <line
          x1={wickX}
          y1={highY}
          x2={wickX}
          y2={bodyTopY}
          stroke={candleColor}
          strokeWidth={1}
        />
        <line
          x1={wickX}
          y1={bodyBottomY}
          x2={wickX}
          y2={lowY}
          stroke={candleColor}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x + 1}
          y={bodyTopY}
          width={Math.max(width - 2, 1)}
          height={Math.max(bodyBottomY - bodyTopY, 1)}
          fill={candleColor}
          stroke={candleColor}
        />
      </g>
    );
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          background: 'rgba(0, 0, 0, 0.9)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '8px',
          padding: '12px',
          color: '#fff',
          fontSize: '12px'
        }}>
          <div style={{ marginBottom: '8px', fontWeight: '600' }}>{data.time}</div>
          <div style={{ color: '#10b981' }}>O: ${data.open.toFixed(2)}</div>
          <div style={{ color: '#00D4FF' }}>H: ${data.high.toFixed(2)}</div>
          <div style={{ color: '#FFA500' }}>L: ${data.low.toFixed(2)}</div>
          <div style={{ color: data.candleColor }}>C: ${data.close.toFixed(2)}</div>
          <div style={{ color: '#888', marginTop: '4px' }}>Vol: {data.volume.toFixed(0)}</div>
          {data.rsi && <div style={{ color: '#2962FF', marginTop: '4px' }}>RSI: {data.rsi.toFixed(1)}</div>}
          {data.mfi && <div style={{ color: '#9C27B0' }}>MFI: {data.mfi.toFixed(1)}</div>}
        </div>
      );
    }
    return null;
  };

  if (error) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '700px',
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
            ⚠️ Chart Error
          </div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>
            {error}
          </div>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '700px',
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
            Loading chart...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.3)',
      borderRadius: '12px',
      padding: '20px'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px'
      }}>
        <div>
          <div style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
            {symbol}
          </div>
          <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
            Interval: {interval} • {chartData.length} candles
          </div>
        </div>
        {stats && (
          <div style={{ display: 'flex', gap: '20px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: '#2962FF', marginBottom: '2px' }}>RSI (14)</div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#2962FF' }}>
                {stats.latestRSI.toFixed(1)}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: '#9C27B0', marginBottom: '2px' }}>MFI (14)</div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#9C27B0' }}>
                {stats.latestMFI.toFixed(1)}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Price Chart with Volume */}
      <div style={{ marginBottom: '20px' }}>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="price"
              domain={['dataMin - 50', 'dataMax + 50']}
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
            />
            <YAxis
              yAxisId="volume"
              orientation="right"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="rgba(0, 212, 255, 0.3)"
              opacity={0.3}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="#00D4FF"
              strokeWidth={2}
              dot={false}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="high"
              stroke="#10b981"
              strokeWidth={1}
              dot={false}
              strokeDasharray="3 3"
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="low"
              stroke="#ef4444"
              strokeWidth={1}
              dot={false}
              strokeDasharray="3 3"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* RSI Chart */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ fontSize: '14px', fontWeight: '600', color: '#2962FF', marginBottom: '8px' }}>
          RSI (14) - Relative Strength Index
        </div>
        <ResponsiveContainer width="100%" height={150}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '70', fill: '#ef4444', fontSize: 10 }} />
            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" label={{ value: '30', fill: '#10b981', fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="rsi"
              stroke="#2962FF"
              strokeWidth={2}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* MFI Chart */}
      <div>
        <div style={{ fontSize: '14px', fontWeight: '600', color: '#9C27B0', marginBottom: '8px' }}>
          MFI (14) - Money Flow Index
        </div>
        <ResponsiveContainer width="100%" height={150}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.5)"
              tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '80', fill: '#ef4444', fontSize: 10 }} />
            <ReferenceLine y={20} stroke="#10b981" strokeDasharray="3 3" label={{ value: '20', fill: '#10b981', fontSize: 10 }} />
            <Line
              type="monotone"
              dataKey="mfi"
              stroke="#9C27B0"
              strokeWidth={2}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
