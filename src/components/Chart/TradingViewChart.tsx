'use client';

import { useEffect, useState, useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';
import { calculateRSI, calculateMFI } from '@/lib/indicators';
import { COLORS } from '@/lib/colors';

interface TradingViewChartProps {
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
  candleRange: [number, number];
  wickRange: [number, number];
  isGreen: boolean;
  rsi?: number;
  mfi?: number;
}

export default function TradingViewChart({ symbol, interval, marketType = 'crypto' }: TradingViewChartProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rawData, setRawData] = useState<KlineData[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Validation: Check if symbol matches market type
        const isCryptoSymbol = symbol.endsWith('USDT') || symbol.endsWith('BUSD');
        const isTraditionalSymbol = !isCryptoSymbol;

        // Skip fetch if there's a mismatch (prevents race condition during market type switch)
        if (marketType === 'traditional' && isCryptoSymbol) {
          console.log(`[TradingViewChart] Skipping fetch: crypto symbol ${symbol} with traditional market type`);
          return;
        }
        if (marketType === 'crypto' && isTraditionalSymbol) {
          console.log(`[TradingViewChart] Skipping fetch: traditional symbol ${symbol} with crypto market type`);
          return;
        }

        setLoading(true);
        setError(null);

        const apiEndpoint = marketType === 'traditional'
          ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=100`
          : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=100`;

        console.log(`[TradingViewChart] Fetching ${marketType}: ${symbol} ${interval} from ${apiEndpoint}`);

        const response = await fetch(apiEndpoint, { cache: 'no-store' });
        const result = await response.json();

        if (!result.success) {
          console.error(`[TradingViewChart] API Error for ${symbol}:`, result.error);
          throw new Error(result.error || 'Failed to fetch data');
        }

        const candles = result.data.klines as KlineData[];
        console.log(`[TradingViewChart] Loaded ${candles.length} candles for ${symbol}`);

        setRawData(candles);
        setLoading(false);
      } catch (err) {
        console.error('[TradingViewChart] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval, marketType]);

  // Memoize processed data to prevent unnecessary recalculations
  const { chartData, stats, priceRange } = useMemo(() => {
    if (rawData.length === 0) {
      return { chartData: [], stats: null, priceRange: { min: 0, max: 0 } };
    }

    // Calculate indicators
    const rsiData = calculateRSI(rawData, 14);
    const mfiData = calculateMFI(rawData, 14);

    const rsiMap = new Map(rsiData.map(r => [r.time, r.value]));
    const mfiMap = new Map(mfiData.map(m => [m.time, m.value]));

    // Transform data
    const transformed: ChartDataPoint[] = rawData.map((candle) => {
      const date = new Date(candle.time * 1000);
      const timeStr = interval.includes('d') || interval.includes('w')
        ? date.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short' })
        : date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });

      const isGreen = candle.close >= candle.open;

      return {
        time: timeStr,
        timestamp: candle.time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        candleRange: [Math.min(candle.open, candle.close), Math.max(candle.open, candle.close)],
        wickRange: [candle.low, candle.high],
        isGreen,
        rsi: rsiMap.get(candle.time),
        mfi: mfiMap.get(candle.time),
      };
    });

    // Calculate stats
    const rsiValues = rsiData.map(r => r.value);
    const mfiValues = mfiData.map(m => m.value);
    const latestRSI = rsiValues[rsiValues.length - 1] || 0;
    const latestMFI = mfiValues[mfiValues.length - 1] || 0;

    const latestCandle = transformed[transformed.length - 1];
    const firstCandle = transformed[0];
    const priceChange = latestCandle.close - firstCandle.open;
    const priceChangePercent = ((priceChange / firstCandle.open) * 100).toFixed(2);

    // Calculate price range for better Y-axis scaling
    const allPrices = rawData.flatMap(c => [c.high, c.low]);
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const padding = (maxPrice - minPrice) * 0.05;

    return {
      chartData: transformed,
      stats: {
        latestRSI,
        latestMFI,
        latestClose: latestCandle.close,
        priceChange: Number(priceChange.toFixed(2)),
        priceChangePercent: Number(priceChangePercent),
        isPositive: priceChange >= 0,
      },
      priceRange: {
        min: minPrice - padding,
        max: maxPrice + padding,
      }
    };
  }, [rawData, interval]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload;

    return (
      <div style={{
        background: 'rgba(10, 10, 10, 0.95)',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        borderRadius: '8px',
        padding: '12px',
        fontSize: '12px'
      }}>
        <div style={{ color: '#fff', fontWeight: '600', marginBottom: '8px' }}>{data.time}</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
          <div style={{ color: '#10b981' }}>O:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.open.toFixed(2)}</div>

          <div style={{ color: '#00D4FF' }}>H:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.high.toFixed(2)}</div>

          <div style={{ color: '#FFA500' }}>L:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.low.toFixed(2)}</div>

          <div style={{ color: data.isGreen ? '#10b981' : '#ef4444' }}>C:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.close.toFixed(2)}</div>
        </div>
        <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
          <div style={{ color: '#888' }}>Vol: {data.volume.toLocaleString()}</div>
          {data.rsi && <div style={{ color: '#2962FF' }}>RSI: {data.rsi.toFixed(1)}</div>}
          {data.mfi && <div style={{ color: '#9C27B0' }}>MFI: {data.mfi.toFixed(1)}</div>}
        </div>
      </div>
    );
  };

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
          textAlign: 'center',
          color: '#00D4FF'
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            border: '4px solid rgba(0, 212, 255, 0.3)',
            borderTop: '4px solid #00D4FF',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            margin: '0 auto 12px'
          }} />
          <div>Grafik yükleniyor...</div>
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
        height: '700px',
        background: 'rgba(239, 68, 68, 0.1)',
        borderRadius: '12px',
        border: '1px solid rgba(239, 68, 68, 0.3)'
      }}>
        <div style={{ textAlign: 'center', color: COLORS.danger, padding: '20px' }}>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
            ⚠️ Grafik Hatası
          </div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>{error}</div>
        </div>
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.4)',
      borderRadius: '12px',
      padding: '20px',
      border: '1px solid rgba(255, 255, 255, 0.1)'
    }}>
      {/* Header with Stats */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        paddingBottom: '16px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
            {symbol}
          </div>
          <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
            {interval} • {chartData.length} mumlar
          </div>
        </div>

        <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '2px' }}>
              Fiyat
            </div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>
              ${stats.latestClose.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div style={{
              fontSize: '14px',
              fontWeight: '600',
              color: stats.isPositive ? '#10b981' : '#ef4444'
            }}>
              {stats.isPositive ? '+' : ''}{stats.priceChangePercent}% ({stats.isPositive ? '+' : ''}${Math.abs(stats.priceChange).toFixed(2)})
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#2962FF', marginBottom: '2px' }}>RSI (14)</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#2962FF' }}>
              {stats.latestRSI.toFixed(1)}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#9C27B0', marginBottom: '2px' }}>MFI (14)</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#9C27B0' }}>
              {stats.latestMFI.toFixed(1)}
            </div>
          </div>
        </div>
      </div>

      {/* Main Price Chart */}
      <div style={{ marginBottom: '16px' }}>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00D4FF" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#00D4FF" stopOpacity={0.05}/>
              </linearGradient>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="price"
              domain={[priceRange.min, priceRange.max]}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <YAxis
              yAxisId="volume"
              orientation="right"
              stroke="rgba(255,255,255,0.2)"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
              tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Volume Bars */}
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="url(#volumeGradient)"
              opacity={0.5}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.isGreen ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'} />
              ))}
            </Bar>

            {/* Price Lines - High, Low, Close */}
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="high"
              stroke="#10b981"
              strokeWidth={1}
              dot={false}
              strokeDasharray="2 2"
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="low"
              stroke="#ef4444"
              strokeWidth={1}
              dot={false}
              strokeDasharray="2 2"
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
              dataKey="open"
              stroke="rgba(255,255,255,0.4)"
              strokeWidth={1}
              dot={false}
              strokeDasharray="1 1"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* RSI Indicator */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#2962FF', marginBottom: '8px' }}>
          RSI (14)
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
            <ReferenceLine y={50} stroke="rgba(255,255,255,0.2)" strokeDasharray="2 2" />
            <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
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

      {/* MFI Indicator */}
      <div>
        <div style={{ fontSize: '13px', fontWeight: '600', color: '#9C27B0', marginBottom: '8px' }}>
          MFI (14)
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <ComposedChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
            <ReferenceLine y={50} stroke="rgba(255,255,255,0.2)" strokeDasharray="2 2" />
            <ReferenceLine y={20} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
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
