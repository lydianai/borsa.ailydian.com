'use client';

import { useEffect, useState, useMemo, useCallback } from 'react';
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
  ReferenceArea,
  Cell,
  Rectangle,
  Scatter,
  Customized,
} from 'recharts';
import { calculateRSI, calculateMFI, calculateMACD, calculateBollingerBands, calculateEMA } from '@/lib/indicators';
import {
  detectSupportResistance,
  calculateFibonacci,
  detectOrderBlocks,
  detectFairValueGaps,
  calculateVolumeProfile,
  findSwingPoints,
  detectTrend,
  calculateValueArea,
  type Candle,
  type SupportResistanceLevel,
  type FibonacciLevel,
  type OrderBlock,
  type FairValueGap,
  type VolumeProfile,
} from '@/lib/chart-analysis';
import { COLORS } from '@/lib/colors';

interface ProfessionalTradingChartProps {
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

interface ProcessedCandle {
  time: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  isBullish: boolean;
  bodyHigh: number;
  bodyLow: number;
  rsi?: number;
  mfi?: number;
  emaFast?: number;
  emaSlow?: number;
  bbUpper?: number;
  bbMiddle?: number;
  bbLower?: number;
}

export default function ProfessionalTradingChart({
  symbol,
  interval,
  marketType = 'crypto',
}: ProfessionalTradingChartProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rawData, setRawData] = useState<KlineData[]>([]);

  // Feature toggles
  const [showSR, setShowSR] = useState(true);
  const [showFib, setShowFib] = useState(false);
  const [showOB, setShowOB] = useState(true);
  const [showFVG, setShowFVG] = useState(true);
  const [showVolProfile, setShowVolProfile] = useState(true);
  const [showEMA, setShowEMA] = useState(true);
  const [showBB, setShowBB] = useState(false);
  const [showRSI, setShowRSI] = useState(true);
  const [showMACD, setShowMACD] = useState(false);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const isCryptoSymbol = symbol.endsWith('USDT') || symbol.endsWith('BUSD');
        const isTraditionalSymbol = !isCryptoSymbol;

        if (marketType === 'traditional' && isCryptoSymbol) {
          console.log(`[ProfessionalChart] Skipping: crypto symbol with traditional type`);
          return;
        }
        if (marketType === 'crypto' && isTraditionalSymbol) {
          console.log(`[ProfessionalChart] Skipping: traditional symbol with crypto type`);
          return;
        }

        setLoading(true);
        setError(null);

        const apiEndpoint =
          marketType === 'traditional'
            ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=200`
            : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=200`;

        console.log(`[ProfessionalChart] Fetching: ${apiEndpoint}`);

        const response = await fetch(apiEndpoint, { cache: 'no-store' });
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to fetch data');
        }

        const candles = result.data.klines as KlineData[];
        console.log(`[ProfessionalChart] Loaded ${candles.length} candles`);

        setRawData(candles);
        setLoading(false);
      } catch (err) {
        console.error('[ProfessionalChart] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval, marketType]);

  // Process data and calculate all indicators
  const chartAnalysis = useMemo(() => {
    if (rawData.length === 0) {
      return {
        processedData: [],
        supportLevels: [],
        resistanceLevels: [],
        fibLevels: [],
        orderBlocks: [],
        fvgs: [],
        volumeProfile: [],
        valueArea: { high: 0, low: 0, poc: 0 },
        trend: 'sideways' as const,
        stats: null,
      };
    }

    // Calculate indicators
    const rsiData = calculateRSI(rawData, 14);
    const mfiData = calculateMFI(rawData, 14);
    const emaFast = calculateEMA(rawData.map(c => c.close), 9);
    const emaSlow = calculateEMA(rawData.map(c => c.close), 21);
    const bb = calculateBollingerBands(rawData, 20, 2);

    const rsiMap = new Map(rsiData.map(r => [r.time, r.value]));
    const mfiMap = new Map(mfiData.map(m => [m.time, m.value]));
    const bbMap = new Map(bb.map(b => [b.time, b]));

    // Process candles
    const processed: ProcessedCandle[] = rawData.map((candle, idx) => {
      const date = new Date(candle.time * 1000);
      const timeStr = interval.includes('d') || interval.includes('w')
        ? date.toLocaleDateString('tr-TR', { day: '2-digit', month: 'short' })
        : date.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' });

      const isBullish = candle.close >= candle.open;
      const bbValues = bbMap.get(candle.time);

      return {
        time: timeStr,
        timestamp: candle.time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        isBullish,
        bodyHigh: Math.max(candle.open, candle.close),
        bodyLow: Math.min(candle.open, candle.close),
        rsi: rsiMap.get(candle.time),
        mfi: mfiMap.get(candle.time),
        emaFast: emaFast[idx],
        emaSlow: emaSlow[idx],
        bbUpper: bbValues?.upper,
        bbMiddle: bbValues?.middle,
        bbLower: bbValues?.lower,
      };
    });

    // Detect support/resistance
    const srLevels = detectSupportResistance(rawData, 20, 0.002);
    const supportLevels = srLevels.filter(l => l.type === 'support');
    const resistanceLevels = srLevels.filter(l => l.type === 'resistance');

    // Calculate Fibonacci (find highest and lowest in recent period)
    const recent50 = rawData.slice(-50);
    const highPrice = Math.max(...recent50.map(c => c.high));
    const lowPrice = Math.min(...recent50.map(c => c.low));
    const trend = detectTrend(rawData, 20);
    const fibLevels = calculateFibonacci(highPrice, lowPrice, trend === 'uptrend' ? 'uptrend' : 'downtrend');

    // SMC features
    const orderBlocks = detectOrderBlocks(rawData, 50);
    const fvgs = detectFairValueGaps(rawData);

    // Volume Profile
    const volumeProfile = calculateVolumeProfile(rawData, 24);
    const valueArea = calculateValueArea(volumeProfile);

    // Stats
    const latestCandle = processed[processed.length - 1];
    const firstCandle = processed[0];
    const priceChange = latestCandle.close - firstCandle.open;
    const priceChangePercent = ((priceChange / firstCandle.open) * 100).toFixed(2);

    const rsiValues = rsiData.map(r => r.value);
    const mfiValues = mfiData.map(m => m.value);
    const latestRSI = rsiValues[rsiValues.length - 1] || 0;
    const latestMFI = mfiValues[mfiValues.length - 1] || 0;

    return {
      processedData: processed,
      supportLevels,
      resistanceLevels,
      fibLevels,
      orderBlocks,
      fvgs,
      volumeProfile,
      valueArea,
      trend,
      stats: {
        latestRSI,
        latestMFI,
        latestClose: latestCandle.close,
        priceChange: Number(priceChange.toFixed(2)),
        priceChangePercent: Number(priceChangePercent),
        isPositive: priceChange >= 0,
        highPrice,
        lowPrice,
      },
    };
  }, [rawData, interval]);


  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload;

    return (
      <div
        style={{
          background: 'rgba(10, 10, 10, 0.95)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '8px',
          padding: '12px',
          fontSize: '11px',
          minWidth: '180px',
        }}
      >
        <div style={{ color: '#fff', fontWeight: '600', marginBottom: '8px', fontSize: '12px' }}>
          {data.time}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', marginBottom: '8px' }}>
          <div style={{ color: '#10b981' }}>O:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.open.toFixed(2)}</div>
          <div style={{ color: '#00D4FF' }}>H:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.high.toFixed(2)}</div>
          <div style={{ color: '#FFA500' }}>L:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.low.toFixed(2)}</div>
          <div style={{ color: data.isBullish ? '#10b981' : '#ef4444' }}>C:</div>
          <div style={{ color: '#fff', textAlign: 'right' }}>${data.close.toFixed(2)}</div>
        </div>
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '6px', marginTop: '6px' }}>
          <div style={{ color: '#888', marginBottom: '2px' }}>Vol: {data.volume.toLocaleString()}</div>
          {data.rsi && <div style={{ color: '#2962FF' }}>RSI: {data.rsi.toFixed(1)}</div>}
          {data.mfi && <div style={{ color: '#9C27B0' }}>MFI: {data.mfi.toFixed(1)}</div>}
          {data.emaFast && <div style={{ color: '#00D4FF' }}>EMA(9): ${data.emaFast.toFixed(2)}</div>}
          {data.emaSlow && <div style={{ color: '#FFA500' }}>EMA(21): ${data.emaSlow.toFixed(2)}</div>}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '800px',
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '12px',
        }}
      >
        <div style={{ textAlign: 'center', color: '#00D4FF' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              border: '4px solid rgba(0, 212, 255, 0.3)',
              borderTop: '4px solid #00D4FF',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 12px',
            }}
          />
          <div>Grafik yükleniyor...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '800px',
          background: 'rgba(239, 68, 68, 0.1)',
          borderRadius: '12px',
          border: '1px solid rgba(239, 68, 68, 0.3)',
        }}
      >
        <div style={{ textAlign: 'center', color: COLORS.danger, padding: '20px' }}>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>⚠️ Grafik Hatası</div>
          <div style={{ fontSize: '14px', opacity: 0.8 }}>{error}</div>
        </div>
      </div>
    );
  }

  if (!chartAnalysis.stats) return null;

  const { processedData, supportLevels, resistanceLevels, fibLevels, orderBlocks, fvgs, volumeProfile, valueArea, trend, stats } =
    chartAnalysis;

  return (
    <div
      style={{
        background: 'rgba(0, 0, 0, 0.4)',
        borderRadius: '12px',
        padding: '20px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '20px',
          paddingBottom: '16px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
            {symbol}
          </div>
          <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
            {interval} • {processedData.length} mumlar • Trend: {trend === 'uptrend' ? '📈 Yükseliş' : trend === 'downtrend' ? '📉 Düşüş' : '➡️ Yatay'}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '2px' }}>Fiyat</div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>
              ${stats.latestClose.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
            <div style={{ fontSize: '14px', fontWeight: '600', color: stats.isPositive ? '#10b981' : '#ef4444' }}>
              {stats.isPositive ? '+' : ''}
              {stats.priceChangePercent}% (${Math.abs(stats.priceChange).toFixed(2)})
            </div>
          </div>

          {showRSI && (
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '12px', color: '#2962FF', marginBottom: '2px' }}>RSI (14)</div>
              <div style={{ fontSize: '20px', fontWeight: '700', color: '#2962FF' }}>{stats.latestRSI.toFixed(1)}</div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                {stats.latestRSI > 70 ? 'Aşırı Alım' : stats.latestRSI < 30 ? 'Aşırı Satım' : 'Nötr'}
              </div>
            </div>
          )}

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#9C27B0', marginBottom: '2px' }}>MFI (14)</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#9C27B0' }}>{stats.latestMFI.toFixed(1)}</div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
              {stats.latestMFI > 80 ? 'Aşırı Alım' : stats.latestMFI < 20 ? 'Aşırı Satım' : 'Nötr'}
            </div>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          marginBottom: '16px',
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setShowSR(!showSR)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showSR ? 'rgba(0, 212, 255, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showSR ? '#00D4FF' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          Destek/Direnç
        </button>
        <button
          onClick={() => setShowFib(!showFib)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showFib ? 'rgba(255, 165, 0, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showFib ? '#FFA500' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          Fibonacci
        </button>
        <button
          onClick={() => setShowOB(!showOB)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showOB ? 'rgba(156, 39, 176, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showOB ? '#9C27B0' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          Order Blocks
        </button>
        <button
          onClick={() => setShowFVG(!showFVG)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showFVG ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showFVG ? '#10b981' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          FVG/Imbalance
        </button>
        <button
          onClick={() => setShowEMA(!showEMA)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showEMA ? 'rgba(41, 98, 255, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showEMA ? '#2962FF' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          EMA (9/21)
        </button>
        <button
          onClick={() => setShowBB(!showBB)}
          style={{
            padding: '6px 12px',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            background: showBB ? 'rgba(255, 193, 7, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            color: showBB ? '#FFC107' : 'rgba(255, 255, 255, 0.7)',
            fontSize: '12px',
            cursor: 'pointer',
          }}
        >
          Bollinger Bands
        </button>
      </div>

      {/* Main Chart */}
      <div style={{ marginBottom: '20px' }}>
        <ResponsiveContainer width="100%" height={500}>
          <ComposedChart data={processedData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="price"
              domain={['dataMin - 100', 'dataMax + 100']}
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 10 }}
            />
            <YAxis
              yAxisId="volume"
              orientation="right"
              stroke="rgba(255,255,255,0.2)"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 9 }}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Candlesticks - using Line for wicks */}
            {processedData.map((candle, idx) => (
              <ReferenceLine
                key={`candle-high-${idx}`}
                yAxisId="price"
                segment={[
                  { x: candle.time, y: candle.low },
                  { x: candle.time, y: candle.high },
                ]}
                stroke={candle.isBullish ? '#10b981' : '#ef4444'}
                strokeWidth={1.5}
                ifOverflow="extendDomain"
              />
            ))}

            {/* Candlestick bodies using ReferenceArea */}
            {processedData.map((candle, idx) => (
              <ReferenceArea
                key={`candle-body-${idx}`}
                yAxisId="price"
                x1={candle.time}
                x2={candle.time}
                y1={candle.open}
                y2={candle.close}
                fill={candle.isBullish ? '#10b981' : '#ef4444'}
                fillOpacity={0.9}
                stroke={candle.isBullish ? '#10b981' : '#ef4444'}
                strokeWidth={6}
                ifOverflow="extendDomain"
              />
            ))}

            {/* Volume bars */}
            <Bar yAxisId="volume" dataKey="volume" opacity={0.25}>
              {processedData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.isBullish ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)'} />
              ))}
            </Bar>

            {/* Support levels */}
            {showSR &&
              supportLevels.slice(0, 3).map((level, idx) => (
                <ReferenceLine
                  key={`support-${idx}`}
                  yAxisId="price"
                  y={level.price}
                  stroke="#10b981"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  opacity={0.6}
                  label={{
                    value: `S: $${level.price.toFixed(2)}`,
                    position: 'right',
                    fill: '#10b981',
                    fontSize: 10,
                  }}
                />
              ))}

            {/* Resistance levels */}
            {showSR &&
              resistanceLevels.slice(0, 3).map((level, idx) => (
                <ReferenceLine
                  key={`resistance-${idx}`}
                  yAxisId="price"
                  y={level.price}
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  opacity={0.6}
                  label={{
                    value: `R: $${level.price.toFixed(2)}`,
                    position: 'right',
                    fill: '#ef4444',
                    fontSize: 10,
                  }}
                />
              ))}

            {/* Fibonacci levels */}
            {showFib &&
              fibLevels.map((fib, idx) => (
                <ReferenceLine
                  key={`fib-${idx}`}
                  yAxisId="price"
                  y={fib.price}
                  stroke="#FFA500"
                  strokeDasharray="2 2"
                  strokeWidth={1}
                  opacity={0.4}
                  label={{
                    value: `${fib.label}`,
                    position: 'left',
                    fill: '#FFA500',
                    fontSize: 9,
                  }}
                />
              ))}

            {/* Fair Value Gaps */}
            {showFVG &&
              fvgs.slice(0, 5).map((fvg, idx) => {
                const startIndex = processedData.findIndex(d => d.timestamp >= fvg.startTime);
                const endIndex = processedData.length - 1;

                if (startIndex === -1) return null;

                return (
                  <ReferenceArea
                    key={`fvg-${idx}`}
                    yAxisId="price"
                    y1={fvg.high}
                    y2={fvg.low}
                    x1={processedData[startIndex]?.time}
                    x2={processedData[endIndex]?.time}
                    fill={fvg.type === 'bullish' ? '#10b981' : '#ef4444'}
                    fillOpacity={0.1}
                    stroke={fvg.type === 'bullish' ? '#10b981' : '#ef4444'}
                    strokeWidth={1}
                    strokeDasharray="3 3"
                    strokeOpacity={0.3}
                  />
                );
              })}

            {/* EMA lines */}
            {showEMA && (
              <>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="emaFast"
                  stroke="#00D4FF"
                  strokeWidth={1.5}
                  dot={false}
                  name="EMA(9)"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="emaSlow"
                  stroke="#FFA500"
                  strokeWidth={1.5}
                  dot={false}
                  name="EMA(21)"
                />
              </>
            )}

            {/* Bollinger Bands */}
            {showBB && (
              <>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbUpper"
                  stroke="#FFC107"
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="3 3"
                  opacity={0.5}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbMiddle"
                  stroke="#FFC107"
                  strokeWidth={1}
                  dot={false}
                  opacity={0.7}
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbLower"
                  stroke="#FFC107"
                  strokeWidth={1}
                  dot={false}
                  strokeDasharray="3 3"
                  opacity={0.5}
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* RSI Chart */}
      {showRSI && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ fontSize: '13px', fontWeight: '600', color: '#2962FF', marginBottom: '8px' }}>
            RSI (14) - Relative Strength Index
          </div>
          <ResponsiveContainer width="100%" height={120}>
            <ComposedChart data={processedData} margin={{ top: 5, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="time"
                stroke="rgba(255,255,255,0.3)"
                tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 9 }}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[0, 100]}
                stroke="rgba(255,255,255,0.3)"
                tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 9 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
              <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" opacity={0.5} />
              <Line type="monotone" dataKey="rsi" stroke="#2962FF" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          marginTop: '16px',
          padding: '12px',
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '8px',
          fontSize: '11px',
        }}
      >
        {showSR && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#10b981', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                Destek ({supportLevels.length})
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#ef4444', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                Direnç ({resistanceLevels.length})
              </span>
            </div>
          </>
        )}
        {showFVG && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '20px', height: '8px', background: 'rgba(16, 185, 129, 0.2)', border: '1px solid #10b981', borderRadius: '2px' }} />
            <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>FVG ({fvgs.length})</span>
          </div>
        )}
        {showOB && orderBlocks.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{ width: '20px', height: '8px', background: 'rgba(156, 39, 176, 0.2)', border: '1px solid #9C27B0', borderRadius: '2px' }} />
            <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>Order Blocks ({orderBlocks.length})</span>
          </div>
        )}
        {showEMA && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#00D4FF', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>EMA(9)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#FFA500', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>EMA(21)</span>
            </div>
          </>
        )}
        <div style={{ marginLeft: 'auto', color: 'rgba(255, 255, 255, 0.5)', fontSize: '10px' }}>
          POC: ${valueArea.poc.toFixed(2)} | VA: ${valueArea.low.toFixed(2)} - ${valueArea.high.toFixed(2)}
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
