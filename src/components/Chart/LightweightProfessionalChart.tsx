'use client';

import { useEffect, useRef, useState, useMemo } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';
import { calculateRSI, calculateMFI, calculateEMA, calculateBollingerBands } from '@/lib/indicators';
import {
  detectSupportResistance,
  calculateFibonacci,
  detectOrderBlocks,
  detectFairValueGaps,
  detectTrend,
  type Candle,
  type SupportResistanceLevel,
  type FibonacciLevel,
  type OrderBlock,
  type FairValueGap,
} from '@/lib/chart-analysis';
import { COLORS } from '@/lib/colors';

interface LightweightProfessionalChartProps {
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

export default function LightweightProfessionalChart({
  symbol,
  interval,
  marketType = 'crypto',
}: LightweightProfessionalChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rawData, setRawData] = useState<KlineData[]>([]);
  const [isMounted, setIsMounted] = useState(false);
  const [chartReady, setChartReady] = useState(false);

  // Feature toggles
  const [showSR, setShowSR] = useState(true);
  const [showFib, setShowFib] = useState(false);
  const [showOB, setShowOB] = useState(false);
  const [showFVG, setShowFVG] = useState(false);
  const [showEMA, setShowEMA] = useState(true);
  const [showBB, setShowBB] = useState(false);

  // Mount check
  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!isMounted || !chartContainerRef.current) {
      console.log('[LightweightChart] Not ready:', { isMounted, hasRef: !!chartContainerRef.current });
      return;
    }

    // Small delay to ensure container has dimensions
    const initChart = () => {
      if (!chartContainerRef.current) return;

      try {
        console.log('[LightweightChart] Initializing chart...');

        const containerWidth = chartContainerRef.current.clientWidth;
        const containerHeight = chartContainerRef.current.clientHeight;
        console.log('[LightweightChart] Container dimensions:', { width: containerWidth, height: containerHeight });

        if (containerWidth === 0) {
          console.error('[LightweightChart] Container width is 0! Retrying in 100ms...');
          setTimeout(initChart, 100);
          return;
        }

        const chart = createChart(chartContainerRef.current, {
          layout: {
            background: { type: ColorType.Solid, color: '#0a0a0a' },
            textColor: '#d1d4dc',
          },
          grid: {
            vertLines: { color: 'rgba(42, 42, 42, 0.5)' },
            horzLines: { color: 'rgba(42, 42, 42, 0.5)' },
          },
          width: chartContainerRef.current.clientWidth,
          height: 600,
          rightPriceScale: {
            borderColor: '#2a2a2a',
            scaleMargins: {
              top: 0.1,
              bottom: 0.2,
            },
          },
          timeScale: {
            borderColor: '#2a2a2a',
            timeVisible: true,
            secondsVisible: false,
          },
          crosshair: {
            mode: 1,
            vertLine: {
              color: 'rgba(255, 255, 255, 0.3)',
              width: 1,
              style: 3,
            },
            horzLine: {
              color: 'rgba(255, 255, 255, 0.3)',
              width: 1,
              style: 3,
            },
          },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candlestickSeries = (chart as any).addCandlestickSeries({
          upColor: '#10b981',
          downColor: '#ef4444',
          borderUpColor: '#10b981',
          borderDownColor: '#ef4444',
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
        });
        candlestickSeriesRef.current = candlestickSeries;

        // Add volume series
        const volumeSeries = (chart as any).addHistogramSeries({
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: '',
        });
        volumeSeries.priceScale().applyOptions({
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        });
        volumeSeriesRef.current = volumeSeries;

        console.log('[LightweightChart] Chart initialized successfully');
        setChartReady(true);
      } catch (err) {
        console.error('[LightweightChart] Chart initialization error:', err);
        setError(err instanceof Error ? err.message : 'Chart initialization failed');
      }
    };

    // Call init with small delay
    const timer = setTimeout(initChart, 50);

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [isMounted]);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const isCryptoSymbol = symbol.endsWith('USDT') || symbol.endsWith('BUSD');
        const isTraditionalSymbol = !isCryptoSymbol;

        if (marketType === 'traditional' && isCryptoSymbol) {
          console.log(`[LightweightChart] Skipping: crypto symbol with traditional type`);
          return;
        }
        if (marketType === 'crypto' && isTraditionalSymbol) {
          console.log(`[LightweightChart] Skipping: traditional symbol with crypto type`);
          return;
        }

        setLoading(true);
        setError(null);

        const apiEndpoint =
          marketType === 'traditional'
            ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=500`
            : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=500`;

        console.log(`[LightweightChart] Fetching: ${apiEndpoint}`);

        const response = await fetch(apiEndpoint, { cache: 'no-store' });
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to fetch data');
        }

        const candles = result.data.klines as KlineData[];
        console.log(`[LightweightChart] Loaded ${candles.length} candles`);

        setRawData(candles);
        setLoading(false);
      } catch (err) {
        console.error('[LightweightChart] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval, marketType]);

  // Calculate analysis data
  const analysisData = useMemo(() => {
    if (rawData.length === 0) {
      return {
        supportLevels: [],
        resistanceLevels: [],
        fibLevels: [],
        orderBlocks: [],
        fvgs: [],
        trend: 'sideways' as const,
        stats: null,
      };
    }

    // Calculate indicators
    const rsiData = calculateRSI(rawData, 14);
    const mfiData = calculateMFI(rawData, 14);
    const emaFast = calculateEMA(rawData.map(c => c.close), 9);
    const emaSlow = calculateEMA(rawData.map(c => c.close), 21);

    // Detect support/resistance
    const srLevels = detectSupportResistance(rawData, 20, 0.002);
    const supportLevels = srLevels.filter(l => l.type === 'support');
    const resistanceLevels = srLevels.filter(l => l.type === 'resistance');

    // Calculate Fibonacci
    const recent50 = rawData.slice(-50);
    const highPrice = Math.max(...recent50.map(c => c.high));
    const lowPrice = Math.min(...recent50.map(c => c.low));
    const trend = detectTrend(rawData, 20);
    const fibLevels = calculateFibonacci(highPrice, lowPrice, trend === 'uptrend' ? 'uptrend' : 'downtrend');

    // SMC features
    const orderBlocks = detectOrderBlocks(rawData, 50);
    const fvgs = detectFairValueGaps(rawData);

    // Stats
    const latestCandle = rawData[rawData.length - 1];
    const firstCandle = rawData[0];
    const priceChange = latestCandle.close - firstCandle.open;
    const priceChangePercent = ((priceChange / firstCandle.open) * 100).toFixed(2);

    const rsiValues = rsiData.map(r => r.value);
    const mfiValues = mfiData.map(m => m.value);
    const latestRSI = rsiValues[rsiValues.length - 1] || 0;
    const latestMFI = mfiValues[mfiValues.length - 1] || 0;

    return {
      supportLevels,
      resistanceLevels,
      fibLevels,
      orderBlocks,
      fvgs,
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
        emaFast: emaFast[emaFast.length - 1],
        emaSlow: emaSlow[emaSlow.length - 1],
      },
    };
  }, [rawData]);

  // Update chart data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current || rawData.length === 0) return;

    // Prepare candlestick data
    const candleData: CandlestickData[] = rawData.map(candle => ({
      time: candle.time as Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    // Prepare volume data
    const volumeData = rawData.map(candle => ({
      time: candle.time as Time,
      value: candle.volume,
      color: candle.close >= candle.open ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)',
    }));

    candlestickSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [rawData]);

  // Add support/resistance lines
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current || !showSR) return;

    const { supportLevels, resistanceLevels } = analysisData;

    // Add support lines
    supportLevels.slice(0, 3).forEach(level => {
      candlestickSeriesRef.current?.createPriceLine({
        price: level.price,
        color: '#10b981',
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: `S: ${level.price.toFixed(2)}`,
      });
    });

    // Add resistance lines
    resistanceLevels.slice(0, 3).forEach(level => {
      candlestickSeriesRef.current?.createPriceLine({
        price: level.price,
        color: '#ef4444',
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: true,
        title: `R: ${level.price.toFixed(2)}`,
      });
    });

    return () => {
      // Remove all price lines when component unmounts or showSR changes
      if (candlestickSeriesRef.current) {
        // Note: lightweight-charts doesn't have a direct way to remove individual price lines
        // We need to recreate the series to clear them
      }
    };
  }, [analysisData, showSR]);

  // Add EMA lines
  useEffect(() => {
    if (!chartRef.current || rawData.length === 0 || !showEMA) return;

    const emaFast = calculateEMA(rawData.map(c => c.close), 9);
    const emaSlow = calculateEMA(rawData.map(c => c.close), 21);

    const emaFastSeries = (chartRef.current as any).addLineSeries({
      color: '#00D4FF',
      lineWidth: 2,
      title: 'EMA(9)',
    });

    const emaSlowSeries = (chartRef.current as any).addLineSeries({
      color: '#FFA500',
      lineWidth: 2,
      title: 'EMA(21)',
    });

    // Prepare data
    const emaFastData = rawData.slice(8).map((candle, idx) => ({
      time: candle.time as Time,
      value: emaFast[idx],
    }));

    const emaSlowData = rawData.slice(20).map((candle, idx) => ({
      time: candle.time as Time,
      value: emaSlow[idx],
    }));

    emaFastSeries.setData(emaFastData);
    emaSlowSeries.setData(emaSlowData);

    return () => {
      if (chartRef.current) {
        chartRef.current.removeSeries(emaFastSeries);
        chartRef.current.removeSeries(emaSlowSeries);
      }
    };
  }, [rawData, showEMA]);

  // Add Bollinger Bands
  useEffect(() => {
    if (!chartRef.current || rawData.length === 0 || !showBB) return;

    const bb = calculateBollingerBands(rawData, 20, 2);

    const bbUpperSeries = (chartRef.current as any).addLineSeries({
      color: '#FFC107',
      lineWidth: 1,
      lineStyle: 2,
      title: 'BB Upper',
    });

    const bbMiddleSeries = (chartRef.current as any).addLineSeries({
      color: '#FFC107',
      lineWidth: 1,
      title: 'BB Middle',
    });

    const bbLowerSeries = (chartRef.current as any).addLineSeries({
      color: '#FFC107',
      lineWidth: 1,
      lineStyle: 2,
      title: 'BB Lower',
    });

    const bbUpperData = bb.map(b => ({ time: b.time as Time, value: b.upper }));
    const bbMiddleData = bb.map(b => ({ time: b.time as Time, value: b.middle }));
    const bbLowerData = bb.map(b => ({ time: b.time as Time, value: b.lower }));

    bbUpperSeries.setData(bbUpperData);
    bbMiddleSeries.setData(bbMiddleData);
    bbLowerSeries.setData(bbLowerData);

    return () => {
      if (chartRef.current) {
        chartRef.current.removeSeries(bbUpperSeries);
        chartRef.current.removeSeries(bbMiddleSeries);
        chartRef.current.removeSeries(bbLowerSeries);
      }
    };
  }, [rawData, showBB]);

  if (loading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '700px',
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
          <div>TradingView tarzı grafik yükleniyor...</div>
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
          height: '700px',
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

  const { stats, trend, supportLevels, resistanceLevels, fvgs, orderBlocks } = analysisData;

  if (!stats) return null;

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
          marginBottom: '16px',
          paddingBottom: '12px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
            {symbol}
          </div>
          <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
            {interval} • {rawData.length} mumlar • Trend:{' '}
            {trend === 'uptrend' ? '📈 Yükseliş' : trend === 'downtrend' ? '📉 Düşüş' : '➡️ Yatay'}
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

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: '#2962FF', marginBottom: '2px' }}>RSI (14)</div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: '#2962FF' }}>{stats.latestRSI.toFixed(1)}</div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
              {stats.latestRSI > 70 ? 'Aşırı Alım' : stats.latestRSI < 30 ? 'Aşırı Satım' : 'Nötr'}
            </div>
          </div>

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
          marginBottom: '12px',
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

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          minHeight: '600px',
          height: '600px',
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          position: 'relative',
        }}
      />

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          flexWrap: 'wrap',
          marginTop: '12px',
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
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>Destek ({supportLevels.length})</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#ef4444', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>Direnç ({resistanceLevels.length})</span>
            </div>
          </>
        )}
        {showFVG && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div
              style={{
                width: '20px',
                height: '8px',
                background: 'rgba(16, 185, 129, 0.2)',
                border: '1px solid #10b981',
                borderRadius: '2px',
              }}
            />
            <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>FVG ({fvgs.length})</span>
          </div>
        )}
        {showOB && orderBlocks.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div
              style={{
                width: '20px',
                height: '8px',
                background: 'rgba(156, 39, 176, 0.2)',
                border: '1px solid #9C27B0',
                borderRadius: '2px',
              }}
            />
            <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>Order Blocks ({orderBlocks.length})</span>
          </div>
        )}
        {showEMA && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#00D4FF', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                EMA(9): ${stats.emaFast?.toFixed(2) || '—'}
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <div style={{ width: '20px', height: '2px', background: '#FFA500', borderRadius: '2px' }} />
              <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                EMA(21): ${stats.emaSlow?.toFixed(2) || '—'}
              </span>
            </div>
          </>
        )}
        <div style={{ marginLeft: 'auto', color: 'rgba(255, 255, 255, 0.5)', fontSize: '10px' }}>
          Aralık: ${stats.lowPrice.toFixed(2)} - ${stats.highPrice.toFixed(2)}
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
