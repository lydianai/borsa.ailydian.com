'use client';

import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';
import { calculateRSI, calculateMFI } from '@/lib/indicators';
import { COLORS } from '@/lib/colors';

interface AdvancedTradingChartProps {
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

export default function AdvancedTradingChart({ symbol, interval, marketType = 'crypto' }: AdvancedTradingChartProps) {
  const mainChartContainerRef = useRef<HTMLDivElement>(null);
  const rsiChartContainerRef = useRef<HTMLDivElement>(null);
  const mfiChartContainerRef = useRef<HTMLDivElement>(null);

  const mainChartRef = useRef<IChartApi | null>(null);
  const rsiChartRef = useRef<IChartApi | null>(null);
  const mfiChartRef = useRef<IChartApi | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  // Create charts on mount
  useEffect(() => {
    if (!mainChartContainerRef.current || !rsiChartContainerRef.current || !mfiChartContainerRef.current) return;

    const chartOptions = {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0a0a' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      rightPriceScale: {
        borderColor: '#2a2a2a',
      },
      timeScale: {
        borderColor: '#2a2a2a',
        timeVisible: true,
        secondsVisible: false,
      },
    };

    // Main chart (candlesticks)
    const mainChart = createChart(mainChartContainerRef.current, {
      ...chartOptions,
      width: mainChartContainerRef.current.clientWidth,
      height: 400,
    });
    mainChartRef.current = mainChart;

    // RSI chart
    const rsiChart = createChart(rsiChartContainerRef.current, {
      ...chartOptions,
      width: rsiChartContainerRef.current.clientWidth,
      height: 150,
    });
    rsiChartRef.current = rsiChart;

    // MFI chart
    const mfiChart = createChart(mfiChartContainerRef.current, {
      ...chartOptions,
      width: mfiChartContainerRef.current.clientWidth,
      height: 150,
    });
    mfiChartRef.current = mfiChart;

    // Handle resize
    const handleResize = () => {
      if (mainChartContainerRef.current && mainChartRef.current) {
        mainChartRef.current.applyOptions({ width: mainChartContainerRef.current.clientWidth });
      }
      if (rsiChartContainerRef.current && rsiChartRef.current) {
        rsiChartRef.current.applyOptions({ width: rsiChartContainerRef.current.clientWidth });
      }
      if (mfiChartContainerRef.current && mfiChartRef.current) {
        mfiChartRef.current.applyOptions({ width: mfiChartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mainChartRef.current) {
        mainChartRef.current.remove();
        mainChartRef.current = null;
      }
      if (rsiChartRef.current) {
        rsiChartRef.current.remove();
        rsiChartRef.current = null;
      }
      if (mfiChartRef.current) {
        mfiChartRef.current.remove();
        mfiChartRef.current = null;
      }
    };
  }, []);

  // Fetch and update data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const apiEndpoint = marketType === 'traditional'
          ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}&limit=200`
          : `/api/charts/klines?symbol=${symbol}&interval=${interval}&limit=200`;

        console.log(`[AdvancedChart] Fetching: ${apiEndpoint}`);

        const response = await fetch(apiEndpoint, { cache: 'no-store' });
        const result = await response.json();

        if (!result.success) {
          throw new Error(result.error || 'Failed to fetch data');
        }

        console.log(`[AdvancedChart] Received ${result.data.klines?.length || 0} candles`);
        setData(result.data);
        setLoading(false);
      } catch (err) {
        console.error('[AdvancedChart] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval, marketType]);

  // Update charts when data changes
  useEffect(() => {
    if (!data || !data.klines || data.klines.length === 0) return;
    if (!mainChartRef.current || !rsiChartRef.current || !mfiChartRef.current) return;

    try {
      const candles = data.klines as KlineData[];

      // Candlestick series (using lightweight-charts v5 API)
      const candlestickSeries = mainChartRef.current.addSeries({
        type: 'Candlestick',
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
      } as any);

      candlestickSeries.setData(
        candles.map(c => ({
          time: c.time as any,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
      );

      // Volume series
      const volumeSeries = mainChartRef.current.addSeries({
        type: 'Histogram',
        color: '#26a69a',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
      } as any);

      volumeSeries.priceScale().applyOptions({
        scaleMargins: {
          top: 0.7,
          bottom: 0,
        },
      });

      volumeSeries.setData(
        candles.map(c => ({
          time: c.time as any,
          value: c.volume,
          color: c.close >= c.open ? '#10b98180' : '#ef444480',
        }))
      );

      // RSI series
      const rsiData = calculateRSI(candles, 14);
      const rsiSeries = rsiChartRef.current.addSeries({
        type: 'Line',
        color: '#2962FF',
        lineWidth: 2,
        title: 'RSI (14)',
      } as any);

      rsiSeries.setData(rsiData.map(r => ({ time: r.time as any, value: r.value })));

      // RSI levels
      const rsiUpperLevel = rsiChartRef.current.addSeries({
        type: 'Line',
        color: '#ef4444',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      } as any);
      rsiUpperLevel.setData(rsiData.map(r => ({ time: r.time as any, value: 70 })));

      const rsiLowerLevel = rsiChartRef.current.addSeries({
        type: 'Line',
        color: '#10b981',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      } as any);
      rsiLowerLevel.setData(rsiData.map(r => ({ time: r.time as any, value: 30 })));

      // MFI series
      const mfiData = calculateMFI(candles, 14);
      const mfiSeries = mfiChartRef.current.addSeries({
        type: 'Line',
        color: '#9C27B0',
        lineWidth: 2,
        title: 'MFI (14)',
      } as any);

      mfiSeries.setData(mfiData.map(m => ({ time: m.time as any, value: m.value })));

      // MFI levels
      const mfiUpperLevel = mfiChartRef.current.addSeries({
        type: 'Line',
        color: '#ef4444',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      } as any);
      mfiUpperLevel.setData(mfiData.map(m => ({ time: m.time as any, value: 80 })));

      const mfiLowerLevel = mfiChartRef.current.addSeries({
        type: 'Line',
        color: '#10b981',
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      } as any);
      mfiLowerLevel.setData(mfiData.map(m => ({ time: m.time as any, value: 20 })));

      // Time scale synchronization
      mainChartRef.current.timeScale().subscribeVisibleLogicalRangeChange((timeRange) => {
        if (timeRange && rsiChartRef.current && mfiChartRef.current) {
          rsiChartRef.current.timeScale().setVisibleLogicalRange(timeRange);
          mfiChartRef.current.timeScale().setVisibleLogicalRange(timeRange);
        }
      });

      console.log(`[AdvancedChart] Charts updated with ${candles.length} candles, ${rsiData.length} RSI, ${mfiData.length} MFI`);
    } catch (err) {
      console.error('[AdvancedChart] Error updating charts:', err);
      setError('Failed to render charts');
    }
  }, [data]);

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

  return (
    <div style={{
      background: 'rgba(0, 0, 0, 0.3)',
      borderRadius: '12px',
      padding: '20px',
      position: 'relative'
    }}>
      {loading && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(0, 0, 0, 0.8)',
          borderRadius: '12px',
          zIndex: 10
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
      )}

      {/* Main Chart */}
      <div>
        <div style={{
          fontSize: '16px',
          fontWeight: '600',
          color: '#FFFFFF',
          marginBottom: '12px'
        }}>
          {symbol} • {interval}
        </div>
        <div ref={mainChartContainerRef} style={{
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }} />
      </div>

      {/* RSI Chart */}
      <div style={{ marginTop: '16px' }}>
        <div style={{
          fontSize: '14px',
          fontWeight: '600',
          color: '#2962FF',
          marginBottom: '8px'
        }}>
          RSI (14) - Relative Strength Index
        </div>
        <div ref={rsiChartContainerRef} style={{
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }} />
      </div>

      {/* MFI Chart */}
      <div style={{ marginTop: '16px' }}>
        <div style={{
          fontSize: '14px',
          fontWeight: '600',
          color: '#9C27B0',
          marginBottom: '8px'
        }}>
          MFI (14) - Money Flow Index
        </div>
        <div ref={mfiChartContainerRef} style={{
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }} />
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
