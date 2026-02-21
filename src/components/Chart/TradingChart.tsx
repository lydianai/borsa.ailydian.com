'use client';

import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, IPriceLine } from 'lightweight-charts';

interface TradingChartProps {
  symbol: string;
  interval: string;
  supportResistanceLevels?: {
    pivot: number;
    resistance: { r1: number; r2: number; r3: number };
    support: { s1: number; s2: number; s3: number };
  };
}

export default function TradingChart({ symbol, interval, supportResistanceLevels }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#0a0a0a' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 600,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: '#2a2a2a',
      },
      rightPriceScale: {
        borderColor: '#2a2a2a',
      },
    });

    chartRef.current = chart;

    // Create candlestick series using addSeries method
    const candlestickSeries = chart.addSeries({
      type: 'Candlestick',
      upColor: '#00ff00',
      downColor: '#ff3333',
      borderDownColor: '#ff3333',
      borderUpColor: '#00ff00',
      wickDownColor: '#ff3333',
      wickUpColor: '#00ff00',
    } as any);

    candlestickSeriesRef.current = candlestickSeries as any;

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
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        candlestickSeriesRef.current = null;
      }
    };
  }, []);

  // Fetch and update chart data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log(`[TradingChart] Fetching data for ${symbol} ${interval}`);

        const response = await fetch(`/api/klines/${symbol}?interval=${interval}&limit=500`, {
          cache: 'no-store'
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.status}`);
        }

        const data = await response.json();

        if (!data.success || !data.data.candles) {
          throw new Error('Invalid data received');
        }

        const candles: CandlestickData<Time>[] = data.data.candles.map((c: any) => ({
          time: c.time as Time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }));

        if (candlestickSeriesRef.current) {
          candlestickSeriesRef.current.setData(candles);
        }

        console.log(`[TradingChart] Loaded ${candles.length} candles`);

        setLoading(false);
      } catch (err) {
        console.error('[TradingChart] Error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load chart');
        setLoading(false);
      }
    };

    if (symbol && interval) {
      fetchData();
    }
  }, [symbol, interval]);

  // Draw support/resistance lines
  useEffect(() => {
    if (!candlestickSeriesRef.current || !supportResistanceLevels) return;

    // Clear previous price lines
    priceLinesRef.current.forEach(line => {
      if (candlestickSeriesRef.current) {
        candlestickSeriesRef.current.removePriceLine(line);
      }
    });
    priceLinesRef.current = [];

    // Add support lines
    const s1 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.support.s1,
      color: '#00bfff',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      axisLabelVisible: true,
      title: 'S1',
    });
    priceLinesRef.current.push(s1);

    const s2 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.support.s2,
      color: '#00bfff',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'S2',
    });
    priceLinesRef.current.push(s2);

    const s3 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.support.s3,
      color: '#00bfff',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'S3',
    });
    priceLinesRef.current.push(s3);

    // Add resistance lines
    const r1 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.resistance.r1,
      color: '#ff6b6b',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'R1',
    });
    priceLinesRef.current.push(r1);

    const r2 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.resistance.r2,
      color: '#ff6b6b',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'R2',
    });
    priceLinesRef.current.push(r2);

    const r3 = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.resistance.r3,
      color: '#ff6b6b',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'R3',
    });
    priceLinesRef.current.push(r3);

    // Add pivot line
    const pivot = candlestickSeriesRef.current.createPriceLine({
      price: supportResistanceLevels.pivot,
      color: '#ffa500',
      lineWidth: 2,
      lineStyle: 0, // Solid
      axisLabelVisible: true,
      title: 'Pivot',
    });
    priceLinesRef.current.push(pivot);

    console.log(`[TradingChart] Added ${priceLinesRef.current.length} price lines`);

  }, [supportResistanceLevels]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-[600px] bg-[#0a0a0a] rounded-lg border border-[#2a2a2a]">
        <div className="text-center">
          <p className="text-red-500 text-lg mb-2">Hata</p>
          <p className="text-gray-400">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[600px]">
      <div ref={chartContainerRef} className="w-full h-full rounded-lg overflow-hidden border border-[#2a2a2a]" />
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0a0a0a]/90 rounded-lg z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-400">Veriler yükleniyor...</p>
          </div>
        </div>
      )}
    </div>
  );
}
