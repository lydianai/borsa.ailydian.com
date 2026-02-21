'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { calculateVWAP } from '@/lib/indicators/vwap';
import { calculateCVD } from '@/lib/indicators/cvd';
import {
  BollingerIcon,
  MovingAverageIcon,
  VWAPIcon,
  FVGIcon,
  OrderBlockIcon,
  SupportResistanceIcon,
  FibonacciIcon,
  RSIIcon,
  MFIIcon,
  VolumeIcon,
  DeltaIcon,
  LiquidityIcon,
  MarketStructureIcon,
  PremiumDiscountIcon,
  POCIcon,
  ValueAreaIcon,
  SessionIcon
} from '@/components/Icons/IndicatorIcons';
import { IndicatorButton } from '@/components/Chart/IndicatorButton';
import {
  detectOrderBlocks,
  detectFairValueGaps,
  detectLiquidityPools,
  calculatePremiumDiscountZones,
  detectMarketStructure,
  type OrderBlock as ICTOrderBlock,
  type FairValueGap as ICTFairValueGap,
  type LiquidityPool,
  type MarketStructure,
  type PremiumDiscountZone
} from '@/lib/indicators/ict-advanced';
import {
  calculateVolumeProfile,
  calculateVWAPWithBands,
  calculateSessionData,
  calculateCumulativeDelta,
  TRADING_SESSIONS,
  type VolumeProfile,
  type VWAPData,
  type SessionData
} from '@/lib/indicators/volume-profile';

const Chart = dynamic(() => import('react-apexcharts'), {
  ssr: false,
  loading: () => <div style={{ padding: '20px', textAlign: 'center', color: '#10b981' }}>ApexCharts yükleniyor...</div>
});

interface ApexCandlestickChartProps {
  symbol: string;
  interval: string;
  isTraditionalMarket?: boolean;
}

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface FVGData {
  type: 'bullish' | 'bearish';
  startTime: number;
  endTime: number;
  high: number;
  low: number;
  filled: boolean;
}

interface OrderBlock {
  type: 'bullish' | 'bearish';
  time: number;
  high: number;
  low: number;
  volume: number;
}

export default function ApexCandlestickChart({
  symbol,
  interval,
  isTraditionalMarket = false
}: ApexCandlestickChartProps) {
  // State management
  const [candleData, setCandleData] = useState<CandleData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sidebar panel state
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Indicator toggle states
  const [showBB, setShowBB] = useState(false);
  const [showMA7, setShowMA7] = useState(false);
  const [showMA25, setShowMA25] = useState(false);
  const [showMA99, setShowMA99] = useState(false);
  const [showVWAP, setShowVWAP] = useState(false);
  const [showFVG, setShowFVG] = useState(false);
  const [showOB, setShowOB] = useState(false);
  const [showSR, setShowSR] = useState(false);
  const [showFib, setShowFib] = useState(false);
  const [showMFI, setShowMFI] = useState(true); // MFI default AÇIK
  const [showRSI, setShowRSI] = useState(false);
  const [showVolume, setShowVolume] = useState(false);
  const [showCVD, setShowCVD] = useState(false);

  // 🎯 ICT Professional Indicators
  const [showLiquidityPools, setShowLiquidityPools] = useState(false);
  const [showMarketStructure, setShowMarketStructure] = useState(false);
  const [showPremiumDiscount, setShowPremiumDiscount] = useState(false);

  // Volume Profile & Sessions
  const [showVolumeProfile, setShowVolumeProfile] = useState(false);
  const [showSessionLevels, setShowSessionLevels] = useState(false);
  const [showPOC, setShowPOC] = useState(false); // Point of Control
  const [showValueArea, setShowValueArea] = useState(false);

  // Data fetching
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        const endpoint = isTraditionalMarket
          ? `/api/charts/traditional-klines?symbol=${symbol}&interval=${interval}`
          : `/api/charts/klines?symbol=${symbol}&interval=${interval}`;

        const response = await fetch(endpoint);
        if (!response.ok) {
          throw new Error(`API hatası: ${response.status}`);
        }

        const data = await response.json();

        if (data.success && data.data) {
          // Get klines array from response
          const candles = data.data.klines || data.data || [];
          setCandleData(Array.isArray(candles) ? candles : []);
        } else {
          throw new Error(data.error || 'Mum verisi alınamadı');
        }
      } catch (err) {
        console.error('Mum verisi alınırken hata:', err);
        setError(err instanceof Error ? err.message : 'Bilinmeyen hata');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [symbol, interval, isTraditionalMarket]);

  // Calculate technical indicators
  const indicators = useMemo(() => {
    if (!Array.isArray(candleData) || candleData.length === 0) return {};

    const closes = candleData.map(c => c.close);
    const _highs = candleData.map(c => c.high);
    const _lows = candleData.map(c => c.low);
    const _volumes = candleData.map(c => c.volume);

    // Simple Moving Averages
    const calculateSMA = (data: number[], period: number) => {
      const sma: (number | null)[] = [];
      for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
          sma.push(null);
        } else {
          const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          sma.push(sum / period);
        }
      }
      return sma;
    };

    // RSI Calculation
    const calculateRSI = (data: number[], period: number = 14) => {
      const rsi: (number | null)[] = [];
      const gains: number[] = [];
      const losses: number[] = [];

      for (let i = 1; i < data.length; i++) {
        const change = data[i] - data[i - 1];
        gains.push(change > 0 ? change : 0);
        losses.push(change < 0 ? Math.abs(change) : 0);
      }

      for (let i = 0; i < data.length; i++) {
        if (i < period) {
          rsi.push(null);
        } else {
          const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
          const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
          const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
          rsi.push(100 - (100 / (1 + rs)));
        }
      }
      return rsi;
    };

    // MFI Calculation
    const calculateMFI = (period: number = 14) => {
      const mfi: (number | null)[] = [];
      const typicalPrices = candleData.map(c => (c.high + c.low + c.close) / 3);
      const moneyFlow = typicalPrices.map((tp, i) => tp * candleData[i].volume);

      for (let i = 0; i < candleData.length; i++) {
        if (i < period) {
          mfi.push(null);
        } else {
          let positiveFlow = 0;
          let negativeFlow = 0;

          for (let j = i - period + 1; j <= i; j++) {
            if (typicalPrices[j] > typicalPrices[j - 1]) {
              positiveFlow += moneyFlow[j];
            } else {
              negativeFlow += moneyFlow[j];
            }
          }

          const moneyRatio = negativeFlow === 0 ? 100 : positiveFlow / negativeFlow;
          mfi.push(100 - (100 / (1 + moneyRatio)));
        }
      }
      return mfi;
    };

    // Bollinger Bands
    const calculateBB = (data: number[], period: number = 20, stdDev: number = 2) => {
      const sma = calculateSMA(data, period);
      const upper: (number | null)[] = [];
      const lower: (number | null)[] = [];

      for (let i = 0; i < data.length; i++) {
        if (i < period - 1 || sma[i] === null) {
          upper.push(null);
          lower.push(null);
        } else {
          const slice = data.slice(i - period + 1, i + 1);
          const mean = sma[i]!;
          const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
          const std = Math.sqrt(variance);
          upper.push(mean + stdDev * std);
          lower.push(mean - stdDev * std);
        }
      }
      return { middle: sma, upper, lower };
    };

    // Fair Value Gaps (FVG)
    const detectFVG = (): FVGData[] => {
      const fvgs: FVGData[] = [];

      for (let i = 2; i < candleData.length; i++) {
        const prev2 = candleData[i - 2];
        const prev1 = candleData[i - 1];
        const current = candleData[i];

        // Bullish FVG: prev2.high < current.low
        if (prev2.high < current.low) {
          fvgs.push({
            type: 'bullish',
            startTime: prev1.time,
            endTime: current.time,
            high: current.low,
            low: prev2.high,
            filled: false
          });
        }

        // Bearish FVG: prev2.low > current.high
        if (prev2.low > current.high) {
          fvgs.push({
            type: 'bearish',
            startTime: prev1.time,
            endTime: current.time,
            high: prev2.low,
            low: current.high,
            filled: false
          });
        }
      }

      // Check if FVGs are filled
      for (const fvg of fvgs) {
        const fvgIndex = candleData.findIndex(c => c.time === fvg.endTime);
        for (let i = fvgIndex + 1; i < candleData.length; i++) {
          const candle = candleData[i];
          if (fvg.type === 'bullish' && candle.low <= fvg.low) {
            fvg.filled = true;
            break;
          }
          if (fvg.type === 'bearish' && candle.high >= fvg.high) {
            fvg.filled = true;
            break;
          }
        }
      }

      return fvgs;
    };

    // Order Blocks
    const detectOrderBlocks = (): OrderBlock[] => {
      const blocks: OrderBlock[] = [];

      for (let i = 5; i < candleData.length; i++) {
        const prev = candleData[i - 1];
        const current = candleData[i];

        // Bullish Order Block: Strong rejection from low
        if (current.close > prev.close && current.volume > prev.volume * 1.5) {
          blocks.push({
            type: 'bullish',
            time: current.time,
            high: current.high,
            low: current.low,
            volume: current.volume
          });
        }

        // Bearish Order Block: Strong rejection from high
        if (current.close < prev.close && current.volume > prev.volume * 1.5) {
          blocks.push({
            type: 'bearish',
            time: current.time,
            high: current.high,
            low: current.low,
            volume: current.volume
          });
        }
      }

      return blocks.slice(-20); // Keep last 20 order blocks
    };

    // Support/Resistance
    const detectSupportResistance = () => {
      const levels: { price: number; type: 'support' | 'resistance'; touches: number }[] = [];
      const tolerance = 0.002; // 0.2% tolerance

      for (let i = 2; i < candleData.length - 2; i++) {
        const candle = candleData[i];

        // Check if this is a local high (resistance)
        if (candle.high > candleData[i - 1].high &&
            candle.high > candleData[i - 2].high &&
            candle.high > candleData[i + 1].high &&
            candle.high > candleData[i + 2].high) {
          levels.push({ price: candle.high, type: 'resistance', touches: 1 });
        }

        // Check if this is a local low (support)
        if (candle.low < candleData[i - 1].low &&
            candle.low < candleData[i - 2].low &&
            candle.low < candleData[i + 1].low &&
            candle.low < candleData[i + 2].low) {
          levels.push({ price: candle.low, type: 'support', touches: 1 });
        }
      }

      // Merge similar levels
      const merged: typeof levels = [];
      for (const level of levels) {
        const existing = merged.find(l =>
          l.type === level.type &&
          Math.abs(l.price - level.price) / level.price < tolerance
        );
        if (existing) {
          existing.touches++;
          existing.price = (existing.price + level.price) / 2; // Average price
        } else {
          merged.push({ ...level });
        }
      }

      return merged.filter(l => l.touches >= 2).slice(-10); // Keep top 10 levels with 2+ touches
    };

    // Fibonacci Retracement
    const calculateFibonacci = () => {
      if (candleData.length < 50) return null;

      const recent = candleData.slice(-50);
      const high = Math.max(...recent.map(c => c.high));
      const low = Math.min(...recent.map(c => c.low));
      const diff = high - low;

      return {
        high,
        low,
        fib236: high - diff * 0.236,
        fib382: high - diff * 0.382,
        fib500: high - diff * 0.500,
        fib618: high - diff * 0.618,
        fib786: high - diff * 0.786
      };
    };

    const ma7 = calculateSMA(closes, 7);
    const ma25 = calculateSMA(closes, 25);
    const ma99 = calculateSMA(closes, 99);
    const bb = calculateBB(closes);
    const rsi = calculateRSI(closes);
    const mfi = calculateMFI();
    const vwap = calculateVWAP(candleData, 'day');
    const cvd = calculateCVD(candleData);
    const fvgs = detectFVG();
    const orderBlocks = detectOrderBlocks();
    const srLevels = detectSupportResistance();
    const fibonacci = calculateFibonacci();

    // 🎯 ICT Professional Indicators
    const ictOrderBlocks = detectOrderBlocks();
    const ictFairValueGaps = detectFairValueGaps(candleData);
    const liquidityPools = detectLiquidityPools(candleData, 5);
    const marketStructure = detectMarketStructure(candleData, 5);
    const premiumDiscountZone = calculatePremiumDiscountZones(candleData, 100);

    // Volume Profile & Session Analysis
    const volumeProfile = calculateVolumeProfile(candleData, 50);
    const vwapBands = calculateVWAPWithBands(candleData);
    const sessionData = calculateSessionData(candleData);
    const cumulativeDelta = calculateCumulativeDelta(candleData);

    // Debug log
    console.log('İndikatörler calculated:', {
      ma7Points: ma7?.filter(v => v !== null).length || 0,
      ma25Points: ma25?.filter(v => v !== null).length || 0,
      ma99Points: ma99?.filter(v => v !== null).length || 0,
      bbPoints: bb?.upper.filter(v => v !== null).length || 0,
      rsiPoints: rsi?.filter(v => v !== null).length || 0,
      mfiPoints: mfi?.filter(v => v !== null).length || 0,
      vwapPoints: vwap?.length || 0,
      cvdPoints: cvd?.length || 0,
      fvgsCount: fvgs?.length || 0,
      orderBlocksCount: orderBlocks?.length || 0,
      srLevelsCount: srLevels?.length || 0,
      fibonacciCalculated: fibonacci ? 'YES' : 'NO',
      // Professional indicators
      ictOrderBlocksCount: ictOrderBlocks?.length || 0,
      ictFairValueGapsCount: ictFairValueGaps?.length || 0,
      liquidityPoolsCount: liquidityPools?.length || 0,
      marketStructureCount: marketStructure?.length || 0,
      volumeProfileLevels: volumeProfile?.levels?.length || 0,
      vwapBandsPoints: vwapBands?.length || 0,
      sessionDataCount: sessionData?.length || 0,
      cumulativeDeltaPoints: cumulativeDelta?.length || 0
    });

    return {
      ma7,
      ma25,
      ma99,
      bb,
      rsi,
      mfi,
      vwap,
      cvd,
      fvgs,
      orderBlocks,
      srLevels,
      fibonacci,
      // Professional indicators
      ictOrderBlocks,
      ictFairValueGaps,
      liquidityPools,
      marketStructure,
      premiumDiscountZone,
      volumeProfile,
      vwapBands,
      sessionData,
      cumulativeDelta
    };
  }, [candleData]);

  // Candlestick series data
  const candleSeries = useMemo(() => {
    if (!Array.isArray(candleData) || candleData.length === 0) {
      return [];
    }

    const mainSeries: any[] = [{
      name: 'Fiyat',
      type: 'candlestick',
      data: candleData.map(c => ({
        x: new Date(c.time * 1000),
        y: [c.open, c.high, c.low, c.close]
      }))
    }];

    // Add Moving Averages
    if (showMA7 && indicators.ma7) {
      mainSeries.push({
        name: 'MA7',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.ma7![i]
        })).filter(d => d.y !== null)
      });
    }

    if (showMA25 && indicators.ma25) {
      mainSeries.push({
        name: 'MA25',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.ma25![i]
        })).filter(d => d.y !== null)
      });
    }

    if (showMA99 && indicators.ma99) {
      mainSeries.push({
        name: 'MA99',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.ma99![i]
        })).filter(d => d.y !== null)
      });
    }

    // Add Bollinger Bands
    if (showBB && indicators.bb) {
      mainSeries.push({
        name: 'BB Upper',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.bb!.upper[i]
        })).filter(d => d.y !== null)
      });
      mainSeries.push({
        name: 'BB Middle',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.bb!.middle[i]
        })).filter(d => d.y !== null)
      });
      mainSeries.push({
        name: 'BB Lower',
        type: 'line',
        data: candleData.map((c, i) => ({
          x: new Date(c.time * 1000),
          y: indicators.bb!.lower[i]
        })).filter(d => d.y !== null)
      });
    }

    // Add VWAP
    if (showVWAP && indicators.vwap) {
      mainSeries.push({
        name: 'VWAP',
        type: 'line',
        data: indicators.vwap.map(v => ({
          x: new Date(v.time * 1000),
          y: v.vwap
        }))
      });
    }

    console.log('📈 Chart series:', {
      total: mainSeries.length,
      names: mainSeries.map(s => s.name),
      dataPoints: mainSeries.map(s => ({ name: s.name, points: s.data?.length || 0 }))
    });

    return mainSeries;
  }, [candleData, indicators, showMA7, showMA25, showMA99, showBB, showVWAP]);

  // Annotations (FVG, Order Blocks, S/R, Fibonacci)
  const annotations = useMemo(() => {
    const annotationsData: any = {
      yaxis: [],
      xaxis: [],
      points: []
    };

    // Fair Value Gaps
    if (showFVG && indicators.fvgs) {
      indicators.fvgs.forEach((fvg: FVGData) => {
        if (!fvg.filled) {
          annotationsData.yaxis.push({
            y: fvg.low,
            y2: fvg.high,
            fillColor: fvg.type === 'bullish' ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)',
            borderColor: fvg.type === 'bullish' ? '#22c55e' : '#ef4444',
            opacity: 0.4,
            label: {
              text: `${fvg.type === 'bullish' ? '↗' : '↘'} FVG`,
              style: {
                color: '#fff',
                background: fvg.type === 'bullish' ? '#22c55e' : '#ef4444'
              }
            }
          });
        }
      });
    }

    // Order Blocks
    if (showOB && indicators.orderBlocks) {
      indicators.orderBlocks.forEach((ob: OrderBlock) => {
        annotationsData.yaxis.push({
          y: ob.low,
          y2: ob.high,
          fillColor: ob.type === 'bullish' ? 'rgba(59, 130, 246, 0.1)' : 'rgba(251, 146, 60, 0.1)',
          borderColor: ob.type === 'bullish' ? '#3b82f6' : '#fb923c',
          opacity: 0.3,
          label: {
            text: 'OB',
            style: {
              color: '#fff',
              background: ob.type === 'bullish' ? '#3b82f6' : '#fb923c'
            }
          }
        });
      });
    }

    // Support/Resistance
    if (showSR && indicators.srLevels) {
      indicators.srLevels.forEach((level: any) => {
        annotationsData.yaxis.push({
          y: level.price,
          borderColor: level.type === 'support' ? '#10b981' : '#f59e0b',
          strokeDashArray: 4,
          label: {
            text: `${level.type === 'support' ? 'Destek' : 'Direnç'}`,
            position: 'left',
            offsetX: 5,
            style: {
              color: '#fff',
              background: level.type === 'support' ? '#10b981' : '#f59e0b',
              fontSize: '10px',
              fontWeight: 600,
              padding: {
                left: 6,
                right: 6,
                top: 3,
                bottom: 3
              }
            }
          }
        });
      });
    }

    // Fibonacci
    if (showFib && indicators.fibonacci) {
      const fib = indicators.fibonacci;
      const levels = [
        { value: fib.high, label: '100%', color: '#ef4444' },
        { value: fib.fib786, label: '78.6%', color: '#f97316' },
        { value: fib.fib618, label: '61.8%', color: '#f59e0b' },
        { value: fib.fib500, label: '50%', color: '#eab308' },
        { value: fib.fib382, label: '38.2%', color: '#84cc16' },
        { value: fib.fib236, label: '23.6%', color: '#22c55e' },
        { value: fib.low, label: '0%', color: '#10b981' }
      ];

      levels.forEach(level => {
        annotationsData.yaxis.push({
          y: level.value,
          borderColor: level.color,
          strokeDashArray: 2,
          opacity: 0.6,
          label: {
            text: level.label,
            style: {
              color: '#fff',
              background: level.color,
              fontSize: '10px'
            }
          }
        });
      });
    }

    // 🎯 ICT Liquidity Pools
    if (showLiquidityPools && indicators.liquidityPools) {
      indicators.liquidityPools.forEach((pool: LiquidityPool) => {
        if (!pool.swept) { // Only show unswept liquidity
          annotationsData.yaxis.push({
            y: pool.price,
            borderColor: pool.type === 'buy_side' ? '#06b6d4' : '#f43f5e',
            strokeDashArray: 6,
            opacity: 0.8,
            label: {
              text: `${pool.type === 'buy_side' ? 'Alış Likiditesi' : 'Satış Likiditesi'}`,
              position: 'left',
              offsetX: 5,
              style: {
                color: '#fff',
                background: pool.type === 'buy_side' ? '#06b6d4' : '#f43f5e',
                fontSize: '10px',
                fontWeight: 600,
                padding: {
                  left: 6,
                  right: 6,
                  top: 3,
                  bottom: 3
                }
              }
            }
          });
        }
      });
    }

    // 🎯 ICT Market Structure
    if (showMarketStructure && indicators.marketStructure) {
      indicators.marketStructure.forEach((ms: MarketStructure) => {
        if (!ms.broken) {
          const color = ms.type.includes('higher') ? '#22c55e' : '#ef4444';
          const label = ms.type === 'higher_high' ? 'HH' :
                       ms.type === 'higher_low' ? 'HL' :
                       ms.type === 'lower_high' ? 'LH' : 'LL';

          annotationsData.points.push({
            x: new Date(ms.time * 1000).getTime(),
            y: ms.price,
            marker: {
              size: 6,
              fillColor: color,
              strokeColor: '#fff',
              strokeWidth: 2
            },
            label: {
              text: label,
              style: {
                color: '#fff',
                background: color,
                fontSize: '9px'
              }
            }
          });
        }
      });
    }

    // 🎯 Premium/Discount Zones
    if (showPremiumDiscount && indicators.premiumDiscountZone) {
      const zone = indicators.premiumDiscountZone;
      const premiumLevel = zone.equilibrium + (zone.high - zone.equilibrium) * 0.618;
      const discountLevel = zone.equilibrium - (zone.equilibrium - zone.low) * 0.618;

      // Premium zone
      annotationsData.yaxis.push({
        y: premiumLevel,
        y2: zone.high,
        fillColor: 'rgba(239, 68, 68, 0.08)',
        borderColor: '#ef4444',
        opacity: 0.2,
        label: {
          text: 'Premium',
          position: 'left',
          offsetX: 5,
          style: {
            color: '#fff',
            background: '#ef4444',
            fontSize: '10px',
            fontWeight: 600,
            padding: {
              left: 6,
              right: 6,
              top: 3,
              bottom: 3
            }
          }
        }
      });

      // Equilibrium
      annotationsData.yaxis.push({
        y: zone.equilibrium,
        borderColor: '#8b5cf6',
        strokeDashArray: 3,
        opacity: 0.6,
        label: {
          text: 'Denge',
          position: 'left',
          offsetX: 5,
          style: {
            color: '#fff',
            background: '#8b5cf6',
            fontSize: '10px',
            fontWeight: 600,
            padding: {
              left: 6,
              right: 6,
              top: 3,
              bottom: 3
            }
          }
        }
      });

      // Discount zone
      annotationsData.yaxis.push({
        y: zone.low,
        y2: discountLevel,
        fillColor: 'rgba(34, 197, 94, 0.08)',
        borderColor: '#22c55e',
        opacity: 0.2,
        label: {
          text: 'İndirim',
          position: 'left',
          offsetX: 5,
          style: {
            color: '#fff',
            background: '#22c55e',
            fontSize: '10px',
            fontWeight: 600,
            padding: {
              left: 6,
              right: 6,
              top: 3,
              bottom: 3
            }
          }
        }
      });
    }

    // Volume Profile POC
    if (showPOC && indicators.volumeProfile) {
      annotationsData.yaxis.push({
        y: indicators.volumeProfile.poc,
        borderColor: '#fbbf24',
        strokeWidth: 2,
        opacity: 0.9,
        label: {
          text: 'POC',
          style: {
            color: '#000',
            background: '#fbbf24',
            fontSize: '10px',
            fontWeight: 'bold'
          }
        }
      });
    }

    // Volume Profile Value Area
    if (showValueArea && indicators.volumeProfile) {
      annotationsData.yaxis.push({
        y: indicators.volumeProfile.valueAreaLow,
        y2: indicators.volumeProfile.valueAreaHigh,
        fillColor: 'rgba(251, 191, 36, 0.05)',
        borderColor: '#fbbf24',
        opacity: 0.3,
        label: {
          text: 'VA',
          style: {
            color: '#000',
            background: '#fbbf24',
            fontSize: '9px'
          }
        }
      });
    }

    // 📊 Session High/Low
    if (showSessionLevels && indicators.sessionData) {
      indicators.sessionData.forEach((session: SessionData) => {
        // Session High
        annotationsData.yaxis.push({
          y: session.high,
          borderColor: session.session.color,
          strokeDashArray: 4,
          opacity: 0.6,
          label: {
            text: `${session.session.name.toUpperCase()} H`,
            style: {
              color: '#fff',
              background: session.session.color,
              fontSize: '8px'
            }
          }
        });

        // Session Low
        annotationsData.yaxis.push({
          y: session.low,
          borderColor: session.session.color,
          strokeDashArray: 4,
          opacity: 0.6,
          label: {
            text: `${session.session.name.toUpperCase()} L`,
            style: {
              color: '#fff',
              background: session.session.color,
              fontSize: '8px'
            }
          }
        });
      });
    }

    return annotationsData;
  }, [showFVG, showOB, showSR, showFib, showLiquidityPools, showMarketStructure, showPremiumDiscount, showPOC, showValueArea, showSessionLevels, indicators]);

  // Main chart options
  const candlestickOptions = useMemo(() => {
    return {
      chart: {
        type: 'candlestick' as const,
        height: 500,
        background: 'transparent',
        toolbar: {
          show: true,
          tools: {
            download: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        },
        animations: {
          enabled: false
        }
      },
      plotOptions: {
        candlestick: {
          colors: {
            upward: '#22c55e',
            downward: '#ef4444'
          },
          wick: {
            useFillColor: true
          }
        }
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          style: {
            colors: '#94a3b8'
          }
        },
        // Sağ tarafta %15 boşluk bırak
        tickAmount: 10,
        min: undefined,
        max: candleData && candleData.length > 0
          ? new Date(candleData[candleData.length - 1].time * 1000).getTime() + (1000 * 60 * 60 * 24 * 5) // 5 gün boşluk
          : undefined
      },
      yaxis: {
        tooltip: {
          enabled: true
        },
        labels: {
          style: {
            colors: '#94a3b8'
          },
          formatter: (value: number) => value.toFixed(2)
        }
      },
      grid: {
        borderColor: '#1e293b',
        strokeDashArray: 3
      },
      tooltip: {
        theme: 'dark',
        x: {
          format: 'dd MMM yyyy HH:mm'
        }
      },
      annotations: annotations,
      stroke: {
        width: [1, 2, 2, 2, 2, 2, 2, 2]
      },
      colors: ['#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'],
      legend: {
        show: true,
        position: 'top' as const,
        labels: {
          colors: '#94a3b8'
        }
      }
    };
  }, [annotations]);

  // RSI chart options
  const rsiOptions = useMemo(() => {
    return {
      chart: {
        type: 'line' as const,
        height: 150,
        background: 'transparent',
        toolbar: { show: false },
        animations: { enabled: false }
      },
      stroke: {
        width: 2,
        colors: ['#8b5cf6']
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          show: false
        }
      },
      yaxis: {
        min: 0,
        max: 100,
        tickAmount: 4,
        labels: {
          style: {
            colors: '#94a3b8'
          }
        }
      },
      grid: {
        borderColor: '#1e293b'
      },
      tooltip: {
        theme: 'dark'
      },
      annotations: {
        yaxis: [
          {
            y: 70,
            borderColor: '#ef4444',
            strokeDashArray: 2,
            label: {
              text: 'Overbought',
              style: {
                color: '#fff',
                background: '#ef4444'
              }
            }
          },
          {
            y: 30,
            borderColor: '#22c55e',
            strokeDashArray: 2,
            label: {
              text: 'Oversold',
              style: {
                color: '#fff',
                background: '#22c55e'
              }
            }
          }
        ]
      }
    };
  }, []);

  // RSI series
  const rsiSeries = useMemo(() => {
    if (!indicators.rsi) return [];
    return [{
      name: 'RSI',
      data: candleData.map((c, i) => ({
        x: new Date(c.time * 1000),
        y: indicators.rsi![i]
      })).filter(d => d.y !== null)
    }];
  }, [candleData, indicators.rsi]);

  // MFI chart options
  const mfiOptions = useMemo(() => {
    return {
      chart: {
        type: 'line' as const,
        height: 150,
        background: 'transparent',
        toolbar: { show: false },
        animations: { enabled: false }
      },
      stroke: {
        width: 2,
        colors: ['#ec4899']
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          show: false
        }
      },
      yaxis: {
        min: 0,
        max: 100,
        tickAmount: 4,
        labels: {
          style: {
            colors: '#94a3b8'
          }
        }
      },
      grid: {
        borderColor: '#1e293b'
      },
      tooltip: {
        theme: 'dark'
      },
      annotations: {
        yaxis: [
          {
            y: 80,
            borderColor: '#ef4444',
            strokeDashArray: 2,
            label: {
              text: 'Aşırı Alım',
              position: 'left',
              offsetX: 5,
              style: {
                color: '#fff',
                background: '#ef4444',
                fontSize: '10px',
                fontWeight: 600,
                padding: {
                  left: 6,
                  right: 6,
                  top: 3,
                  bottom: 3
                }
              }
            }
          },
          {
            y: 20,
            borderColor: '#22c55e',
            strokeDashArray: 2,
            label: {
              text: 'Aşırı Satım',
              position: 'left',
              offsetX: 5,
              style: {
                color: '#fff',
                background: '#22c55e',
                fontSize: '10px',
                fontWeight: 600,
                padding: {
                  left: 6,
                  right: 6,
                  top: 3,
                  bottom: 3
                }
              }
            }
          }
        ]
      }
    };
  }, []);

  // MFI series
  const mfiSeries = useMemo(() => {
    if (!indicators.mfi) return [];
    return [{
      name: 'MFI',
      data: candleData.map((c, i) => ({
        x: new Date(c.time * 1000),
        y: indicators.mfi![i]
      })).filter(d => d.y !== null)
    }];
  }, [candleData, indicators.mfi]);

  // Volume chart options
  const volumeOptions = useMemo(() => {
    return {
      chart: {
        type: 'bar' as const,
        height: 150,
        background: 'transparent',
        toolbar: { show: false },
        animations: { enabled: false }
      },
      plotOptions: {
        bar: {
          columnWidth: '80%',
          colors: {
            ranges: [{
              from: 0,
              to: Number.MAX_VALUE,
              color: '#3b82f6'
            }]
          }
        }
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          show: false
        }
      },
      yaxis: {
        labels: {
          style: {
            colors: '#94a3b8'
          }
        }
      },
      grid: {
        borderColor: '#1e293b'
      },
      tooltip: {
        theme: 'dark'
      }
    };
  }, []);

  // Volume series
  const volumeSeries = useMemo(() => {
    return [{
      name: 'Volume',
      data: candleData.map(c => ({
        x: new Date(c.time * 1000),
        y: c.volume
      }))
    }];
  }, [candleData]);

  // CVD chart options
  const cvdOptions = useMemo(() => {
    return {
      chart: {
        type: 'line' as const,
        height: 150,
        background: 'transparent',
        toolbar: { show: false },
        animations: { enabled: false }
      },
      stroke: {
        width: 2,
        colors: ['#14b8a6']
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          show: false
        }
      },
      yaxis: {
        labels: {
          style: {
            colors: '#94a3b8'
          }
        }
      },
      grid: {
        borderColor: '#1e293b'
      },
      tooltip: {
        theme: 'dark'
      }
    };
  }, []);

  // CVD series
  const cvdSeries = useMemo(() => {
    if (!indicators.cvd) return [];
    return [{
      name: 'CVD',
      data: indicators.cvd.map(c => ({
        x: new Date(c.time * 1000),
        y: c.cumulativeDelta
      }))
    }];
  }, [indicators.cvd]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-gradient-to-b from-[#0a0a0a] to-[#111111] rounded-lg">
        <div className="text-slate-400">Grafik verileri yükleniyor...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-gradient-to-b from-[#0a0a0a] to-[#111111] rounded-lg">
        <div className="text-red-400">Hata: {error}</div>
      </div>
    );
  }

  return (
    <div style={{ width: '100%' }} className="space-y-4">
      {/* Toggle Button - Chart Üstünde Sağda */}
      <div style={{
        display: 'flex',
        justifyContent: 'flex-end',
        marginBottom: '12px'
      }}>
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          style={{
            padding: '10px 18px',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            background: isSidebarOpen
              ? 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
              : 'linear-gradient(135deg, rgba(10, 10, 15, 0.96) 0%, rgba(15, 15, 20, 0.96) 100%)',
            backdropFilter: 'blur(20px) saturate(150%)',
            boxShadow: isSidebarOpen
              ? '0 8px 32px rgba(59, 130, 246, 0.4), 0 0 0 1px rgba(59, 130, 246, 0.6)'
              : '0 4px 16px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.08)',
            color: '#ffffff',
            fontSize: '13px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
          onMouseEnter={(e) => {
            if (!isSidebarOpen) {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 6px 24px rgba(59, 130, 246, 0.3), 0 0 0 1px rgba(59, 130, 246, 0.4)';
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            if (!isSidebarOpen) {
              e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.08)';
            }
          }}
        >
          <span style={{ fontSize: '16px' }}>📊</span>
          <span>İndikatörler</span>
          <span style={{
            fontSize: '12px',
            transform: isSidebarOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.3s ease'
          }}>▼</span>
        </button>
      </div>

      {/* Collapsible Sidebar Panel */}
      <div
        style={{
          position: 'fixed',
          top: '0',
          right: isSidebarOpen ? '0' : '-380px',
          width: '360px',
          height: '100vh',
          zIndex: 1000,
          background: 'linear-gradient(135deg, rgba(10, 10, 15, 0.98) 0%, rgba(15, 15, 20, 0.98) 100%)',
          backdropFilter: 'blur(50px) saturate(200%)',
          borderLeft: '1px solid rgba(255, 255, 255, 0.12)',
          boxShadow: '-8px 0 32px rgba(0, 0, 0, 0.8)',
          transition: 'right 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
          overflowY: 'auto',
          padding: '20px'
        }}
        className="custom-scrollbar"
      >
        {/* Sidebar Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '20px',
          paddingBottom: '12px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <h3 style={{
            fontSize: '16px',
            fontWeight: '700',
            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            letterSpacing: '0.3px'
          }}>
            📊 İndikatörler
          </h3>
          <button
            onClick={() => setIsSidebarOpen(false)}
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '6px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              background: 'rgba(255, 255, 255, 0.05)',
              color: 'rgba(255, 255, 255, 0.7)',
              fontSize: '16px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)';
              e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
            }}
          >
            ✕
          </button>
        </div>

        {/* Teknik İndikatörler */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{
            fontSize: '11px',
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.5)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '8px',
            paddingLeft: '4px'
          }}>
            Teknik İndikatörler
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr',
            gap: '16px'
          }}>
        <IndicatorButton
          active={showBB}
          onClick={() => setShowBB(!showBB)}
          icon={<BollingerIcon size={16} />}
          label="Bollinger Bantları"
          description="Volatiliteye göre dinamik destek ve direnç bantları. Fiyat hareketlerinin normalliğini ölçer ve aşırı alım/satım bölgelerini gösterir."
          gradientFrom="#3b82f6"
          gradientTo="#2563eb"
          glowColor="rgba(59, 130, 246, 0.4)"
        />

        <IndicatorButton
          active={showMA7}
          onClick={() => setShowMA7(!showMA7)}
          icon={<MovingAverageIcon size={16} />}
          label="MA 7"
          description="7 periyotluk hareketli ortalama. Kısa vadeli trend yönünü gösterir ve hızlı fiyat değişimlerini yumuşatır."
          gradientFrom="#3b82f6"
          gradientTo="#2563eb"
          glowColor="rgba(59, 130, 246, 0.4)"
        />

        <IndicatorButton
          active={showMA25}
          onClick={() => setShowMA25(!showMA25)}
          icon={<MovingAverageIcon size={16} />}
          label="MA 25"
          description="25 periyotluk hareketli ortalama. Orta vadeli trend yönünü belirler ve destek/direnç görevi görür."
          gradientFrom="#f59e0b"
          gradientTo="#d97706"
          glowColor="rgba(245, 158, 11, 0.4)"
        />

        <IndicatorButton
          active={showMA99}
          onClick={() => setShowMA99(!showMA99)}
          icon={<MovingAverageIcon size={16} />}
          label="MA 99"
          description="99 periyotluk hareketli ortalama. Uzun vadeli ana trendi gösterir ve güçlü destek/direnç seviyesi oluşturur."
          gradientFrom="#8b5cf6"
          gradientTo="#7c3aed"
          glowColor="rgba(139, 92, 246, 0.4)"
        />

        <IndicatorButton
          active={showVWAP}
          onClick={() => setShowVWAP(!showVWAP)}
          icon={<VWAPIcon size={16} />}
          label="VWAP"
          description="Hacim Ağırlıklı Ortalama Fiyat. Kurumsal işlemlerin ortalama fiyatını gösterir ve intraday ticaret için kritik seviyedir."
          gradientFrom="#14b8a6"
          gradientTo="#0d9488"
          glowColor="rgba(20, 184, 166, 0.4)"
        />
          </div>
        </div>

        {/* ICT & SMC İndikatörler */}
        <div style={{ marginBottom: '16px' }}>
          <h4 style={{
            fontSize: '11px',
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.5)',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '8px',
            paddingLeft: '4px'
          }}>
            ICT & Smart Money
          </h4>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr',
            gap: '16px'
          }}>
        <IndicatorButton
          active={showFVG}
          onClick={() => setShowFVG(!showFVG)}
          icon={<FVGIcon size={16} />}
          label="FVG / İmbalans"
          description="Fair Value Gap - Fiyat boşlukları ve dengesizlik bölgeleri. Piyasa etkinliğinin olmadığı ve fiyatın geri dönebileceği alanlar."
          gradientFrom="#10b981"
          gradientTo="#059669"
          glowColor="rgba(16, 185, 129, 0.4)"
        />

        <IndicatorButton
          active={showOB}
          onClick={() => setShowOB(!showOB)}
          icon={<OrderBlockIcon size={16} />}
          label="Emir Blokları"
          description="Kurumsal alım/satım emirlerinin yoğunlaştığı bölgeler. Güçlü destek ve direnç seviyelerini işaret eder."
          gradientFrom="#f59e0b"
          gradientTo="#d97706"
          glowColor="rgba(245, 158, 11, 0.4)"
        />

        <IndicatorButton
          active={showSR}
          onClick={() => setShowSR(!showSR)}
          icon={<SupportResistanceIcon size={16} />}
          label="Destek / Direnç"
          description="Fiyatın durduğu veya geri döndüğü kritik seviyeler. Alım ve satım kararları için önemli referans noktaları."
          gradientFrom="#06b6d4"
          gradientTo="#0891b2"
          glowColor="rgba(6, 182, 212, 0.4)"
        />

        <IndicatorButton
          active={showFib}
          onClick={() => setShowFib(!showFib)}
          icon={<FibonacciIcon size={16} />}
          label="Fibonacci"
          description="Geri çekilme ve uzatma seviyeleri. 0.236, 0.382, 0.5, 0.618, 0.786 oranlarında potansiyel destek/direnç bölgeleri."
          gradientFrom="#ec4899"
          gradientTo="#db2777"
          glowColor="rgba(236, 72, 153, 0.4)"
        />

        <IndicatorButton
          active={showRSI}
          onClick={() => setShowRSI(!showRSI)}
          icon={<RSIIcon size={16} />}
          label="RSI"
          description="Göreceli Güç Endeksi. 0-100 arası değer alır. 70 üzeri aşırı alım, 30 altı aşırı satım bölgesi. Momentum göstergesi."
          gradientFrom="#8b5cf6"
          gradientTo="#7c3aed"
          glowColor="rgba(139, 92, 246, 0.4)"
        />

        <IndicatorButton
          active={showMFI}
          onClick={() => setShowMFI(!showMFI)}
          icon={<MFIIcon size={16} />}
          label="MFI"
          description="Para Akış Endeksi. Hacmi de hesaba katan RSI benzeri gösterge. Alım/satım baskısını ve likidite hareketini ölçer."
          gradientFrom="#ec4899"
          gradientTo="#db2777"
          glowColor="rgba(236, 72, 153, 0.4)"
        />

        <IndicatorButton
          active={showVolume}
          onClick={() => setShowVolume(!showVolume)}
          icon={<VolumeIcon size={16} />}
          label="İşlem Hacmi"
          description="Gerçekleşen işlem miktarı. Yüksek hacim trend doğrulaması, düşük hacim ise zayıf hareket anlamına gelir."
          gradientFrom="#3b82f6"
          gradientTo="#2563eb"
          glowColor="rgba(59, 130, 246, 0.4)"
        />

        <IndicatorButton
          active={showCVD}
          onClick={() => setShowCVD(!showCVD)}
          icon={<DeltaIcon size={16} />}
          label="Kümülatif Delta"
          description="Alış ve satış hacmi farkının birikimi. Kurumsal para akışını ve piyasa baskısının yönünü gösterir."
          gradientFrom="#14b8a6"
          gradientTo="#0d9488"
          glowColor="rgba(20, 184, 166, 0.4)"
        />

        <IndicatorButton
          active={showLiquidityPools}
          onClick={() => setShowLiquidityPools(!showLiquidityPools)}
          icon={<LiquidityIcon size={16} />}
          label="Likidite Havuzları"
          description="Stop-loss emirlerinin yoğunlaştığı bölgeler. Fiyatın bu bölgeleri hedefleyerek likidite topladığı alanlar."
          gradientFrom="#06b6d4"
          gradientTo="#0891b2"
          glowColor="rgba(6, 182, 212, 0.4)"
        />

        <IndicatorButton
          active={showMarketStructure}
          onClick={() => setShowMarketStructure(!showMarketStructure)}
          icon={<MarketStructureIcon size={16} />}
          label="Piyasa Yapısı"
          description="Higher Highs, Higher Lows (yükseliş) veya Lower Highs, Lower Lows (düşüş) formasyonları. Trend yönünü belirler."
          gradientFrom="#10b981"
          gradientTo="#059669"
          glowColor="rgba(16, 185, 129, 0.4)"
        />

        <IndicatorButton
          active={showPremiumDiscount}
          onClick={() => setShowPremiumDiscount(!showPremiumDiscount)}
          icon={<PremiumDiscountIcon size={16} />}
          label="Premium / İndirim"
          description="Fibonacci 0.618 üzeri premium (pahalı), 0.382 altı discount (ucuz) bölgeler. Alım/satım için optimal fiyat seviyeleri."
          gradientFrom="#f59e0b"
          gradientTo="#d97706"
          glowColor="rgba(245, 158, 11, 0.4)"
        />

        <IndicatorButton
          active={showPOC}
          onClick={() => setShowPOC(!showPOC)}
          icon={<POCIcon size={16} />}
          label="POC"
          description="Point of Control - En yüksek hacmin gerçekleştiği fiyat seviyesi. Adil değer ve dengenin göstergesi."
          gradientFrom="#ec4899"
          gradientTo="#db2777"
          glowColor="rgba(236, 72, 153, 0.4)"
        />

        <IndicatorButton
          active={showValueArea}
          onClick={() => setShowValueArea(!showValueArea)}
          icon={<ValueAreaIcon size={16} />}
          label="Değer Alanı"
          description="Toplam hacmin %70'inin gerçekleştiği fiyat aralığı. Piyasanın kabul ettiği adil değer bölgesi."
          gradientFrom="#8b5cf6"
          gradientTo="#7c3aed"
          glowColor="rgba(139, 92, 246, 0.4)"
        />

        <IndicatorButton
          active={showSessionLevels}
          onClick={() => setShowSessionLevels(!showSessionLevels)}
          icon={<SessionIcon size={16} />}
          label="Seans Seviyeleri"
          description="Asya, Londra ve New York seanslarının en yüksek/en düşük seviyeleri. Intraday destek/direnç noktaları."
          gradientFrom="#f59e0b"
          gradientTo="#d97706"
          glowColor="rgba(245, 158, 11, 0.4)"
        />
          </div>
        </div>
      </div>

      {/* Backdrop overlay when sidebar is open */}
      {isSidebarOpen && (
        <div
          onClick={() => setIsSidebarOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999,
            transition: 'opacity 0.3s ease'
          }}
        />
      )}

      {/* Main Candlestick Chart */}
      <div className="bg-gradient-to-b from-[#0a0a0a] to-[#111111] rounded-lg border border-slate-800 p-4" style={{ minHeight: '500px' }}>
        {candleData.length === 0 ? (
          <div style={{ padding: '20px', textAlign: 'center', color: '#94a3b8' }}>
            Mum verileri yükleniyor...
          </div>
        ) : (
          <Chart
            options={candlestickOptions}
            series={candleSeries}
            type="candlestick"
            height={500}
          />
        )}
      </div>

      {/* RSI Grafiği */}
      {showRSI && (
        <div className="rounded-2xl border border-white/10 p-5"
          style={{
            background: 'linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(17, 17, 17, 0.9) 100%)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          }}
        >
          <h3 style={{
            background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '14px',
            fontWeight: '700',
            marginBottom: '12px',
            letterSpacing: '0.5px'
          }}>
            RSI - Göreceli Güç Endeksi (14)
          </h3>
          {rsiSeries.length > 0 && (
            <Chart
              options={rsiOptions}
              series={rsiSeries}
              type="line"
              height={150}
            />
          )}
        </div>
      )}

      {/* MFI Grafiği */}
      {showMFI && (
        <div className="rounded-2xl border border-white/10 p-5"
          style={{
            background: 'linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(17, 17, 17, 0.9) 100%)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          }}
        >
          <h3 style={{
            background: 'linear-gradient(135deg, #ec4899 0%, #db2777 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '14px',
            fontWeight: '700',
            marginBottom: '12px',
            letterSpacing: '0.5px'
          }}>
            MFI - Para Akış Endeksi
          </h3>
          {mfiSeries.length > 0 && (
            <Chart
              options={mfiOptions}
              series={mfiSeries}
              type="line"
              height={150}
            />
          )}
        </div>
      )}

      {/* İşlem Hacmi Grafiği */}
      {showVolume && (
        <div className="rounded-2xl border border-white/10 p-5"
          style={{
            background: 'linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(17, 17, 17, 0.9) 100%)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          }}
        >
          <h3 style={{
            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '14px',
            fontWeight: '700',
            marginBottom: '12px',
            letterSpacing: '0.5px'
          }}>
            📊 İşlem Hacmi
          </h3>
          <Chart
            options={volumeOptions}
            series={volumeSeries}
            type="bar"
            height={150}
          />
        </div>
      )}

      {/* 📈 Kümülatif Delta Grafiği */}
      {showCVD && (
        <div className="rounded-2xl border border-white/10 p-5"
          style={{
            background: 'linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(17, 17, 17, 0.9) 100%)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          }}
        >
          <h3 style={{
            background: 'linear-gradient(135deg, #14b8a6 0%, #0d9488 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontSize: '14px',
            fontWeight: '700',
            marginBottom: '12px',
            letterSpacing: '0.5px'
          }}>
            CVD - Kümülatif Hacim Deltası
          </h3>
          {cvdSeries.length > 0 && (
            <Chart
              options={cvdOptions}
              series={cvdSeries}
              type="line"
              height={150}
            />
          )}
        </div>
      )}
    </div>
  );
}
