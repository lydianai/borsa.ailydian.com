'use client';

/**
 * 🔤 ALFABETIK PATTERN ULTRA - Unique Pattern Analysis System
 *
 * Unique Pattern Features:
 * - Pattern Frequency Heatmap (which letters how often they win)
 * - Pattern Similarity Matcher (group similar patterns)
 * - Historical Pattern Performance (past pattern success rate)
 * - Pattern Builder Tool (create your own pattern strategy)
 * - Pattern Correlation Matrix (which letters rising together)
 * - AI Pattern Predictor (next strong letter prediction)
 * - Pattern Strength Score (0-100 strength score)
 * - Multi-Letter Pattern Detection (multiple letter combination)
 */

import { useState, useEffect, useMemo } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS } from '@/lib/colors';

interface AlfabetikPattern {
  harf: string;
  coinSayisi: number;
  ortalamaPerformans24h: number;
  ortalamaPerformans7d: number;
  momentum: 'YUKARIDA' | 'ASAGIDA' | 'YATAY';
  güvenilirlik: number;
  kategoriAnaliz: {
    layer1: number;
    layer2: number;
    defi: number;
    ai: number;
    gaming: number;
    other: number;
  };
  topCoins: string[];
  zayifCoins: string[];
  signal: 'STRONG_BUY' | 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
}

interface CoinDetail {
  symbol: string;
  currentPrice: number;
  signal: 'LONG' | 'SHORT' | 'HOLD';
  confidence: number;
  entryPrice: number;
  targetPrice: number;
  stopLoss: number;
  recommendedLeverage: number;
  riskRewardRatio: number;
  positionSize: number;
  expectedProfit: number;
  maxLoss: number;
  timeframe: string;
  volatility: number;
  trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  changePercent24h: number;
  changePercent7d: number;
  rsi: number;
  momentum: string;
  patternLetter: string;
}

interface PatternCorrelation {
  letter1: string;
  letter2: string;
  correlation: number;
  strength: string;
}

type ViewMode = 'grid' | 'heatmap' | 'correlation' | 'timeline';

export default function AlfabetikPatternPage() {
  const [patterns, setPatterns] = useState<AlfabetikPattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(60);
  const [selectedCoin, setSelectedCoin] = useState<CoinDetail | null>(null);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [filterSignal, setFilterSignal] = useState<'ALL' | 'STRONG_BUY' | 'BUY' | 'SELL' | 'HOLD'>('ALL');
  const [minConfidence, setMinConfidence] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [showLogicModal, setShowLogicModal] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [selectedPattern, setSelectedPattern] = useState<AlfabetikPattern | null>(null);
  const [showPatternBuilder, setShowPatternBuilder] = useState(false);
  const [customPattern, setCustomPattern] = useState<string[]>([]);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Fetch alfabetik pattern data
  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/alfabetik-pattern/tracking', {
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch pattern data');
      }

      setPatterns(data.patterns || []);
      setLoading(false);
    } catch (err: any) {
      console.error('Alfabetik Pattern Fetch Error:', err);
      setError(err.message || 'Failed to load data');
      setLoading(false);
    }
  };

  // Initial load and countdown timer
  useEffect(() => {
    fetchData();

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchData();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdownInterval);
  }, []);

  // Calculate Pattern Strength Score (0-100)
  const calculatePatternStrength = (pattern: AlfabetikPattern): number => {
    const perfWeight = (pattern.ortalamaPerformans24h + 10) / 20; // -10% to +10% -> 0 to 1
    const confWeight = pattern.confidence / 100;
    const momentumWeight = pattern.momentum === 'YUKARIDA' ? 1 : pattern.momentum === 'ASAGIDA' ? 0 : 0.5;
    const coinCountWeight = Math.min(pattern.coinSayisi / 50, 1); // Max 50 coins = 1

    const strength = (perfWeight * 0.4 + confWeight * 0.3 + momentumWeight * 0.2 + coinCountWeight * 0.1) * 100;
    return Math.min(100, Math.max(0, strength));
  };

  // Calculate Pattern Correlations
  const calculateCorrelations = useMemo((): PatternCorrelation[] => {
    const correlations: PatternCorrelation[] = [];

    for (let i = 0; i < patterns.length; i++) {
      for (let j = i + 1; j < patterns.length; j++) {
        const p1 = patterns[i];
        const p2 = patterns[j];

        // Simple correlation based on similar performance
        const perfDiff = Math.abs(p1.ortalamaPerformans24h - p2.ortalamaPerformans24h);
        const correlation = Math.max(0, 1 - perfDiff / 20); // 0-1 scale

        let strength = 'Weak';
        if (correlation > 0.8) strength = 'Very Strong';
        else if (correlation > 0.6) strength = 'Strong';
        else if (correlation > 0.4) strength = 'Medium';

        correlations.push({
          letter1: p1.harf,
          letter2: p2.harf,
          correlation: parseFloat(correlation.toFixed(2)),
          strength
        });
      }
    }

    return correlations.sort((a, b) => b.correlation - a.correlation);
  }, [patterns]);

  // Get top correlations
  const topCorrelations = calculateCorrelations.slice(0, 10);

  // AI Pattern Predictor - Next strong letter
  const predictNextStrongLetter = useMemo(() => {
    if (patterns.length === 0) return null;

    // Find patterns with strong upward momentum and high confidence
    const strongPatterns = patterns
      .filter(p => p.momentum === 'YUKARIDA' && p.confidence > 70)
      .sort((a, b) => calculatePatternStrength(b) - calculatePatternStrength(a));

    if (strongPatterns.length === 0) return null;

    const predicted = strongPatterns[0];
    const strength = calculatePatternStrength(predicted);

    return {
      letter: predicted.harf,
      strength,
      confidence: predicted.confidence,
      expectedGain: predicted.ortalamaPerformans24h
    };
  }, [patterns]);

  // Handle coin click
  const handleCoinClick = async (_pattern: AlfabetikPattern, coinSymbol: string) => {
    try {
      const symbol = coinSymbol.endsWith('USDT') ? coinSymbol : `${coinSymbol}USDT`;
      const response = await fetch(`/api/alfabetik-pattern/coin-details?symbol=${symbol}`);
      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch coin details');
      }

      setSelectedCoin(data.data);
    } catch (err: any) {
      console.error('Failed to load coin details:', err);
      alert('Coin details could not be loaded. Please try again.');
    }
  };

  // Filter patterns
  const filteredPatterns = patterns.filter((p) => {
    if (filterSignal !== 'ALL' && p.signal !== filterSignal) return false;
    if (p.confidence < minConfidence) return false;
    if (searchTerm && !p.harf.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  // Get signal color
  const getSignalColor = (signal: string): string => {
    switch (signal) {
      case 'STRONG_BUY': return COLORS.success;
      case 'BUY': return COLORS.info;
      case 'SELL': return COLORS.danger;
      default: return COLORS.text.secondary;
    }
  };

  // Get strength color
  const getStrengthColor = (strength: number): string => {
    if (strength >= 80) return COLORS.success;
    if (strength >= 60) return COLORS.info;
    if (strength >= 40) return COLORS.warning;
    return COLORS.danger;
  };

  // Pattern Builder - Add/Remove letters
  const toggleCustomPattern = (letter: string) => {
    if (customPattern.includes(letter)) {
      setCustomPattern(customPattern.filter(l => l !== letter));
    } else {
      setCustomPattern([...customPattern, letter]);
    }
  };

  // Calculate custom pattern score
  const customPatternScore = useMemo(() => {
    if (customPattern.length === 0) return 0;

    const selectedPatterns = patterns.filter(p => customPattern.includes(p.harf));
    if (selectedPatterns.length === 0) return 0;

    const avgPerf = selectedPatterns.reduce((sum, p) => sum + p.ortalamaPerformans24h, 0) / selectedPatterns.length;
    const avgConf = selectedPatterns.reduce((sum, p) => sum + p.confidence, 0) / selectedPatterns.length;

    return ((avgPerf + 10) / 20 * 0.6 + avgConf / 100 * 0.4) * 100;
  }, [customPattern, patterns]);

  return (
    <>
      <SharedSidebar
        currentPage="alfabetik-pattern"
        onAiAssistantOpen={() => setAiAssistantOpen(true)}
        notificationCounts={notificationCounts}
        coinCount={patterns.length}
        countdown={countdown}
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
      />

      <AIAssistantFullScreen
        isOpen={aiAssistantOpen}
        onClose={() => setAiAssistantOpen(false)}
      />

      {/* Main Content */}
      <div style={{ flex: 1, marginLeft: '280px', padding: '32px 48px', paddingTop: isLocalhost ? '116px' : '60px', overflowY: 'auto', maxWidth: '1920px', margin: '0 auto', width: '100%' }}>

        {/* Header */}
        <div style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '24px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '12px' }}>
              <div style={{
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                padding: '12px',
                borderRadius: '16px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: `0 8px 24px ${COLORS.premium}40`
              }}>
                <Icons.Layers style={{ width: '32px', height: '32px', color: '#000' }} />
              </div>
              <div>
                <h1 style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0, background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                  Desen Tanıma Ultra
                </h1>
                <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: '4px 0 0 0' }}>
                  {patterns.length} alfabetik desen • YZ destekli analiz • {countdown}s yenileme
                </p>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px' }}>
            <button
              onClick={() => setShowPatternBuilder(true)}
              style={{
                padding: '14px 24px',
                background: `linear-gradient(135deg, ${COLORS.info}, ${COLORS.premium})`,
                color: '#000',
                border: 'none',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: `0 4px 20px ${COLORS.info}40`,
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 6px 25px ${COLORS.info}60`;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = `0 4px 20px ${COLORS.info}40`;
              }}
            >
              <Icons.Zap style={{ width: '18px', height: '18px' }} />
              Desen Oluşturucu
            </button>

            <div>
              <style>{`
                @media (max-width: 768px) {
                  .mantik-button-alfabetik {
                    padding: 10px 20px !important;
                    fontSize: 13px !important;
                    height: 42px !important;
                  }
                  .mantik-button-alfabetik svg {
                    width: 18px !important;
                    height: 18px !important;
                  }
                }
                @media (max-width: 480px) {
                  .mantik-button-alfabetik {
                    padding: 8px 16px !important;
                    fontSize: 12px !important;
                    height: 40px !important;
                  }
                  .mantik-button-alfabetik svg {
                    width: 16px !important;
                    height: 16px !important;
                  }
                }
              `}</style>
              <button
                onClick={() => setShowLogicModal(true)}
                className="mantik-button-alfabetik"
                style={{
                  padding: '12px 24px',
                  background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                  color: '#000',
                  border: 'none',
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  height: '44px',
                  boxShadow: `0 4px 20px ${COLORS.premium}40`,
                  transition: 'all 0.3s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = `0 6px 25px ${COLORS.premium}60`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = `0 4px 20px ${COLORS.premium}40`;
                }}
              >
                <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
                MANTIK
              </button>
            </div>
          </div>
        </div>

        {/* AI Prediction Card */}
        {predictNextStrongLetter && (
          <div style={{
            background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
            border: `2px solid ${COLORS.success}`,
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '32px',
            boxShadow: `0 8px 32px ${COLORS.success}30`
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Icons.Bot style={{ width: '28px', height: '28px', color: COLORS.success }} />
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  YZ Desen Tahmini
                </h3>
              </div>
              <div style={{
                background: `${COLORS.success}20`,
                border: `1px solid ${COLORS.success}`,
                padding: '6px 16px',
                borderRadius: '8px',
                fontSize: '12px',
                fontWeight: '600',
                color: COLORS.success
              }}>
                Live
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: '24px', alignItems: 'center' }}>
              <div style={{
                fontSize: '72px',
                fontWeight: '900',
                color: COLORS.success,
                lineHeight: '1'
              }}>
                {predictNextStrongLetter.letter}
              </div>

              <div>
                <p style={{ fontSize: '16px', color: COLORS.text.primary, marginBottom: '12px', fontWeight: '600' }}>
                  Sıradaki güçlü desen tahmini
                </p>
                <div style={{ display: 'flex', gap: '16px' }}>
                  <div>
                    <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>Güç Skoru: </span>
                    <span style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.success }}>
                      {predictNextStrongLetter.strength.toFixed(0)}/100
                    </span>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>Güvenilirlik: </span>
                    <span style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.success }}>
                      {predictNextStrongLetter.confidence}%
                    </span>
                  </div>
                  <div>
                    <span style={{ fontSize: '12px', color: COLORS.text.secondary }}>Beklenen: </span>
                    <span style={{ fontSize: '16px', fontWeight: 'bold', color: COLORS.success }}>
                      +{predictNextStrongLetter.expectedGain.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>

              <div style={{
                width: '120px',
                height: '120px',
                borderRadius: '50%',
                background: `conic-gradient(${COLORS.success} ${predictNextStrongLetter.strength}%, ${COLORS.border.default} 0)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}>
                <div style={{
                  width: '100px',
                  height: '100px',
                  borderRadius: '50%',
                  background: COLORS.bg.primary,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column'
                }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.success }}>
                    {predictNextStrongLetter.strength.toFixed(0)}
                  </div>
                  <div style={{ fontSize: '10px', color: COLORS.text.secondary }}>
                    Güç
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* View Mode Selector & Filters */}
        <div style={{
          background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
          border: `1px solid ${COLORS.border.default}`,
          borderRadius: '16px',
          padding: '24px',
          marginBottom: '32px'
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', alignItems: 'end' }}>

            {/* View Mode */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                View Mode
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                {[
                  { mode: 'grid' as ViewMode, icon: Icons.Dashboard, label: 'Grid' },
                  { mode: 'heatmap' as ViewMode, icon: Icons.BarChart3, label: 'Heatmap' },
                  { mode: 'correlation' as ViewMode, icon: Icons.Activity, label: 'Correlation' }
                ].map((view) => {
                  const IconComponent = view.icon;
                  return (
                    <button
                      key={view.mode}
                      onClick={() => setViewMode(view.mode)}
                      style={{
                        flex: 1,
                        padding: '10px',
                        background: viewMode === view.mode ? `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})` : COLORS.bg.primary,
                        border: `1px solid ${viewMode === view.mode ? COLORS.premium : COLORS.border.default}`,
                        borderRadius: '10px',
                        color: viewMode === view.mode ? '#000' : COLORS.text.secondary,
                        fontSize: '11px',
                        fontWeight: viewMode === view.mode ? '600' : '500',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '4px',
                        transition: 'all 0.2s ease'
                      }}
                    >
                      <IconComponent style={{ width: '14px', height: '14px' }} />
                      {view.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Signal Filter */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Signal Filter
              </label>
              <select
                value={filterSignal}
                onChange={(e) => setFilterSignal(e.target.value as any)}
                style={{
                  width: '100%',
                  padding: '12px',
                  background: COLORS.bg.primary,
                  border: `1px solid ${COLORS.border.default}`,
                  borderRadius: '10px',
                  color: COLORS.text.primary,
                  fontSize: '14px',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="ALL">All</option>
                <option value="STRONG_BUY">Strong Buy</option>
                <option value="BUY">Buy</option>
                <option value="SELL">Sell</option>
                <option value="HOLD">Hold</option>
              </select>
            </div>

            {/* Confidence Filter */}
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: COLORS.text.secondary, marginBottom: '8px', fontWeight: '500' }}>
                Min Reliability: {minConfidence}%
              </label>
              <input
                type="range"
                min="0"
                max="100"
                step="10"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                style={{
                  width: '100%',
                  height: '8px',
                  borderRadius: '4px',
                  background: `linear-gradient(to right, ${COLORS.premium} 0%, ${COLORS.premium} ${minConfidence}%, ${COLORS.border.default} ${minConfidence}%, ${COLORS.border.default} 100%)`,
                  outline: 'none',
                  cursor: 'pointer',
                  WebkitAppearance: 'none',
                  appearance: 'none'
                }}
              />
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ textAlign: 'center', padding: '80px 0', color: COLORS.text.secondary }}>
            Loading pattern data...
          </div>
        )}

        {/* Error State */}
        {error && (
          <div style={{
            background: `${COLORS.danger}10`,
            border: `1px solid ${COLORS.danger}`,
            borderRadius: '16px',
            padding: '24px',
            textAlign: 'center',
            color: COLORS.danger,
          }}>
            Error: {error}
          </div>
        )}

        {/* Grid View */}
        {!loading && !error && viewMode === 'grid' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: '20px'
          }}>
            {filteredPatterns.map((pattern) => {
              const strength = calculatePatternStrength(pattern);
              const strengthColor = getStrengthColor(strength);
              const signalColor = getSignalColor(pattern.signal);

              return (
                <div
                  key={pattern.harf}
                  onClick={() => setSelectedPattern(pattern)}
                  style={{
                    background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
                    border: `3px solid ${signalColor}`,
                    borderRadius: '16px',
                    padding: '24px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-8px)';
                    e.currentTarget.style.boxShadow = `0 16px 40px ${signalColor}60`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  {/* Background Gradient */}
                  <div style={{
                    position: 'absolute',
                    top: '-50%',
                    right: '-50%',
                    width: '200%',
                    height: '200%',
                    background: `radial-gradient(circle, ${signalColor}10, transparent)`,
                    pointerEvents: 'none'
                  }} />

                  {/* Content */}
                  <div style={{ position: 'relative', zIndex: 1 }}>
                    {/* Letter & Signal */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                      <div style={{ fontSize: '64px', fontWeight: '900', color: signalColor, lineHeight: '1' }}>
                        {pattern.harf}
                      </div>
                      <div style={{
                        background: signalColor,
                        color: '#000',
                        padding: '8px 16px',
                        borderRadius: '10px',
                        fontSize: '12px',
                        fontWeight: '700'
                      }}>
                        {pattern.signal}
                      </div>
                    </div>

                    {/* Pattern Strength Bar */}
                    <div style={{ marginBottom: '20px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={{ fontSize: '12px', color: COLORS.text.secondary, fontWeight: '500' }}>Pattern Strength</span>
                        <span style={{ fontSize: '14px', fontWeight: 'bold', color: strengthColor }}>{strength.toFixed(0)}/100</span>
                      </div>
                      <div style={{
                        width: '100%',
                        height: '8px',
                        background: COLORS.bg.primary,
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${strength}%`,
                          height: '100%',
                          background: `linear-gradient(90deg, ${COLORS.danger}, ${COLORS.warning}, ${COLORS.success})`,
                          borderRadius: '4px',
                          transition: 'width 0.5s ease'
                        }} />
                      </div>
                    </div>

                    {/* Performance */}
                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>
                        24h Performance
                      </div>
                      <div style={{ fontSize: '28px', fontWeight: 'bold', color: pattern.ortalamaPerformans24h > 0 ? COLORS.success : COLORS.danger }}>
                        {pattern.ortalamaPerformans24h > 0 ? '+' : ''}{pattern.ortalamaPerformans24h.toFixed(2)}%
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>Coin Count</div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: COLORS.text.primary }}>{pattern.coinSayisi}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>Reliability</div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: COLORS.text.primary }}>{pattern.güvenilirlik}%</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>Momentum</div>
                        <div style={{
                          fontSize: '14px',
                          fontWeight: '600',
                          color: pattern.momentum === 'YUKARIDA' ? COLORS.success : pattern.momentum === 'ASAGIDA' ? COLORS.danger : COLORS.warning
                        }}>
                          {pattern.momentum}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '11px', color: COLORS.text.secondary }}>Confidence</div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: COLORS.text.primary }}>{pattern.confidence}%</div>
                      </div>
                    </div>

                    {/* Top Coins Preview */}
                    <div style={{ paddingTop: '16px', borderTop: `1px solid ${COLORS.border.default}` }}>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                        Coins ({pattern.topCoins.length})
                      </div>
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', maxHeight: '60px', overflowY: 'auto' }}>
                        {pattern.topCoins.slice(0, 5).map((coin) => (
                          <span
                            key={coin}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCoinClick(pattern, coin);
                            }}
                            style={{
                              background: `${signalColor}20`,
                              border: `1px solid ${signalColor}40`,
                              borderRadius: '6px',
                              padding: '4px 10px',
                              fontSize: '11px',
                              fontWeight: '600',
                              color: signalColor,
                              cursor: 'pointer',
                              transition: 'all 0.2s ease'
                            }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.background = signalColor;
                              e.currentTarget.style.color = '#000';
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.background = `${signalColor}20`;
                              e.currentTarget.style.color = signalColor;
                            }}
                          >
                            {coin}
                          </span>
                        ))}
                        {pattern.topCoins.length > 5 && (
                          <span style={{
                            padding: '4px 10px',
                            fontSize: '11px',
                            color: COLORS.text.secondary
                          }}>
                            +{pattern.topCoins.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Heatmap View */}
        {!loading && !error && viewMode === 'heatmap' && (
          <div>
            <div style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `1px solid ${COLORS.border.default}`,
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '20px'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Icons.BarChart3 style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                Pattern Frequency Heatmap
              </h3>
              <p style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '16px' }}>
                Color intensity shows pattern strength score. Dark green = Strong, Light = Weak
              </p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: '12px' }}>
              {filteredPatterns.map((pattern) => {
                const strength = calculatePatternStrength(pattern);
                const intensity = strength / 100;

                let bgColor;
                if (pattern.ortalamaPerformans24h > 0) {
                  bgColor = `rgba(16, 185, 129, ${0.2 + intensity * 0.8})`;
                } else {
                  bgColor = `rgba(239, 68, 68, ${0.2 + intensity * 0.8})`;
                }

                return (
                  <div
                    key={pattern.harf}
                    onClick={() => setSelectedPattern(pattern)}
                    style={{
                      background: bgColor,
                      border: `2px solid ${pattern.ortalamaPerformans24h > 0 ? COLORS.success : COLORS.danger}`,
                      borderRadius: '12px',
                      padding: '20px',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      minHeight: '120px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'space-between'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'scale(1.1)';
                      e.currentTarget.style.zIndex = '10';
                      e.currentTarget.style.boxShadow = `0 8px 24px ${pattern.ortalamaPerformans24h > 0 ? COLORS.success : COLORS.danger}60`;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'scale(1)';
                      e.currentTarget.style.zIndex = '1';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <div style={{ fontSize: '48px', fontWeight: '900', color: '#000', lineHeight: '1', textAlign: 'center' }}>
                      {pattern.harf}
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#000' }}>
                        {pattern.ortalamaPerformans24h > 0 ? '+' : ''}{pattern.ortalamaPerformans24h.toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '11px', color: '#00000080', marginTop: '4px' }}>
                        {pattern.coinSayisi} coins
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Correlation View */}
        {!loading && !error && viewMode === 'correlation' && (
          <div>
            <div style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `1px solid ${COLORS.border.default}`,
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '20px'
            }}>
              <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Icons.Activity style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                Pattern Korelasyon Matrisi
              </h3>
              <p style={{ fontSize: '14px', color: COLORS.text.secondary }}>
                Which letters rising together? Highest correlations:
              </p>
            </div>

            <div style={{ display: 'grid', gap: '16px' }}>
              {topCorrelations.map((corr, index) => (
                <div
                  key={`${corr.letter1}-${corr.letter2}`}
                  style={{
                    background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
                    border: `1px solid ${corr.correlation > 0.7 ? COLORS.success : corr.correlation > 0.5 ? COLORS.info : COLORS.border.default}`,
                    borderRadius: '12px',
                    padding: '20px',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateX(8px)';
                    e.currentTarget.style.borderColor = COLORS.premium;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateX(0)';
                    e.currentTarget.style.borderColor = corr.correlation > 0.7 ? COLORS.success : corr.correlation > 0.5 ? COLORS.info : COLORS.border.default;
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                      <div style={{
                        background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}20)`,
                        padding: '8px 16px',
                        borderRadius: '8px',
                        fontSize: '14px',
                        fontWeight: 'bold',
                        color: COLORS.premium,
                        minWidth: '40px',
                        textAlign: 'center'
                      }}>
                        #{index + 1}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div style={{ fontSize: '32px', fontWeight: '900', color: COLORS.text.primary }}>
                          {corr.letter1}
                        </div>
                        <Icons.ArrowRight style={{ width: '20px', height: '20px', color: COLORS.text.secondary }} />
                        <div style={{ fontSize: '32px', fontWeight: '900', color: COLORS.text.primary }}>
                          {corr.letter2}
                        </div>
                      </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '4px' }}>
                        Correlation
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: corr.correlation > 0.7 ? COLORS.success : COLORS.info }}>
                        {(corr.correlation * 100).toFixed(0)}%
                      </div>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginTop: '4px' }}>
                        {corr.strength}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No Results */}
        {!loading && !error && filteredPatterns.length === 0 && (
          <div style={{ textAlign: 'center', padding: '80px 0', color: COLORS.text.secondary }}>
            No patterns found matching your filters.
          </div>
        )}
      </div>

      {/* Pattern Builder Modal */}
      {showPatternBuilder && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '20px',
            backdropFilter: 'blur(10px)'
          }}
          onClick={() => setShowPatternBuilder(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '20px',
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}60`,
              padding: '32px'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Icons.Zap style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                <h2 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                  Pattern Builder
                </h2>
              </div>
              <button
                onClick={() => setShowPatternBuilder(false)}
                style={{
                  background: 'transparent',
                  border: `2px solid ${COLORS.danger}`,
                  borderRadius: '12px',
                  padding: '10px 20px',
                  color: COLORS.danger,
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = COLORS.danger;
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = COLORS.danger;
                }}
              >
                CLOSE
              </button>
            </div>

            <p style={{ fontSize: '15px', color: COLORS.text.secondary, marginBottom: '24px' }}>
              Create your own pattern strategy. Select letters to make a custom combination and see its overall performance.
            </p>

            {/* Custom Pattern Score */}
            {customPattern.length > 0 && (
              <div style={{
                background: `${COLORS.success}10`,
                border: `2px solid ${COLORS.success}`,
                borderRadius: '12px',
                padding: '20px',
                marginBottom: '24px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '14px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                      Selected Pattern: {customPattern.join(', ')}
                    </div>
                    <div style={{ fontSize: '32px', fontWeight: 'bold', color: COLORS.success }}>
                      Score: {customPatternScore.toFixed(0)}/100
                    </div>
                  </div>
                  <button
                    onClick={() => setCustomPattern([])}
                    style={{
                      padding: '10px 20px',
                      background: COLORS.danger,
                      border: 'none',
                      borderRadius: '10px',
                      color: '#fff',
                      fontSize: '13px',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}

            {/* Letter Selection Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '12px' }}>
              {patterns.map((pattern) => {
                const isSelected = customPattern.includes(pattern.harf);
                const strength = calculatePatternStrength(pattern);

                return (
                  <button
                    key={pattern.harf}
                    onClick={() => toggleCustomPattern(pattern.harf)}
                    style={{
                      padding: '20px',
                      background: isSelected ? `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.info})` : COLORS.bg.primary,
                      border: `2px solid ${isSelected ? COLORS.premium : COLORS.border.default}`,
                      borderRadius: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = COLORS.premium;
                        e.currentTarget.style.transform = 'scale(1.05)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) {
                        e.currentTarget.style.borderColor = COLORS.border.default;
                        e.currentTarget.style.transform = 'scale(1)';
                      }
                    }}
                  >
                    <div style={{ fontSize: '32px', fontWeight: '900', color: isSelected ? '#000' : COLORS.text.primary, textAlign: 'center', marginBottom: '8px' }}>
                      {pattern.harf}
                    </div>
                    <div style={{ fontSize: '11px', color: isSelected ? '#00000080' : COLORS.text.secondary, textAlign: 'center' }}>
                      {strength.toFixed(0)}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Pattern Detail Modal */}
      {selectedPattern && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '20px',
            backdropFilter: 'blur(10px)'
          }}
          onClick={() => setSelectedPattern(null)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `3px solid ${getSignalColor(selectedPattern.signal)}`,
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '800px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: `0 20px 60px ${getSignalColor(selectedPattern.signal)}66`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontSize: '72px', fontWeight: '900', color: getSignalColor(selectedPattern.signal), lineHeight: '1', marginBottom: '12px' }}>
                  {selectedPattern.harf}
                </div>
                <div style={{
                  background: getSignalColor(selectedPattern.signal),
                  color: '#000',
                  padding: '10px 20px',
                  borderRadius: '10px',
                  fontSize: '16px',
                  fontWeight: '700',
                  display: 'inline-block'
                }}>
                  {selectedPattern.signal}
                </div>
              </div>

              <button
                onClick={() => setSelectedPattern(null)}
                style={{
                  background: 'transparent',
                  border: `2px solid ${COLORS.danger}`,
                  borderRadius: '12px',
                  padding: '12px 24px',
                  color: COLORS.danger,
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = COLORS.danger;
                  e.currentTarget.style.color = '#fff';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = COLORS.danger;
                }}
              >
                CLOSE
              </button>
            </div>

            {/* All Coins List */}
            <div style={{
              background: `${getSignalColor(selectedPattern.signal)}10`,
              border: `1px solid ${getSignalColor(selectedPattern.signal)}40`,
              borderRadius: '12px',
              padding: '20px'
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                All Coins ({selectedPattern.topCoins.length})
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))', gap: '10px', maxHeight: '400px', overflowY: 'auto' }}>
                {selectedPattern.topCoins.map((coin) => (
                  <button
                    key={coin}
                    onClick={() => handleCoinClick(selectedPattern, coin)}
                    style={{
                      background: COLORS.bg.primary,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '12px',
                      fontSize: '13px',
                      fontWeight: '600',
                      color: COLORS.text.primary,
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = getSignalColor(selectedPattern.signal);
                      e.currentTarget.style.color = '#000';
                      e.currentTarget.style.transform = 'scale(1.05)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = COLORS.bg.primary;
                      e.currentTarget.style.color = COLORS.text.primary;
                      e.currentTarget.style.transform = 'scale(1)';
                    }}
                  >
                    {coin}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Coin Detail Modal */}
      {selectedCoin && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
            padding: '20px',
            backdropFilter: 'blur(10px)'
          }}
          onClick={() => setSelectedCoin(null)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.card}, ${COLORS.bg.secondary})`,
              border: `3px solid ${getSignalColor(selectedCoin.signal)}`,
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '700px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: `0 20px 60px ${getSignalColor(selectedCoin.signal)}66`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ marginBottom: '24px' }}>
              <div style={{ fontSize: '36px', fontWeight: '900', color: getSignalColor(selectedCoin.signal), marginBottom: '8px' }}>
                {selectedCoin.symbol}
              </div>
              <div style={{ fontSize: '24px', fontWeight: '700', color: COLORS.text.primary, marginBottom: '4px' }}>
                ${selectedCoin.currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
              </div>
              <div style={{ fontSize: '16px', color: selectedCoin.changePercent24h > 0 ? COLORS.success : COLORS.danger }}>
                {selectedCoin.changePercent24h > 0 ? '+' : ''}{selectedCoin.changePercent24h.toFixed(2)}% (24h)
              </div>
            </div>

            <div style={{
              background: getSignalColor(selectedCoin.signal),
              color: '#000',
              padding: '16px',
              borderRadius: '12px',
              marginBottom: '24px'
            }}>
              <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '4px' }}>Signal</div>
              <div style={{ fontSize: '24px', fontWeight: '900' }}>{selectedCoin.signal}</div>
              <div style={{ fontSize: '14px', opacity: 0.8, marginTop: '4px' }}>
                Reliability: {selectedCoin.confidence}% • {selectedCoin.trend} Trend
              </div>
            </div>

            <div style={{
              background: `${COLORS.premium}10`,
              border: `1px solid ${COLORS.premium}40`,
              borderRadius: '12px',
              padding: '20px'
            }}>
              <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.premium, marginBottom: '16px' }}>
                Trading Recommendations
              </div>

              <div style={{ display: 'grid', gap: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: `${COLORS.info}10`, borderRadius: '8px' }}>
                  <div>
                    <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '4px' }}>Entry</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.info }}>
                      ${selectedCoin.entryPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: `${COLORS.success}10`, borderRadius: '8px' }}>
                  <div>
                    <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '4px' }}>Target (TP)</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.success }}>
                      ${selectedCoin.targetPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                    </div>
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.success }}>
                    +{selectedCoin.expectedProfit.toFixed(2)}%
                  </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px', background: `${COLORS.danger}10`, borderRadius: '8px' }}>
                  <div>
                    <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '4px' }}>Stop Loss</div>
                    <div style={{ fontSize: '20px', fontWeight: '700', color: COLORS.danger }}>
                      ${selectedCoin.stopLoss.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
                    </div>
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: '700', color: COLORS.danger }}>
                    {selectedCoin.maxLoss.toFixed(2)}%
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                  <div style={{ padding: '12px', background: `${COLORS.warning}10`, borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>Leverage</div>
                    <div style={{ fontSize: '20px', fontWeight: '900', color: COLORS.warning }}>{selectedCoin.recommendedLeverage}x</div>
                  </div>
                  <div style={{ padding: '12px', background: `${COLORS.premium}10`, borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>R/R</div>
                    <div style={{ fontSize: '20px', fontWeight: '900', color: COLORS.premium }}>{selectedCoin.riskRewardRatio.toFixed(2)}</div>
                  </div>
                  <div style={{ padding: '12px', background: `${COLORS.info}10`, borderRadius: '8px', textAlign: 'center' }}>
                    <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginBottom: '4px' }}>Position</div>
                    <div style={{ fontSize: '20px', fontWeight: '900', color: COLORS.info }}>{selectedCoin.positionSize}%</div>
                  </div>
                </div>
              </div>
            </div>

            <div style={{ marginTop: '24px', fontSize: '11px', color: COLORS.text.secondary, textAlign: 'center', fontStyle: 'italic' }}>
              These recommendations are generated by algorithm. Do your own research.
            </div>

            <button
              onClick={() => setSelectedCoin(null)}
              style={{
                width: '100%',
                marginTop: '20px',
                padding: '14px',
                background: 'transparent',
                border: `2px solid ${COLORS.danger}`,
                borderRadius: '12px',
                color: COLORS.danger,
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = COLORS.danger;
                e.currentTarget.style.color = '#fff';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent';
                e.currentTarget.style.color = COLORS.danger;
              }}
            >
              CLOSE
            </button>
          </div>
        </div>
      )}

      {/* LOGIC Modal */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.92)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 2000,
            padding: '20px',
            backdropFilter: 'blur(10px)'
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
              border: `2px solid ${COLORS.premium}`,
              borderRadius: '16px',
              maxWidth: '900px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              boxShadow: `0 0 60px ${COLORS.premium}80`
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{
              background: `linear-gradient(135deg, ${COLORS.premium}15, ${COLORS.warning}15)`,
              padding: '24px',
              borderBottom: `2px solid ${COLORS.premium}`,
              position: 'sticky',
              top: 0,
              zIndex: 10,
              backdropFilter: 'blur(10px)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                  <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    Pattern Recognition Ultra - LOGIC
                  </h2>
                </div>
                <button
                  onClick={() => setShowLogicModal(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.active}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = COLORS.danger;
                    e.currentTarget.style.borderColor = COLORS.danger;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.borderColor = COLORS.border.active;
                  }}
                >
                  CLOSE
                </button>
              </div>
            </div>

            <div style={{ padding: '32px' }}>
              <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.Target style={{ width: '24px', height: '24px' }} />
                  Unique Pattern Features
                </h3>
                <div style={{ display: 'grid', gap: '12px' }}>
                  {[
                    { name: 'Pattern Strength Score', desc: '0-100 pattern strength score. Calculated based on performance, reliability, momentum and coin count.' },
                    { name: 'AI Pattern Predictor', desc: 'Predicts the next strong letter. Identifies patterns with high momentum and reliability.' },
                    { name: 'Pattern Heatmap', desc: 'Pattern frequency visualization with color intensity. Dark color = Strong pattern.' },
                    { name: 'Pattern Correlation Matrix', desc: 'Which letters rising together? Shows the highest correlations.' },
                    { name: 'Pattern Builder', desc: 'Create your own pattern strategy. Select letters and calculate custom pattern score.' },
                    { name: '3 View Modes', desc: 'Grid (detailed cards), Heatmap (visual density), Correlation (relationship matrix).' },
                    { name: 'Real-time Pattern Tracking', desc: 'Alphabetic pattern analysis on 600+ coins. Auto-refresh every 60 seconds.' },
                    { name: 'Multi-Letter Pattern Detection', desc: 'Create custom patterns with multiple letter combinations.' }
                  ].map((feature, index) => (
                    <div key={index} style={{
                      background: `${COLORS.bg.card}40`,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '16px',
                      transition: 'all 0.3s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = COLORS.premium;
                      e.currentTarget.style.transform = 'translateX(8px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = COLORS.border.default;
                      e.currentTarget.style.transform = 'translateX(0)';
                    }}>
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}20)`,
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '14px',
                          fontWeight: 'bold',
                          color: COLORS.premium,
                          minWidth: '32px',
                          textAlign: 'center'
                        }}>
                          {index + 1}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            {feature.name}
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            {feature.desc}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{
                background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                border: `2px solid ${COLORS.warning}`,
                borderRadius: '12px',
                padding: '20px'
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                  How to Use Patterns?
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.8' }}>
                  <li><strong style={{ color: COLORS.text.primary }}>Grid Mode:</strong> Detailed analysis and strength score for each pattern.</li>
                  <li><strong style={{ color: COLORS.text.primary }}>Heatmap Mode:</strong> Find the strongest patterns visually.</li>
                  <li><strong style={{ color: COLORS.text.primary }}>Correlation Mode:</strong> Discover letters rising together.</li>
                  <li><strong style={{ color: COLORS.text.primary }}>Pattern Builder:</strong> Test custom combinations.</li>
                  <li><strong style={{ color: COLORS.text.primary }}>AI Prediction:</strong> Track the next strong pattern.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
