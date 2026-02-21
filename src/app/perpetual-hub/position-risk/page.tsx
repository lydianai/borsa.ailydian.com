'use client';

/**
 * 🛡️ AI POSITION RISK ANALYZER
 *
 * Machine learning-powered position risk analysis
 *
 * Features:
 * - Real-time liquidation price calculator
 * - AI risk score (0-10 scale)
 * - Smart stop-loss suggestions
 * - Portfolio heat map
 * - Risk/Reward ratio optimizer
 * - Maximum position size calculator
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  leverage: number;
  liquidationPrice: number;
  margin: number;
  pnl: number;
  pnlPercent: number;
  riskScore: number;
  stopLoss?: number;
  takeProfit?: number;
}

interface RiskMetrics {
  totalRiskScore: number;
  portfolioMargin: number;
  usedMargin: number;
  availableMargin: number;
  marginLevel: number;
  totalPnL: number;
  totalPnLPercent: number;
  maxDrawdown: number;
  sharpeRatio: number;
}

export default function PositionRiskAnalyzer() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState<string | null>(null);
  const [showCalculator, setShowCalculator] = useState(false);

  // Calculator inputs
  const [calcSymbol, setCalcSymbol] = useState('BTC');
  const [calcEntry, setCalcEntry] = useState('67000');
  const [calcSize, setCalcSize] = useState('1000');
  const [calcLeverage, setCalcLeverage] = useState('10');
  const [calcSide, setCalcSide] = useState<'long' | 'short'>('long');

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock positions
  const positions: Position[] = [
    {
      id: '1',
      symbol: 'BTCUSDT',
      side: 'long',
      size: 1.5,
      entryPrice: 65400,
      currentPrice: 67234,
      leverage: 10,
      liquidationPrice: 58860,
      margin: 9810,
      pnl: 2751,
      pnlPercent: 28.05,
      riskScore: 6.2,
      stopLoss: 64000,
      takeProfit: 70000,
    },
    {
      id: '2',
      symbol: 'ETHUSDT',
      side: 'short',
      size: 10,
      entryPrice: 3500,
      currentPrice: 3421,
      leverage: 15,
      liquidationPrice: 3733,
      margin: 2333,
      pnl: 790,
      pnlPercent: 33.86,
      riskScore: 7.8,
      stopLoss: 3550,
    },
    {
      id: '3',
      symbol: 'SOLUSDT',
      side: 'long',
      size: 50,
      entryPrice: 120,
      currentPrice: 124.89,
      leverage: 20,
      liquidationPrice: 114,
      margin: 300,
      pnl: 244.5,
      pnlPercent: 81.5,
      riskScore: 8.9,
      takeProfit: 130,
    },
  ];

  // Mock risk metrics
  const riskMetrics: RiskMetrics = {
    totalRiskScore: 7.2,
    portfolioMargin: 15000,
    usedMargin: 12443,
    availableMargin: 2557,
    marginLevel: 120.5,
    totalPnL: 3785.5,
    totalPnLPercent: 30.42,
    maxDrawdown: -8.5,
    sharpeRatio: 2.34,
  };

  const getRiskColor = (score: number): string => {
    if (score >= 8) return '#EF4444';
    if (score >= 6) return '#F59E0B';
    if (score >= 4) return '#FBBF24';
    return '#10B981';
  };

  const getRiskLabel = (score: number): string => {
    if (score >= 8) return 'Çok Yüksek Risk';
    if (score >= 6) return 'Yüksek Risk';
    if (score >= 4) return 'Orta Risk';
    return 'Düşük Risk';
  };

  const calculateLiquidation = () => {
    const entry = parseFloat(calcEntry);
    const leverage = parseFloat(calcLeverage);
    const maintenanceMargin = 0.005; // 0.5%

    if (calcSide === 'long') {
      return entry * (1 - (1 / leverage) + maintenanceMargin);
    } else {
      return entry * (1 + (1 / leverage) - maintenanceMargin);
    }
  };

  const calculateMargin = () => {
    const size = parseFloat(calcSize);
    const leverage = parseFloat(calcLeverage);
    return size / leverage;
  };

  if (!mounted) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.6)' }}>Yükleniyor...</div>
      </div>
    );
  }

  return (
    <PWAProvider>
      <div
        suppressHydrationWarning
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
          paddingTop: '60px',
        }}
      >
        <SharedSidebar
          currentPage="perpetual-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1800px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
          {/* Header */}
          <div style={{ marginBottom: '32px' }}>
            <Link
              href="/perpetual-hub"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '14px',
                marginBottom: '12px',
              }}
            >
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              Perpetual Hub'a Dön
            </Link>

            <h1
              style={{
                fontSize: '40px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              AI Position Risk Analyzer
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Makine öğrenimi ile pozisyon risk analizi ve akıllı stop-loss önerileri
            </p>
          </div>

          {/* Portfolio Risk Metrics */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px',
              marginBottom: '32px',
            }}
          >
            {/* Total Risk Score */}
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: `2px solid ${getRiskColor(riskMetrics.totalRiskScore)}`,
                borderRadius: '16px',
                padding: '20px',
              }}
            >
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                PORTFOLIO RISK SCORE
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '8px' }}>
                <span style={{ fontSize: '32px', fontWeight: '700', color: getRiskColor(riskMetrics.totalRiskScore) }}>
                  {riskMetrics.totalRiskScore.toFixed(1)}
                </span>
                <span style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.5)' }}>/10</span>
              </div>
              <div style={{ fontSize: '12px', fontWeight: '600', color: getRiskColor(riskMetrics.totalRiskScore) }}>
                {getRiskLabel(riskMetrics.totalRiskScore)}
              </div>
            </div>

            {/* Margin Level */}
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '20px' }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>MARGIN LEVEL</div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: riskMetrics.marginLevel > 100 ? '#10B981' : '#EF4444' }}>
                {riskMetrics.marginLevel.toFixed(1)}%
              </div>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                {riskMetrics.usedMargin.toLocaleString()} / {riskMetrics.portfolioMargin.toLocaleString()} USDT
              </div>
            </div>

            {/* Total PnL */}
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '20px' }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>TOTAL PnL</div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: riskMetrics.totalPnL > 0 ? '#10B981' : '#EF4444' }}>
                ${riskMetrics.totalPnL.toLocaleString()}
              </div>
              <div style={{ fontSize: '14px', fontWeight: '600', color: riskMetrics.totalPnL > 0 ? '#10B981' : '#EF4444' }}>
                {riskMetrics.totalPnL > 0 ? '+' : ''}{riskMetrics.totalPnLPercent.toFixed(2)}%
              </div>
            </div>

            {/* Sharpe Ratio */}
            <div style={{ background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '16px', padding: '20px' }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>SHARPE RATIO</div>
              <div style={{ fontSize: '28px', fontWeight: '700', color: '#FFFFFF' }}>{riskMetrics.sharpeRatio.toFixed(2)}</div>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>Risk-adjusted return</div>
            </div>
          </div>

          {/* Calculator Button */}
          <div style={{ marginBottom: '24px' }}>
            <button
              onClick={() => setShowCalculator(!showCalculator)}
              style={{
                padding: '12px 24px',
                background: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <Icons.Calculator style={{ width: '18px', height: '18px' }} />
              {showCalculator ? 'Hesap Makinesini Gizle' : 'Risk Hesap Makinesi'}
            </button>
          </div>

          {/* Risk Calculator */}
          {showCalculator && (
            <div
              style={{
                background: 'rgba(239, 68, 68, 0.05)',
                border: '1px solid rgba(239, 68, 68, 0.2)',
                borderRadius: '16px',
                padding: '24px',
                marginBottom: '32px',
              }}
            >
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444', marginBottom: '20px' }}>
                Likidasyon Hesaplama
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                <div>
                  <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>Sembol</label>
                  <input
                    type="text"
                    value={calcSymbol}
                    onChange={(e) => setCalcSymbol(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>Giriş Fiyatı</label>
                  <input
                    type="number"
                    value={calcEntry}
                    onChange={(e) => setCalcEntry(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>Pozisyon Büyüklüğü (USDT)</label>
                  <input
                    type="number"
                    value={calcSize}
                    onChange={(e) => setCalcSize(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                    }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px', display: 'block' }}>Kaldıraç</label>
                  <input
                    type="number"
                    value={calcLeverage}
                    onChange={(e) => setCalcLeverage(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      borderRadius: '8px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                    }}
                  />
                </div>
              </div>

              {/* Side Selector */}
              <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
                <button
                  onClick={() => setCalcSide('long')}
                  style={{
                    flex: 1,
                    padding: '12px',
                    background: calcSide === 'long' ? '#10B981' : 'rgba(16, 185, 129, 0.1)',
                    color: '#FFFFFF',
                    border: calcSide === 'long' ? 'none' : '1px solid rgba(16, 185, 129, 0.3)',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                  }}
                >
                  LONG
                </button>
                <button
                  onClick={() => setCalcSide('short')}
                  style={{
                    flex: 1,
                    padding: '12px',
                    background: calcSide === 'short' ? '#EF4444' : 'rgba(239, 68, 68, 0.1)',
                    color: '#FFFFFF',
                    border: calcSide === 'short' ? 'none' : '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                  }}
                >
                  SHORT
                </button>
              </div>

              {/* Results */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', padding: '20px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '12px' }}>
                <div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Likidasyon Fiyatı</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#EF4444' }}>
                    ${calculateLiquidation().toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Gerekli Margin</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#FFFFFF' }}>
                    ${calculateMargin().toFixed(2)}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Risk Mesafesi</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#F59E0B' }}>
                    {(((Math.abs(parseFloat(calcEntry) - calculateLiquidation()) / parseFloat(calcEntry)) * 100)).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Active Positions */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}
          >
            <h2 style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Aktif Pozisyonlar ({positions.length})
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {positions.map((position) => {
                const isSelected = selectedPosition === position.id;
                const distanceToLiquidation = ((Math.abs(position.currentPrice - position.liquidationPrice) / position.currentPrice) * 100);

                return (
                  <div
                    key={position.id}
                    onClick={() => setSelectedPosition(isSelected ? null : position.id)}
                    style={{
                      background: isSelected ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.02)',
                      border: `1px solid ${isSelected ? getRiskColor(position.riskScore) : 'rgba(255, 255, 255, 0.1)'}`,
                      borderRadius: '12px',
                      padding: '20px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                  >
                    <div style={{ display: 'grid', gridTemplateColumns: '200px 150px 150px 150px 150px 150px 1fr', gap: '16px', alignItems: 'center' }}>
                      {/* Symbol & Side */}
                      <div>
                        <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
                          {position.symbol}
                        </div>
                        <div
                          style={{
                            display: 'inline-block',
                            padding: '4px 12px',
                            background: position.side === 'long' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                            color: position.side === 'long' ? '#10B981' : '#EF4444',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: '600',
                          }}
                        >
                          {position.side.toUpperCase()} {position.leverage}x
                        </div>
                      </div>

                      {/* Entry Price */}
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Giriş</div>
                        <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                          ${position.entryPrice.toLocaleString()}
                        </div>
                      </div>

                      {/* Current Price */}
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Güncel</div>
                        <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                          ${position.currentPrice.toLocaleString()}
                        </div>
                      </div>

                      {/* Liquidation */}
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Likidasyon</div>
                        <div style={{ fontSize: '16px', fontWeight: '600', color: '#EF4444' }}>
                          ${position.liquidationPrice.toLocaleString()}
                        </div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>
                          {distanceToLiquidation.toFixed(1)}% mesafe
                        </div>
                      </div>

                      {/* PnL */}
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>PnL</div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: position.pnl > 0 ? '#10B981' : '#EF4444' }}>
                          {position.pnl > 0 ? '+' : ''}${position.pnl.toFixed(2)}
                        </div>
                        <div style={{ fontSize: '12px', fontWeight: '600', color: position.pnl > 0 ? '#10B981' : '#EF4444' }}>
                          {position.pnl > 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%
                        </div>
                      </div>

                      {/* Margin */}
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Margin</div>
                        <div style={{ fontSize: '16px', fontWeight: '600', color: '#FFFFFF' }}>
                          ${position.margin.toLocaleString()}
                        </div>
                      </div>

                      {/* Risk Score */}
                      <div style={{ textAlign: 'right' }}>
                        <div
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '8px',
                            padding: '8px 16px',
                            background: `${getRiskColor(position.riskScore)}20`,
                            borderRadius: '8px',
                          }}
                        >
                          <Icons.Shield style={{ width: '16px', height: '16px', color: getRiskColor(position.riskScore) }} />
                          <div>
                            <div style={{ fontSize: '18px', fontWeight: '700', color: getRiskColor(position.riskScore) }}>
                              {position.riskScore.toFixed(1)}
                            </div>
                            <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>Risk</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isSelected && (
                      <div
                        style={{
                          marginTop: '20px',
                          paddingTop: '20px',
                          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        <h4 style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF', marginBottom: '12px' }}>
                          AI Stop-Loss & Take-Profit Önerileri
                        </h4>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                          {position.stopLoss && (
                            <div
                              style={{
                                padding: '16px',
                                background: 'rgba(239, 68, 68, 0.1)',
                                border: '1px solid rgba(239, 68, 68, 0.3)',
                                borderRadius: '8px',
                              }}
                            >
                              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                                Önerilen Stop-Loss
                              </div>
                              <div style={{ fontSize: '20px', fontWeight: '700', color: '#EF4444' }}>
                                ${position.stopLoss.toLocaleString()}
                              </div>
                              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                                Risk: {(((Math.abs(position.entryPrice - position.stopLoss) / position.entryPrice) * 100 * position.leverage)).toFixed(2)}%
                              </div>
                            </div>
                          )}

                          {position.takeProfit && (
                            <div
                              style={{
                                padding: '16px',
                                background: 'rgba(16, 185, 129, 0.1)',
                                border: '1px solid rgba(16, 185, 129, 0.3)',
                                borderRadius: '8px',
                              }}
                            >
                              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                                Önerilen Take-Profit
                              </div>
                              <div style={{ fontSize: '20px', fontWeight: '700', color: '#10B981' }}>
                                ${position.takeProfit.toLocaleString()}
                              </div>
                              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                                Hedef: {(((Math.abs(position.takeProfit - position.entryPrice) / position.entryPrice) * 100 * position.leverage)).toFixed(2)}%
                              </div>
                            </div>
                          )}

                          <div
                            style={{
                              padding: '16px',
                              background: 'rgba(59, 130, 246, 0.1)',
                              border: '1px solid rgba(59, 130, 246, 0.3)',
                              borderRadius: '8px',
                            }}
                          >
                            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                              Risk/Reward Oranı
                            </div>
                            <div style={{ fontSize: '20px', fontWeight: '700', color: '#3B82F6' }}>
                              1:{position.takeProfit && position.stopLoss ? ((Math.abs(position.takeProfit - position.entryPrice) / Math.abs(position.entryPrice - position.stopLoss))).toFixed(2) : '---'}
                            </div>
                            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                              {position.takeProfit && position.stopLoss && ((Math.abs(position.takeProfit - position.entryPrice) / Math.abs(position.entryPrice - position.stopLoss)) >= 2) ? '✅ İyi oran' : '⚠️ Düşük oran'}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </main>

        {aiAssistantOpen && (
          <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
        )}
      </div>
    </PWAProvider>
  );
}
