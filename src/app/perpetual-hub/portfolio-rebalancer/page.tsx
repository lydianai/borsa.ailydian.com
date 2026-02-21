'use client';

/**
 * ⚖️ DYNAMIC PORTFOLIO REBALANCER
 *
 * AI-powered portfolio optimization and automatic rebalancing
 *
 * Features:
 * - Risk parity allocation
 * - Correlation-based rebalancing
 * - Black Swan hedging strategies
 * - Dynamic position sizing
 * - Multi-asset portfolio optimization
 * - Rebalancing triggers and alerts
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface PortfolioAsset {
  symbol: string;
  currentAllocation: number; // percentage
  targetAllocation: number; // percentage
  currentValue: number; // USD
  targetValue: number; // USD
  rebalanceAction: 'BUY' | 'SELL' | 'HOLD';
  rebalanceAmount: number; // USD
  volatility: number; // annualized
  sharpeRatio: number;
  correlation: number; // to portfolio
}

interface RebalancingStrategy {
  name: string;
  description: string;
  assets: {
    symbol: string;
    weight: number;
  }[];
  expectedReturn: number;
  expectedVolatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

export default function PortfolioRebalancer() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [totalPortfolioValue, setTotalPortfolioValue] = useState('100000');
  const [rebalanceThreshold, setRebalanceThreshold] = useState('5');
  const [selectedStrategy, setSelectedStrategy] = useState<string>('risk-parity');

  useEffect(() => {
    setMounted(true);
  }, []);

  const portfolioValue = parseFloat(totalPortfolioValue);

  // Mock current portfolio
  const currentPortfolio: PortfolioAsset[] = [
    {
      symbol: 'BTCUSDT',
      currentAllocation: 45,
      targetAllocation: 40,
      currentValue: 45000,
      targetValue: 40000,
      rebalanceAction: 'SELL',
      rebalanceAmount: 5000,
      volatility: 0.65,
      sharpeRatio: 1.2,
      correlation: 0.85,
    },
    {
      symbol: 'ETHUSDT',
      currentAllocation: 28,
      targetAllocation: 30,
      currentValue: 28000,
      targetValue: 30000,
      rebalanceAction: 'BUY',
      rebalanceAmount: 2000,
      volatility: 0.72,
      sharpeRatio: 1.1,
      correlation: 0.78,
    },
    {
      symbol: 'SOLUSDT',
      currentAllocation: 12,
      targetAllocation: 15,
      currentValue: 12000,
      targetValue: 15000,
      rebalanceAction: 'BUY',
      rebalanceAmount: 3000,
      volatility: 0.85,
      sharpeRatio: 0.9,
      correlation: 0.65,
    },
    {
      symbol: 'BNBUSDT',
      currentAllocation: 10,
      targetAllocation: 10,
      currentValue: 10000,
      targetValue: 10000,
      rebalanceAction: 'HOLD',
      rebalanceAmount: 0,
      volatility: 0.58,
      sharpeRatio: 1.0,
      correlation: 0.72,
    },
    {
      symbol: 'ADAUSDT',
      currentAllocation: 5,
      targetAllocation: 5,
      currentValue: 5000,
      targetValue: 5000,
      rebalanceAction: 'HOLD',
      rebalanceAmount: 0,
      volatility: 0.95,
      sharpeRatio: 0.7,
      correlation: 0.55,
    },
  ];

  // Rebalancing strategies
  const strategies: RebalancingStrategy[] = [
    {
      name: 'Risk Parity',
      description: 'Equal risk contribution from each asset based on volatility',
      assets: [
        { symbol: 'BTCUSDT', weight: 35 },
        { symbol: 'ETHUSDT', weight: 30 },
        { symbol: 'SOLUSDT', weight: 20 },
        { symbol: 'BNBUSDT', weight: 10 },
        { symbol: 'ADAUSDT', weight: 5 },
      ],
      expectedReturn: 45.2,
      expectedVolatility: 38.5,
      sharpeRatio: 1.17,
      maxDrawdown: 32.5,
    },
    {
      name: 'Max Sharpe',
      description: 'Optimize for maximum risk-adjusted returns (Sharpe Ratio)',
      assets: [
        { symbol: 'BTCUSDT', weight: 50 },
        { symbol: 'ETHUSDT', weight: 35 },
        { symbol: 'BNBUSDT', weight: 15 },
      ],
      expectedReturn: 52.8,
      expectedVolatility: 42.3,
      sharpeRatio: 1.25,
      maxDrawdown: 38.2,
    },
    {
      name: 'Min Variance',
      description: 'Minimize portfolio volatility for stable returns',
      assets: [
        { symbol: 'BTCUSDT', weight: 40 },
        { symbol: 'BNBUSDT', weight: 35 },
        { symbol: 'ETHUSDT', weight: 25 },
      ],
      expectedReturn: 38.5,
      expectedVolatility: 32.1,
      sharpeRatio: 1.20,
      maxDrawdown: 28.5,
    },
    {
      name: 'Black Swan Hedge',
      description: 'Diversified with low-correlation assets for tail risk protection',
      assets: [
        { symbol: 'BTCUSDT', weight: 30 },
        { symbol: 'ETHUSDT', weight: 25 },
        { symbol: 'SOLUSDT', weight: 20 },
        { symbol: 'BNBUSDT', weight: 15 },
        { symbol: 'ADAUSDT', weight: 10 },
      ],
      expectedReturn: 42.0,
      expectedVolatility: 36.8,
      sharpeRatio: 1.14,
      maxDrawdown: 30.5,
    },
  ];

  const currentStrategy = strategies.find(s => s.name.toLowerCase().replace(' ', '-') === selectedStrategy) || strategies[0];

  // Calculate portfolio metrics
  const portfolioVolatility = currentPortfolio.reduce((sum, asset) =>
    sum + Math.pow(asset.volatility * asset.currentAllocation / 100, 2), 0
  );
  const portfolioSharpe = currentPortfolio.reduce((sum, asset) =>
    sum + (asset.sharpeRatio * asset.currentAllocation / 100), 0
  );

  // Rebalancing needed
  const needsRebalancing = currentPortfolio.some(asset =>
    Math.abs(asset.currentAllocation - asset.targetAllocation) > parseFloat(rebalanceThreshold)
  );

  const totalRebalanceAmount = currentPortfolio.reduce((sum, asset) =>
    sum + Math.abs(asset.rebalanceAmount), 0
  );

  if (!mounted) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.6)' }}>Loading...</div>
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
          paddingTop: '80px',
        }}
      >
        <SharedSidebar
          currentPage="perpetual-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1600px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
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
              Back to Perpetual Hub
            </Link>

            <h1
              style={{
                fontSize: '40px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #EC4899 0%, #BE185D 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Dynamic Portfolio Rebalancer
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              AI-powered portfolio optimization and automatic rebalancing
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Portfolio Value (USD):
              </label>
              <input
                type="number"
                value={totalPortfolioValue}
                onChange={(e) => setTotalPortfolioValue(e.target.value)}
                style={{
                  width: '150px',
                  padding: '12px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Rebalance Threshold (%):
              </label>
              <input
                type="number"
                value={rebalanceThreshold}
                onChange={(e) => setRebalanceThreshold(e.target.value)}
                style={{
                  width: '80px',
                  padding: '12px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '8px',
                  color: '#FFFFFF',
                  fontSize: '14px',
                }}
              />
            </div>

            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              style={{
                padding: '12px 16px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '14px',
                fontWeight: '600',
              }}
            >
              <option value="risk-parity">Risk Parity</option>
              <option value="max-sharpe">Max Sharpe</option>
              <option value="min-variance">Min Variance</option>
              <option value="black-swan-hedge">Black Swan Hedge</option>
            </select>
          </div>

          {/* Rebalancing Alert */}
          {needsRebalancing && (
            <div style={{
              padding: '20px',
              background: 'rgba(236, 72, 153, 0.1)',
              border: '2px solid #EC4899',
              borderRadius: '12px',
              marginBottom: '32px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <div>
                <div style={{ fontSize: '16px', fontWeight: '700', color: '#EC4899', marginBottom: '8px' }}>
                  ⚠️ Rebalancing Recommended
                </div>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                  Your portfolio has drifted from target allocations. Total rebalance: ${totalRebalanceAmount.toFixed(2)}
                </div>
              </div>
              <button style={{
                padding: '12px 24px',
                background: '#EC4899',
                border: 'none',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '14px',
                fontWeight: '700',
                cursor: 'pointer',
              }}>
                Execute Rebalance
              </button>
            </div>
          )}

          {/* Portfolio Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Portfolio Value
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#FFFFFF' }}>
                ${portfolioValue.toLocaleString()}
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Portfolio Volatility
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#F59E0B' }}>
                {(Math.sqrt(portfolioVolatility) * 100).toFixed(1)}%
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Sharpe Ratio
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#10B981' }}>
                {portfolioSharpe.toFixed(2)}
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Assets
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#3B82F6' }}>
                {currentPortfolio.length}
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '32px' }}>
            {/* Current Portfolio */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                Current Portfolio & Rebalancing Actions
              </h3>

              {/* Table Header */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '120px 100px 100px 100px 100px',
                gap: '12px',
                padding: '12px 16px',
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                marginBottom: '12px',
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(255, 255, 255, 0.5)',
              }}>
                <div>ASSET</div>
                <div>CURRENT %</div>
                <div>TARGET %</div>
                <div>ACTION</div>
                <div>AMOUNT</div>
              </div>

              {/* Portfolio Rows */}
              {currentPortfolio.map((asset, index) => {
                const needsAction = Math.abs(asset.currentAllocation - asset.targetAllocation) > parseFloat(rebalanceThreshold);

                return (
                  <div
                    key={index}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '120px 100px 100px 100px 100px',
                      gap: '12px',
                      padding: '16px',
                      background: needsAction ? 'rgba(236, 72, 153, 0.05)' : 'rgba(255, 255, 255, 0.02)',
                      border: needsAction ? '1px solid rgba(236, 72, 153, 0.3)' : '1px solid rgba(255, 255, 255, 0.05)',
                      borderRadius: '12px',
                      marginBottom: '8px',
                      alignItems: 'center',
                    }}
                  >
                    <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                      {asset.symbol.replace('USDT', '')}
                    </div>

                    <div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                        {asset.currentAllocation}%
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        ${asset.currentValue.toLocaleString()}
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#3B82F6' }}>
                        {asset.targetAllocation}%
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        ${asset.targetValue.toLocaleString()}
                      </div>
                    </div>

                    <div>
                      <div style={{
                        padding: '6px 12px',
                        background: asset.rebalanceAction === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : asset.rebalanceAction === 'SELL' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                        border: `1px solid ${asset.rebalanceAction === 'BUY' ? '#10B981' : asset.rebalanceAction === 'SELL' ? '#EF4444' : 'rgba(255, 255, 255, 0.2)'}`,
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '700',
                        color: asset.rebalanceAction === 'BUY' ? '#10B981' : asset.rebalanceAction === 'SELL' ? '#EF4444' : 'rgba(255, 255, 255, 0.7)',
                        textAlign: 'center',
                      }}>
                        {asset.rebalanceAction}
                      </div>
                    </div>

                    <div style={{ fontSize: '14px', fontWeight: '700', color: asset.rebalanceAction === 'HOLD' ? 'rgba(255, 255, 255, 0.5)' : '#FFFFFF' }}>
                      {asset.rebalanceAmount > 0 ? `$${asset.rebalanceAmount.toLocaleString()}` : '-'}
                    </div>
                  </div>
                );
              })}

              {/* Allocation Chart */}
              <div style={{ marginTop: '32px' }}>
                <h4 style={{ fontSize: '14px', fontWeight: '700', color: 'rgba(255, 255, 255, 0.8)', marginBottom: '16px' }}>
                  Asset Allocation
                </h4>
                <div style={{ display: 'flex', height: '40px', borderRadius: '8px', overflow: 'hidden' }}>
                  {currentPortfolio.map((asset, index) => {
                    const colors = ['#10B981', '#3B82F6', '#F59E0B', '#EC4899', '#8B5CF6'];
                    return (
                      <div
                        key={index}
                        style={{
                          width: `${asset.currentAllocation}%`,
                          background: colors[index % colors.length],
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#FFFFFF',
                          fontSize: '12px',
                          fontWeight: '700',
                        }}
                      >
                        {asset.currentAllocation > 8 && `${asset.currentAllocation}%`}
                      </div>
                    );
                  })}
                </div>
                <div style={{ display: 'flex', gap: '16px', marginTop: '12px', flexWrap: 'wrap' }}>
                  {currentPortfolio.map((asset, index) => {
                    const colors = ['#10B981', '#3B82F6', '#F59E0B', '#EC4899', '#8B5CF6'];
                    return (
                      <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <div style={{ width: '12px', height: '12px', borderRadius: '2px', background: colors[index % colors.length] }} />
                        <span style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)' }}>
                          {asset.symbol.replace('USDT', '')}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right Column - Strategies */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Current Strategy */}
              <div style={{
                background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(190, 24, 93, 0.05) 100%)',
                border: '2px solid #EC4899',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                  SELECTED STRATEGY
                </div>
                <h3 style={{ fontSize: '20px', fontWeight: '900', color: '#FFFFFF', marginBottom: '8px' }}>
                  {currentStrategy.name}
                </h3>
                <p style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '16px', lineHeight: '1.5' }}>
                  {currentStrategy.description}
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                      Expected Return
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: '#10B981' }}>
                      {currentStrategy.expectedReturn.toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                      Volatility
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: '#F59E0B' }}>
                      {currentStrategy.expectedVolatility.toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                      Sharpe Ratio
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: '#3B82F6' }}>
                      {currentStrategy.sharpeRatio.toFixed(2)}
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                      Max Drawdown
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: '#EF4444' }}>
                      -{currentStrategy.maxDrawdown.toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)', marginBottom: '8px' }}>
                  Target Weights:
                </div>
                {currentStrategy.assets.map((asset, index) => (
                  <div
                    key={index}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '8px',
                      background: 'rgba(0, 0, 0, 0.2)',
                      borderRadius: '6px',
                      marginBottom: '6px',
                    }}
                  >
                    <span style={{ fontSize: '13px', color: '#FFFFFF' }}>
                      {asset.symbol.replace('USDT', '')}
                    </span>
                    <span style={{ fontSize: '13px', fontWeight: '700', color: '#EC4899' }}>
                      {asset.weight}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Other Strategies */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  Compare Strategies
                </h3>

                {strategies.filter(s => s.name !== currentStrategy.name).map((strategy, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '16px',
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid rgba(255, 255, 255, 0.05)',
                      borderRadius: '8px',
                      marginBottom: '12px',
                      cursor: 'pointer',
                    }}
                    onClick={() => setSelectedStrategy(strategy.name.toLowerCase().replace(' ', '-'))}
                  >
                    <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF', marginBottom: '8px' }}>
                      {strategy.name}
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                      <span>Return: {strategy.expectedReturn.toFixed(1)}%</span>
                      <span>Sharpe: {strategy.sharpeRatio.toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>
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
