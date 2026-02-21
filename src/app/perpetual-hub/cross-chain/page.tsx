'use client';

/**
 * 🌉 CROSS-CHAIN PERPETUAL AGGREGATOR
 *
 * Compare and route perpetual trades across multiple blockchains
 *
 * Features:
 * - Multi-chain price comparison (Arbitrum, Optimism, Base, Polygon, etc.)
 * - Gas cost calculator for each chain
 * - Bridge time and cost analysis
 * - Best execution route finder
 * - Chain TVL and liquidity metrics
 * - Cross-chain arbitrage opportunities
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface ChainMarket {
  chain: string;
  protocol: string;
  price: number;
  liquidity: number;
  tvl: number;
  tradingFee: number; // percentage
  fundingRate: number; // percentage per 8h
  gasEstimate: number; // USD
  avgBlockTime: number; // seconds
  maxLeverage: number;
  spreadBps: number; // basis points
  volumeUSD24h: number;
}

interface BridgeRoute {
  fromChain: string;
  toChain: string;
  bridgeProvider: string;
  estimatedTime: number; // minutes
  bridgeFee: number; // USD
  security: 'HIGH' | 'MEDIUM' | 'LOW';
}

export default function CrossChainAggregator() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedPair, setSelectedPair] = useState('BTC/USDT');
  const [tradeSize, setTradeSize] = useState('10000');
  const [sortBy, setSortBy] = useState<'price' | 'total-cost' | 'liquidity'>('total-cost');

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock chain market data
  const chainMarkets: ChainMarket[] = [
    {
      chain: 'Arbitrum',
      protocol: 'GMX',
      price: 98234.50,
      liquidity: 45000000,
      tvl: 450000000,
      tradingFee: 0.07,
      fundingRate: 0.01,
      gasEstimate: 0.50,
      avgBlockTime: 0.25,
      maxLeverage: 50,
      spreadBps: 5,
      volumeUSD24h: 85000000,
    },
    {
      chain: 'Arbitrum',
      protocol: 'Vertex',
      price: 98240.20,
      liquidity: 12500000,
      tvl: 125000000,
      tradingFee: 0.05,
      fundingRate: 0.008,
      gasEstimate: 0.45,
      avgBlockTime: 0.25,
      maxLeverage: 25,
      spreadBps: 8,
      volumeUSD24h: 42000000,
    },
    {
      chain: 'Optimism',
      protocol: 'Synthetix',
      price: 98228.80,
      liquidity: 8500000,
      tvl: 95000000,
      tradingFee: 0.10,
      fundingRate: 0.012,
      gasEstimate: 0.35,
      avgBlockTime: 2,
      maxLeverage: 10,
      spreadBps: 12,
      volumeUSD24h: 28000000,
    },
    {
      chain: 'Base',
      protocol: 'Aevo',
      price: 98245.10,
      liquidity: 6200000,
      tvl: 58000000,
      tradingFee: 0.04,
      fundingRate: 0.015,
      gasEstimate: 0.15,
      avgBlockTime: 2,
      maxLeverage: 20,
      spreadBps: 15,
      volumeUSD24h: 18000000,
    },
    {
      chain: 'Polygon',
      protocol: 'Gains Network',
      price: 98251.30,
      liquidity: 6500000,
      tvl: 65000000,
      tradingFee: 0.08,
      fundingRate: 0.009,
      gasEstimate: 0.05,
      avgBlockTime: 2,
      maxLeverage: 150,
      spreadBps: 10,
      volumeUSD24h: 28000000,
    },
    {
      chain: 'dYdX Chain',
      protocol: 'dYdX v4',
      price: 98230.00,
      liquidity: 38000000,
      tvl: 380000000,
      tradingFee: 0.03,
      fundingRate: 0.007,
      gasEstimate: 0.02,
      avgBlockTime: 1,
      maxLeverage: 20,
      spreadBps: 3,
      volumeUSD24h: 950000000,
    },
  ];

  // Bridge routes
  const bridgeRoutes: BridgeRoute[] = [
    { fromChain: 'Ethereum', toChain: 'Arbitrum', bridgeProvider: 'Arbitrum Bridge', estimatedTime: 10, bridgeFee: 5, security: 'HIGH' },
    { fromChain: 'Ethereum', toChain: 'Optimism', bridgeProvider: 'Optimism Bridge', estimatedTime: 10, bridgeFee: 4, security: 'HIGH' },
    { fromChain: 'Ethereum', toChain: 'Base', bridgeProvider: 'Base Bridge', estimatedTime: 10, bridgeFee: 3, security: 'HIGH' },
    { fromChain: 'Ethereum', toChain: 'Polygon', bridgeProvider: 'Polygon PoS Bridge', estimatedTime: 30, bridgeFee: 8, security: 'MEDIUM' },
    { fromChain: 'Arbitrum', toChain: 'Optimism', bridgeProvider: 'Stargate', estimatedTime: 15, bridgeFee: 6, security: 'MEDIUM' },
    { fromChain: 'Arbitrum', toChain: 'Base', bridgeProvider: 'Across Protocol', estimatedTime: 5, bridgeFee: 3, security: 'MEDIUM' },
  ];

  // Calculate total cost including slippage
  const calculateTotalCost = (market: ChainMarket): number => {
    const tradeSizeNum = parseFloat(tradeSize);
    const slippage = (tradeSizeNum / market.liquidity) * 100 * 0.5; // Simplified slippage model
    const tradingFeeCost = (tradeSizeNum * market.tradingFee) / 100;
    const spreadCost = (tradeSizeNum * market.spreadBps) / 10000;
    return market.gasEstimate + tradingFeeCost + spreadCost + slippage;
  };

  // Sort markets
  const sortedMarkets = [...chainMarkets].sort((a, b) => {
    if (sortBy === 'price') return a.price - b.price;
    if (sortBy === 'total-cost') return calculateTotalCost(a) - calculateTotalCost(b);
    if (sortBy === 'liquidity') return b.liquidity - a.liquidity;
    return 0;
  });

  const bestMarket = sortedMarkets[0];

  // Detect arbitrage opportunities
  const arbitrageOpportunities = chainMarkets.map((market) => {
    const otherMarkets = chainMarkets.filter(m => m.chain !== market.chain);
    const maxPriceOther = Math.max(...otherMarkets.map(m => m.price));
    const minPriceOther = Math.min(...otherMarkets.map(m => m.price));

    const buyHereSellThere = maxPriceOther - market.price;
    const sellHereBuyThere = market.price - minPriceOther;

    return {
      market,
      arbitrageSpread: Math.max(buyHereSellThere, sellHereBuyThere),
      direction: buyHereSellThere > sellHereBuyThere ? 'BUY_HERE' : 'SELL_HERE',
    };
  }).filter(opp => opp.arbitrageSpread > 10) // Only show if spread > $10
    .sort((a, b) => b.arbitrageSpread - a.arbitrageSpread);

  const getChainColor = (chain: string): string => {
    const colors: Record<string, string> = {
      'Arbitrum': '#28A0F0',
      'Optimism': '#FF0420',
      'Base': '#0052FF',
      'Polygon': '#8247E5',
      'dYdX Chain': '#6966FF',
    };
    return colors[chain] || '#FFFFFF';
  };

  const getSecurityColor = (security: string): string => {
    if (security === 'HIGH') return '#10B981';
    if (security === 'MEDIUM') return '#F59E0B';
    return '#EF4444';
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

        <main style={{ maxWidth: '1800px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
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
                background: 'linear-gradient(135deg, #A855F7 0%, #7C3AED 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Cross-Chain Perpetual Aggregator
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Multi-blockchain perpetual trading comparison and best route finding
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', flexWrap: 'wrap' }}>
            <select
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
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
              <option value="BTC/USDT">BTC/USDT</option>
              <option value="ETH/USDT">ETH/USDT</option>
            </select>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Trade Size (USD):
              </label>
              <input
                type="number"
                value={tradeSize}
                onChange={(e) => setTradeSize(e.target.value)}
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

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
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
              <option value="total-cost">Sort: Total Cost</option>
              <option value="price">Sort: Price</option>
              <option value="liquidity">Sort: Liquidity</option>
            </select>
          </div>

          {/* Best Route Recommendation */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%)',
            border: '2px solid #A855F7',
            borderRadius: '20px',
            padding: '32px',
            marginBottom: '32px',
          }}>
            <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px' }}>
              ✨ BEST ROUTE RECOMMENDATION
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '24px' }}>
              <div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  Chain
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: getChainColor(bestMarket.chain) }} />
                  <div style={{ fontSize: '20px', fontWeight: '900', color: '#FFFFFF' }}>
                    {bestMarket.chain}
                  </div>
                </div>
                <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)', marginTop: '4px' }}>
                  Protocol: {bestMarket.protocol}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  Price
                </div>
                <div style={{ fontSize: '28px', fontWeight: '900', color: '#10B981' }}>
                  ${bestMarket.price.toFixed(2)}
                </div>
                <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
                  Spread: {bestMarket.spreadBps} bps
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  Total Cost
                </div>
                <div style={{ fontSize: '28px', fontWeight: '900', color: '#F59E0B' }}>
                  ${calculateTotalCost(bestMarket).toFixed(2)}
                </div>
                <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
                  Fee + Gas + Slippage
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                  Liquidity
                </div>
                <div style={{ fontSize: '28px', fontWeight: '900', color: '#3B82F6' }}>
                  ${(bestMarket.liquidity / 1000000).toFixed(1)}M
                </div>
                <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
                  Max Leverage: {bestMarket.maxLeverage}x
                </div>
              </div>
            </div>
          </div>

          {/* Chain Markets Table */}
          <div style={{
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '32px',
          }}>
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Chain Comparison
            </h3>

            {/* Table Header */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '150px 120px 100px 100px 100px 100px 120px 100px',
              gap: '12px',
              padding: '12px 16px',
              background: 'rgba(0, 0, 0, 0.3)',
              borderRadius: '8px',
              marginBottom: '12px',
              fontSize: '11px',
              fontWeight: '600',
              color: 'rgba(255, 255, 255, 0.5)',
            }}>
              <div>CHAIN / PROTOCOL</div>
              <div>PRICE</div>
              <div>SPREAD</div>
              <div>TRADING FEE</div>
              <div>GAS</div>
              <div>SLIPPAGE</div>
              <div>TOTAL COST</div>
              <div>LIQUIDITY</div>
            </div>

            {/* Table Rows */}
            {sortedMarkets.map((market, index) => {
              const totalCost = calculateTotalCost(market);
              const isBest = market === bestMarket;

              return (
                <div
                  key={index}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '150px 120px 100px 100px 100px 100px 120px 100px',
                    gap: '12px',
                    padding: '16px',
                    background: isBest ? 'rgba(168, 85, 247, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                    border: isBest ? '1px solid #A855F7' : '1px solid rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    marginBottom: '8px',
                    alignItems: 'center',
                  }}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: getChainColor(market.chain) }} />
                      <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                        {market.chain}
                      </div>
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                      {market.protocol}
                    </div>
                  </div>

                  <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                    ${market.price.toFixed(2)}
                  </div>

                  <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    {market.spreadBps} bps
                  </div>

                  <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    {market.tradingFee}%
                  </div>

                  <div style={{ fontSize: '13px', fontWeight: '600', color: market.gasEstimate < 0.3 ? '#10B981' : '#F59E0B' }}>
                    ${market.gasEstimate}
                  </div>

                  <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    ~${((parseFloat(tradeSize) / market.liquidity) * 100 * 0.5).toFixed(2)}
                  </div>

                  <div style={{ fontSize: '16px', fontWeight: '700', color: isBest ? '#10B981' : '#F59E0B' }}>
                    ${totalCost.toFixed(2)}
                    {isBest && <span style={{ marginLeft: '8px', fontSize: '12px' }}>✨</span>}
                  </div>

                  <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    ${(market.liquidity / 1000000).toFixed(1)}M
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
            {/* Arbitrage Opportunities */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                💰 Cross-Chain Arbitrage Opportunities
              </h3>

              {arbitrageOpportunities.length === 0 && (
                <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255, 255, 255, 0.5)' }}>
                  No arbitrage opportunities detected at this time
                </div>
              )}

              {arbitrageOpportunities.map((opp, index) => (
                <div
                  key={index}
                  style={{
                    padding: '16px',
                    background: 'rgba(16, 185, 129, 0.1)',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    borderRadius: '12px',
                    marginBottom: '12px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: getChainColor(opp.market.chain) }} />
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                          {opp.market.chain} - {opp.market.protocol}
                        </div>
                      </div>
                      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>
                        {opp.direction === 'BUY_HERE' ? '🟢 Buy here, sell elsewhere' : '🔴 Sell here, buy elsewhere'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '24px', fontWeight: '900', color: '#10B981' }}>
                        ${opp.arbitrageSpread.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        Potential Profit
                      </div>
                    </div>
                  </div>

                  <div style={{ padding: '10px', background: 'rgba(0, 0, 0, 0.2)', borderRadius: '6px', fontSize: '11px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    ⚠️ Bridge fees and gas costs must be calculated. Risk: Cross-chain delay time.
                  </div>
                </div>
              ))}
            </div>

            {/* Bridge Routes */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                🌉 Bridge Routes
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {bridgeRoutes.map((route, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '16px',
                      background: 'rgba(255, 255, 255, 0.02)',
                      border: '1px solid rgba(255, 255, 255, 0.05)',
                      borderRadius: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                        {route.fromChain} → {route.toChain}
                      </div>
                      <div style={{
                        padding: '4px 10px',
                        background: `${getSecurityColor(route.security)}20`,
                        border: `1px solid ${getSecurityColor(route.security)}`,
                        borderRadius: '6px',
                        fontSize: '11px',
                        fontWeight: '600',
                        color: getSecurityColor(route.security),
                      }}>
                        {route.security} SECURITY
                      </div>
                    </div>

                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px' }}>
                      via {route.bridgeProvider}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Estimated Time
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#3B82F6' }}>
                          ~{route.estimatedTime} min
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          Bridge Fee
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: '#F59E0B' }}>
                          ${route.bridgeFee}
                        </div>
                      </div>
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
