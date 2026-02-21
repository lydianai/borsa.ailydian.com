'use client';

/**
 * 🐋 WHALE TRACKER & ALERTS
 *
 * Real-time whale position tracking and smart alerts
 *
 * Features:
 * - Large position monitoring (>$1M)
 * - Wallet clustering analysis
 * - Historical whale win rate tracking
 * - Real-time transaction alerts
 * - Whale vs Retail sentiment comparison
 * - Copy trading signals
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface WhalePosition {
  walletAddress: string;
  walletLabel?: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  size: number; // USD value
  leverage: number;
  pnl: number;
  pnlPercent: number;
  openTime: number;
  liquidationPrice: number;
  exchange: string;
}

interface WhaleTransaction {
  timestamp: number;
  walletAddress: string;
  walletLabel?: string;
  action: 'OPEN_LONG' | 'OPEN_SHORT' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'INCREASE' | 'DECREASE';
  symbol: string;
  size: number;
  price: number;
  txHash: string;
}

interface WhaleStats {
  walletAddress: string;
  walletLabel?: string;
  totalTrades: number;
  winRate: number;
  avgPnL: number;
  totalVolume: number;
  activePositions: number;
  largestWin: number;
  largestLoss: number;
  lastActivityTime: number;
}

export default function WhaleTracker() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('ALL');
  const [minPositionSize, setMinPositionSize] = useState('1000000');
  const [alertsEnabled, setAlertsEnabled] = useState(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock whale positions
  const whalePositions: WhalePosition[] = [
    {
      walletAddress: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
      walletLabel: 'Whale #1 (Known Trader)',
      symbol: 'BTCUSDT',
      side: 'LONG',
      entryPrice: 97850,
      currentPrice: 98234.5,
      size: 5800000,
      leverage: 10,
      pnl: 228500,
      pnlPercent: 3.94,
      openTime: Date.now() - 3600000 * 12,
      liquidationPrice: 88965,
      exchange: 'Binance',
    },
    {
      walletAddress: '0x8A9C67fee641579dEbA04928c4BC45F66e26343A',
      walletLabel: 'Alameda Research (Historical)',
      symbol: 'ETHUSDT',
      side: 'SHORT',
      entryPrice: 3480,
      currentPrice: 3456.78,
      size: 2400000,
      leverage: 5,
      pnl: 158400,
      pnlPercent: 6.6,
      openTime: Date.now() - 3600000 * 8,
      liquidationPrice: 4176,
      exchange: 'dYdX',
    },
    {
      walletAddress: '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
      walletLabel: 'Whale #3',
      symbol: 'BTCUSDT',
      side: 'SHORT',
      entryPrice: 98500,
      currentPrice: 98234.5,
      size: 3200000,
      leverage: 20,
      pnl: -171200,
      pnlPercent: -5.35,
      openTime: Date.now() - 3600000 * 4,
      liquidationPrice: 103425,
      exchange: 'Bybit',
    },
    {
      walletAddress: '0x1a9C8182C09F50C8318d769245beA52c32BE35BC',
      symbol: 'SOLUSDT',
      side: 'LONG',
      entryPrice: 242.5,
      currentPrice: 248.3,
      size: 1850000,
      leverage: 15,
      pnl: 98560,
      pnlPercent: 5.33,
      openTime: Date.now() - 3600000 * 2,
      liquidationPrice: 226.03,
      exchange: 'OKX',
    },
  ];

  // Mock recent transactions
  const recentTransactions: WhaleTransaction[] = [
    {
      timestamp: Date.now() - 120000,
      walletAddress: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
      walletLabel: 'Whale #1',
      action: 'INCREASE',
      symbol: 'BTCUSDT',
      size: 1200000,
      price: 98210,
      txHash: '0xabc123...',
    },
    {
      timestamp: Date.now() - 480000,
      walletAddress: '0x8A9C67fee641579dEbA04928c4BC45F66e26343A',
      walletLabel: 'Alameda Research',
      action: 'OPEN_SHORT',
      symbol: 'ETHUSDT',
      size: 2400000,
      price: 3480,
      txHash: '0xdef456...',
    },
    {
      timestamp: Date.now() - 720000,
      walletAddress: '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
      walletLabel: 'Whale #3',
      action: 'OPEN_SHORT',
      symbol: 'BTCUSDT',
      size: 3200000,
      price: 98500,
      txHash: '0xghi789...',
    },
  ];

  // Mock whale stats
  const whaleStats: WhaleStats[] = [
    {
      walletAddress: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
      walletLabel: 'Whale #1',
      totalTrades: 128,
      winRate: 68.5,
      avgPnL: 12500,
      totalVolume: 45000000,
      activePositions: 3,
      largestWin: 285000,
      largestLoss: -125000,
      lastActivityTime: Date.now() - 120000,
    },
    {
      walletAddress: '0x8A9C67fee641579dEbA04928c4BC45F66e26343A',
      walletLabel: 'Alameda Research',
      totalTrades: 342,
      winRate: 71.2,
      avgPnL: 18200,
      totalVolume: 180000000,
      activePositions: 5,
      largestWin: 950000,
      largestLoss: -420000,
      lastActivityTime: Date.now() - 480000,
    },
    {
      walletAddress: '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
      walletLabel: 'Whale #3',
      totalTrades: 95,
      winRate: 54.7,
      avgPnL: 8400,
      totalVolume: 28000000,
      activePositions: 2,
      largestWin: 180000,
      largestLoss: -95000,
      lastActivityTime: Date.now() - 720000,
    },
  ];

  const filteredPositions = whalePositions.filter((pos) => {
    if (selectedSymbol !== 'ALL' && pos.symbol !== selectedSymbol) return false;
    if (pos.size < parseFloat(minPositionSize)) return false;
    return true;
  });

  const formatAddress = (address: string): string => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatTime = (timestamp: number): string => {
    const now = Date.now();
    const diff = now - timestamp;
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);

    if (hours > 0) return `${hours}h ${minutes}m ago`;
    return `${minutes}m ago`;
  };

  const getActionColor = (action: string): string => {
    if (action.includes('LONG')) return '#10B981';
    if (action.includes('SHORT')) return '#EF4444';
    return '#F59E0B';
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
                background: 'linear-gradient(135deg, #14B8A6 0%, #0D9488 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Whale Tracker & Alerts
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Large position tracking and real-time whale transaction alerts
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', flexWrap: 'wrap', alignItems: 'center' }}>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
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
              <option value="ALL">All Symbols</option>
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
              <option value="SOLUSDT">SOL/USDT</option>
            </select>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Min Position Size:
              </label>
              <input
                type="number"
                value={minPositionSize}
                onChange={(e) => setMinPositionSize(e.target.value)}
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

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={alertsEnabled}
                onChange={(e) => setAlertsEnabled(e.target.checked)}
                style={{ width: '18px', height: '18px' }}
              />
              <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                🔔 Real-time Alerts
              </span>
            </label>

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10B981', animation: 'pulse 2s infinite' }} />
              <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>TRACKING</div>
            </div>
          </div>

          {/* Summary Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                🐋 Active Whale Positions
              </div>
              <div style={{ fontSize: '36px', fontWeight: '900', color: '#14B8A6' }}>
                {filteredPositions.length}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Total Value: ${(filteredPositions.reduce((sum, p) => sum + p.size, 0) / 1000000).toFixed(1)}M
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                📊 Long vs Short
              </div>
              <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '24px', fontWeight: '900', color: '#10B981' }}>
                    {filteredPositions.filter(p => p.side === 'LONG').length}
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>LONG</div>
                </div>
                <div style={{ fontSize: '20px', color: 'rgba(255, 255, 255, 0.3)' }}>vs</div>
                <div>
                  <div style={{ fontSize: '24px', fontWeight: '900', color: '#EF4444' }}>
                    {filteredPositions.filter(p => p.side === 'SHORT').length}
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>SHORT</div>
                </div>
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                💰 Total PnL
              </div>
              <div style={{ fontSize: '36px', fontWeight: '900', color: filteredPositions.reduce((sum, p) => sum + p.pnl, 0) >= 0 ? '#10B981' : '#EF4444' }}>
                {filteredPositions.reduce((sum, p) => sum + p.pnl, 0) >= 0 ? '+' : ''}
                ${(filteredPositions.reduce((sum, p) => sum + p.pnl, 0) / 1000).toFixed(1)}k
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Unrealized P&L
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                📈 Avg Win Rate
              </div>
              <div style={{ fontSize: '36px', fontWeight: '900', color: '#3B82F6' }}>
                {(whaleStats.reduce((sum, s) => sum + s.winRate, 0) / whaleStats.length).toFixed(1)}%
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                From {whaleStats.length} tracked whales
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '32px' }}>
            {/* Active Whale Positions */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                🐋 Active Whale Positions
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {filteredPositions.map((position, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '20px',
                      background: position.side === 'LONG' ? 'rgba(16, 185, 129, 0.05)' : 'rgba(239, 68, 68, 0.05)',
                      border: `2px solid ${position.side === 'LONG' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                      borderRadius: '12px',
                    }}
                  >
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '16px' }}>
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          {formatAddress(position.walletAddress)}
                          {position.walletLabel && ` - ${position.walletLabel}`}
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ fontSize: '20px', fontWeight: '900', color: '#FFFFFF' }}>
                            {position.symbol}
                          </div>
                          <div style={{
                            padding: '4px 10px',
                            background: position.side === 'LONG' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: '700',
                            color: position.side === 'LONG' ? '#10B981' : '#EF4444',
                          }}>
                            {position.side}
                          </div>
                          <div style={{
                            padding: '4px 10px',
                            background: 'rgba(255, 255, 255, 0.1)',
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '600',
                            color: 'rgba(255, 255, 255, 0.7)',
                          }}>
                            {position.leverage}x
                          </div>
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '24px', fontWeight: '900', color: position.pnl >= 0 ? '#10B981' : '#EF4444' }}>
                          {position.pnl >= 0 ? '+' : ''}${(position.pnl / 1000).toFixed(1)}k
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: position.pnl >= 0 ? '#10B981' : '#EF4444' }}>
                          ({position.pnl >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%)
                        </div>
                      </div>
                    </div>

                    {/* Position Details */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Position Size</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          ${(position.size / 1000000).toFixed(2)}M
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Entry Price</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          ${position.entryPrice.toFixed(2)}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Current Price</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          ${position.currentPrice.toFixed(2)}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Liquidation</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444' }}>
                          ${position.liquidationPrice.toFixed(2)}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Exchange</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          {position.exchange}
                        </div>
                      </div>

                      <div>
                        <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Open Time</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          {formatTime(position.openTime)}
                        </div>
                      </div>
                    </div>

                    {/* Copy Trading Button */}
                    <button style={{
                      width: '100%',
                      padding: '12px',
                      background: 'rgba(20, 184, 166, 0.2)',
                      border: '1px solid #14B8A6',
                      borderRadius: '8px',
                      color: '#14B8A6',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                    }}>
                      📋 Copy This Position
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Recent Whale Transactions */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  🔔 Recent Transactions
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {recentTransactions.map((tx, index) => (
                    <div
                      key={index}
                      style={{
                        padding: '16px',
                        background: 'rgba(255, 255, 255, 0.02)',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                        <div style={{
                          padding: '4px 10px',
                          background: `${getActionColor(tx.action)}20`,
                          border: `1px solid ${getActionColor(tx.action)}`,
                          borderRadius: '6px',
                          fontSize: '11px',
                          fontWeight: '700',
                          color: getActionColor(tx.action),
                        }}>
                          {tx.action.replace('_', ' ')}
                        </div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                          {formatTime(tx.timestamp)}
                        </div>
                      </div>

                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '8px' }}>
                        {tx.walletLabel || formatAddress(tx.walletAddress)}
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                            {tx.symbol}
                          </div>
                          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                            ${tx.price.toFixed(2)}
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                            ${(tx.size / 1000000).toFixed(2)}M
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Whale Performance Stats */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  📊 Whale Performance
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {whaleStats.slice(0, 3).map((stats, index) => (
                    <div
                      key={index}
                      style={{
                        padding: '16px',
                        background: 'rgba(255, 255, 255, 0.02)',
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                      }}
                    >
                      <div style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF', marginBottom: '12px' }}>
                        {stats.walletLabel || formatAddress(stats.walletAddress)}
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Win Rate</div>
                          <div style={{ fontSize: '16px', fontWeight: '700', color: stats.winRate >= 60 ? '#10B981' : '#F59E0B' }}>
                            {stats.winRate.toFixed(1)}%
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Avg PnL</div>
                          <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                            ${(stats.avgPnL / 1000).toFixed(1)}k
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Total Trades</div>
                          <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                            {stats.totalTrades}
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Active Pos</div>
                          <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                            {stats.activePositions}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
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
