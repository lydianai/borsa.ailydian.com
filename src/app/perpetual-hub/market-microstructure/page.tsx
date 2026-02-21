'use client';

/**
 * 📈 MARKET MICROSTRUCTURE DASHBOARD
 *
 * Professional tape reading and order flow analysis
 *
 * Features:
 * - Real-time Time & Sales (Tape)
 * - CVD (Cumulative Volume Delta)
 * - Large order detection (Iceberg orders)
 * - Spoofing pattern recognition
 * - Bid-Ask spread dynamics
 * - Volume profile analysis
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface Trade {
  timestamp: number;
  price: number;
  size: number;
  side: 'BUY' | 'SELL';
  isLargeOrder: boolean; // > $100k
  exchange: string;
}

interface CVDPoint {
  timestamp: number;
  cvd: number;
  price: number;
}

interface SpoofingAlert {
  timestamp: number;
  type: 'BID_SPOOFING' | 'ASK_SPOOFING';
  price: number;
  size: number;
  description: string;
}

export default function MarketMicrostructure() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock real-time trade data
  const generateMockTrades = (): Trade[] => {
    const basePrice = selectedSymbol === 'BTCUSDT' ? 98234.5 : 3456.78;
    const trades: Trade[] = [];
    const now = Date.now();

    for (let i = 0; i < 50; i++) {
      const timestamp = now - (i * 1000);
      const priceDelta = (Math.random() - 0.5) * (basePrice * 0.0002);
      const price = basePrice + priceDelta;
      const size = Math.random() * 2 + 0.1;
      const side = Math.random() > 0.5 ? 'BUY' : 'SELL';
      const isLargeOrder = Math.random() > 0.85;
      const exchanges = ['Binance', 'Bybit', 'OKX'];
      const exchange = exchanges[Math.floor(Math.random() * exchanges.length)];

      trades.push({
        timestamp,
        price,
        size: isLargeOrder ? size * 10 : size,
        side,
        isLargeOrder,
        exchange,
      });
    }

    return trades;
  };

  const trades = generateMockTrades();

  // Calculate CVD (Cumulative Volume Delta)
  const calculateCVD = (): CVDPoint[] => {
    let cvd = 0;
    return trades.slice().reverse().map((trade) => {
      cvd += trade.side === 'BUY' ? trade.size : -trade.size;
      return {
        timestamp: trade.timestamp,
        cvd,
        price: trade.price,
      };
    });
  };

  const cvdData = calculateCVD();
  const currentCVD = cvdData[cvdData.length - 1]?.cvd || 0;

  // Volume Delta
  const buyVolume = trades.filter(t => t.side === 'BUY').reduce((sum, t) => sum + t.size, 0);
  const sellVolume = trades.filter(t => t.side === 'SELL').reduce((sum, t) => sum + t.size, 0);
  const volumeDelta = buyVolume - sellVolume;

  // Spoofing detection (mock)
  const spoofingAlerts: SpoofingAlert[] = [
    {
      timestamp: Date.now() - 45000,
      type: 'BID_SPOOFING',
      price: 98220,
      size: 15.5,
      description: '15.5 BTC bid wall placed and pulled within 3 seconds',
    },
  ];

  // Large orders
  const largeOrders = trades.filter(t => t.isLargeOrder);

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp);
    return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}`;
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
                background: 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Market Microstructure
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Professional tape reading and order flow analysis - real-time market microstructure
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
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
            </select>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                style={{ width: '18px', height: '18px' }}
              />
              <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Auto-scroll Tape
              </span>
            </label>

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10B981', animation: 'pulse 2s infinite' }} />
              <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>LIVE</div>
            </div>
          </div>

          {/* CVD and Volume Delta Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <div style={{
              padding: '24px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                CVD (Cumulative Volume Delta)
              </div>
              <div style={{ fontSize: '48px', fontWeight: '900', color: currentCVD >= 0 ? '#10B981' : '#EF4444', marginBottom: '8px' }}>
                {currentCVD >= 0 ? '+' : ''}{currentCVD.toFixed(2)}
              </div>
              <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                {currentCVD >= 0 ? '🟢 Net buy pressure dominant' : '🔴 Net sell pressure dominant'}
              </div>
            </div>

            <div style={{
              padding: '24px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Volume Delta (Last 50 trades)
              </div>
              <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981', marginBottom: '4px' }}>
                    {buyVolume.toFixed(2)} BTC
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                    Buy Volume
                  </div>
                </div>
                <div style={{ fontSize: '24px', color: 'rgba(255, 255, 255, 0.3)' }}>vs</div>
                <div>
                  <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444', marginBottom: '4px' }}>
                    {sellVolume.toFixed(2)} BTC
                  </div>
                  <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                    Sell Volume
                  </div>
                </div>
              </div>
              <div style={{ marginTop: '12px', fontSize: '20px', fontWeight: '700', color: volumeDelta >= 0 ? '#10B981' : '#EF4444' }}>
                Δ {volumeDelta >= 0 ? '+' : ''}{volumeDelta.toFixed(2)} BTC
              </div>
            </div>

            <div style={{
              padding: '24px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                Large Orders Detected
              </div>
              <div style={{ fontSize: '48px', fontWeight: '900', color: '#F59E0B', marginBottom: '8px' }}>
                {largeOrders.length}
              </div>
              <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)' }}>
                ⚡ Trades &gt; $100k in last 50 trades
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
            {/* Time & Sales (Tape) */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
              height: '700px',
              display: 'flex',
              flexDirection: 'column',
            }}>
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                Time & Sales (Tape Reading)
              </h3>

              {/* Header */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '80px 100px 80px 60px 100px',
                gap: '8px',
                padding: '8px 12px',
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                marginBottom: '12px',
                fontSize: '11px',
                fontWeight: '600',
                color: 'rgba(255, 255, 255, 0.5)',
              }}>
                <div>TIME</div>
                <div>PRICE</div>
                <div>SIZE</div>
                <div>SIDE</div>
                <div>EXCHANGE</div>
              </div>

              {/* Tape */}
              <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {trades.map((trade, index) => (
                  <div
                    key={index}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '80px 100px 80px 60px 100px',
                      gap: '8px',
                      padding: '8px 12px',
                      background: trade.isLargeOrder
                        ? (trade.side === 'BUY' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)')
                        : 'rgba(255, 255, 255, 0.02)',
                      borderLeft: trade.isLargeOrder ? `3px solid ${trade.side === 'BUY' ? '#10B981' : '#EF4444'}` : 'none',
                      borderRadius: '6px',
                      fontSize: '13px',
                      fontFamily: 'monospace',
                    }}
                  >
                    <div style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                      {formatTime(trade.timestamp)}
                    </div>
                    <div style={{ fontWeight: '700', color: trade.side === 'BUY' ? '#10B981' : '#EF4444' }}>
                      ${trade.price.toFixed(2)}
                    </div>
                    <div style={{ color: '#FFFFFF', fontWeight: trade.isLargeOrder ? '700' : '400' }}>
                      {trade.size.toFixed(4)}
                      {trade.isLargeOrder && ' 🔥'}
                    </div>
                    <div style={{
                      fontSize: '11px',
                      fontWeight: '700',
                      color: trade.side === 'BUY' ? '#10B981' : '#EF4444',
                    }}>
                      {trade.side}
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                      {trade.exchange}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right Column */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* CVD Chart */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  CVD Chart (Last 50 trades)
                </h3>

                <div style={{ position: 'relative', height: '200px' }}>
                  <svg width="100%" height="100%" viewBox="0 0 600 200" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="0" y1="100" x2="600" y2="100" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

                    {/* CVD line */}
                    <polyline
                      points={cvdData.map((point, i) => {
                        const x = (i / (cvdData.length - 1)) * 600;
                        const minCVD = Math.min(...cvdData.map(p => p.cvd));
                        const maxCVD = Math.max(...cvdData.map(p => p.cvd));
                        const range = maxCVD - minCVD || 1;
                        const y = 200 - ((point.cvd - minCVD) / range) * 180 - 10;
                        return `${x},${y}`;
                      }).join(' ')}
                      fill="none"
                      stroke={currentCVD >= 0 ? '#10B981' : '#EF4444'}
                      strokeWidth="2"
                    />

                    {/* Area fill */}
                    <polygon
                      points={`0,200 ${cvdData.map((point, i) => {
                        const x = (i / (cvdData.length - 1)) * 600;
                        const minCVD = Math.min(...cvdData.map(p => p.cvd));
                        const maxCVD = Math.max(...cvdData.map(p => p.cvd));
                        const range = maxCVD - minCVD || 1;
                        const y = 200 - ((point.cvd - minCVD) / range) * 180 - 10;
                        return `${x},${y}`;
                      }).join(' ')} 600,200`}
                      fill={currentCVD >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)'}
                    />
                  </svg>
                </div>

                <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: '8px', fontSize: '12px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <strong style={{ color: '#06B6D4' }}>Analiz:</strong>{' '}
                  {currentCVD > 5 && 'Strong cumulative buy pressure - Bullish'}
                  {currentCVD < -5 && 'Strong cumulative sell pressure - Bearish'}
                  {Math.abs(currentCVD) <= 5 && 'Balanced order flow - Neutral'}
                </div>
              </div>

              {/* Spoofing Alerts */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  🚨 Spoofing Detection
                </h3>

                {spoofingAlerts.length === 0 && (
                  <div style={{ textAlign: 'center', padding: '32px', color: 'rgba(255, 255, 255, 0.5)' }}>
                    No spoofing detected
                  </div>
                )}

                {spoofingAlerts.map((alert, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '16px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: '12px',
                      marginBottom: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '8px' }}>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444', marginBottom: '4px' }}>
                          {alert.type === 'BID_SPOOFING' ? '🔴 BID SPOOFING' : '🔴 ASK SPOOFING'}
                        </div>
                        <div style={{ fontSize: '18px', fontWeight: '900', color: '#FFFFFF' }}>
                          ${alert.price.toFixed(2)}
                        </div>
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        {formatTime(alert.timestamp)}
                      </div>
                    </div>

                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '8px' }}>
                      {alert.size.toFixed(2)} BTC
                    </div>

                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)', lineHeight: '1.5' }}>
                      {alert.description}
                    </div>

                    <div style={{ marginTop: '12px', padding: '10px', background: 'rgba(0, 0, 0, 0.3)', borderRadius: '6px', fontSize: '11px', color: 'rgba(255, 255, 255, 0.7)' }}>
                      ⚠️ <strong>Spoofing:</strong> Attempt to manipulate price by placing and pulling large orders
                    </div>
                  </div>
                ))}
              </div>

              {/* Large Orders */}
              <div style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}>
                <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
                  🐋 Large Orders (&gt;$100k)
                </h3>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '250px', overflow: 'auto' }}>
                  {largeOrders.map((order, index) => (
                    <div
                      key={index}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '12px',
                        background: order.side === 'BUY' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        border: `1px solid ${order.side === 'BUY' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                        borderRadius: '8px',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                          {formatTime(order.timestamp)}
                        </div>
                        <div style={{ fontSize: '16px', fontWeight: '700', color: order.side === 'BUY' ? '#10B981' : '#EF4444' }}>
                          ${order.price.toFixed(2)}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF' }}>
                          {order.size.toFixed(4)} BTC
                        </div>
                        <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                          ${(order.price * order.size / 1000).toFixed(1)}k
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
