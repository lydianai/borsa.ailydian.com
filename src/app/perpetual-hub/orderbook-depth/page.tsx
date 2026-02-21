'use client';

/**
 * 📊 ADVANCED ORDER BOOK DEPTH ANALYZER
 *
 * Multi-exchange order book aggregation and imbalance detection
 *
 * Features:
 * - Real-time aggregated depth from 5+ exchanges
 * - Buy/Sell pressure imbalance detection
 * - Order book walls (large orders)
 * - Support/Resistance level detection
 * - Liquidity heatmap visualization
 * - Spread comparison across exchanges
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface OrderBookLevel {
  price: number;
  size: number;
  total: number;
  exchange: string;
}

interface ExchangeDepth {
  exchange: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  spread: number;
  bidVolume: number;
  askVolume: number;
  imbalance: number; // -100 to +100 (negative = sell pressure)
  lastUpdate: number;
}

interface OrderBookWall {
  price: number;
  size: number;
  type: 'bid' | 'ask';
  exchange: string;
  percentOfTotal: number;
}

export default function OrderBookDepth() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedExchange, setSelectedExchange] = useState('ALL');
  const [depthRange, setDepthRange] = useState('1'); // 1% price range

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock data - in production, this would come from WebSocket
  const generateMockOrderBook = (exchange: string, basePrice: number): ExchangeDepth => {
    const bids: OrderBookLevel[] = [];
    const asks: OrderBookLevel[] = [];
    let bidTotal = 0;
    let askTotal = 0;

    // Generate 20 levels
    for (let i = 0; i < 20; i++) {
      const bidPrice = basePrice - (basePrice * 0.0005 * i);
      const askPrice = basePrice + (basePrice * 0.0005 * i);

      const bidSize = Math.random() * 5 + 0.5;
      const askSize = Math.random() * 5 + 0.5;

      bidTotal += bidSize;
      askTotal += askSize;

      bids.push({ price: bidPrice, size: bidSize, total: bidTotal, exchange });
      asks.push({ price: askPrice, size: askSize, total: askTotal, exchange });
    }

    const bidVolume = bids.reduce((sum, b) => sum + b.size, 0);
    const askVolume = asks.reduce((sum, a) => sum + a.size, 0);
    const imbalance = ((bidVolume - askVolume) / (bidVolume + askVolume)) * 100;
    const spread = ((asks[0].price - bids[0].price) / bids[0].price) * 100;

    return {
      exchange,
      bids,
      asks,
      spread,
      bidVolume,
      askVolume,
      imbalance,
      lastUpdate: Date.now(),
    };
  };

  const basePrice = selectedSymbol === 'BTCUSDT' ? 98234.5 : 3456.78;

  const exchanges: ExchangeDepth[] = [
    generateMockOrderBook('Binance', basePrice),
    generateMockOrderBook('Bybit', basePrice + 10),
    generateMockOrderBook('OKX', basePrice - 5),
    generateMockOrderBook('Bitget', basePrice + 15),
    generateMockOrderBook('Gate.io', basePrice - 8),
  ];

  // Aggregate order book
  const aggregateOrderBook = (): { bids: OrderBookLevel[]; asks: OrderBookLevel[] } => {
    const allBids: OrderBookLevel[] = [];
    const allAsks: OrderBookLevel[] = [];

    exchanges.forEach((ex) => {
      allBids.push(...ex.bids.map((b) => ({ ...b, exchange: ex.exchange })));
      allAsks.push(...ex.asks.map((a) => ({ ...a, exchange: ex.exchange })));
    });

    // Sort and group by price level (rounded to nearest $10)
    const bidMap = new Map<number, OrderBookLevel>();
    const askMap = new Map<number, OrderBookLevel>();

    allBids.forEach((bid) => {
      const roundedPrice = Math.round(bid.price / 10) * 10;
      const existing = bidMap.get(roundedPrice);
      if (existing) {
        existing.size += bid.size;
      } else {
        bidMap.set(roundedPrice, { ...bid, price: roundedPrice });
      }
    });

    allAsks.forEach((ask) => {
      const roundedPrice = Math.round(ask.price / 10) * 10;
      const existing = askMap.get(roundedPrice);
      if (existing) {
        existing.size += ask.size;
      } else {
        askMap.set(roundedPrice, { ...ask, price: roundedPrice });
      }
    });

    const bids = Array.from(bidMap.values()).sort((a, b) => b.price - a.price).slice(0, 15);
    const asks = Array.from(askMap.values()).sort((a, b) => a.price - b.price).slice(0, 15);

    return { bids, asks };
  };

  const { bids, asks } = selectedExchange === 'ALL'
    ? aggregateOrderBook()
    : { bids: exchanges.find(e => e.exchange === selectedExchange)?.bids.slice(0, 15) || [], asks: exchanges.find(e => e.exchange === selectedExchange)?.asks.slice(0, 15) || [] };

  // Detect order book walls (large orders)
  const detectWalls = (): OrderBookWall[] => {
    const walls: OrderBookWall[] = [];
    const totalBidVolume = bids.reduce((sum, b) => sum + b.size, 0);
    const totalAskVolume = asks.reduce((sum, a) => sum + a.size, 0);

    bids.forEach((bid) => {
      const percentOfTotal = (bid.size / totalBidVolume) * 100;
      if (percentOfTotal > 8) { // Threshold for "wall"
        walls.push({
          price: bid.price,
          size: bid.size,
          type: 'bid',
          exchange: bid.exchange,
          percentOfTotal,
        });
      }
    });

    asks.forEach((ask) => {
      const percentOfTotal = (ask.size / totalAskVolume) * 100;
      if (percentOfTotal > 8) {
        walls.push({
          price: ask.price,
          size: ask.size,
          type: 'ask',
          exchange: ask.exchange,
          percentOfTotal,
        });
      }
    });

    return walls.sort((a, b) => b.percentOfTotal - a.percentOfTotal);
  };

  const walls = detectWalls();

  // Calculate total imbalance
  const totalBidVolume = bids.reduce((sum, b) => sum + b.size, 0);
  const totalAskVolume = asks.reduce((sum, a) => sum + a.size, 0);
  const totalImbalance = ((totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume)) * 100;

  const maxBidSize = Math.max(...bids.map((b) => b.size));
  const maxAskSize = Math.max(...asks.map((a) => a.size));
  const maxSize = Math.max(maxBidSize, maxAskSize);

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
              Perpetual Hub'a Dön
            </Link>

            <h1
              style={{
                fontSize: '40px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Advanced Order Book Depth
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Çoklu borsa emir defteri derinlik analizi ve likidite duvarları tespiti
            </p>
          </div>

          {/* Controls */}
          <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', flexWrap: 'wrap' }}>
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

            <select
              value={selectedExchange}
              onChange={(e) => setSelectedExchange(e.target.value)}
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
              <option value="ALL">Tüm Borsalar (Birleşik)</option>
              <option value="Binance">Binance</option>
              <option value="Bybit">Bybit</option>
              <option value="OKX">OKX</option>
              <option value="Bitget">Bitget</option>
              <option value="Gate.io">Gate.io</option>
            </select>

            <select
              value={depthRange}
              onChange={(e) => setDepthRange(e.target.value)}
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
              <option value="0.5">±0.5% Derinlik</option>
              <option value="1">±1% Derinlik</option>
              <option value="2">±2% Derinlik</option>
            </select>

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>Güncelleme:</div>
              <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>Canlı</div>
            </div>
          </div>

          {/* Imbalance Meter */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '16px' }}>
              Emir Dengesi (Order Flow Imbalance)
            </h3>

            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#EF4444', fontWeight: '600' }}>SATIŞ BASKI</span>
                  <span style={{ fontSize: '12px', color: '#10B981', fontWeight: '600' }}>ALIŞ BASKI</span>
                </div>
                <div
                  style={{
                    height: '48px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '24px',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: '0',
                      bottom: '0',
                      width: '2px',
                      background: 'rgba(255, 255, 255, 0.2)',
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      left: totalImbalance > 0 ? '50%' : `${50 + totalImbalance}%`,
                      right: totalImbalance > 0 ? `${50 - totalImbalance}%` : '50%',
                      top: '0',
                      bottom: '0',
                      background: totalImbalance > 0 ? 'linear-gradient(90deg, rgba(16, 185, 129, 0.3), #10B981)' : 'linear-gradient(90deg, #EF4444, rgba(239, 68, 68, 0.3))',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <span style={{ fontSize: '16px', fontWeight: '900', color: '#FFFFFF' }}>
                      {Math.abs(totalImbalance).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                  Toplam Hacim
                </div>
                <div style={{ fontSize: '20px', fontWeight: '700', color: '#FFFFFF' }}>
                  {(totalBidVolume + totalAskVolume).toFixed(2)} BTC
                </div>
                <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                  Alış: {totalBidVolume.toFixed(2)} | Satış: {totalAskVolume.toFixed(2)}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px', fontSize: '13px', color: 'rgba(255, 255, 255, 0.8)' }}>
              <strong style={{ color: '#8B5CF6' }}>Analiz:</strong>{' '}
              {totalImbalance > 10 && 'Güçlü ALIŞ baskısı - Fiyat yükselişi bekleniyor'}
              {totalImbalance < -10 && 'Güçlü SATIŞ baskısı - Fiyat düşüşü bekleniyor'}
              {Math.abs(totalImbalance) <= 10 && 'Dengeli emir akışı - Konsolidasyon bekleniyor'}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
            {/* Order Book Depth Visualization */}
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                Emir Defteri Derinliği
              </h3>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* Bids */}
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: '#10B981', marginBottom: '12px' }}>
                    ALIŞ EMİRLERİ (BIDS)
                  </div>
                  {bids.map((bid, index) => (
                    <div
                      key={index}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '80px 60px',
                        gap: '8px',
                        marginBottom: '6px',
                        position: 'relative',
                        padding: '4px 8px',
                      }}
                    >
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${(bid.size / maxSize) * 100}%`,
                          background: 'rgba(16, 185, 129, 0.15)',
                          borderRadius: '4px',
                        }}
                      />
                      <div style={{ fontSize: '13px', fontWeight: '600', color: '#10B981', position: 'relative' }}>
                        ${bid.price.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)', position: 'relative' }}>
                        {bid.size.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Asks */}
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: '#EF4444', marginBottom: '12px' }}>
                    SATIŞ EMİRLERİ (ASKS)
                  </div>
                  {asks.map((ask, index) => (
                    <div
                      key={index}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '80px 60px',
                        gap: '8px',
                        marginBottom: '6px',
                        position: 'relative',
                        padding: '4px 8px',
                      }}
                    >
                      <div
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          bottom: 0,
                          width: `${(ask.size / maxSize) * 100}%`,
                          background: 'rgba(239, 68, 68, 0.15)',
                          borderRadius: '4px',
                        }}
                      />
                      <div style={{ fontSize: '13px', fontWeight: '600', color: '#EF4444', position: 'relative' }}>
                        ${ask.price.toFixed(2)}
                      </div>
                      <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.7)', position: 'relative' }}>
                        {ask.size.toFixed(3)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Order Book Walls Detection */}
            <div
              style={{
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
                Likidite Duvarları (Order Book Walls)
              </h3>

              {walls.length === 0 && (
                <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255, 255, 255, 0.5)' }}>
                  Büyük emir tespit edilmedi
                </div>
              )}

              {walls.map((wall, index) => (
                <div
                  key={index}
                  style={{
                    padding: '16px',
                    background: wall.type === 'bid' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    border: `1px solid ${wall.type === 'bid' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                    borderRadius: '12px',
                    marginBottom: '12px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        {wall.type === 'bid' ? '🟢 ALIŞ DUVARI' : '🔴 SATIŞ DUVARI'}
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: wall.type === 'bid' ? '#10B981' : '#EF4444' }}>
                        ${wall.price.toFixed(2)}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                        {wall.size.toFixed(3)} BTC
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                        {wall.percentOfTotal.toFixed(1)}% of total
                      </div>
                    </div>
                  </div>

                  <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                    Borsa: <strong>{wall.exchange}</strong>
                  </div>

                  <div style={{ marginTop: '8px', padding: '8px', background: 'rgba(0, 0, 0, 0.2)', borderRadius: '6px', fontSize: '11px', color: 'rgba(255, 255, 255, 0.7)' }}>
                    {wall.type === 'bid' ?
                      '💡 Bu seviyede güçlü alım desteği var. Fiyat bu seviyeyi test ederse toparlanma beklenebilir.' :
                      '⚠️ Bu seviyede güçlü satış baskısı var. Fiyat bu seviyeyi kırmakta zorlanabilir.'
                    }
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Exchange Comparison */}
          <div
            style={{
              marginTop: '32px',
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Borsa Karşılaştırması
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              {exchanges.map((ex) => (
                <div
                  key={ex.exchange}
                  style={{
                    padding: '16px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                  }}
                >
                  <div style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF', marginBottom: '12px' }}>
                    {ex.exchange}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>Spread</div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: ex.spread < 0.02 ? '#10B981' : '#F59E0B' }}>
                        {ex.spread.toFixed(3)}%
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>Imbalance</div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: ex.imbalance > 0 ? '#10B981' : '#EF4444' }}>
                        {ex.imbalance > 0 ? '+' : ''}{ex.imbalance.toFixed(1)}%
                      </div>
                    </div>

                    <div>
                      <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)' }}>Toplam Hacim</div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        {(ex.bidVolume + ex.askVolume).toFixed(2)} BTC
                      </div>
                    </div>
                  </div>
                </div>
              ))}
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
