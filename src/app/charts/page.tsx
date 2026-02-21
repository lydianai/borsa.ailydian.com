'use client';

/**
 * 📊 PREMIUM CHARTS PAGE
 * Professional candlestick charts with support/resistance levels
 *
 * Features:
 * - Real-time OHLCV data from Binance & Traditional Markets
 * - Multiple timeframe support (1m - 1w)
 * - Support/resistance levels visualization
 * - 580+ crypto pairs + Traditional assets
 * - Auto-refresh
 * - Lightweight Charts integration
 */

import React, { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { COLORS } from '@/lib/colors';
import { Icons } from '@/components/Icons';
import dynamic from 'next/dynamic';

// Dynamically import ApexCharts Candlestick Chart
const ApexCandlestickChart = dynamic(() => import('@/components/Chart/ApexCandlestickChart'), {
  ssr: false,
  loading: () => (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '600px',
      background: 'rgba(255, 255, 255, 0.03)',
      borderRadius: '16px'
    }}>
      <div style={{ textAlign: 'center', color: '#10b981' }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '4px solid rgba(16, 185, 129, 0.3)',
          borderTop: '4px solid #10b981',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 12px'
        }} />
        <div>Grafik yükleniyor...</div>
      </div>
    </div>
  )
});

// Coin data interface
interface CoinData {
  symbol: string;
  price: number;
  changePercent24h: number;
}

// Traditional asset interface
interface TraditionalAsset {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  category: string;
}

// Market type
type MarketType = 'crypto' | 'traditional';

// Timeframe options
const TIMEFRAMES = [
  { value: '1m', label: '1m' },
  { value: '3m', label: '3m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '30m', label: '30m' },
  { value: '1h', label: '1h' },
  { value: '2h', label: '2h' },
  { value: '4h', label: '4h' },
  { value: '1d', label: '1d' },
  { value: '1w', label: '1w' },
];

// Traditional market assets grouped by category
const TRADITIONAL_ASSETS: TraditionalAsset[] = [
  // Precious Metals
  { symbol: 'XAU', name: 'Gold', price: 0, change24h: 0, category: 'Metals' },
  { symbol: 'XAG', name: 'Silver', price: 0, change24h: 0, category: 'Metals' },
  { symbol: 'XPD', name: 'Palladium', price: 0, change24h: 0, category: 'Metals' },
  { symbol: 'XCU', name: 'Copper', price: 0, change24h: 0, category: 'Metals' },

  // Major Forex
  { symbol: 'EUR', name: 'Euro', price: 0, change24h: 0, category: 'Forex' },
  { symbol: 'GBP', name: 'British Pound', price: 0, change24h: 0, category: 'Forex' },
  { symbol: 'JPY', name: 'Japanese Yen', price: 0, change24h: 0, category: 'Forex' },
  { symbol: 'CHF', name: 'Swiss Franc', price: 0, change24h: 0, category: 'Forex' },

  // Indices
  { symbol: 'DXY', name: 'US Dollar Index', price: 0, change24h: 0, category: 'Indices' },
  { symbol: 'SPX', name: 'S&P 500', price: 0, change24h: 0, category: 'Indices' },
  { symbol: 'NDX', name: 'NASDAQ 100', price: 0, change24h: 0, category: 'Indices' },
  { symbol: 'DJI', name: 'Dow Jones', price: 0, change24h: 0, category: 'Indices' },

  // Energy
  { symbol: 'BRENT', name: 'Brent Crude', price: 0, change24h: 0, category: 'Energy' },
  { symbol: 'WTI', name: 'WTI Crude', price: 0, change24h: 0, category: 'Energy' },
  { symbol: 'NATGAS', name: 'Natural Gas', price: 0, change24h: 0, category: 'Energy' },
];

export default function ChartsPage() {
  // State
  const [mounted, setMounted] = useState(false);
  const [marketType, setMarketType] = useState<MarketType>('crypto');
  const [coins, setCoins] = useState<CoinData[]>([]);
  const [traditionalAssets, setTraditionalAssets] = useState<TraditionalAsset[]>(TRADITIONAL_ASSETS);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [searchTerm, setSearchTerm] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [countdown, setCountdown] = useState(60);

  // Prevent hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch available coins
  useEffect(() => {
    if (!mounted) return;

    const fetchCoins = async () => {
      try {
        const response = await fetch('/api/binance/futures');
        const data = await response.json();

        if (data.success && data.data?.all) {
          const coinData = data.data.all.map((c: any) => ({
            symbol: c.symbol,
            price: parseFloat(c.price),
            changePercent24h: parseFloat(c.changePercent24h || '0')
          }));
          setCoins(coinData);
        }
      } catch (err) {
        console.error('Error fetching coins:', err);
      }
    };

    fetchCoins();
  }, [mounted]);

  // Handle market type change
  useEffect(() => {
    if (!mounted) return;

    // Switch to default symbol when changing market type
    if (marketType === 'crypto') {
      setSelectedSymbol('BTCUSDT');
    } else {
      setSelectedSymbol('XAU'); // Default to Gold for traditional markets
    }
  }, [marketType, mounted]);

  // Auto-refresh countdown
  useEffect(() => {
    if (!autoRefresh || !mounted) return;

    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          // Refresh will happen automatically through TradingChart component
          return 60;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, mounted]);

  // Filter coins/assets by search based on market type
  const filteredCoins = marketType === 'crypto'
    ? coins.filter((coin) =>
        coin.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

  const filteredTraditionalAssets = marketType === 'traditional'
    ? traditionalAssets.filter((asset) =>
        asset.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
        asset.name.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : [];

  // Get current asset data
  const currentCoin = marketType === 'crypto'
    ? coins.find((c) => c.symbol === selectedSymbol)
    : undefined;

  const currentTraditionalAsset = marketType === 'traditional'
    ? traditionalAssets.find((a) => a.symbol === selectedSymbol)
    : undefined;

  if (!mounted) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#000000',
        color: '#FFFFFF'
      }}>
        Yükleniyor...
      </div>
    );
  }

  return (
      <div className="dashboard-container">
        <SharedSidebar currentPage="charts" coinCount={coins.length} countdown={countdown} />

        <div className="dashboard-main" style={{ marginTop: '60px', paddingTop: '80px' }}>
          <main className="dashboard-content" style={{
            padding: '24px',
            maxWidth: '100%',
            background: '#000000',
            minHeight: '100vh'
          }}>

            {/* Header Section */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '20px',
              marginBottom: '24px'
            }}>

              {/* Title + Current Price */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '12px'
              }}>
                <h1 className="neon-text" style={{
                  fontSize: '28px',
                  fontWeight: '700',
                  background: 'linear-gradient(135deg, #00D4FF, #FFFFFF)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  margin: 0
                }}>
                  📊 Premium Grafikler
                </h1>

                {/* Current Price Display */}
                {(currentCoin || currentTraditionalAsset) && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    padding: '12px 20px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    {/* Symbol Name */}
                    <div style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      color: 'rgba(255, 255, 255, 0.7)'
                    }}>
                      {marketType === 'crypto'
                        ? selectedSymbol
                        : currentTraditionalAsset?.name || selectedSymbol}
                    </div>

                    {/* Price */}
                    <div style={{
                      fontSize: '24px',
                      fontWeight: '700',
                      color: '#FFFFFF'
                    }}>
                      ${(currentCoin?.price || currentTraditionalAsset?.price || 0).toLocaleString('en-US', { maximumFractionDigits: 2 })}
                    </div>

                    {/* 24h Change */}
                    <div style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      padding: '4px 10px',
                      borderRadius: '6px',
                      background: (currentCoin?.changePercent24h || currentTraditionalAsset?.change24h || 0) >= 0
                        ? 'rgba(16, 185, 129, 0.15)'
                        : 'rgba(239, 68, 68, 0.15)',
                      color: (currentCoin?.changePercent24h || currentTraditionalAsset?.change24h || 0) >= 0
                        ? COLORS.success
                        : COLORS.danger
                    }}>
                      {(currentCoin?.changePercent24h || currentTraditionalAsset?.change24h || 0) >= 0 ? '↑' : '↓'}
                      {' '}{Math.abs(currentCoin?.changePercent24h || currentTraditionalAsset?.change24h || 0).toFixed(2)}%
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div style={{
                display: 'flex',
                gap: '16px',
                flexWrap: 'wrap',
                alignItems: 'flex-start'
              }}>

                {/* Market Type Selector */}
                <div style={{
                  display: 'flex',
                  gap: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  padding: '4px',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <button
                    onClick={() => setMarketType('crypto')}
                    style={{
                      padding: '10px 20px',
                      fontSize: '13px',
                      fontWeight: '600',
                      background: marketType === 'crypto'
                        ? 'linear-gradient(135deg, #00D4FF, #0EA5E9)'
                        : 'transparent',
                      color: marketType === 'crypto' ? '#000000' : '#FFFFFF',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    <Icons.TrendingUp style={{ width: '14px', height: '14px' }} />
                    Kripto
                  </button>
                  <button
                    onClick={() => setMarketType('traditional')}
                    style={{
                      padding: '10px 20px',
                      fontSize: '13px',
                      fontWeight: '600',
                      background: marketType === 'traditional'
                        ? 'linear-gradient(135deg, #F59E0B, #F97316)'
                        : 'transparent',
                      color: marketType === 'traditional' ? '#000000' : '#FFFFFF',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }}
                  >
                    <Icons.BarChart style={{ width: '14px', height: '14px' }} />
                    Geleneksel
                  </button>
                </div>

                {/* Symbol Selector */}
                <div style={{ flex: '1 1 300px', minWidth: '250px', position: 'relative' }}>
                  <div style={{
                    position: 'relative',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    <Icons.Search style={{
                      position: 'absolute',
                      left: '14px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      width: '18px',
                      height: '18px',
                      color: 'rgba(255, 255, 255, 0.5)',
                      pointerEvents: 'none'
                    }} />
                    <input
                      type="text"
                      placeholder={marketType === 'crypto' ? "Coin ara (örn: BTC, ETH)..." : "Varlık ara (örn: Altın, EUR)..."}
                      value={searchTerm}
                      onChange={(e) => {
                        setSearchTerm(e.target.value);
                        setShowDropdown(e.target.value.length > 0);
                      }}
                      onFocus={() => setShowDropdown(searchTerm.length > 0)}
                      style={{
                        width: '100%',
                        height: '44px',
                        background: 'transparent',
                        border: 'none',
                        borderRadius: '12px',
                        padding: '0 16px 0 44px',
                        fontSize: '14px',
                        color: '#fff',
                        outline: 'none'
                      }}
                    />
                  </div>

                  {/* Dropdown for Crypto */}
                  {showDropdown && marketType === 'crypto' && filteredCoins.length > 0 && (
                    <div style={{
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      right: 0,
                      marginTop: '4px',
                      maxHeight: '300px',
                      overflowY: 'auto',
                      background: 'rgba(17, 17, 17, 0.98)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '12px',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
                      zIndex: 1000
                    }}>
                      {filteredCoins.slice(0, 50).map((coin) => (
                        <div
                          key={coin.symbol}
                          onClick={() => {
                            setSelectedSymbol(coin.symbol);
                            setSearchTerm('');
                            setShowDropdown(false);
                          }}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '12px 16px',
                            cursor: 'pointer',
                            background: selectedSymbol === coin.symbol
                              ? 'rgba(255, 255, 255, 0.1)'
                              : 'transparent',
                            transition: 'all 0.2s'
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = selectedSymbol === coin.symbol
                              ? 'rgba(255, 255, 255, 0.1)'
                              : 'transparent';
                          }}
                        >
                          <span style={{
                            fontSize: '14px',
                            fontWeight: '600',
                            color: '#FFFFFF'
                          }}>
                            {coin.symbol}
                          </span>
                          <span style={{
                            fontSize: '12px',
                            fontWeight: '600',
                            color: coin.changePercent24h >= 0 ? COLORS.success : COLORS.danger
                          }}>
                            {coin.changePercent24h >= 0 ? '↑' : '↓'} {Math.abs(coin.changePercent24h).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Dropdown for Traditional Markets */}
                  {showDropdown && marketType === 'traditional' && filteredTraditionalAssets.length > 0 && (
                    <div style={{
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      right: 0,
                      marginTop: '4px',
                      maxHeight: '400px',
                      overflowY: 'auto',
                      background: 'rgba(17, 17, 17, 0.98)',
                      border: '1px solid rgba(255, 255, 255, 0.2)',
                      borderRadius: '12px',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
                      zIndex: 1000
                    }}>
                      {/* Group by category */}
                      {['Metals', 'Forex', 'Indices', 'Energy'].map((category) => {
                        const assetsInCategory = filteredTraditionalAssets.filter(a => a.category === category);
                        if (assetsInCategory.length === 0) return null;

                        return (
                          <React.Fragment key={category}>
                            <div style={{
                              padding: '8px 16px',
                              fontSize: '11px',
                              fontWeight: '700',
                              color: 'rgba(255, 255, 255, 0.5)',
                              textTransform: 'uppercase',
                              borderBottom: '1px solid rgba(255, 255, 255, 0.05)'
                            }}>
                              {category}
                            </div>
                            {assetsInCategory.map((asset) => (
                              <div
                                key={asset.symbol}
                                onClick={() => {
                                  setSelectedSymbol(asset.symbol);
                                  setSearchTerm('');
                                  setShowDropdown(false);
                                }}
                                style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'space-between',
                                  padding: '12px 16px',
                                  cursor: 'pointer',
                                  background: selectedSymbol === asset.symbol
                                    ? 'rgba(255, 255, 255, 0.1)'
                                    : 'transparent',
                                  transition: 'all 0.2s'
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.background = selectedSymbol === asset.symbol
                                    ? 'rgba(255, 255, 255, 0.1)'
                                    : 'transparent';
                                }}
                              >
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                                  <span style={{
                                    fontSize: '14px',
                                    fontWeight: '600',
                                    color: '#FFFFFF'
                                  }}>
                                    {asset.symbol}
                                  </span>
                                  <span style={{
                                    fontSize: '11px',
                                    color: 'rgba(255, 255, 255, 0.5)'
                                  }}>
                                    {asset.name}
                                  </span>
                                </div>
                                <span style={{
                                  fontSize: '12px',
                                  fontWeight: '600',
                                  color: asset.change24h >= 0 ? COLORS.success : COLORS.danger
                                }}>
                                  {asset.change24h >= 0 ? '↑' : '↓'} {Math.abs(asset.change24h).toFixed(2)}%
                                </span>
                              </div>
                            ))}
                          </React.Fragment>
                        );
                      })}
                    </div>
                  )}

                  {/* Quick Select based on market type */}
                  <div style={{
                    display: 'flex',
                    gap: '6px',
                    marginTop: '8px',
                    flexWrap: 'wrap'
                  }}>
                    {marketType === 'crypto' ? (
                      // Top 15 Crypto Coins
                      coins.slice(0, 15).map((coin) => (
                        <button
                          key={coin.symbol}
                          onClick={() => setSelectedSymbol(coin.symbol)}
                          style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            fontWeight: '600',
                            background: selectedSymbol === coin.symbol
                              ? 'linear-gradient(135deg, #00D4FF, #0EA5E9)'
                              : 'rgba(255, 255, 255, 0.05)',
                            color: selectedSymbol === coin.symbol ? '#000000' : '#FFFFFF',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                          }}
                          onMouseEnter={(e) => {
                            if (selectedSymbol !== coin.symbol) {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (selectedSymbol !== coin.symbol) {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                            }
                          }}
                        >
                          {coin.symbol.replace('USDT', '')}
                        </button>
                      ))
                    ) : (
                      // Top Traditional Assets
                      traditionalAssets.slice(0, 12).map((asset) => (
                        <button
                          key={asset.symbol}
                          onClick={() => setSelectedSymbol(asset.symbol)}
                          style={{
                            padding: '6px 12px',
                            fontSize: '11px',
                            fontWeight: '600',
                            background: selectedSymbol === asset.symbol
                              ? 'linear-gradient(135deg, #F59E0B, #F97316)'
                              : 'rgba(255, 255, 255, 0.05)',
                            color: selectedSymbol === asset.symbol ? '#000000' : '#FFFFFF',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            transition: 'all 0.2s'
                          }}
                          onMouseEnter={(e) => {
                            if (selectedSymbol !== asset.symbol) {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (selectedSymbol !== asset.symbol) {
                              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                            }
                          }}
                        >
                          {asset.symbol}
                        </button>
                      ))
                    )}
                  </div>
                </div>

                {/* Timeframe Selector */}
                <div style={{
                  display: 'flex',
                  gap: '6px',
                  flexWrap: 'wrap'
                }}>
                  {TIMEFRAMES.map((tf) => (
                    <button
                      key={tf.value}
                      onClick={() => setSelectedTimeframe(tf.value)}
                      style={{
                        padding: '10px 16px',
                        fontSize: '13px',
                        fontWeight: '600',
                        background: selectedTimeframe === tf.value
                          ? 'linear-gradient(135deg, #FFFFFF, #E0E0E0)'
                          : 'rgba(255, 255, 255, 0.05)',
                        color: selectedTimeframe === tf.value ? '#000000' : '#FFFFFF',
                        border: selectedTimeframe === tf.value
                          ? 'none'
                          : '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                        minWidth: '50px'
                      }}
                      onMouseEnter={(e) => {
                        if (selectedTimeframe !== tf.value) {
                          e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (selectedTimeframe !== tf.value) {
                          e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        }
                      }}
                    >
                      {tf.label}
                    </button>
                  ))}
                </div>

                {/* Auto-refresh Toggle */}
                <button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '10px 16px',
                    fontSize: '13px',
                    fontWeight: '600',
                    background: autoRefresh
                      ? 'rgba(16, 185, 129, 0.15)'
                      : 'rgba(255, 255, 255, 0.05)',
                    color: autoRefresh ? COLORS.success : '#FFFFFF',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                >
                  <Icons.Clock style={{ width: '16px', height: '16px' }} />
                  {autoRefresh ? `${countdown}s` : 'Kapalı'}
                </button>
              </div>
            </div>

            {/* Chart Container - Using SimpleChart Component */}
            <div style={{
              background: 'rgba(255, 255, 255, 0.03)',
              borderRadius: '16px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '20px'
            }}>
              <ApexCandlestickChart
                symbol={selectedSymbol}
                interval={selectedTimeframe}
                isTraditionalMarket={marketType === 'traditional'}
              />
            </div>

          </main>
        </div>

        {/* Global Styles */}
        <style jsx global>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .dashboard-container {
          min-height: 100vh;
          background: #000000;
        }

        .dashboard-main {
          width: 100%;
        }

        .dashboard-content {
          width: 100%;
          max-width: 100%;
        }

        .neon-text {
          text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }

        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }
      `}</style>
      </div>
  );
}
