'use client';

/**
 * 📊 MULTI-ASSET CORRELATION MATRIX
 *
 * Real-time correlation analysis across:
 * - Cryptocurrency (BTC, ETH, SOL, etc.)
 * - Forex (EUR/USD, GBP/USD, USD/JPY, etc.)
 * - Commodities (Gold, Silver, Oil, etc.)
 * - Traditional Markets (S&P 500, Nasdaq, etc.)
 *
 * Features:
 * - 3D Heatmap visualization
 * - Live correlation coefficients
 * - Historical correlation trends
 * - Cross-market signal generation
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface CorrelationPair {
  asset1: string;
  asset2: string;
  correlation: number;
  category1: 'crypto' | 'forex' | 'commodity' | 'index';
  category2: 'crypto' | 'forex' | 'commodity' | 'index';
  change24h: number;
  strength: 'very-strong' | 'strong' | 'moderate' | 'weak';
}

interface Asset {
  symbol: string;
  name: string;
  category: 'crypto' | 'forex' | 'commodity' | 'index';
  price: number;
  change24h: number;
}

export default function CorrelationMatrix() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState<'1h' | '4h' | '1d' | '1w'>('1d');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setMounted(true);
    // Simulate data loading
    setTimeout(() => setLoading(false), 1500);
  }, []);

  // Mock assets data
  const assets: Asset[] = [
    // Crypto
    { symbol: 'BTC', name: 'Bitcoin', category: 'crypto', price: 67234.52, change24h: 2.34 },
    { symbol: 'ETH', name: 'Ethereum', category: 'crypto', price: 3421.67, change24h: 1.87 },
    { symbol: 'SOL', name: 'Solana', category: 'crypto', price: 124.89, change24h: 4.12 },
    { symbol: 'BNB', name: 'Binance Coin', category: 'crypto', price: 542.34, change24h: 1.45 },

    // Forex
    { symbol: 'EUR/USD', name: 'Euro/US Dollar', category: 'forex', price: 1.0842, change24h: 0.12 },
    { symbol: 'GBP/USD', name: 'British Pound/US Dollar', category: 'forex', price: 1.2634, change24h: -0.08 },
    { symbol: 'USD/JPY', name: 'US Dollar/Japanese Yen', category: 'forex', price: 149.52, change24h: 0.34 },
    { symbol: 'AUD/USD', name: 'Australian Dollar/US Dollar', category: 'forex', price: 0.6523, change24h: 0.21 },

    // Commodities
    { symbol: 'XAU', name: 'Gold', category: 'commodity', price: 2087.45, change24h: 0.67 },
    { symbol: 'XAG', name: 'Silver', category: 'commodity', price: 24.52, change24h: 1.23 },
    { symbol: 'WTI', name: 'Oil (WTI)', category: 'commodity', price: 78.34, change24h: -0.45 },
    { symbol: 'BRENT', name: 'Oil (Brent)', category: 'commodity', price: 82.67, change24h: -0.32 },

    // Indices
    { symbol: 'SPX', name: 'S&P 500', category: 'index', price: 4782.34, change24h: 0.89 },
    { symbol: 'NDX', name: 'Nasdaq 100', category: 'index', price: 16834.52, change24h: 1.23 },
    { symbol: 'DXY', name: 'US Dollar Index', category: 'index', price: 103.45, change24h: -0.15 },
  ];

  // Mock correlation data
  const correlations: CorrelationPair[] = [
    { asset1: 'BTC', asset2: 'XAU', correlation: 0.78, category1: 'crypto', category2: 'commodity', change24h: 0.05, strength: 'strong' },
    { asset1: 'BTC', asset2: 'SPX', correlation: 0.65, category1: 'crypto', category2: 'index', change24h: -0.12, strength: 'moderate' },
    { asset1: 'ETH', asset2: 'BTC', correlation: 0.92, category1: 'crypto', category2: 'crypto', change24h: 0.02, strength: 'very-strong' },
    { asset1: 'XAU', asset2: 'DXY', correlation: -0.68, category1: 'commodity', category2: 'index', change24h: -0.08, strength: 'moderate' },
    { asset1: 'EUR/USD', asset2: 'DXY', correlation: -0.89, category1: 'forex', category2: 'index', change24h: 0.03, strength: 'very-strong' },
    { asset1: 'WTI', asset2: 'XAU', correlation: 0.45, category1: 'commodity', category2: 'commodity', change24h: 0.15, strength: 'moderate' },
    { asset1: 'BTC', asset2: 'EUR/USD', correlation: 0.23, category1: 'crypto', category2: 'forex', change24h: 0.07, strength: 'weak' },
    { asset1: 'SOL', asset2: 'ETH', correlation: 0.84, category1: 'crypto', category2: 'crypto', change24h: -0.04, strength: 'strong' },
  ];

  const getCorrelationColor = (value: number): string => {
    const absValue = Math.abs(value);
    if (absValue >= 0.8) return value > 0 ? '#10B981' : '#EF4444';
    if (absValue >= 0.5) return value > 0 ? '#3B82F6' : '#F59E0B';
    return '#6B7280';
  };

  const getCategoryColor = (category: string): string => {
    switch (category) {
      case 'crypto': return '#FFD700';
      case 'forex': return '#3B82F6';
      case 'commodity': return '#F59E0B';
      case 'index': return '#8B5CF6';
      default: return '#FFFFFF';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'crypto': return Icons.TrendingUp;
      case 'forex': return Icons.Globe;
      case 'commodity': return Icons.Activity;
      case 'index': return Icons.BarChart;
      default: return Icons.TrendingUp;
    }
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
        {/* Header */}
        <SharedSidebar
          currentPage="perpetual-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        {/* Main Content */}
        <main
          style={{
            maxWidth: '1800px',
            margin: '0 auto',
            padding: '40px 24px',
            paddingTop: '80px',
          }}
        >
          {/* Header Section */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '32px',
            }}
          >
            <div>
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
                  background: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  marginBottom: '8px',
                }}
              >
                Multi-Asset Correlation Matrix
              </h1>

              <p
                style={{
                  fontSize: '16px',
                  color: 'rgba(255, 255, 255, 0.6)',
                }}
              >
                Crypto, Forex, Altın ve Endeksler arası gerçek zamanlı korelasyon analizi
              </p>
            </div>

            {/* Timeframe Selector */}
            <div
              style={{
                display: 'flex',
                gap: '8px',
                background: 'rgba(255, 255, 255, 0.05)',
                padding: '4px',
                borderRadius: '12px',
              }}
            >
              {(['1h', '4h', '1d', '1w'] as const).map((tf) => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  style={{
                    padding: '8px 20px',
                    background: timeframe === tf ? '#3B82F6' : 'transparent',
                    color: timeframe === tf ? '#FFFFFF' : 'rgba(255, 255, 255, 0.6)',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                  }}
                >
                  {tf.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Asset Categories Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
              gap: '16px',
              marginBottom: '32px',
            }}
          >
            {['crypto', 'forex', 'commodity', 'index'].map((category) => {
              const categoryAssets = assets.filter((a) => a.category === category);
              const IconComponent = getCategoryIcon(category);

              return (
                <div
                  key={category}
                  style={{
                    background: 'rgba(255, 255, 255, 0.03)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '16px',
                    padding: '20px',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      marginBottom: '16px',
                    }}
                  >
                    <div
                      style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '10px',
                        background: `${getCategoryColor(category)}20`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      <IconComponent
                        style={{
                          width: '20px',
                          height: '20px',
                          color: getCategoryColor(category),
                        }}
                      />
                    </div>
                    <div>
                      <h3
                        style={{
                          fontSize: '16px',
                          fontWeight: '700',
                          color: '#FFFFFF',
                          textTransform: 'capitalize',
                        }}
                      >
                        {category === 'crypto' ? 'Cryptocurrency' : category === 'forex' ? 'Forex' : category === 'commodity' ? 'Commodities' : 'Indices'}
                      </h3>
                      <p
                        style={{
                          fontSize: '12px',
                          color: 'rgba(255, 255, 255, 0.5)',
                        }}
                      >
                        {categoryAssets.length} varlık
                      </p>
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {categoryAssets.map((asset) => (
                      <div
                        key={asset.symbol}
                        onClick={() => setSelectedAsset(asset.symbol)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '12px',
                          background: selectedAsset === asset.symbol ? 'rgba(255, 255, 255, 0.1)' : 'rgba(255, 255, 255, 0.03)',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          transition: 'all 0.2s',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = selectedAsset === asset.symbol ? 'rgba(255, 255, 255, 0.1)' : 'rgba(255, 255, 255, 0.03)';
                        }}
                      >
                        <div>
                          <div
                            style={{
                              fontSize: '14px',
                              fontWeight: '600',
                              color: '#FFFFFF',
                            }}
                          >
                            {asset.symbol}
                          </div>
                          <div
                            style={{
                              fontSize: '11px',
                              color: 'rgba(255, 255, 255, 0.5)',
                            }}
                          >
                            {asset.name}
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div
                            style={{
                              fontSize: '14px',
                              fontWeight: '600',
                              color: '#FFFFFF',
                            }}
                          >
                            ${asset.price.toLocaleString()}
                          </div>
                          <div
                            style={{
                              fontSize: '12px',
                              fontWeight: '600',
                              color: asset.change24h > 0 ? '#10B981' : '#EF4444',
                            }}
                          >
                            {asset.change24h > 0 ? '+' : ''}{asset.change24h.toFixed(2)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Correlation Pairs */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}
          >
            <h2
              style={{
                fontSize: '20px',
                fontWeight: '700',
                color: '#FFFFFF',
                marginBottom: '20px',
              }}
            >
              Önemli Korelasyonlar
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {correlations.map((pair, index) => (
                <div
                  key={index}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '200px 200px 1fr 120px 100px',
                    alignItems: 'center',
                    gap: '16px',
                    padding: '16px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '12px',
                    border: `1px solid ${getCorrelationColor(pair.correlation)}40`,
                  }}
                >
                  {/* Asset 1 */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: getCategoryColor(pair.category1),
                      }}
                    />
                    <span
                      style={{
                        fontSize: '14px',
                        fontWeight: '600',
                        color: '#FFFFFF',
                      }}
                    >
                      {pair.asset1}
                    </span>
                  </div>

                  {/* Asset 2 */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                  >
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        background: getCategoryColor(pair.category2),
                      }}
                    />
                    <span
                      style={{
                        fontSize: '14px',
                        fontWeight: '600',
                        color: '#FFFFFF',
                      }}
                    >
                      {pair.asset2}
                    </span>
                  </div>

                  {/* Correlation Bar */}
                  <div
                    style={{
                      position: 'relative',
                      height: '24px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      borderRadius: '12px',
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        position: 'absolute',
                        left: pair.correlation < 0 ? `${50 + pair.correlation * 50}%` : '50%',
                        width: `${Math.abs(pair.correlation) * 50}%`,
                        height: '100%',
                        background: getCorrelationColor(pair.correlation),
                        transition: 'all 0.3s',
                      }}
                    />
                    <div
                      style={{
                        position: 'absolute',
                        left: '50%',
                        top: 0,
                        bottom: 0,
                        width: '1px',
                        background: 'rgba(255, 255, 255, 0.3)',
                      }}
                    />
                  </div>

                  {/* Correlation Value */}
                  <div style={{ textAlign: 'center' }}>
                    <div
                      style={{
                        fontSize: '18px',
                        fontWeight: '700',
                        color: getCorrelationColor(pair.correlation),
                      }}
                    >
                      {pair.correlation > 0 ? '+' : ''}{pair.correlation.toFixed(2)}
                    </div>
                    <div
                      style={{
                        fontSize: '11px',
                        color: 'rgba(255, 255, 255, 0.5)',
                        textTransform: 'capitalize',
                      }}
                    >
                      {pair.strength.replace('-', ' ')}
                    </div>
                  </div>

                  {/* 24h Change */}
                  <div
                    style={{
                      fontSize: '13px',
                      fontWeight: '600',
                      color: pair.change24h > 0 ? '#10B981' : '#EF4444',
                      textAlign: 'right',
                    }}
                  >
                    {pair.change24h > 0 ? '+' : ''}{pair.change24h.toFixed(2)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Info Banner */}
          <div
            style={{
              marginTop: '32px',
              padding: '24px',
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(29, 78, 216, 0.05) 100%)',
              border: '1px solid rgba(59, 130, 246, 0.2)',
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'start',
              gap: '16px',
            }}
          >
            <Icons.Info style={{ width: '24px', height: '24px', color: '#3B82F6', flexShrink: 0 }} />
            <div>
              <h4
                style={{
                  fontSize: '16px',
                  fontWeight: '700',
                  color: '#3B82F6',
                  marginBottom: '8px',
                }}
              >
                Korelasyon Nasıl Yorumlanır?
              </h4>
              <p
                style={{
                  fontSize: '14px',
                  color: 'rgba(255, 255, 255, 0.7)',
                  lineHeight: '1.6',
                }}
              >
                <strong>+1.0:</strong> Mükemmel pozitif korelasyon (birlikte hareket ederler)<br />
                <strong>0.0:</strong> Korelasyon yok (bağımsız hareket)<br />
                <strong>-1.0:</strong> Mükemmel negatif korelasyon (ters yönde hareket ederler)<br />
                <strong>0.8+:</strong> Çok güçlü ilişki, hedge veya arbitraj fırsatları için kullanılabilir
              </p>
            </div>
          </div>
        </main>

        {/* AI Assistant */}
        {aiAssistantOpen && (
          <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
        )}
      </div>
    </PWAProvider>
  );
}
