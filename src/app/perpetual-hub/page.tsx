'use client';

/**
 * ⚡ PERPETUAL INTELLIGENCE HUB
 *
 * Advanced perpetual futures intelligence platform
 * Inspired by Avantis.fi with unique AI-powered features
 *
 * Core Features (5):
 * - Multi-Asset Correlation Matrix (Crypto + Forex + Metals)
 * - AI Position Risk Analyzer
 * - Futures Sentiment Hedger
 * - Leverage Efficiency Optimizer
 * - Liquidity Flow Intelligence
 *
 * Advanced Features (6):
 * - Advanced Order Book Depth Analyzer
 * - Smart Contract Risk Scanner
 * - Market Microstructure Dashboard
 * - Cross-Chain Perpetual Aggregator
 * - Whale Tracker & Alerts
 * - Dynamic Portfolio Rebalancer
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface FeatureCard {
  id: string;
  title: string;
  description: string;
  icon: any;
  color: string;
  gradient: string;
  href: string;
  comingSoon?: boolean;
  stats?: {
    label: string;
    value: string;
    trend?: 'up' | 'down' | 'neutral';
  }[];
}

export default function PerpetualHub() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [activeFeature, setActiveFeature] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Feature cards
  const features: FeatureCard[] = [
    {
      id: 'correlation-matrix',
      title: 'Multi-Asset Correlation Matrix',
      description: 'Real-time correlation analysis between Crypto, Forex, and Gold. Visualize market connections with 3D heatmap.',
      icon: Icons.GitBranch,
      color: '#3B82F6',
      gradient: 'linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)',
      href: '/perpetual-hub/correlation-matrix',
      stats: [
        { label: 'Active Assets', value: '247', trend: 'up' },
        { label: 'Correlation', value: '0.82', trend: 'up' },
        { label: 'Update', value: '1s', trend: 'neutral' },
      ],
    },
    {
      id: 'position-risk',
      title: 'AI Position Risk Analyzer',
      description: 'Machine learning position risk scoring. Smart stop-loss recommendations and liquidation prediction.',
      icon: Icons.Shield,
      color: '#EF4444',
      gradient: 'linear-gradient(135deg, #EF4444 0%, #DC2626 100%)',
      href: '/perpetual-hub/position-risk',
      stats: [
        { label: 'Risk Score', value: '7.2/10', trend: 'down' },
        { label: 'Liquidation', value: '$67,234', trend: 'neutral' },
        { label: 'Recommended', value: '15x', trend: 'up' },
      ],
    },
    {
      id: 'sentiment-hedge',
      title: 'Futures Sentiment Hedger',
      description: 'Whale vs Retail position analysis. AI-powered contrarian signal generation and Long/Short ratio tracking.',
      icon: Icons.TrendingUp,
      color: '#10B981',
      gradient: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
      href: '/perpetual-hub/sentiment-hedge',
      stats: [
        { label: 'L/S Ratio', value: '2.34', trend: 'up' },
        { label: 'Whale', value: '67%', trend: 'up' },
        { label: 'Signal', value: 'LONG', trend: 'up' },
      ],
    },
    {
      id: 'leverage-optimizer',
      title: 'Leverage Efficiency Optimizer',
      description: 'Risk-adjusted return calculation. Optimal leverage recommendation and funding cost simulation.',
      icon: Icons.Activity,
      color: '#F59E0B',
      gradient: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
      href: '/perpetual-hub/leverage-optimizer',
      stats: [
        { label: 'Optimal Lvg', value: '12.5x', trend: 'neutral' },
        { label: 'Expected', value: '+24.3%', trend: 'up' },
        { label: 'Cost', value: '0.02%', trend: 'down' },
      ],
    },
    {
      id: 'liquidity-flow',
      title: 'Liquidity Flow Intelligence',
      description: 'DEX vs CEX liquidity comparison. Best execution route and slippage prediction engine.',
      icon: Icons.Globe,
      color: '#8B5CF6',
      gradient: 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)',
      href: '/perpetual-hub/liquidity-flow',
      stats: [
        { label: 'CEX Liq.', value: '$2.4B', trend: 'up' },
        { label: 'DEX Liq.', value: '$890M', trend: 'up' },
        { label: 'Slippage', value: '0.12%', trend: 'down' },
      ],
    },
    {
      id: 'orderbook-depth',
      title: 'Advanced Order Book Depth',
      description: 'Multi-exchange order book aggregation. Liquidity walls and order flow imbalance detection.',
      icon: Icons.Layers,
      color: '#8B5CF6',
      gradient: 'linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)',
      href: '/perpetual-hub/orderbook-depth',
      stats: [
        { label: 'Exchanges', value: '5', trend: 'neutral' },
        { label: 'Imbalance', value: '+12.4%', trend: 'up' },
        { label: 'Depth', value: '±1%', trend: 'neutral' },
      ],
    },
    {
      id: 'contract-scanner',
      title: 'Smart Contract Risk Scanner',
      description: 'Security analysis of DeFi perpetual protocols. Audit scores and rug pull risk assessment.',
      icon: Icons.Shield,
      color: '#DC2626',
      gradient: 'linear-gradient(135deg, #DC2626 0%, #991B1B 100%)',
      href: '/perpetual-hub/contract-scanner',
      stats: [
        { label: 'Low Risk', value: '3', trend: 'up' },
        { label: 'High Risk', value: '1', trend: 'down' },
        { label: 'Insured', value: '4', trend: 'up' },
      ],
    },
    {
      id: 'market-microstructure',
      title: 'Market Microstructure',
      description: 'Professional tape reading. CVD analysis, spoofing detection and large order tracking.',
      icon: Icons.Activity,
      color: '#06B6D4',
      gradient: 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)',
      href: '/perpetual-hub/market-microstructure',
      stats: [
        { label: 'CVD', value: '+15.8', trend: 'up' },
        { label: 'Large Orders', value: '12', trend: 'neutral' },
        { label: 'Delta', value: '+8.2', trend: 'up' },
      ],
    },
    {
      id: 'cross-chain',
      title: 'Cross-Chain Aggregator',
      description: 'Multi-chain price comparison. Bridge cost analysis and arbitrage opportunities.',
      icon: Icons.Globe,
      color: '#A855F7',
      gradient: 'linear-gradient(135deg, #A855F7 0%, #7C3AED 100%)',
      href: '/perpetual-hub/cross-chain',
      stats: [
        { label: 'Chains', value: '6', trend: 'up' },
        { label: 'Best', value: 'Arbitrum', trend: 'neutral' },
        { label: 'Arbitrage', value: '$12', trend: 'up' },
      ],
    },
    {
      id: 'whale-tracker',
      title: 'Whale Tracker & Alerts',
      description: 'Large position tracking. Real-time whale transaction alerts and performance analysis.',
      icon: Icons.Users,
      color: '#14B8A6',
      gradient: 'linear-gradient(135deg, #14B8A6 0%, #0D9488 100%)',
      href: '/perpetual-hub/whale-tracker',
      stats: [
        { label: 'Active Whales', value: '4', trend: 'up' },
        { label: 'Avg Win Rate', value: '64.8%', trend: 'up' },
        { label: 'Total PnL', value: '+$315k', trend: 'up' },
      ],
    },
    {
      id: 'portfolio-rebalancer',
      title: 'Portfolio Rebalancer',
      description: 'AI-powered portfolio optimization. Risk parity and automatic rebalancing.',
      icon: Icons.TrendingUp,
      color: '#EC4899',
      gradient: 'linear-gradient(135deg, #EC4899 0%, #BE185D 100%)',
      href: '/perpetual-hub/portfolio-rebalancer',
      stats: [
        { label: 'Sharpe', value: '1.17', trend: 'up' },
        { label: 'Volatility', value: '38.5%', trend: 'down' },
        { label: 'Assets', value: '5', trend: 'neutral' },
      ],
    },
  ];

  // Market stats
  const marketStats = [
    { label: 'Total Volume 24h', value: '$127.4B', change: '+12.4%', trend: 'up' as const },
    { label: 'Open Interest', value: '$42.8B', change: '+8.2%', trend: 'up' as const },
    { label: 'Funding Rate', value: '0.0125%', change: '-0.003%', trend: 'down' as const },
    { label: 'Active Traders', value: '284K', change: '+15.7%', trend: 'up' as const },
  ];

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
            maxWidth: '1600px',
            margin: '0 auto',
            padding: '40px 24px',
            paddingTop: '80px',
          }}
        >
          {/* Hero Section */}
          <div
            style={{
              marginBottom: '48px',
              textAlign: 'center',
            }}
          >
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '16px',
                padding: '8px 20px',
                background: 'rgba(255, 215, 0, 0.1)',
                borderRadius: '50px',
                border: '1px solid rgba(255, 215, 0, 0.3)',
              }}
            >
              <span style={{ fontSize: '24px' }}>⚡</span>
              <span
                style={{
                  fontSize: '14px',
                  fontWeight: '700',
                  color: '#FFD700',
                  textTransform: 'uppercase',
                  letterSpacing: '1px',
                }}
              >
                Perpetual Intelligence Hub
              </span>
            </div>

            <h1
              style={{
                fontSize: '48px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '16px',
                letterSpacing: '-1px',
              }}
            >
              Gelecek Piyasalarında Üstünlük
            </h1>

            <p
              style={{
                fontSize: '18px',
                color: 'rgba(255, 255, 255, 0.7)',
                maxWidth: '800px',
                margin: '0 auto',
                lineHeight: '1.6',
              }}
            >
              Yapay zeka destekli analiz araçları ile perpetual futures piyasasında rekabet avantajı kazanın.
              Multi-asset korelasyon, risk yönetimi ve likidite optimizasyonu tek bir platformda.
            </p>
          </div>

          {/* Market Stats */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '16px',
              marginBottom: '48px',
            }}
          >
            {marketStats.map((stat, index) => (
              <div
                key={index}
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '16px',
                  padding: '20px',
                  transition: 'all 0.3s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <div
                  style={{
                    fontSize: '12px',
                    color: 'rgba(255, 255, 255, 0.5)',
                    marginBottom: '8px',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                  }}
                >
                  {stat.label}
                </div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'baseline',
                    gap: '12px',
                  }}
                >
                  <span
                    style={{
                      fontSize: '28px',
                      fontWeight: '700',
                      color: '#FFFFFF',
                    }}
                  >
                    {stat.value}
                  </span>
                  <span
                    style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      color: stat.trend === 'up' ? '#10B981' : stat.trend === 'down' ? '#EF4444' : '#6B7280',
                    }}
                  >
                    {stat.change}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Feature Cards */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
              gap: '24px',
            }}
          >
            {features.map((feature) => {
              const IconComponent = feature.icon;
              const isActive = activeFeature === feature.id;

              return (
                <Link
                  key={feature.id}
                  href={feature.href}
                  style={{ textDecoration: 'none' }}
                  onMouseEnter={() => setActiveFeature(feature.id)}
                  onMouseLeave={() => setActiveFeature(null)}
                >
                  <div
                    style={{
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: `1px solid ${isActive ? feature.color : 'rgba(255, 255, 255, 0.1)'}`,
                      borderRadius: '20px',
                      padding: '32px',
                      height: '100%',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      transform: isActive ? 'translateY(-8px)' : 'translateY(0)',
                      boxShadow: isActive
                        ? `0 20px 40px rgba(0, 0, 0, 0.5), 0 0 60px ${feature.color}40`
                        : '0 4px 16px rgba(0, 0, 0, 0.2)',
                      cursor: 'pointer',
                    }}
                  >
                    {/* Icon */}
                    <div
                      style={{
                        width: '64px',
                        height: '64px',
                        borderRadius: '16px',
                        background: feature.gradient,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        marginBottom: '24px',
                        boxShadow: `0 8px 24px ${feature.color}40`,
                      }}
                    >
                      <IconComponent style={{ width: '32px', height: '32px', color: '#FFFFFF' }} />
                    </div>

                    {/* Title */}
                    <h3
                      style={{
                        fontSize: '24px',
                        fontWeight: '700',
                        color: '#FFFFFF',
                        marginBottom: '12px',
                      }}
                    >
                      {feature.title}
                    </h3>

                    {/* Description */}
                    <p
                      style={{
                        fontSize: '14px',
                        color: 'rgba(255, 255, 255, 0.6)',
                        lineHeight: '1.6',
                        marginBottom: '24px',
                      }}
                    >
                      {feature.description}
                    </p>

                    {/* Stats */}
                    {feature.stats && (
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: 'repeat(3, 1fr)',
                          gap: '16px',
                          paddingTop: '24px',
                          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        {feature.stats.map((stat, idx) => (
                          <div key={idx}>
                            <div
                              style={{
                                fontSize: '10px',
                                color: 'rgba(255, 255, 255, 0.4)',
                                marginBottom: '4px',
                                textTransform: 'uppercase',
                              }}
                            >
                              {stat.label}
                            </div>
                            <div
                              style={{
                                fontSize: '16px',
                                fontWeight: '700',
                                color:
                                  stat.trend === 'up'
                                    ? '#10B981'
                                    : stat.trend === 'down'
                                    ? '#EF4444'
                                    : '#FFFFFF',
                              }}
                            >
                              {stat.value}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Coming Soon Badge */}
                    {feature.comingSoon && (
                      <div
                        style={{
                          marginTop: '16px',
                          padding: '8px 16px',
                          background: 'rgba(255, 215, 0, 0.1)',
                          border: '1px solid rgba(255, 215, 0, 0.3)',
                          borderRadius: '50px',
                          display: 'inline-block',
                        }}
                      >
                        <span
                          style={{
                            fontSize: '11px',
                            fontWeight: '700',
                            color: '#FFD700',
                            textTransform: 'uppercase',
                          }}
                        >
                          Çok Yakında
                        </span>
                      </div>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Info Banner */}
          <div
            style={{
              marginTop: '48px',
              padding: '32px',
              background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 165, 0, 0.05) 100%)',
              border: '1px solid rgba(255, 215, 0, 0.2)',
              borderRadius: '20px',
              textAlign: 'center',
            }}
          >
            <h3
              style={{
                fontSize: '20px',
                fontWeight: '700',
                color: '#FFD700',
                marginBottom: '12px',
              }}
            >
              🚀 Yeni Özellikler Yakında
            </h3>
            <p
              style={{
                fontSize: '14px',
                color: 'rgba(255, 255, 255, 0.7)',
                maxWidth: '700px',
                margin: '0 auto',
              }}
            >
              Yapay zeka destekli perpetual futures analiz araçlarımız sürekli geliştirilmektedir.
              Daha fazla özellik için takipte kalın!
            </p>
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
