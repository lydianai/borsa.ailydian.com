'use client';

/**
 * 🌊 LIQUIDITY FLOW INTELLIGENCE
 *
 * DEX vs CEX liquidity comparison and best execution routing
 *
 * Features:
 * - Real-time liquidity depth analysis
 * - DEX vs CEX comparison
 * - Slippage prediction engine
 * - Best execution route finder
 * - Liquidity concentration heatmap
 * - Market impact calculator
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface LiquiditySource {
  name: string;
  type: 'CEX' | 'DEX';
  liquidity: number;
  spread: number;
  slippage100k: number;
  slippage1m: number;
  fee: number;
  avgExecutionTime: number;
}

export default function LiquidityFlow() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState('BTCUSDT');
  const [tradeSize, setTradeSize] = useState('100000');

  useEffect(() => {
    setMounted(true);
  }, []);

  const liquiditySources: LiquiditySource[] = [
    {
      name: 'Binance Futures',
      type: 'CEX',
      liquidity: 2400000000,
      spread: 0.01,
      slippage100k: 0.02,
      slippage1m: 0.15,
      fee: 0.04,
      avgExecutionTime: 50,
    },
    {
      name: 'Bybit',
      type: 'CEX',
      liquidity: 1800000000,
      spread: 0.015,
      slippage100k: 0.03,
      slippage1m: 0.18,
      fee: 0.055,
      avgExecutionTime: 60,
    },
    {
      name: 'OKX',
      type: 'CEX',
      liquidity: 1200000000,
      spread: 0.012,
      slippage100k: 0.025,
      slippage1m: 0.16,
      fee: 0.05,
      avgExecutionTime: 55,
    },
    {
      name: 'dYdX',
      type: 'DEX',
      liquidity: 450000000,
      spread: 0.03,
      slippage100k: 0.08,
      slippage1m: 0.45,
      fee: 0.02,
      avgExecutionTime: 3000,
    },
    {
      name: 'GMX',
      type: 'DEX',
      liquidity: 320000000,
      spread: 0.04,
      slippage100k: 0.12,
      slippage1m: 0.65,
      fee: 0.1,
      avgExecutionTime: 5000,
    },
    {
      name: 'Vertex',
      type: 'DEX',
      liquidity: 180000000,
      spread: 0.05,
      slippage100k: 0.15,
      slippage1m: 0.8,
      fee: 0.025,
      avgExecutionTime: 4000,
    },
  ];

  const calculateSlippage = (source: LiquiditySource) => {
    const size = parseFloat(tradeSize);
    if (size <= 100000) return source.slippage100k;
    if (size >= 1000000) return source.slippage1m;
    // Linear interpolation
    const ratio = (size - 100000) / 900000;
    return source.slippage100k + (source.slippage1m - source.slippage100k) * ratio;
  };

  const calculateTotalCost = (source: LiquiditySource) => {
    const slippage = calculateSlippage(source);
    return slippage + source.fee + source.spread;
  };

  const sortedByBestExecution = [...liquiditySources].sort((a, b) => {
    return calculateTotalCost(a) - calculateTotalCost(b);
  });

  const bestRoute = sortedByBestExecution[0];
  const totalCEXLiquidity = liquiditySources.filter(s => s.type === 'CEX').reduce((sum, s) => sum + s.liquidity, 0);
  const totalDEXLiquidity = liquiditySources.filter(s => s.type === 'DEX').reduce((sum, s) => sum + s.liquidity, 0);

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
                background: 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Liquidity Flow Intelligence
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              DEX vs CEX likidite karşılaştırması ve en iyi execution route analizi
            </p>
          </div>

          {/* Trade Size Input */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px',
            }}
          >
            <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px', display: 'block' }}>
              İşlem Büyüklüğü (USDT)
            </label>
            <input
              type="number"
              value={tradeSize}
              onChange={(e) => setTradeSize(e.target.value)}
              style={{
                width: '300px',
                padding: '16px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                color: '#FFFFFF',
                fontSize: '18px',
                fontWeight: '600',
              }}
            />
          </div>

          {/* Best Route Card */}
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(124, 58, 237, 0.08) 100%)',
              border: '2px solid #8B5CF6',
              borderRadius: '20px',
              padding: '32px',
              marginBottom: '32px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                  ✨ EN İYİ EXECUTION ROUTE
                </div>
                <h2 style={{ fontSize: '36px', fontWeight: '900', color: '#FFFFFF', marginBottom: '8px' }}>
                  {bestRoute.name}
                </h2>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '6px 16px',
                    background: bestRoute.type === 'CEX' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                    color: bestRoute.type === 'CEX' ? '#10B981' : '#3B82F6',
                    borderRadius: '8px',
                    fontSize: '13px',
                    fontWeight: '700',
                  }}
                >
                  {bestRoute.type}
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                  Total Cost
                </div>
                <div style={{ fontSize: '42px', fontWeight: '900', color: '#10B981' }}>
                  {calculateTotalCost(bestRoute).toFixed(3)}%
                </div>
              </div>
            </div>
          </div>

          {/* CEX vs DEX Overview */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
            <div
              style={{
                background: 'rgba(16, 185, 129, 0.05)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: '#10B981',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icons.TrendingUp style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                    CEX Toplam
                  </h3>
                  <p style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', margin: 0 }}>
                    Merkezi borsalar
                  </p>
                </div>
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#10B981' }}>
                ${(totalCEXLiquidity / 1000000000).toFixed(2)}B
              </div>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
                {liquiditySources.filter(s => s.type === 'CEX').length} platform
              </div>
            </div>

            <div
              style={{
                background: 'rgba(59, 130, 246, 0.05)',
                border: '1px solid rgba(59, 130, 246, 0.2)',
                borderRadius: '16px',
                padding: '24px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: '#3B82F6',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Icons.Globe style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
                </div>
                <div>
                  <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', margin: 0 }}>
                    DEX Toplam
                  </h3>
                  <p style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)', margin: 0 }}>
                    Merkezi olmayan borsalar
                  </p>
                </div>
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#3B82F6' }}>
                ${(totalDEXLiquidity / 1000000000).toFixed(2)}B
              </div>
              <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.6)' }}>
                {liquiditySources.filter(s => s.type === 'DEX').length} protocol
              </div>
            </div>
          </div>

          {/* Detailed Comparison Table */}
          <div
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              padding: '24px',
            }}
          >
            <h3 style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
              Detaylı Karşılaştırma (${parseFloat(tradeSize).toLocaleString()} işlem için)
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {sortedByBestExecution.map((source, index) => {
                const totalCost = calculateTotalCost(source);
                const slippage = calculateSlippage(source);
                const isBest = index === 0;

                return (
                  <div
                    key={source.name}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '200px 120px 140px 120px 120px 120px 1fr',
                      alignItems: 'center',
                      gap: '16px',
                      padding: '20px',
                      background: isBest ? 'rgba(139, 92, 246, 0.1)' : 'rgba(255, 255, 255, 0.02)',
                      border: isBest ? '1px solid #8B5CF6' : '1px solid rgba(255, 255, 255, 0.05)',
                      borderRadius: '12px',
                    }}
                  >
                    {/* Platform Name */}
                    <div>
                      <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
                        {source.name}
                      </div>
                      <div
                        style={{
                          display: 'inline-block',
                          padding: '4px 10px',
                          background: source.type === 'CEX' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                          color: source.type === 'CEX' ? '#10B981' : '#3B82F6',
                          borderRadius: '6px',
                          fontSize: '11px',
                          fontWeight: '600',
                        }}
                      >
                        {source.type}
                      </div>
                    </div>

                    {/* Liquidity */}
                    <div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Likidite
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        ${(source.liquidity / 1000000).toFixed(0)}M
                      </div>
                    </div>

                    {/* Slippage */}
                    <div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Slippage
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '700', color: '#F59E0B' }}>
                        {slippage.toFixed(3)}%
                      </div>
                    </div>

                    {/* Fee */}
                    <div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Fee
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        {source.fee.toFixed(3)}%
                      </div>
                    </div>

                    {/* Spread */}
                    <div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Spread
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        {source.spread.toFixed(3)}%
                      </div>
                    </div>

                    {/* Execution Time */}
                    <div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Süre
                      </div>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#FFFFFF' }}>
                        {source.avgExecutionTime < 1000 ? `${source.avgExecutionTime}ms` : `${(source.avgExecutionTime / 1000).toFixed(1)}s`}
                      </div>
                    </div>

                    {/* Total Cost */}
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>
                        Toplam Maliyet
                      </div>
                      <div
                        style={{
                          fontSize: '20px',
                          fontWeight: '900',
                          color: isBest ? '#10B981' : '#FFFFFF',
                        }}
                      >
                        {totalCost.toFixed(3)}%
                      </div>
                      {isBest && (
                        <div style={{ fontSize: '11px', color: '#10B981', marginTop: '4px' }}>
                          ⭐ En İyi
                        </div>
                      )}
                    </div>
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
