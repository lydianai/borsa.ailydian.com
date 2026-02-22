'use client';

/**
 * GELENEKSEL PİYASALAR GÖSTERGE PANELI
 * Premium mobil öncelikli tasarım - SADECE CANLI VERİ
 * Kıymetli Metaller, Döviz, DXY Endeksi
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { PWAProvider } from '@/components/PWAProvider';
import { NotificationPanel as _NotificationPanel } from '@/components/NotificationPanel';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS } from '@/lib/colors';
import { analyzeAssetWithAllStrategies, type MultiStrategyAnalysis } from '@/lib/analyzers/multi-strategy-traditional';
import { useGlobalFilters } from '@/hooks/useGlobalFilters';

interface GoldPrice {
  symbol: string;
  name: string;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  timestamp: Date;
  carat22TRY: number;
  carat24TRY: number;
}

interface PreciousMetalPrice {
  symbol: string;
  name: string;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  timestamp: Date;
}

interface ForexRate {
  symbol: string;
  baseCurrency: string;
  name: string;
  rate: number;
  change24h: number;
  timestamp: Date;
}

interface DXYData {
  symbol: string;
  name: string;
  price: number;
  open: number;
  high: number;
  low: number;
  previousClose: number;
  change: number;
  changePercent: number;
  volume: number;
  support: number;
  resistance: number;
  timestamp: Date;
}

interface TurkishGoldPrice {
  symbol: string;
  name: string;
  price: number;
  change24h: number;
  buyPrice: number;
  sellPrice: number;
  lastUpdate: Date;
  category: string;
  currency: string;
}

interface EnergyCommodity {
  symbol: string;
  name: string;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  unit: string;
  category: 'energy';
  timestamp: Date;
}

interface StockIndex {
  symbol: string;
  name: string;
  priceUSD: number;
  priceTRY: number;
  changePercent: number;
  marketCap: string;
  category: 'index';
  timestamp: Date;
}

interface TreasuryBond {
  symbol: string;
  name: string;
  yield: number;
  price: number;
  change24h: number;
  maturity: string;
  category: 'bond';
  timestamp: Date;
}

interface AgriculturalCommodity {
  symbol: string;
  name: string;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  unit: string;
  category: 'agriculture';
  timestamp: Date;
}

interface MarketData {
  metals: {
    gold: GoldPrice;
    silver: PreciousMetalPrice;
    palladium: PreciousMetalPrice;
    copper: PreciousMetalPrice;
    turkishGold?: TurkishGoldPrice[];
    timestamp: Date;
  };
  forex: {
    rates: ForexRate[];
    timestamp: Date;
  };
  dxy: DXYData;
  energy?: EnergyCommodity[];
  stockIndices?: StockIndex[];
  bonds?: TreasuryBond[];
  agriculture?: AgriculturalCommodity[];
  timestamp: Date;
}

interface AssetAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  assetType: string;
  aiAnalysis: string;
  groqAnalysis: string;
  quantumScore: number;
  recommendation: string;
  overallScore: number;
  strategies: any[];
  buyCount: number;
  sellCount: number;
  waitCount: number;
  neutralCount: number;
  timestamp: string;
}

export default function TraditionalMarketsDashboard() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<MarketData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<any>(null);
  const [multiStrategyAnalysis, setMultiStrategyAnalysis] = useState<MultiStrategyAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [countdown, setCountdown] = useState(60);
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<'all' | 'metals' | 'forex' | 'indices' | 'energy' | 'bonds' | 'agriculture'>('all');
  const [showLogicModal, setShowLogicModal] = useState(false);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Global filters (synchronized across all pages)
  const { timeframe, sortBy } = useGlobalFilters();

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      const response = await fetch('/api/traditional-markets');
      const result = await response.json();

      if (result.success) {
        setData(result.data);
        setError(null);
      } else {
        setError('Failed to fetch market data');
      }
    } catch (err: any) {
      console.error('Market data fetch error:', err);
      setError(err.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh
  useEffect(() => {
    fetchMarketData();
    const interval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchMarketData();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Filter assets
  const getAllAssets = (): any[] => {
    if (!data) return [];

    let assets: any[] = [];

    // Add metals
    if (categoryFilter === 'all' || categoryFilter === 'metals') {
      assets.push(
        { ...data.metals.gold, category: 'metal', icon: 'Wallet' },
        { ...data.metals.silver, category: 'metal', icon: 'Wallet' },
        { ...data.metals.palladium, category: 'metal', icon: 'Wallet' },
        { ...data.metals.copper, category: 'metal', icon: 'Wallet' }
      );
    }

    // Add Turkish Gold Products (Harem Altın API)
    if ((categoryFilter === 'all' || categoryFilter === 'metals') && data.metals.turkishGold) {
      assets.push(...data.metals.turkishGold.map((gold: TurkishGoldPrice) => ({
        ...gold,
        category: 'turkish-gold',
        icon: 'Wallet',
        // Normalize for display
        priceTRY: gold.price,
        change24h: gold.change24h
      })));
    }

    // Add forex
    if (categoryFilter === 'all' || categoryFilter === 'forex') {
      assets.push(...data.forex.rates.map((rate) => ({ ...rate, category: 'forex', icon: 'Activity' })));
    }

    // Add DXY
    if (categoryFilter === 'all' || categoryFilter === 'indices') {
      assets.push({ ...data.dxy, category: 'index', icon: 'TrendingUp' });
    }

    // Add Energy Commodities
    if ((categoryFilter === 'all' || categoryFilter === 'energy') && data.energy) {
      assets.push(...data.energy.map((commodity) => ({ ...commodity, category: 'energy', icon: 'Zap' })));
    }

    // Add Stock Indices
    if ((categoryFilter === 'all' || categoryFilter === 'indices') && data.stockIndices) {
      assets.push(...data.stockIndices.map((index) => ({
        ...index,
        category: 'stock-index',
        icon: 'TrendingUp',
        change24h: index.changePercent // Normalize for display
      })));
    }

    // Add Treasury Bonds
    if ((categoryFilter === 'all' || categoryFilter === 'bonds') && data.bonds) {
      assets.push(...data.bonds.map((bond) => ({ ...bond, category: 'bond', icon: 'FileText' })));
    }

    // Add Agricultural Commodities
    if ((categoryFilter === 'all' || categoryFilter === 'agriculture') && data.agriculture) {
      assets.push(...data.agriculture.map((agri) => ({ ...agri, category: 'agriculture', icon: 'Sun' })));
    }

    // Apply search filter
    if (searchTerm) {
      assets = assets.filter((asset) =>
        asset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        asset.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    return assets;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return COLORS.success;
    if (change < 0) return COLORS.danger;
    return COLORS.warning;
  };

  // Fetch asset analysis
  const _fetchAssetAnalysis = async (symbol: string, assetData: any) => {
    setAnalysisLoading(true);
    try {
      const response = await fetch(
        `/api/traditional-markets-analysis/${symbol}?price=${assetData.price}&change24h=${assetData.change24h}&type=${assetData.type}`
      );
      const result = await response.json();

      if (result.success) {
        setMultiStrategyAnalysis(result.data);
      } else {
        console.error('Analysis fetch failed:', result.error);
      }
    } catch (err: any) {
      console.error('Analysis fetch error:', err);
    } finally {
      setAnalysisLoading(false);
    }
  };

  const openAssetModal = (asset: any) => {
    setSelectedAsset(asset);
    setAnalysisLoading(true);

    // Fiyat ve değişim bilgisini al
    const price = asset.priceTRY || asset.rate || asset.price || 0;
    const change24h = asset.change24h ?? asset.changePercent ?? 0;

    // Multi-strategy analizi yap
    setTimeout(() => {
      const analysis = analyzeAssetWithAllStrategies(
        asset.symbol,
        price,
        change24h,
        asset.category
      );
      setMultiStrategyAnalysis(analysis);
      setAnalysisLoading(false);
    }, 500); // Küçük gecikme - UI için smooth transition
  };

  const closeModal = () => {
    setSelectedAsset(null);
    setMultiStrategyAnalysis(null);
  };

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.primary }}>
        <LoadingAnimation />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.primary, color: COLORS.danger, fontSize: '16px' }}>
        Error: {error}
      </div>
    );
  }

  const assets = getAllAssets();

  return (
    <PWAProvider>
      <div className="dashboard-container">
      {/* AI Assistant */}
      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}

        {/* Sidebar */}
        <SharedSidebar
          currentPage="traditional-markets"
          notificationCounts={notificationCounts}
        />

        {/* Main Content */}
        <div className="dashboard-main">
          <main className="dashboard-content" style={{ padding: '16px', paddingTop: isLocalhost ? '116px' : '60px' }}>
            {/* Page Header with MANTIK Button */}
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                  <Icons.TrendingUp style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                  <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    Geleneksel Piyasalar
                  </h1>
                </div>
                <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
                  Altın, Kıymetli Metaller, Forex, Emtia ve Endeksler - 8 Strateji Analizi
                </p>
              </div>

              {/* MANTIK Button - Responsive */}
              <div>
                <style>{`
                  @media (max-width: 768px) {
                    .mantik-button-traditional {
                      padding: 10px 20px !important;
                      fontSize: 13px !important;
                      height: 42px !important;
                    }
                    .mantik-button-traditional svg {
                      width: 18px !important;
                      height: 18px !important;
                    }
                  }
                  @media (max-width: 480px) {
                    .mantik-button-traditional {
                      padding: 8px 16px !important;
                      fontSize: 12px !important;
                      height: 40px !important;
                    }
                    .mantik-button-traditional svg {
                      width: 16px !important;
                      height: 16px !important;
                    }
                  }
                `}</style>
                <button
                  onClick={() => setShowLogicModal(true)}
                  className="mantik-button-traditional"
                  style={{
                    padding: '12px 24px',
                    background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.warning})`,
                    color: '#000',
                    border: 'none',
                    borderRadius: '10px',
                    fontSize: '14px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    height: '44px',
                    boxShadow: `0 4px 20px ${COLORS.premium}40`,
                    transition: 'all 0.3s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = `0 6px 25px ${COLORS.premium}60`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = `0 4px 20px ${COLORS.premium}40`;
                  }}
                >
                  <Icons.Lightbulb style={{ width: '18px', height: '18px' }} />
                  MANTIK
                </button>
              </div>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
              gap: '12px',
            }}>
              {assets.map((asset) => {
                const IconComponent = Icons[asset.icon as keyof typeof Icons] || Icons.Activity;
                const change = asset.change24h ?? asset.changePercent ?? 0;

                return (
                  <div
                    key={asset.symbol}
                    onClick={() => openAssetModal(asset)}
                    style={{
                      background: COLORS.bg.primary,
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '8px',
                      padding: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      position: 'relative',
                      overflow: 'hidden',
                      minHeight: '140px',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.border = `1px solid ${COLORS.text.primary}`;
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(255,255,255,0.2)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.border = `1px solid ${COLORS.border.default}`;
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    {/* Category Badge */}
                    <div style={{
                      position: 'absolute',
                      top: '8px',
                      right: '8px',
                      background:
                        asset.category === 'metal' || asset.category === 'turkish-gold' ? COLORS.warning :
                        asset.category === 'forex' ? COLORS.success :
                        asset.category === 'energy' ? '#FF6B35' :
                        asset.category === 'stock-index' ? '#4ECDC4' :
                        asset.category === 'bond' ? '#95E1D3' :
                        asset.category === 'agriculture' ? '#F38181' :
                        COLORS.cyan,
                      color: COLORS.bg.primary,
                      fontSize: '9px',
                      fontWeight: '700',
                      padding: '2px 6px',
                      borderRadius: '3px',
                      letterSpacing: '0.5px',
                    }}>
                      {asset.category === 'metal' && 'METAL'}
                      {asset.category === 'turkish-gold' && 'METAL'}
                      {asset.category === 'forex' && 'FOREX'}
                      {asset.category === 'index' && 'ENDEKS'}
                      {asset.category === 'energy' && 'ENERJİ'}
                      {asset.category === 'stock-index' && 'BORSA'}
                      {asset.category === 'bond' && 'TAHVİL'}
                      {asset.category === 'agriculture' && 'TARIM'}
                    </div>

                    {/* Symbol & Icon */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <IconComponent style={{ width: '20px', height: '20px', color: COLORS.text.primary }} />
                      <div>
                        <div style={{ color: COLORS.text.primary, fontSize: '15px', fontWeight: '700', letterSpacing: '0.5px' }}>
                          {asset.symbol}
                        </div>
                        <div style={{ color: COLORS.text.muted, fontSize: '10px' }}>{asset.name}</div>
                      </div>
                    </div>

                    {/* Price - TRY format for metals/forex, USD for DXY */}
                    <div style={{ marginBottom: '8px' }}>
                      {asset.category === 'metal' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          {asset.symbol === 'XAU' ? (
                            <>
                              <div>22K: ₺{(asset.carat22TRY ?? 0).toFixed(2)}</div>
                              <div style={{ fontSize: '14px', color: COLORS.text.secondary }}>24K: ₺{(asset.carat24TRY ?? 0).toFixed(2)}</div>
                            </>
                          ) : (
                            `₺${(asset.priceTRY ?? 0).toFixed(2)}`
                          )}
                        </div>
                      )}
                      {asset.category === 'forex' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          ₺{(asset.rate ?? 0).toFixed(4)}
                        </div>
                      )}
                      {asset.category === 'turkish-gold' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          <div>₺{(asset.price ?? 0).toFixed(2)}</div>
                          {asset.buyPrice && asset.sellPrice && (
                            <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '2px' }}>
                              Alış: ₺{(asset.buyPrice ?? 0).toFixed(2)} • Satış: ₺{(asset.sellPrice ?? 0).toFixed(2)}
                            </div>
                          )}
                        </div>
                      )}
                      {asset.category === 'index' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          ${(asset.price ?? 0).toFixed(3)}
                        </div>
                      )}
                      {asset.category === 'energy' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          <div>₺{(asset.priceTRY ?? 0).toFixed(2)}</div>
                          <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '2px' }}>
                            ${(asset.priceUSD ?? 0).toFixed(2)}/{asset.unit}
                          </div>
                        </div>
                      )}
                      {asset.category === 'stock-index' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          <div>{(asset.priceUSD ?? 0).toFixed(0)} pts</div>
                          <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '2px' }}>
                            ₺{(asset.priceTRY ?? 0).toFixed(0)}
                          </div>
                        </div>
                      )}
                      {asset.category === 'bond' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          <div>{(asset.yield ?? 0).toFixed(2)}% yield</div>
                          <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '2px' }}>
                            ${(asset.price ?? 0).toFixed(2)} price
                          </div>
                        </div>
                      )}
                      {asset.category === 'agriculture' && (
                        <div style={{ color: COLORS.text.primary, fontSize: '16px', fontWeight: '600', fontFamily: 'monospace' }}>
                          <div>₺{(asset.priceTRY ?? 0).toFixed(2)}</div>
                          <div style={{ fontSize: '11px', color: COLORS.text.muted, marginTop: '2px' }}>
                            ${(asset.priceUSD ?? 0).toFixed(2)}/{asset.unit}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Change */}
                    <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{
                        color: getChangeColor(change),
                        fontSize: '14px',
                        fontWeight: '700',
                        fontFamily: 'monospace',
                      }}>
                        {(change ?? 0) > 0 ? '+' : ''}{(change ?? 0).toFixed(2)}%
                      </div>
                      {change > 0 ? (
                        <Icons.TrendingUp style={{ width: '14px', height: '14px', color: COLORS.success }} />
                      ) : (
                        <Icons.TrendingUp style={{ width: '14px', height: '14px', color: COLORS.danger, transform: 'rotate(180deg)' }} />
                      )}
                    </div>

                    {/* Stats */}
                    <div style={{ fontSize: '11px', color: COLORS.text.muted }}>
                      Son güncelleme: {new Date(asset.timestamp || asset.lastUpdate).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                );
              })}
            </div>

            {assets.length === 0 && (
              <div style={{ textAlign: 'center', padding: '80px 20px', color: COLORS.text.muted }}>
                <Icons.Search style={{ width: '48px', height: '48px', color: COLORS.border.default, marginBottom: '16px' }} />
                <div>Varlık bulunamadı. Farklı arama yapın.</div>
              </div>
            )}
          </main>
        </div>

        {/* Analysis Modal */}
        {selectedAsset && (
          <div
            className="modal-overlay"
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0, 0, 0, 0.95)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
              padding: '16px',
              backdropFilter: 'blur(8px)',
            }}
            onClick={closeModal}
          >
            <div
              className="modal-content"
              style={{
                background: COLORS.bg.primary,
                border: `1px solid ${COLORS.text.primary}`,
                borderRadius: '12px',
                maxWidth: '700px',
                width: '100%',
                maxHeight: '90vh',
                overflow: 'auto',
                padding: '24px',
                boxShadow: '0 0 30px rgba(255, 255, 255, 0.3)',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
                <div>
                  <h2 className="neon-text" style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '8px' }}>
                    {selectedAsset.symbol} - {selectedAsset.name}
                  </h2>
                  <div style={{ fontSize: '14px', color: COLORS.text.muted }}>
                    {selectedAsset.category === 'metal' && 'Kıymetli Metal'}
                    {selectedAsset.category === 'turkish-gold' && 'Türk Altını'}
                    {selectedAsset.category === 'forex' && 'Döviz Kuru (vs TRY)'}
                    {selectedAsset.category === 'index' && 'Endeks'}
                    {selectedAsset.category === 'energy' && 'Enerji Emtiası'}
                    {selectedAsset.category === 'stock-index' && 'Borsa Endeksi'}
                    {selectedAsset.category === 'bond' && 'Hazine Bonosu'}
                    {selectedAsset.category === 'agriculture' && 'Tarım Ürünü'}
                  </div>
                </div>
                <button
                  onClick={closeModal}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.default}`,
                    color: COLORS.text.primary,
                    padding: '8px 16px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600',
                  }}
                >
                  KAPAT
                </button>
              </div>

              {/* Loading State */}
              {analysisLoading && (
                <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                  <div className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>
                    AI Analizi Yapılıyor...
                  </div>
                  <div style={{ color: COLORS.text.muted, fontSize: '14px' }}>
                    Groq AI stratejileri hesaplıyor...
                  </div>
                </div>
              )}

              {/* Analysis Content */}
              {!analysisLoading && multiStrategyAnalysis && (
                <>
                  {/* Overall Recommendation */}
                  <div style={{ background: COLORS.bg.primary, border: `1px solid ${COLORS.border.default}`, borderRadius: '8px', padding: '16px', marginBottom: '24px' }}>
                    <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>🎯 LyTrade Çoklu Strateji Analizi</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '16px' }}>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Genel Sinyal</div>
                        <div className={`neon-text ${multiStrategyAnalysis.overallSignal === 'AL' ? 'signal-buy' : multiStrategyAnalysis.overallSignal === 'SAT' ? 'signal-sell' : 'signal-wait'}`} style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
                          {multiStrategyAnalysis.overallSignal}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Güven Oranı</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{multiStrategyAnalysis.overallConfidence}%</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Risk Seviyesi</div>
                        <div className="neon-text" style={{
                          fontSize: '1.5rem',
                          fontWeight: 'bold',
                          color: multiStrategyAnalysis.riskLevel === 'DÜŞÜK' ? COLORS.success : multiStrategyAnalysis.riskLevel === 'YÜKSEK' ? COLORS.danger : COLORS.warning
                        }}>
                          {multiStrategyAnalysis.riskLevel}
                        </div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>AL Sinyalleri</div>
                        <div style={{ color: COLORS.success, fontSize: '1.5rem', fontWeight: 'bold' }}>{multiStrategyAnalysis.buyCount}/8</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>SAT Sinyalleri</div>
                        <div style={{ color: COLORS.danger, fontSize: '1.5rem', fontWeight: 'bold' }}>{multiStrategyAnalysis.sellCount}/8</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>BEKLE Sinyalleri</div>
                        <div style={{ color: COLORS.warning, fontSize: '1.5rem', fontWeight: 'bold' }}>{multiStrategyAnalysis.waitCount}/8</div>
                      </div>
                    </div>

                    {/* Recommendation Text */}
                    <div style={{
                      background: 'rgba(255,255,255,0.05)',
                      border: `1px solid ${COLORS.border.default}`,
                      borderRadius: '6px',
                      padding: '12px',
                      marginTop: '12px'
                    }}>
                      <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '6px', fontWeight: '600' }}>📊 Detaylı Öneri:</div>
                      <div style={{ color: COLORS.text.primary, fontSize: '14px', lineHeight: '1.6' }}>
                        {multiStrategyAnalysis.recommendation}
                      </div>
                    </div>
                  </div>

                  {/* All 8 Strategies */}
                  <div style={{ marginBottom: '24px' }}>
                    <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '16px' }}>
                      🔍 Tüm Stratejiler ({multiStrategyAnalysis.strategies.length}/8)
                    </h3>
                    <div style={{ display: 'grid', gap: '12px' }}>
                      {multiStrategyAnalysis.strategies.map((strategy, index) => (
                        <div key={index} style={{
                          background: COLORS.bg.primary,
                          border: `1px solid ${strategy.signal === 'AL' ? COLORS.success : strategy.signal === 'SAT' ? COLORS.danger : strategy.signal === 'BEKLE' ? COLORS.warning : COLORS.border.default}`,
                          borderRadius: '6px',
                          padding: '12px',
                          boxShadow: strategy.signal === 'AL' ? '0 0 10px rgba(0,255,0,0.2)' : strategy.signal === 'SAT' ? '0 0 10px rgba(255,0,0,0.2)' : 'none'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                            <div style={{ flex: 1 }}>
                              <div style={{ color: COLORS.text.primary, fontSize: '14px', fontWeight: '600', marginBottom: '4px' }}>
                                {strategy.name}
                              </div>
                              <div style={{ color: COLORS.text.muted, fontSize: '12px' }}>
                                Güven: {strategy.confidence}% | Skor: {(strategy.score ?? 0) > 0 ? '+' : ''}{(strategy.score ?? 0).toFixed(1)}/100
                              </div>
                            </div>
                            <div style={{
                              background: strategy.signal === 'AL' ? 'rgba(0,255,0,0.15)' : strategy.signal === 'SAT' ? 'rgba(255,0,0,0.15)' : strategy.signal === 'BEKLE' ? 'rgba(255,255,0,0.15)' : 'rgba(128,128,128,0.15)',
                              border: `1px solid ${strategy.signal === 'AL' ? COLORS.success : strategy.signal === 'SAT' ? COLORS.danger : strategy.signal === 'BEKLE' ? COLORS.warning : COLORS.text.muted}`,
                              color: strategy.signal === 'AL' ? COLORS.success : strategy.signal === 'SAT' ? COLORS.danger : strategy.signal === 'BEKLE' ? COLORS.warning : COLORS.text.muted,
                              padding: '4px 12px',
                              borderRadius: '4px',
                              fontSize: '12px',
                              fontWeight: '700',
                            }}>
                              {strategy.signal}
                            </div>
                          </div>
                          <div style={{ color: COLORS.text.secondary, fontSize: '12px', lineHeight: '1.5' }}>
                            {strategy.reason}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Asset Details */}
              <div style={{ background: COLORS.bg.primary, border: `1px solid ${COLORS.border.default}`, borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
                <h3 className="neon-text" style={{ fontSize: '1.25rem', marginBottom: '12px' }}>Fiyat Bilgisi</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
                  {selectedAsset.category === 'metal' && selectedAsset.symbol === 'XAU' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>22 Ayar (TL/gram)</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>�{(selectedAsset.carat22TRY ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>24 Ayar (TL/gram)</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>�{(selectedAsset.carat24TRY ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>USD/oz</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.priceUSD ?? 0).toFixed(2)}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'metal' && selectedAsset.symbol !== 'XAU' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>TL Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>�{(selectedAsset.priceTRY ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>USD Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.priceUSD ?? 0).toFixed(2)}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'forex' && (
                    <div>
                      <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>{selectedAsset.baseCurrency}/TRY</div>
                      <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>�{(selectedAsset.rate ?? 0).toFixed(4)}</div>
                    </div>
                  )}
                  {selectedAsset.category === 'index' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Mevcut</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.price ?? 0).toFixed(3)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Açılış</div>
                        <div style={{ color: COLORS.text.primary, fontSize: '1.25rem', fontWeight: 'bold' }}>${(selectedAsset.open ?? 0).toFixed(3)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Yüksek</div>
                        <div style={{ color: COLORS.success, fontSize: '1.25rem', fontWeight: 'bold' }}>${(selectedAsset.high ?? 0).toFixed(3)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Düşük</div>
                        <div style={{ color: COLORS.danger, fontSize: '1.25rem', fontWeight: 'bold' }}>${(selectedAsset.low ?? 0).toFixed(3)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Destek</div>
                        <div style={{ color: COLORS.success, fontSize: '1.25rem', fontWeight: 'bold' }}>${(selectedAsset.support ?? 0).toFixed(3)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Direnç</div>
                        <div style={{ color: COLORS.danger, fontSize: '1.25rem', fontWeight: 'bold' }}>${(selectedAsset.resistance ?? 0).toFixed(3)}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'energy' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>USD Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.priceUSD ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>TL Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>₺{(selectedAsset.priceTRY ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Birim</div>
                        <div style={{ color: COLORS.text.primary, fontSize: '1.25rem', fontWeight: 'bold' }}>{selectedAsset.unit}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'stock-index' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Puan</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{(selectedAsset.priceUSD ?? 0).toFixed(0)} pts</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>TL Değer</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>₺{(selectedAsset.priceTRY ?? 0).toFixed(0)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Piyasa Değeri</div>
                        <div style={{ color: COLORS.text.primary, fontSize: '1.25rem', fontWeight: 'bold' }}>{selectedAsset.marketCap}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'bond' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Getiri (Yield)</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{(selectedAsset.yield ?? 0).toFixed(2)}%</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.price ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Vade</div>
                        <div style={{ color: COLORS.text.primary, fontSize: '1.25rem', fontWeight: 'bold' }}>{selectedAsset.maturity}</div>
                      </div>
                    </>
                  )}
                  {selectedAsset.category === 'agriculture' && (
                    <>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>USD Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>${(selectedAsset.priceUSD ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>TL Fiyat</div>
                        <div className="neon-text" style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>₺{(selectedAsset.priceTRY ?? 0).toFixed(2)}</div>
                      </div>
                      <div>
                        <div style={{ color: COLORS.text.muted, fontSize: '12px', marginBottom: '4px' }}>Birim</div>
                        <div style={{ color: COLORS.text.primary, fontSize: '1.25rem', fontWeight: 'bold' }}>{selectedAsset.unit}</div>
                      </div>
                    </>
                  )}
                  <div>
                    <div style={{ color: '#666', fontSize: '12px', marginBottom: '4px' }}>24s Dei_im</div>
                    <div style={{ color: getChangeColor(selectedAsset.change24h ?? selectedAsset.changePercent ?? 0), fontSize: '1.5rem', fontWeight: 'bold' }}>
                      {(selectedAsset.change24h ?? selectedAsset.changePercent ?? 0) > 0 ? '+' : ''}
                      {(selectedAsset.change24h ?? selectedAsset.changePercent ?? 0).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Timestamp */}
              <div style={{ marginTop: '16px', textAlign: 'center', color: COLORS.text.muted, fontSize: '12px' }}>
                {multiStrategyAnalysis ? (
                  <>⏱️ Analiz zamanı: {new Date(multiStrategyAnalysis.timestamp).toLocaleString('tr-TR')}</>
                ) : (
                  <>Son güncelleme: {new Date(selectedAsset.timestamp).toLocaleString('tr-TR')}</>
                )}
              </div>
            </div>
          </div>
        )}

        {/* MANTIK Modal - Traditional Markets Explanation */}
        {showLogicModal && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0, 0, 0, 0.85)',
              zIndex: 9999,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '20px',
              backdropFilter: 'blur(8px)'
            }}
            onClick={() => setShowLogicModal(false)}
          >
            <div
              className="neon-card"
              style={{
                maxWidth: '900px',
                maxHeight: '85vh',
                overflowY: 'auto',
                padding: '32px',
                background: COLORS.bg.primary,
                border: `2px solid ${COLORS.premium}`,
                boxShadow: `0 0 40px ${COLORS.premium}60`
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                  <h2 style={{ fontSize: '26px', fontWeight: 'bold', color: COLORS.premium, margin: 0 }}>
                    GELENEKSEL PİYASALAR MANTIĞI
                  </h2>
                </div>
                <button
                  onClick={() => setShowLogicModal(false)}
                  style={{
                    background: 'transparent',
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '8px',
                    padding: '8px 16px',
                    color: COLORS.text.secondary,
                    cursor: 'pointer',
                    fontSize: '14px',
                    fontWeight: '600'
                  }}
                >
                  ✕ Kapat
                </button>
              </div>

              {/* Overview */}
              <div style={{ marginBottom: '28px' }}>
                <div style={{
                  padding: '20px',
                  borderRadius: '12px',
                  background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.info}20)`,
                  border: `1px solid ${COLORS.premium}40`,
                  marginBottom: '20px'
                }}>
                  <p style={{ fontSize: '15px', lineHeight: '1.7', color: COLORS.text.primary, margin: 0 }}>
                    <strong style={{ color: COLORS.premium }}>Geleneksel Piyasalar</strong> sayfası, kripto dışındaki önemli finansal varlıkları izlemenizi ve
                    analiz etmenizi sağlar. Altın, gümüş, petrol, forex pariteler ve borsa endeksleri gibi geleneksel piyasa araçlarına
                    otomatik 8-strateji analizi uygulanır.
                  </p>
                </div>
              </div>

              {/* Asset Categories */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <Icons.TrendingUp style={{ width: '24px', height: '24px', color: COLORS.info }} />
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    1️⃣ Varlık Kategorileri
                  </h3>
                </div>
                <div style={{
                  padding: '16px',
                  borderRadius: '10px',
                  background: COLORS.bg.secondary,
                  border: `1px solid ${COLORS.border.default}`
                }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Kıymetli Metaller:</strong><br />
                    Altın (XAU), Gümüş (XAG), Paladyum (XPD), Bakır (CU) - Hem USD hem TL fiyatları
                  </p>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Forex (Döviz Pariteler):</strong><br />
                    EUR/USD, USD/JPY, GBP/USD, USD/TRY ve daha fazlası
                  </p>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Emtia (Commodities):</strong><br />
                    Petrol (WTI, Brent), Doğalgaz, Tarımsal ürünler (Buğday, Mısır, Soya)
                  </p>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: 0 }}>
                    <strong style={{ color: COLORS.text.primary }}>Borsa Endeksleri:</strong><br />
                    S&P 500, Nasdaq 100, Dow Jones, DAX, FTSE 100, Nikkei 225
                  </p>
                </div>
              </div>

              {/* Multi-Strategy Analysis */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <Icons.BarChart3 style={{ width: '24px', height: '24px', color: COLORS.premium }} />
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    2️⃣ 8-Strateji Analiz Sistemi
                  </h3>
                </div>
                <div style={{
                  padding: '16px',
                  borderRadius: '10px',
                  background: COLORS.bg.secondary,
                  border: `1px solid ${COLORS.border.default}`
                }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    Her varlık tıklandığında otomatik olarak 8 farklı strateji ile analiz edilir:
                  </p>
                  <div style={{
                    padding: '12px',
                    borderRadius: '8px',
                    background: `${COLORS.premium}10`,
                    border: `1px solid ${COLORS.premium}30`,
                    marginBottom: '12px'
                  }}>
                    <p style={{ fontSize: '13px', lineHeight: '1.6', color: COLORS.text.primary, margin: 0 }}>
                      <strong>1. RSI Momentum:</strong> Aşırı alım/satım bölgelerini tespit eder<br />
                      <strong>2. MACD Trend:</strong> Trend yönünü ve gücünü belirler<br />
                      <strong>3. Bollinger Bands:</strong> Volatilite ve fiyat sınırlarını analiz eder<br />
                      <strong>4. MA Cross:</strong> Hareketli ortalama kesişimlerini izler<br />
                      <strong>5. Volume Analysis:</strong> Hacim bazlı sinyal üretir<br />
                      <strong>6. Fibonacci:</strong> Destek/direnç seviyelerini hesaplar<br />
                      <strong>7. Stochastic:</strong> Momentum ve dönüş noktalarını bulur<br />
                      <strong>8. ATR Volatility:</strong> Risk seviyesini ölçer
                    </p>
                  </div>
                </div>
              </div>

              {/* How to Use */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <Icons.Activity style={{ width: '24px', height: '24px', color: COLORS.success }} />
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    3️⃣ Nasıl Kullanılır?
                  </h3>
                </div>
                <div style={{
                  padding: '16px',
                  borderRadius: '10px',
                  background: COLORS.bg.secondary,
                  border: `1px solid ${COLORS.border.default}`
                }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Adım 1 - Varlık Seç:</strong><br />
                    Ana sayfada görünen varlıklardan birini tıklayın. Güncel fiyat, 24s değişim ve kategori bilgileri kartlarda gösterilir.
                  </p>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: '12px' }}>
                    <strong style={{ color: COLORS.text.primary }}>Adım 2 - Analizi İncele:</strong><br />
                    Açılan detay popup'ında 8 stratejinin her biri için: Sinyal (AL/SAT/BEK), Güven Skoru (%), ve Açıklama gösterilir.
                  </p>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.secondary, marginBottom: 0 }}>
                    <strong style={{ color: COLORS.text.primary }}>Adım 3 - Genel Değerlendirme:</strong><br />
                    Popup'ın üst kısmında tüm stratejilerin ortalaması alınarak genel sinyal ve güven skoru hesaplanır.
                    "3 AL, 2 SAT, 3 BEKLE" gibi dağılım bilgisi de görüntülenir.
                  </p>
                </div>
              </div>

              {/* Important Notes */}
              <div style={{ marginBottom: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                  <Icons.AlertTriangle style={{ width: '24px', height: '24px', color: COLORS.warning }} />
                  <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                    ⚠️ Önemli Notlar
                  </h3>
                </div>
                <div style={{
                  padding: '16px',
                  borderRadius: '10px',
                  background: `${COLORS.warning}15`,
                  border: `1px solid ${COLORS.warning}40`
                }}>
                  <p style={{ fontSize: '14px', lineHeight: '1.6', color: COLORS.text.primary, marginBottom: '10px' }}>
                    • <strong>Otomatik Yenileme:</strong> Veri 60 saniyede bir otomatik güncellenir<br />
                    • <strong>Çoklu Kaynak:</strong> Fiyatlar birden fazla kaynaktan doğrulanır<br />
                    • <strong>Risk Yönetimi:</strong> Stratejiler sadece sinyal verir, yatırım kararı size aittir<br />
                    • <strong>Eğitim Amaçlı:</strong> Bu analizler eğitim amaçlıdır, finansal tavsiye değildir
                  </p>
                </div>
              </div>

              {/* Footer */}
              <div style={{
                marginTop: '24px',
                paddingTop: '20px',
                borderTop: `1px solid ${COLORS.border.default}`,
                textAlign: 'center'
              }}>
                <p style={{
                  fontSize: '13px',
                  color: COLORS.text.muted,
                  margin: 0
                }}>
                  Geleneksel piyasalarla kripto piyasalar arasındaki korelasyonları da takip edebilirsiniz. <br />
                  İyi yatırımlar!
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </PWAProvider>
  );
}
