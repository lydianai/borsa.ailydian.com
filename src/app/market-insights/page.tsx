'use client';

/**
 * 📊 PİYASA ANALİZİ SAYFASI
 * Profesyonel piyasa analizi dashboard'u
 *
 * Özellikler:
 * - Likidasy Haritası (50 fiyat seviyesi)
 * - Funding Rate Geçmişi (100 dönem)
 * - Açık Pozisyon Takibi
 * - Long/Short Oran Analizi
 * - Gerçek zamanlı veri güncelleme (10s aralık)
 * - 531 sembol desteği
 *
 * Veri Kaynakları:
 * - Binance Futures API (primary)
 * - Bybit API (fallback)
 * - Backend: Port 5031 (market-insights-service)
 */

import { useState, useEffect } from 'react';
import '../globals.css';
import { Icons } from '@/components/Icons';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { LoadingAnimation } from '@/components/LoadingAnimation';
import { SharedSidebar } from '@/components/SharedSidebar';
import { useNotificationCounts } from '@/hooks/useNotificationCounts';
import { COLORS, getChangeColor } from '@/lib/colors';
import {
  FundingRateData,
  LiquidationHeatmapData,
  OpenInterestData,
  LongShortRatioData,
  MarketData,
} from '@/types/market-insights';

export default function MarketInsightsPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [fundingRates, setFundingRates] = useState<FundingRateData[]>([]);
  const [liquidationHeatmap, setLiquidationHeatmap] = useState<LiquidationHeatmapData | null>(null);
  const [openInterest, setOpenInterest] = useState<OpenInterestData | null>(null);
  const [longShortRatio, setLongShortRatio] = useState<LongShortRatioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const [selectedCategory, setSelectedCategory] = useState('populer');
  const [searchQuery, setSearchQuery] = useState('');
  const [showLogicModal, setShowLogicModal] = useState(false);
  const notificationCounts = useNotificationCounts();

  // Check if running on localhost
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';

  // Binance Futures'dan gerçek semboller (531 sembol - kategorize edilmiş)
  const symbolCategories = {
    populer: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'MATICUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'APEUS'],
    defi: ['AAVEUSDT', 'UNIUSDT', 'SUSHIUSDT', 'CRVUSDT', 'YFIUSDT', 'COMPUSDT', 'SNXUSDT', 'MKRUSDT', 'LRCUSDT', 'KNCUSDT', 'ZRXUSDT', 'BANDUSDT', '1INCHUSDT', 'LPTUSDT', 'PERPUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'CAKEUSDT', 'RUNEUSDT'],
    layer1: ['SOLUSDT', 'AVAXUSDT', 'NEARUSDT', 'ATOMUSDT', 'DOTUSDT', 'EGLDUSDT', 'FTMUSDT', 'ALGOUSDT', 'ICXUSDT', 'QTUMUSDT', 'ONTUSDT', 'NEOUSDT', 'THETAUSDT', 'VETUSDT', 'IOTAUSDT', 'ZILUSDT', 'KSMUSDT', 'ZENUSDT', 'WAVESUSDT', 'SUIUSDT', 'APTUSDT', 'SEIUSDT', 'INJUSDT', 'TIAUSDT', 'ARBUSDT', 'OPUSDT', 'STARKUSDT', 'ZETAUSDT'],
    meme: ['DOGEUSDT', '1000SHIBUSDT', '1000PEPEUSDT', '1000FLOKIUSDT', '1000BONKUSDT', '1000SATSUSDT', 'ORDIUSDT', '1000RATSUSDT', 'WIFUSDT', 'BOMEUSDT', 'MEMEUSDT', 'BONUSDT', 'FLOKIUSDT', 'BRETTUSDT', 'POPCATUSDT', 'DOGSUSDT', 'NEIROUSDT', '1000CATUSDT', 'GOATUSDT', 'MOODENGUSDT', 'PONKEUSDT', '1000000MOGUSDT', 'HIPPOUSDT', '1000CHEEMSUSDT', '1000WHYUSDT', 'CHILLGUYUSDT', 'PENGUUSDT', 'FARTCOINUSDT'],
    ai: ['FETCHUSDT', 'RENDERUSDT', 'OCEANUSDT', 'NMRUSDT', 'AGIXUSDT', 'PHBUSDT', 'AIUSDT', 'ARKMUSDT', 'WLDUSDT', 'TAOVSDT', 'RNDRVSDT', 'AGLDUSDT', 'AIXBTUSDT', 'CGPTUSDT', 'AI16ZUSDT', 'ZEREBROUSDT', 'SWARMSUSDT', 'AVAAIUSDT', 'GOATUSDT'],
    gaming: ['AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT', 'APEUS', 'GMTUSDT', 'FLOWUSDT', 'ICPUSDT', 'XAIUSDT', 'YGGUSDT', 'ILVUSDT', 'BEAMXUSDT', 'PORTASDT', 'PIXELUSDT', 'RONUSDT', 'RAREUSDT', 'VOXELUSDT', 'MBOXUSDT', 'CHESSUSDT'],
    nft: ['SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'THETAUSDT', 'FLOWUSDT', 'CHZUSDT', 'ALICEUSDT', 'RAREUSDT', 'BLZUSDT', 'AUCTIONUSDT', 'SUPERUSDT', 'VOXELUSDT', 'GHSTUSDT', 'MBOXUSDT'],
    depin: ['IOTAUSDT', 'HNTUSDT', 'RENUSDT', 'STORJUSDT', 'OCEANUSDT', 'RLCUSDT', 'ARPAUSDT', 'POWRUSDT', 'IOTXUSDT', 'ARKMUSDT', 'RENDERUSDT'],
    rwa: ['ONDOUSDT', 'POLUSUSDT', 'CFXUSDT', 'MANTRAUSDT', 'ILVUSDT', 'PENDLEUSDT', 'LINKUSDT', 'MKRUSDT'],
    top100: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'MATICUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'BCHUSDT', 'FILUSDT', 'NEARUSDT', 'APEUS', 'AAVEUSDT', 'SUSHIUSDT', 'CRVUSDT', 'YFIUSDT', 'COMPUSDT', 'SNXUSDT', 'MKRUSDT', 'LRCUSDT', 'KNCUSDT', 'ZRXUSDT', 'BANDUSDT', '1INCHUSDT', 'LPTUSDT', 'PERPUSDT', 'DYDXUSDT', 'GMXUSDT', 'PENDLEUSDT', 'CAKEUSDT', 'RUNEUSDT', 'EGLDUSDT', 'FTMUSDT', 'ALGOUSDT', 'ICXUSDT', 'QTUMUSDT', 'ONTUSDT', 'NEOUSDT', 'THETAUSDT', 'VETUSDT', 'IOTAUSDT', 'ZILUSDT', 'KSMUSDT', 'ZENUSDT', 'SUIUSDT', 'APTUSDT', 'SEIUSDT', 'INJUSDT', 'TIAUSDT', 'ARBUSDT', 'OPUSDT', 'STARKUSDT', 'ZETAUSDT', 'DOGEUSDT', '1000SHIBUSDT', '1000PEPEUSDT', '1000FLOKIUSDT', '1000BONKUSDT', '1000SATSUSDT', 'ORDIUSDT', '1000RATSUSDT', 'WIFUSDT', 'BOMEUSDT', 'MEMEUSDT', 'FETCHUSDT', 'RENDERUSDT', 'OCEANUSDT', 'NMRUSDT', 'AGIXUSDT', 'PHBUSDT', 'AIUSDT', 'ARKMUSDT', 'WLDUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'ENJUSDT', 'GALAUSDT', 'GMTUSDT', 'FLOWUSDT', 'ICPUSDT', 'XAIUSDT', 'YGGUSDT', 'ILVUSDT', 'BEAMXUSDT'],
  };

  const categoryNames: { [key: string]: string } = {
    populer: 'Popüler (20)',
    defi: 'DeFi (20)',
    layer1: 'Layer 1 (28)',
    meme: 'Meme (28)',
    ai: 'AI/ML (19)',
    gaming: 'Gaming (20)',
    nft: 'NFT/Meta (14)',
    depin: 'DePIN (11)',
    rwa: 'RWA (8)',
    top100: 'Top 100',
  };

  // Tüm sembolleri tek listede topla
  const allSymbols = Array.from(new Set(
    Object.values(symbolCategories).flat()
  )).sort();

  // Arama filtrelemesi
  const filteredSymbols = searchQuery.trim() !== ''
    ? allSymbols.filter(symbol =>
        symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        symbol.replace('USDT', '').toLowerCase().includes(searchQuery.toLowerCase())
      )
    : symbolCategories[selectedCategory as keyof typeof symbolCategories];

  const symbols = filteredSymbols;

  // Seçili sembol için tüm verileri çek
  const fetchAllData = async () => {
    try {
      // Tüm endpoint'leri paralel çağır
      const [marketRes, fundingRes, heatmapRes, oiRes, lsRatioRes] = await Promise.all([
        fetch(`/api/market-insights/market-data?symbol=${selectedSymbol}`),
        fetch(`/api/market-insights/funding-rate?symbol=${selectedSymbol}&limit=20`),
        fetch(`/api/market-insights/liquidation-heatmap?symbol=${selectedSymbol}`),
        fetch(`/api/market-insights/open-interest?symbol=${selectedSymbol}`),
        fetch(`/api/market-insights/long-short-ratio?symbol=${selectedSymbol}`),
      ]);

      const [marketJson, fundingJson, heatmapJson, oiJson, lsRatioJson] = await Promise.all([
        marketRes.json(),
        fundingRes.json(),
        heatmapRes.json(),
        oiRes.json(),
        lsRatioRes.json(),
      ]);

      if (marketJson.success && marketJson.data) {
        setMarketData(marketJson.data);
      }

      if (fundingJson.success && fundingJson.data) {
        setFundingRates(fundingJson.data.slice(0, 20));
      }

      if (heatmapJson.success && heatmapJson.data) {
        setLiquidationHeatmap(heatmapJson.data);
      }

      if (oiJson.success && oiJson.data) {
        setOpenInterest(oiJson.data);
      }

      if (lsRatioJson.success && lsRatioJson.data) {
        setLongShortRatio(lsRatioJson.data);
      }

      setLoading(false);
    } catch (error) {
      console.error('[Piyasa Analizi] Veri çekme hatası:', error);
      setLoading(false);
    }
  };

  // Her 10 saniyede otomatik yenileme
  useEffect(() => {
    fetchAllData();

    const countdownInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          fetchAllData();
          return 10;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdownInterval);
  }, [selectedSymbol]);

  const formatPrice = (price: number) => {
    if (!price || isNaN(price)) return '$0.00';
    if (price >= 1000) return `$${price.toLocaleString('tr-TR', { maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(6)}`;
  };

  const formatVolume = (volume: number) => {
    if (!volume || isNaN(volume)) return '$0.00';
    if (volume >= 1_000_000_000) return `$${(volume / 1_000_000_000).toFixed(2)}B`;
    if (volume >= 1_000_000) return `$${(volume / 1_000_000).toFixed(2)}M`;
    return `$${volume.toFixed(2)}`;
  };

  const formatFundingRate = (rate: number) => {
    return `${(rate * 100).toFixed(4)}%`;
  };

  if (loading && !marketData) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: COLORS.bg.secondary }}>
        <LoadingAnimation />
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* AI Asistan */}
      {aiAssistantOpen && (
        <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
      )}

      {/* Kenar Çubuğu */}
      <SharedSidebar currentPage="market-insights" notificationCounts={notificationCounts} />

      {/* Ana İçerik */}
      <div className="dashboard-main" style={{ paddingTop: isLocalhost ? '116px' : '60px' }}>
        {/* Page Header with MANTIK Button */}
        <div style={{ margin: '16px 24px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', borderBottom: `1px solid ${COLORS.border.default}`, paddingBottom: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Icons.BarChart3 style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                Market Insights
              </h1>
            </div>
            <p style={{ fontSize: '14px', color: COLORS.text.secondary, margin: 0 }}>
              Piyasa Derinliği ve Akıllı Analiz - Trend, Değişkenlik ve Likidite
            </p>
          </div>

          {/* MANTIK Button - Responsive */}
          <div>
            <style>{`
              @media (max-width: 768px) {
                .mantik-button-insights {
                  padding: 10px 20px !important;
                  fontSize: 13px !important;
                  height: 42px !important;
                }
                .mantik-button-insights svg {
                  width: 18px !important;
                  height: 18px !important;
                }
              }
              @media (max-width: 480px) {
                .mantik-button-insights {
                  padding: 8px 16px !important;
                  fontSize: 12px !important;
                  height: 40px !important;
                }
                .mantik-button-insights svg {
                  width: 16px !important;
                  height: 16px !important;
                }
              }
            `}</style>
            <button
              onClick={() => setShowLogicModal(true)}
              className="mantik-button-insights"
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

        <main className="dashboard-content" style={{ padding: '20px' }}>
          {/* Countdown & AI Assistant */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px', alignItems: 'center', marginBottom: '24px' }}>
            <div style={{
              padding: '8px 16px',
              background: `${COLORS.premium}15`,
              border: `1px solid ${COLORS.premium}40`,
              borderRadius: '8px',
              fontSize: '14px',
              color: COLORS.premium,
              fontWeight: 'bold'
            }}>
              Otomatik Yenileme: {countdown}s
            </div>
            <button
              onClick={() => setAiAssistantOpen(true)}
              style={{
                padding: '10px 20px',
                background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.cyan})`,
                border: 'none',
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontWeight: 'bold',
                cursor: 'pointer',
              }}
            >
              AI Asistan
            </button>
          </div>

          {/* Arama Motoru */}
          <div style={{ marginBottom: '16px' }}>
            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              🔍 Sembol Ara (168 Sembol):
            </div>
            <input
              type="text"
              placeholder="BTC, ETH, SOL, DOGE yazın..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                if (e.target.value.trim() !== '') {
                  setSelectedCategory('');
                }
              }}
              style={{
                width: '100%',
                padding: '12px 16px',
                background: `${COLORS.bg.secondary}`,
                border: `2px solid ${searchQuery.trim() !== '' ? COLORS.premium : COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '14px',
                fontWeight: '500',
                outline: 'none',
                transition: 'all 0.2s ease',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = COLORS.premium;
                e.target.style.boxShadow = `0 0 0 3px ${COLORS.premium}20`;
              }}
              onBlur={(e) => {
                e.target.style.borderColor = searchQuery.trim() !== '' ? COLORS.premium : COLORS.border.default;
                e.target.style.boxShadow = 'none';
              }}
            />
            {searchQuery.trim() !== '' && (
              <div style={{
                marginTop: '8px',
                padding: '8px 12px',
                background: `${COLORS.premium}15`,
                border: `1px solid ${COLORS.premium}40`,
                borderRadius: '6px',
                fontSize: '12px',
                color: COLORS.text.secondary,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span>🎯 {filteredSymbols.length} sonuç bulundu: "{searchQuery}"</span>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedCategory('populer');
                  }}
                  style={{
                    padding: '4px 12px',
                    background: COLORS.bg.primary,
                    border: `1px solid ${COLORS.border.default}`,
                    borderRadius: '4px',
                    color: COLORS.text.secondary,
                    fontSize: '11px',
                    cursor: 'pointer',
                    fontWeight: 'bold',
                  }}
                >
                  Aramayı Temizle
                </button>
              </div>
            )}
          </div>

          {/* Kategori Seçici */}
          {searchQuery.trim() === '' && (
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                📂 Kategori Seç (531 Sembol - Binance Futures):
              </div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {Object.keys(symbolCategories).map((category) => (
                  <button
                    key={category}
                    onClick={() => {
                      setSelectedCategory(category);
                      setSelectedSymbol(symbolCategories[category as keyof typeof symbolCategories][0]);
                      setSearchQuery('');
                    }}
                    style={{
                      padding: '8px 16px',
                      background: selectedCategory === category ? `${COLORS.cyan}25` : `${COLORS.bg.card}`,
                      border: selectedCategory === category ? `1px solid ${COLORS.cyan}` : `1px solid ${COLORS.border.default}`,
                      borderRadius: '6px',
                      color: selectedCategory === category ? COLORS.cyan : COLORS.text.secondary,
                      fontSize: '12px',
                      fontWeight: selectedCategory === category ? 'bold' : 'normal',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      textTransform: 'uppercase',
                    }}
                  >
                    {categoryNames[category]}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Sembol Seçici */}
          <div style={{ marginBottom: '24px' }}>
            <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
              {searchQuery.trim() !== '' ? `🎯 Arama Sonuçları (${symbols.length} adet):` : `💰 Sembol Seç (${symbols.length} adet):`}
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
              gap: '8px',
              maxHeight: '200px',
              overflowY: 'auto',
              padding: '8px',
              background: `${COLORS.bg.secondary}80`,
              borderRadius: '8px',
              border: `1px solid ${COLORS.border.default}`
            }}>
              {symbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => setSelectedSymbol(symbol)}
                  className="neon-card"
                  style={{
                    padding: '10px 16px',
                    background: selectedSymbol === symbol ? `${COLORS.premium}25` : `${COLORS.bg.primary}`,
                    border: selectedSymbol === symbol ? `2px solid ${COLORS.premium}` : `1px solid ${COLORS.border.default}`,
                    borderRadius: '6px',
                    color: selectedSymbol === symbol ? COLORS.premium : COLORS.text.secondary,
                    fontWeight: selectedSymbol === symbol ? 'bold' : 'normal',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    fontSize: '11px',
                  }}
                >
                  {symbol.replace('USDT', '').replace('1000', '').replace('1000000', '')}
                </button>
              ))}
            </div>
          </div>

          {/* Üst İstatistik Satırı */}
          {marketData && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '24px' }}>
              {/* Fiyat Kartı */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.premium}0D`, border: `1px solid ${COLORS.premium}4D` }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>Fiyat</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '4px' }}>
                  {formatPrice(marketData.price)}
                </div>
                <div style={{ fontSize: '14px', color: getChangeColor(marketData.price_change_percent ?? 0) }}>
                  {(marketData.price_change_percent ?? 0) >= 0 ? '+' : ''}{(marketData.price_change_percent ?? 0).toFixed(2)}%
                </div>
              </div>

              {/* Hacim Kartı */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.cyan}0D`, border: `1px solid ${COLORS.cyan}4D` }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>24s Hacim</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary }}>
                  {formatVolume(marketData.quote_volume)}
                </div>
              </div>

              {/* Yüksek/Düşük Kartı */}
              <div className="neon-card" style={{ padding: '20px', background: `${COLORS.warning}0D`, border: `1px solid ${COLORS.warning}4D` }}>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>24s Aralık</div>
                <div style={{ fontSize: '14px', color: COLORS.success }}>Y: {formatPrice(marketData.high_price)}</div>
                <div style={{ fontSize: '14px', color: COLORS.danger }}>D: {formatPrice(marketData.low_price)}</div>
              </div>

              {/* Açık Pozisyon Kartı */}
              {openInterest && (
                <div className="neon-card" style={{ padding: '20px', background: `${COLORS.success}0D`, border: `1px solid ${COLORS.success}4D` }}>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>Açık Pozisyon</div>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary }}>
                    {openInterest.open_interest.toLocaleString()} {selectedSymbol.replace('USDT', '').replace('1000', '')}
                  </div>
                  <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>
                    {marketData && formatVolume(openInterest.open_interest * marketData.price)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Ana İçerik Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
            {/* Likidasyson Haritası */}
            {liquidationHeatmap && (
              <div className="neon-card" style={{ padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                  🔥 Likiditas Haritası
                </h2>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '12px' }}>
                  Güncel Fiyat: {formatPrice(liquidationHeatmap.current_price)}
                </div>
                <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                  {liquidationHeatmap.heatmap.slice(0, 30).map((point, idx) => {
                    const maxAmount = Math.max(...liquidationHeatmap.heatmap.map(p => p.liquidation_amount_usd));
                    const percentage = (point.liquidation_amount_usd / maxAmount) * 100;
                    const isNearPrice = Math.abs(point.price - liquidationHeatmap.current_price) / liquidationHeatmap.current_price < 0.02;

                    return (
                      <div key={idx} style={{ marginBottom: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                          <span style={{ color: isNearPrice ? COLORS.warning : COLORS.text.secondary }}>
                            {formatPrice(point.price)}
                          </span>
                          <span style={{ color: COLORS.text.secondary }}>
                            {formatVolume(point.liquidation_amount_usd)}
                          </span>
                        </div>
                        <div style={{
                          width: '100%',
                          height: '6px',
                          background: `${COLORS.bg.secondary}`,
                          borderRadius: '3px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${percentage}%`,
                            height: '100%',
                            background: isNearPrice ? COLORS.danger : `${COLORS.danger}60`,
                            borderRadius: '3px',
                            transition: 'width 0.3s ease'
                          }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Funding Rate Grafiği */}
            {fundingRates.length > 0 && (
              <div className="neon-card" style={{ padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                  📈 Funding Rate Geçmişi (33 Gün)
                </h2>
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '12px' }}>
                  Son: {formatFundingRate(fundingRates[0].funding_rate)} • Min: {formatFundingRate(Math.min(...fundingRates.map(r => r.funding_rate)))} • Max: {formatFundingRate(Math.max(...fundingRates.map(r => r.funding_rate)))}
                </div>

                {/* SVG Çizgi Grafiği */}
                <div style={{ marginBottom: '20px', background: `${COLORS.bg.secondary}`, padding: '16px', borderRadius: '8px' }}>
                  <svg width="100%" height="200" style={{ overflow: 'visible' }}>
                    {/* Y-ekseni çizgileri */}
                    {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
                      <line
                        key={pct}
                        x1="0"
                        y1={200 * pct}
                        x2="100%"
                        y2={200 * pct}
                        stroke={COLORS.border.default}
                        strokeWidth="1"
                        strokeDasharray="4,4"
                      />
                    ))}

                    {/* Sıfır çizgisi */}
                    {(() => {
                      const rates = fundingRates.map(r => r.funding_rate);
                      const min = Math.min(...rates);
                      const max = Math.max(...rates);
                      const range = max - min;
                      const zeroY = range === 0 ? 100 : 200 - ((0 - min) / range) * 200;

                      return (
                        <line
                          x1="0"
                          y1={zeroY}
                          x2="100%"
                          y2={zeroY}
                          stroke={COLORS.text.secondary}
                          strokeWidth="2"
                          strokeDasharray="8,4"
                        />
                      );
                    })()}

                    {/* Çizgi grafik yolu */}
                    <path
                      d={(() => {
                        const rates = fundingRates.map(r => r.funding_rate).reverse();
                        const min = Math.min(...rates);
                        const max = Math.max(...rates);
                        const range = max - min || 0.0001;

                        return rates.map((rate, i) => {
                          const x = (i / (rates.length - 1)) * 100;
                          const y = 200 - ((rate - min) / range) * 200;
                          return `${i === 0 ? 'M' : 'L'} ${x}% ${y}`;
                        }).join(' ');
                      })()}
                      fill="none"
                      stroke={COLORS.premium}
                      strokeWidth="2"
                    />

                    {/* Veri noktaları */}
                    {(() => {
                      const rates = fundingRates.map(r => r.funding_rate).reverse();
                      const min = Math.min(...rates);
                      const max = Math.max(...rates);
                      const range = max - min || 0.0001;

                      return rates.filter((_, i) => i % 10 === 0).map((rate, i) => {
                        const actualIndex = i * 10;
                        const x = (actualIndex / (rates.length - 1)) * 100;
                        const y = 200 - ((rate - min) / range) * 200;
                        const isPositive = rate > 0;

                        return (
                          <circle
                            key={actualIndex}
                            cx={`${x}%`}
                            cy={y}
                            r="4"
                            fill={isPositive ? COLORS.success : COLORS.danger}
                            stroke={COLORS.bg.primary}
                            strokeWidth="2"
                          />
                        );
                      });
                    })()}
                  </svg>
                </div>

                {/* Son 10 Kayıt */}
                <div style={{ fontSize: '12px', color: COLORS.text.secondary, marginBottom: '8px' }}>
                  Son 10 Kayıt:
                </div>
                <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                  {fundingRates.slice(0, 10).map((rate, idx) => {
                    const isPositive = rate.funding_rate > 0;
                    const date = new Date(rate.funding_time);

                    return (
                      <div key={idx} style={{
                        marginBottom: '8px',
                        padding: '10px',
                        background: `${isPositive ? COLORS.success : COLORS.danger}0D`,
                        border: `1px solid ${isPositive ? COLORS.success : COLORS.danger}20`,
                        borderRadius: '6px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div>
                          <div style={{
                            fontSize: '14px',
                            fontWeight: 'bold',
                            color: isPositive ? COLORS.success : COLORS.danger
                          }}>
                            {formatFundingRate(rate.funding_rate)}
                          </div>
                          <div style={{ fontSize: '10px', color: COLORS.text.secondary, marginTop: '2px' }}>
                            {date.toLocaleString('tr-TR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </div>
                        <div style={{
                          fontSize: '11px',
                          padding: '3px 6px',
                          background: isPositive ? `${COLORS.success}20` : `${COLORS.danger}20`,
                          borderRadius: '4px',
                          color: isPositive ? COLORS.success : COLORS.danger,
                          fontWeight: 'bold'
                        }}>
                          {isPositive ? 'LONG ÖDER' : 'SHORT ÖDER'}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Long/Short Oranı */}
            {longShortRatio && (
              <div className="neon-card" style={{ padding: '24px' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.text.primary, marginBottom: '16px' }}>
                  ⚖️ Long/Short Oranı
                </h2>
                <div style={{ marginBottom: '20px' }}>
                  {/* Pasta Grafik Görselleştirme */}
                  <div style={{
                    width: '200px',
                    height: '200px',
                    margin: '0 auto 20px',
                    borderRadius: '50%',
                    background: `conic-gradient(
                      ${COLORS.success} 0% ${longShortRatio.long_percentage}%,
                      ${COLORS.danger} ${longShortRatio.long_percentage}% 100%
                    )`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    position: 'relative'
                  }}>
                    <div style={{
                      width: '140px',
                      height: '140px',
                      borderRadius: '50%',
                      background: COLORS.bg.primary,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexDirection: 'column'
                    }}>
                      <div style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary }}>
                        {longShortRatio.long_percentage.toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>LONG</div>
                    </div>
                  </div>

                  {/* İstatistikler */}
                  <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.success }}>
                        {longShortRatio.long_percentage.toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Long</div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                        {longShortRatio.long_account} hesap
                      </div>
                    </div>
                    <div style={{ width: '1px', background: COLORS.border.default }} />
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.danger }}>
                        {longShortRatio.short_percentage.toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '12px', color: COLORS.text.secondary }}>Short</div>
                      <div style={{ fontSize: '11px', color: COLORS.text.secondary, marginTop: '4px' }}>
                        {longShortRatio.short_account} hesap
                      </div>
                    </div>
                  </div>
                </div>

                {/* Duygu Göstergesi */}
                <div style={{
                  marginTop: '20px',
                  padding: '12px',
                  background: longShortRatio.long_percentage > 60 ? `${COLORS.success}15` :
                              longShortRatio.short_percentage > 60 ? `${COLORS.danger}15` : `${COLORS.warning}15`,
                  border: `1px solid ${
                    longShortRatio.long_percentage > 60 ? COLORS.success :
                    longShortRatio.short_percentage > 60 ? COLORS.danger : COLORS.warning
                  }40`,
                  borderRadius: '6px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', color: COLORS.text.primary }}>
                    {longShortRatio.long_percentage > 60 ? '🚀 Yükseliş Beklentisi' :
                     longShortRatio.short_percentage > 60 ? '🐻 Düşüş Beklentisi' :
                     '⚖️ Nötr Piyasa'}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Veri Kaynakları Alt Bilgi */}
          <div style={{
            marginTop: '24px',
            padding: '16px',
            background: COLORS.bg.primary,
            border: `1px solid ${COLORS.border.default}`,
            borderRadius: '8px',
            fontSize: '12px',
            color: COLORS.text.secondary,
            textAlign: 'center'
          }}>
            ✅ Gerçek Veri: Binance Futures API • Backend: Port 5031 • 10 saniyede bir güncelleme • 531 sembol desteği
          </div>

          {/* MANTIK Modal */}
          {showLogicModal && (
            <div
              style={{
                position: 'fixed',
                inset: 0,
                background: 'rgba(0, 0, 0, 0.92)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 2000,
                padding: '20px',
                backdropFilter: 'blur(10px)',
              }}
              onClick={() => setShowLogicModal(false)}
            >
              <div
                style={{
                  background: `linear-gradient(145deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
                  border: `2px solid ${COLORS.premium}`,
                  borderRadius: '16px',
                  maxWidth: '900px',
                  width: '100%',
                  maxHeight: '90vh',
                  overflow: 'auto',
                  boxShadow: `0 0 60px ${COLORS.premium}80`,
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* Modal Header */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.premium}15, ${COLORS.warning}15)`,
                  padding: '24px',
                  borderBottom: `2px solid ${COLORS.premium}`,
                  position: 'sticky',
                  top: 0,
                  zIndex: 10,
                  backdropFilter: 'blur(10px)',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Icons.Lightbulb style={{ width: '32px', height: '32px', color: COLORS.premium }} />
                      <h2 style={{ fontSize: '24px', fontWeight: 'bold', color: COLORS.text.primary, margin: 0 }}>
                        Market Insights MANTIK
                      </h2>
                    </div>
                    <button
                      onClick={() => setShowLogicModal(false)}
                      style={{
                        background: 'transparent',
                        border: `1px solid ${COLORS.border.active}`,
                        color: COLORS.text.primary,
                        padding: '8px 16px',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '600',
                        transition: 'all 0.2s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = COLORS.danger;
                        e.currentTarget.style.borderColor = COLORS.danger;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.borderColor = COLORS.border.active;
                      }}
                    >
                      KAPAT
                    </button>
                  </div>
                </div>

                {/* Modal Content */}
                <div style={{ padding: '24px' }}>
                  {/* Overview */}
                  <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icons.BarChart3 style={{ width: '24px', height: '24px' }} />
                      Genel Bakış
                    </h3>
                    <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8', marginBottom: '12px' }}>
                      Market Insights, profesyonel piyasa derinliği analizi sunan gelişmiş bir araçtır.
                      Binance Futures API kullanılarak likidite haritası, funding rate, açık pozisyon ve long/short oranları gerçek zamanlı izlenir.
                    </p>
                    <p style={{ fontSize: '15px', color: COLORS.text.secondary, lineHeight: '1.8' }}>
                      531 sembol desteği ile tüm kripto piyasasının derinlemesine analizini yaparak, akıllı para hareketlerini ve market sentiment'ı takip edebilirsiniz.
                    </p>
                  </div>

                  {/* Key Features */}
                  <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icons.Zap style={{ width: '24px', height: '24px' }} />
                      Temel Özellikler
                    </h3>
                    <div style={{ display: 'grid', gap: '12px' }}>
                      {[
                        { name: 'Piyasa Derinliği Analizi', desc: 'Likidite haritası ile önemli destek ve direnç seviyelerini görselleştirin.' },
                        { name: 'Trend Gücü Göstergeleri', desc: 'Funding rate ve açık pozisyon verileri ile trend gücünü ölçün.' },
                        { name: 'Volatilite Metrikleri', desc: 'Gerçek zamanlı volatilite takibi ile piyasa hareketlerini öngörün.' },
                        { name: 'Likidite Takibi', desc: '50 fiyat seviyesi analizi ile büyük likidite bölgelerini tespit edin.' },
                        { name: 'Akıllı Para Akışı', desc: 'Büyük yatırımcıların hareketlerini long/short oranları ile izleyin.' },
                        { name: 'Market Faz Tespiti', desc: 'Piyasanın hangi fazda olduğunu (yükseliş, düşüş, konsolidasyon) belirleyin.' }
                      ].map((feature, index) => (
                        <div key={index} style={{
                          background: `${COLORS.bg.card}40`,
                          border: `1px solid ${COLORS.border.default}`,
                          borderRadius: '8px',
                          padding: '16px',
                          transition: 'all 0.3s ease',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = COLORS.premium;
                          e.currentTarget.style.transform = 'translateX(8px)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = COLORS.border.default;
                          e.currentTarget.style.transform = 'translateX(0)';
                        }}>
                          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                            <div style={{
                              background: `linear-gradient(135deg, ${COLORS.premium}20, ${COLORS.warning}20)`,
                              padding: '8px 12px',
                              borderRadius: '6px',
                              fontSize: '14px',
                              fontWeight: 'bold',
                              color: COLORS.premium,
                              minWidth: '32px',
                              textAlign: 'center',
                            }}>
                              {index + 1}
                            </div>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontSize: '15px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                                {feature.name}
                              </div>
                              <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                                {feature.desc}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Usage Guide */}
                  <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '20px', fontWeight: 'bold', color: COLORS.premium, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icons.BarChart3 style={{ width: '24px', height: '24px' }} />
                      Kullanım Rehberi
                    </h3>
                    <div style={{ display: 'grid', gap: '16px' }}>
                      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.success}, ${COLORS.success}dd)`,
                          color: '#000',
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '18px',
                          fontWeight: 'bold',
                          flexShrink: 0,
                        }}>
                          1
                        </div>
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            Sembol Seçimi
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            Kategori veya arama ile 531 sembol arasından istediğiniz kripto parayı seçin.
                          </div>
                        </div>
                      </div>

                      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.info}, ${COLORS.info}dd)`,
                          color: '#000',
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '18px',
                          fontWeight: 'bold',
                          flexShrink: 0,
                        }}>
                          2
                        </div>
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            Likidite Haritasını İnceleyin
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            30 fiyat seviyesi görselleştirilir. Yoğun likidite bölgeleri destek/direnç olarak kullanılabilir.
                          </div>
                        </div>
                      </div>

                      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.warning}, ${COLORS.warning}dd)`,
                          color: '#000',
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '18px',
                          fontWeight: 'bold',
                          flexShrink: 0,
                        }}>
                          3
                        </div>
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            Funding Rate Takibi
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            100 dönem funding rate geçmişi ile long veya short tarafının baskınlığını görün.
                          </div>
                        </div>
                      </div>

                      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        <div style={{
                          background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.premium}dd)`,
                          color: '#000',
                          width: '40px',
                          height: '40px',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '18px',
                          fontWeight: 'bold',
                          flexShrink: 0,
                        }}>
                          4
                        </div>
                        <div>
                          <div style={{ fontSize: '16px', fontWeight: '600', color: COLORS.text.primary, marginBottom: '6px' }}>
                            Long/Short Oranı
                          </div>
                          <div style={{ fontSize: '14px', color: COLORS.text.secondary, lineHeight: '1.6' }}>
                            Trader sentiment'ını pasta grafik ile görselleştirin ve karşıt pozisyon fırsatlarını yakalayın.
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Important Notes */}
                  <div style={{
                    background: `linear-gradient(135deg, ${COLORS.warning}15, ${COLORS.danger}15)`,
                    border: `2px solid ${COLORS.warning}`,
                    borderRadius: '12px',
                    padding: '20px',
                  }}>
                    <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: COLORS.warning, marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Icons.AlertTriangle style={{ width: '22px', height: '22px' }} />
                      Önemli Notlar
                    </h3>
                    <ul style={{ margin: 0, paddingLeft: '20px', color: COLORS.text.secondary, fontSize: '14px', lineHeight: '1.8' }}>
                      <li style={{ marginBottom: '8px' }}>
                        <strong style={{ color: COLORS.text.primary }}>Gelişmiş Analitik:</strong> Profesyonel seviye piyasa derinliği ve likidite analizi.
                      </li>
                      <li style={{ marginBottom: '8px' }}>
                        <strong style={{ color: COLORS.text.primary }}>Çoklu Zaman Dilimleri:</strong> Farklı zaman dilimlerinde funding rate ve açık pozisyon analizi.
                      </li>
                      <li style={{ marginBottom: '8px' }}>
                        <strong style={{ color: COLORS.text.primary }}>Gerçek Zamanlı İzleme:</strong> Her 10 saniyede otomatik veri güncellemesi.
                      </li>
                      <li>
                        <strong style={{ color: COLORS.text.primary }}>Eğitici İçgörüler:</strong> Her metrik için açıklayıcı bilgiler ve kullanım önerileri.
                      </li>
                    </ul>
                  </div>
                </div>

                {/* Modal Footer */}
                <div style={{
                  background: `linear-gradient(135deg, ${COLORS.bg.card}, ${COLORS.bg.primary})`,
                  padding: '20px 24px',
                  borderTop: `1px solid ${COLORS.border.default}`,
                  textAlign: 'center',
                }}>
                  <p style={{ margin: 0, fontSize: '13px', color: COLORS.text.secondary }}>
                    Market Insights ile piyasayı derinlemesine analiz edin
                  </p>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
