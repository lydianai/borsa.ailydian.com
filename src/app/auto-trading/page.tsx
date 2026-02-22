/**
 * 💹 OTOMATİK TİCARET KONTROL MERKEZİ
 *
 * Binance Futures USDT-M tüm çiftlerini izleyen otomatik ticaret sistemi
 * Özellikler:
 * - Tüm Binance Futures USDT-M çiftleri (200+ coin)
 * - Yapay Zeka destekli sinyal üretimi
 * - Risk yönetimi ve pozisyon kontrolü
 * - Gerçek zamanlı performans takibi
 *
 * BEYAZ ŞAPKA: Sadece eğitim ve demo modu - GERÇEK TİCARET YAPILMAZ
 */

'use client';

import { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { Icons as _Icons } from '@/components/Icons';

interface OtomatikTicaretIstatistik {
  enabled: boolean;
  totalCoins: number;
  activeSignals: number;
  performance24h: number;
  riskLevel: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK';
  tradingPairs?: string[];
  lastUpdate?: string;
}

export default function OtomatikTicaretSayfasi() {
  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
  const [istatistik, setIstatistik] = useState<OtomatikTicaretIstatistik | null>(null);
  const [yukleniyor, setYukleniyor] = useState(true);
  const [hata, setHata] = useState<string | null>(null);
  const [seciliTab, setSeciliTab] = useState<'genel' | 'coinler' | 'sinyaller'>('genel');
  const [showLogicModal, setShowLogicModal] = useState(false);

  useEffect(() => {
    otomatikTicaretIstatistikAl();
    const interval = setInterval(otomatikTicaretIstatistikAl, 30000);
    return () => clearInterval(interval);
  }, []);

  const otomatikTicaretIstatistikAl = async () => {
    try {
      const response = await fetch('/api/auto-trading');
      const data = await response.json();

      if (data.success && data.data) {
        // API'den gelen verileri Türkçe'ye çevir
        const turkceIstatistik: OtomatikTicaretIstatistik = {
          enabled: data.data.enabled ?? false,
          totalCoins: data.data.totalCoins ?? 0,
          activeSignals: data.data.activeSignals ?? 0,
          performance24h: data.data.performance24h ?? 0,
          riskLevel: riskSeviyesiTurkce(data.data.riskLevel),
          tradingPairs: data.data.tradingPairs ?? [],
          lastUpdate: data.timestamp,
        };
        setIstatistik(turkceIstatistik);
      }
      setHata(null);
    } catch (err) {
      setHata('Otomatik Ticaret API bağlantı hatası');
      console.error('Otomatik Ticaret veri alma hatası:', err);
    } finally {
      setYukleniyor(false);
    }
  };

  const riskSeviyesiTurkce = (risk: string | undefined): 'DÜŞÜK' | 'ORTA' | 'YÜKSEK' => {
    if (!risk) return 'ORTA';
    switch (risk.toUpperCase()) {
      case 'LOW':
        return 'DÜŞÜK';
      case 'MEDIUM':
        return 'ORTA';
      case 'HIGH':
        return 'YÜKSEK';
      default:
        return 'ORTA';
    }
  };

  const riskRengiAl = (risk: 'DÜŞÜK' | 'ORTA' | 'YÜKSEK') => {
    switch (risk) {
      case 'DÜŞÜK':
        return { bg: 'rgba(16, 185, 129, 0.15)', border: '#10B981', text: '#10B981' };
      case 'ORTA':
        return { bg: 'rgba(245, 158, 11, 0.15)', border: '#F59E0B', text: '#F59E0B' };
      case 'YÜKSEK':
        return { bg: 'rgba(239, 68, 68, 0.15)', border: '#EF4444', text: '#EF4444' };
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0A0A0A' }}>
      <SharedSidebar currentPage="auto-trading" />

      <main
        style={{
          marginTop: '0px',
          padding: '24px',
          paddingTop: isLocalhost ? '116px' : '60px',
          maxWidth: '1920px',
          margin: '120px auto 0',
        }}
      >
        {/* Premium Header */}
        <div
          style={{
            background:
              'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.15) 50%, rgba(245, 158, 11, 0.1) 100%)',
            backdropFilter: 'blur(30px)',
            border: '2px solid rgba(245, 158, 11, 0.5)',
            borderRadius: '24px',
            padding: '40px',
            marginBottom: '32px',
            boxShadow:
              '0 20px 60px rgba(245, 158, 11, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Animated background effect */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background:
                'radial-gradient(circle at 30% 50%, rgba(245, 158, 11, 0.15) 0%, transparent 50%)',
              animation: 'pulse 4s ease-in-out infinite',
            }}
          />

          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', position: 'relative' }}>
            <div
              style={{
                width: '72px',
                height: '72px',
                background: 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)',
                borderRadius: '20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '36px',
                boxShadow: '0 10px 30px rgba(245, 158, 11, 0.5)',
              }}
            >
              💹
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
                <h1
                  style={{
                    fontSize: '40px',
                    fontWeight: '800',
                    background: 'linear-gradient(135deg, #FFFFFF 0%, rgba(255,255,255,0.7) 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    margin: 0,
                    letterSpacing: '-0.5px',
                  }}
                >
  Otomatik İşlem Kontrol Merkezi
                </h1>

                {/* MANTIK Button - Responsive */}
                <div>
                  <style>{`
                    @media (max-width: 768px) {
                      .mantik-button-autotrading {
                        padding: 10px 20px !important;
                        fontSize: 13px !important;
                        height: 42px !important;
                      }
                      .mantik-button-autotrading span {
                        fontSize: 18px !important;
                      }
                    }
                    @media (max-width: 480px) {
                      .mantik-button-autotrading {
                        padding: 8px 16px !important;
                        fontSize: 12px !important;
                        height: 40px !important;
                      }
                      .mantik-button-autotrading span {
                        fontSize: 16px !important;
                      }
                    }
                  `}</style>
                  <button
                    onClick={() => setShowLogicModal(true)}
                    className="mantik-button-autotrading"
                    style={{
                      background: 'linear-gradient(135deg, #8B5CF6, #7C3AED)',
                      border: '2px solid rgba(139, 92, 246, 0.5)',
                      borderRadius: '10px',
                      padding: '12px 24px',
                      color: '#FFFFFF',
                      fontSize: '14px',
                      fontWeight: '700',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      boxShadow: '0 4px 16px rgba(139, 92, 246, 0.3)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      height: '44px'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 24px rgba(139, 92, 246, 0.5)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 4px 16px rgba(139, 92, 246, 0.3)';
                    }}
                  >
                    <span style={{ fontSize: '18px' }}>🧠</span>
                    MANTIK
                  </button>
                </div>
              </div>
              <p
                style={{
                  fontSize: '16px',
                  color: 'rgba(255,255,255,0.8)',
                  margin: '12px 0 0',
                  fontWeight: '500',
                }}
              >
                Binance Futures USDT-M Tüm Çiftler • Yapay Zeka Destekli • Anlık İzleme
              </p>
            </div>

            {istatistik && (
              <div
                style={{
                  background: 'rgba(0, 0, 0, 0.4)',
                  backdropFilter: 'blur(10px)',
                  padding: '16px 24px',
                  borderRadius: '16px',
                  border: '2px solid rgba(255, 255, 255, 0.1)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>
                  Toplam Coin
                </div>
                <div style={{ fontSize: '32px', fontWeight: '800', color: '#F59E0B' }}>
                  {istatistik.totalCoins || 0}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* DEMO MODE Uyarısı */}
        <div
          style={{
            background:
              'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%)',
            backdropFilter: 'blur(20px)',
            border: '2px solid rgba(239, 68, 68, 0.5)',
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '32px',
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            boxShadow: '0 10px 40px rgba(239, 68, 68, 0.2)',
          }}
        >
          <div
            style={{
              fontSize: '40px',
              filter: 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.6))',
            }}
          >
            ⚠️
          </div>
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                color: '#EF4444',
                marginBottom: '8px',
              }}
            >
              DEMO MODU - Sadece Eğitim Amaçlı
            </div>
            <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.7)', lineHeight: '1.6' }}>
              Bu sistem sadece eğitim ve analiz amaçlıdır. Gerçek işlem yapılmaz. Tüm
              veriler simülasyon amaçlıdır ve etik kurallara uygundur.
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div
          style={{
            display: 'flex',
            gap: '12px',
            marginBottom: '24px',
            background: 'rgba(26, 26, 26, 0.6)',
            backdropFilter: 'blur(10px)',
            padding: '8px',
            borderRadius: '16px',
            border: '2px solid rgba(255, 255, 255, 0.05)',
          }}
        >
          {[
            { id: 'genel', label: '📊 Genel Bakış', icon: '📊' },
            { id: 'coinler', label: '🪙 İzlenen Kripto Paralar', icon: '🪙' },
            { id: 'sinyaller', label: '⚡ Aktif Sinyaller', icon: '⚡' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSeciliTab(tab.id as any)}
              style={{
                flex: 1,
                padding: '16px 24px',
                background:
                  seciliTab === tab.id
                    ? 'linear-gradient(135deg, #F59E0B 0%, #D97706 100%)'
                    : 'transparent',
                border: seciliTab === tab.id ? '2px solid #F59E0B' : '2px solid transparent',
                borderRadius: '12px',
                color: seciliTab === tab.id ? '#FFFFFF' : 'rgba(255,255,255,0.6)',
                fontSize: '15px',
                fontWeight: '700',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow:
                  seciliTab === tab.id ? '0 8px 24px rgba(245, 158, 11, 0.4)' : 'none',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        {yukleniyor ? (
          <div
            style={{
              background: 'rgba(26, 26, 26, 0.95)',
              backdropFilter: 'blur(20px)',
              borderRadius: '24px',
              padding: '80px',
              textAlign: 'center',
              border: '2px solid rgba(255, 255, 255, 0.05)',
            }}
          >
            <div style={{ fontSize: '64px', marginBottom: '24px', filter: 'grayscale(0.3)' }}>
              💹
            </div>
            <div style={{ fontSize: '18px', color: 'rgba(255,255,255,0.6)', fontWeight: '500' }}>
              Otomatik Ticaret verileri yükleniyor...
            </div>
          </div>
        ) : hata ? (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '2px solid rgba(239, 68, 68, 0.4)',
              borderRadius: '16px',
              padding: '40px',
              textAlign: 'center',
              color: '#EF4444',
              fontSize: '16px',
              fontWeight: '600',
            }}
          >
            ❌ {hata}
          </div>
        ) : istatistik ? (
          <>
            {/* Genel Bakış Tab */}
            {seciliTab === 'genel' && (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                  gap: '24px',
                }}
              >
                {/* Sistem Durumu */}
                <div
                  style={{
                    background:
                      'linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
                    backdropFilter: 'blur(30px)',
                    border: `2px solid ${istatistik.enabled ? 'rgba(16, 185, 129, 0.5)' : 'rgba(107, 114, 128, 0.5)'}`,
                    borderRadius: '20px',
                    padding: '32px',
                    boxShadow: `0 15px 50px ${istatistik.enabled ? 'rgba(16, 185, 129, 0.3)' : 'rgba(0, 0, 0, 0.3)'}`,
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '100px',
                      height: '100px',
                      background: `radial-gradient(circle, ${istatistik.enabled ? 'rgba(16, 185, 129, 0.2)' : 'rgba(107, 114, 128, 0.2)'} 0%, transparent 70%)`,
                    }}
                  />
                  <div
                    style={{
                      fontSize: '14px',
                      color: 'rgba(255,255,255,0.6)',
                      marginBottom: '12px',
                      fontWeight: '600',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    Sistem Durumu
                  </div>
                  <div
                    style={{
                      fontSize: '40px',
                      fontWeight: '800',
                      color: istatistik.enabled ? '#10B981' : '#6B7280',
                      marginBottom: '12px',
                    }}
                  >
                    {istatistik.enabled ? 'AKTİF' : 'PASİF'}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
                    {istatistik.enabled ? '🟢 Sistem çalışıyor' : '🔴 Sistem durdu'}
                  </div>
                </div>

                {/* Aktif Sinyaller */}
                <div
                  style={{
                    background:
                      'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%)',
                    backdropFilter: 'blur(30px)',
                    border: '2px solid rgba(16, 185, 129, 0.5)',
                    borderRadius: '20px',
                    padding: '32px',
                    boxShadow: '0 15px 50px rgba(16, 185, 129, 0.3)',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '100px',
                      height: '100px',
                      background:
                        'radial-gradient(circle, rgba(16, 185, 129, 0.3) 0%, transparent 70%)',
                    }}
                  />
                  <div
                    style={{
                      fontSize: '14px',
                      color: 'rgba(255,255,255,0.6)',
                      marginBottom: '12px',
                      fontWeight: '600',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    Aktif Sinyaller
                  </div>
                  <div
                    style={{
                      fontSize: '40px',
                      fontWeight: '800',
                      color: '#10B981',
                      marginBottom: '12px',
                    }}
                  >
                    {istatistik.activeSignals}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
                    ⚡ Toplam aktif sinyal sayısı
                  </div>
                </div>

                {/* 24h Performans */}
                <div
                  style={{
                    background:
                      (istatistik.performance24h ?? 0) >= 0
                        ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.1) 100%)'
                        : 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%)',
                    backdropFilter: 'blur(30px)',
                    border: `2px solid ${(istatistik.performance24h ?? 0) >= 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)'}`,
                    borderRadius: '20px',
                    padding: '32px',
                    boxShadow: `0 15px 50px ${(istatistik.performance24h ?? 0) >= 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '100px',
                      height: '100px',
                      background: `radial-gradient(circle, ${(istatistik.performance24h ?? 0) >= 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'} 0%, transparent 70%)`,
                    }}
                  />
                  <div
                    style={{
                      fontSize: '14px',
                      color: 'rgba(255,255,255,0.6)',
                      marginBottom: '12px',
                      fontWeight: '600',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    24 Saatlik Performans
                  </div>
                  <div
                    style={{
                      fontSize: '40px',
                      fontWeight: '800',
                      color: (istatistik.performance24h ?? 0) >= 0 ? '#10B981' : '#EF4444',
                      marginBottom: '12px',
                    }}
                  >
                    {(istatistik.performance24h ?? 0) > 0 ? '+' : ''}
                    {(istatistik.performance24h ?? 0).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
                    {(istatistik.performance24h ?? 0) >= 0
                      ? '📈 Pozitif performans'
                      : '📉 Negatif performans'}
                  </div>
                </div>

                {/* Risk Seviyesi */}
                <div
                  style={{
                    background: `linear-gradient(135deg, ${riskRengiAl(istatistik.riskLevel).bg} 0%, rgba(10, 10, 10, 0.95) 100%)`,
                    backdropFilter: 'blur(30px)',
                    border: `2px solid ${riskRengiAl(istatistik.riskLevel).border}`,
                    borderRadius: '20px',
                    padding: '32px',
                    boxShadow: `0 15px 50px ${riskRengiAl(istatistik.riskLevel).border}40`,
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      right: 0,
                      width: '100px',
                      height: '100px',
                      background: `radial-gradient(circle, ${riskRengiAl(istatistik.riskLevel).border}30 0%, transparent 70%)`,
                    }}
                  />
                  <div
                    style={{
                      fontSize: '14px',
                      color: 'rgba(255,255,255,0.6)',
                      marginBottom: '12px',
                      fontWeight: '600',
                      textTransform: 'uppercase',
                      letterSpacing: '1px',
                    }}
                  >
                    Risk Seviyesi
                  </div>
                  <div
                    style={{
                      fontSize: '40px',
                      fontWeight: '800',
                      color: riskRengiAl(istatistik.riskLevel).text,
                      marginBottom: '12px',
                    }}
                  >
                    {istatistik.riskLevel}
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(255,255,255,0.5)' }}>
                    {istatistik.riskLevel === 'DÜŞÜK'
                      ? '✅ Güvenli seviye'
                      : istatistik.riskLevel === 'ORTA'
                        ? '⚠️ Dikkat gerekli'
                        : '🚨 Yüksek risk'}
                  </div>
                </div>
              </div>
            )}

            {/* İzlenen Coinler Tab */}
            {seciliTab === 'coinler' && (
              <div
                style={{
                  background: 'rgba(26, 26, 26, 0.95)',
                  backdropFilter: 'blur(30px)',
                  borderRadius: '24px',
                  padding: '32px',
                  border: '2px solid rgba(255, 255, 255, 0.05)',
                }}
              >
                <div
                  style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: '#FFFFFF',
                    marginBottom: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  🪙 İzlenen Binance Futures USDT-M Kripto Paraları
                  <span
                    style={{
                      fontSize: '14px',
                      background: '#F59E0B',
                      padding: '6px 12px',
                      borderRadius: '8px',
                      fontWeight: '800',
                    }}
                  >
                    {istatistik.tradingPairs?.length || 0} Kripto
                  </span>
                </div>

                {istatistik.tradingPairs && istatistik.tradingPairs.length > 0 ? (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))',
                      gap: '12px',
                      maxHeight: '600px',
                      overflowY: 'auto',
                      padding: '8px',
                    }}
                  >
                    {istatistik.tradingPairs.map((pair, index) => (
                      <div
                        key={`${pair}-${index}`}
                        style={{
                          background:
                            'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%)',
                          border: '2px solid rgba(59, 130, 246, 0.3)',
                          borderRadius: '12px',
                          padding: '12px',
                          textAlign: 'center',
                          transition: 'all 0.3s ease',
                          cursor: 'pointer',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background =
                            'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.1) 100%)';
                          e.currentTarget.style.borderColor = '#3B82F6';
                          e.currentTarget.style.transform = 'translateY(-2px)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background =
                            'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%)';
                          e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
                          e.currentTarget.style.transform = 'translateY(0)';
                        }}
                      >
                        <div
                          style={{
                            fontSize: '16px',
                            fontWeight: '700',
                            color: '#3B82F6',
                          }}
                        >
                          {pair}
                        </div>
                        <div
                          style={{
                            fontSize: '11px',
                            color: 'rgba(255,255,255,0.5)',
                            marginTop: '4px',
                          }}
                        >
                          USDT
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '60px',
                      color: 'rgba(255,255,255,0.5)',
                    }}
                  >
                    <div style={{ fontSize: '48px', marginBottom: '16px' }}>🪙</div>
                    <div style={{ fontSize: '16px' }}>Henüz izlenen kripto para yok</div>
                  </div>
                )}
              </div>
            )}

            {/* Aktif Sinyaller Tab */}
            {seciliTab === 'sinyaller' && (
              <div
                style={{
                  background: 'rgba(26, 26, 26, 0.95)',
                  backdropFilter: 'blur(30px)',
                  borderRadius: '24px',
                  padding: '32px',
                  border: '2px solid rgba(255, 255, 255, 0.05)',
                }}
              >
                <div
                  style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: '#FFFFFF',
                    marginBottom: '24px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  ⚡ Aktif İşlem Sinyalleri
                  <span
                    style={{
                      fontSize: '14px',
                      background: '#10B981',
                      padding: '6px 12px',
                      borderRadius: '8px',
                      fontWeight: '800',
                    }}
                  >
                    {istatistik.activeSignals} Sinyal
                  </span>
                </div>

                <div
                  style={{
                    textAlign: 'center',
                    padding: '60px',
                    color: 'rgba(255,255,255,0.5)',
                  }}
                >
                  <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚡</div>
                  <div style={{ fontSize: '16px' }}>
                    Aktif sinyal detayları yakında eklenecek
                  </div>
                </div>
              </div>
            )}
          </>
        ) : null}

        {/* Bilgilendirme Bölümü */}
        <div
          style={{
            background:
              'linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
            backdropFilter: 'blur(30px)',
            border: '2px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '24px',
            padding: '40px',
            marginTop: '32px',
            boxShadow: '0 15px 50px rgba(0, 0, 0, 0.3)',
          }}
        >
          <h2
            style={{
              fontSize: '24px',
              fontWeight: '700',
              color: '#FFFFFF',
              marginBottom: '24px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}
          >
            ℹ️ Otomatik İşlem Nasıl Çalışır?
          </h2>
          <div
            style={{
              fontSize: '15px',
              color: 'rgba(255,255,255,0.8)',
              lineHeight: '2',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '20px',
            }}
          >
            <div
              style={{
                background: 'rgba(59, 130, 246, 0.1)',
                padding: '20px',
                borderRadius: '12px',
                border: '2px solid rgba(59, 130, 246, 0.2)',
              }}
            >
              <div style={{ fontSize: '18px', marginBottom: '12px' }}>📊</div>
              <strong>Tüm Kripto Paraları İzle:</strong> Binance Futures USDT-M piyasasındaki tüm
              kripto para çiftlerini anlık olarak takip eder
            </div>
            <div
              style={{
                background: 'rgba(16, 185, 129, 0.1)',
                padding: '20px',
                borderRadius: '12px',
                border: '2px solid rgba(16, 185, 129, 0.2)',
              }}
            >
              <div style={{ fontSize: '18px', marginBottom: '12px' }}>🤖</div>
              <strong>Yapay Zeka Sinyalleri:</strong> Çoklu yapay zeka stratejilerinden (Quantum Pro,
              Transformer, Gradient Boosting) gelen sinyalleri toplar ve analiz eder
            </div>
            <div
              style={{
                background: 'rgba(245, 158, 11, 0.1)',
                padding: '20px',
                borderRadius: '12px',
                border: '2px solid rgba(245, 158, 11, 0.2)',
              }}
            >
              <div style={{ fontSize: '18px', marginBottom: '12px' }}>🛡️</div>
              <strong>Risk Yönetimi:</strong> Belirlenmiş risk yönetimi kurallarına göre
              sinyalleri filtreler ve pozisyon büyüklüğünü kontrol eder
            </div>
            <div
              style={{
                background: 'rgba(139, 92, 246, 0.1)',
                padding: '20px',
                borderRadius: '12px',
                border: '2px solid rgba(139, 92, 246, 0.2)',
              }}
            >
              <div style={{ fontSize: '18px', marginBottom: '12px' }}>⚡</div>
              <strong>Sinyal Üretimi:</strong> Confidence threshold değerini aşan ve risk
              kriterlerine uyan sinyalleri işaretler ve raporlar
            </div>
            <div
              style={{
                background: 'rgba(239, 68, 68, 0.1)',
                padding: '20px',
                borderRadius: '12px',
                border: '2px solid rgba(239, 68, 68, 0.2)',
                gridColumn: 'span 2',
              }}
            >
              <div style={{ fontSize: '18px', marginBottom: '12px' }}>🎓</div>
              <strong>DEMO MODU:</strong> Bu sistem tamamen eğitim ve analiz amaçlıdır. Gerçek
              işlem yapmaz, sadece sinyal analizi ve performans takibi sağlar. Tüm işlemler
              simülasyon ortamında gerçekleşir ve etik kurallara uygundur.
            </div>
          </div>
        </div>

        {/* Son Güncelleme */}
        {istatistik?.lastUpdate && (
          <div
            style={{
              textAlign: 'center',
              marginTop: '24px',
              fontSize: '13px',
              color: 'rgba(255,255,255,0.4)',
            }}
          >
            Son Güncelleme: {new Date(istatistik.lastUpdate).toLocaleString('tr-TR')}
          </div>
        )}
      </main>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
      `}</style>

      {/* MANTIK Modal */}
      {showLogicModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '24px'
          }}
          onClick={() => setShowLogicModal(false)}
        >
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
              backdropFilter: 'blur(20px)',
              border: '2px solid rgba(139, 92, 246, 0.3)',
              borderRadius: '24px',
              padding: '32px',
              maxWidth: '800px',
              maxHeight: '80vh',
              overflow: 'auto',
              boxShadow: '0 20px 60px rgba(139, 92, 246, 0.3)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ fontSize: '28px', fontWeight: '700', color: '#8B5CF6', margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span>🧠</span>
                OTOMATİK İŞLEM MANTIĞI
              </h2>
              <button
                onClick={() => setShowLogicModal(false)}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '2px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '12px',
                  width: '40px',
                  height: '40px',
                  color: '#FFFFFF',
                  fontSize: '20px',
                  cursor: 'pointer',
                  transition: 'all 0.3s'
                }}
              >
                ✕
              </button>
            </div>

            {/* Content sections */}
            <div style={{ color: 'rgba(255, 255, 255, 0.9)', lineHeight: '1.8' }}>

              {/* Section 1: Purpose */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#10B981', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  📌 Sayfa Amacı
                </h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0 }}>
                  Otomatik İşlem Kontrol Merkezi, Binance Futures USDT-M piyasasındaki tüm kripto çiftlerini izleyen ve yapay zeka destekli sinyal üreten demo işlem sistemidir. SADECE EĞİTİM AMAÇLIDIR - gerçek işlem yapmaz.
                </p>
              </div>

              {/* Section 2: How It Works */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#3B82F6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ⚙️ Nasıl Çalışır?
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Binance Futures API'den 200+ USDT-M çiftinin anlık verilerini alır</li>
                  <li>Quantum Pro, Transformer ve Gradient Boosting gibi çoklu yapay zeka stratejileri kullanır</li>
                  <li>Her kripto para için sinyal güçlendirme ve risk skoru hesaplar</li>
                  <li>Güven eşiği değerini aşan sinyalleri işaretler ve raporlar</li>
                  <li>Demo modda performans takibi ve analiz yapar (gerçek işlem yapmaz)</li>
                </ul>
              </div>

              {/* Section 3: Key Features */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#F59E0B', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ✨ Önemli Özellikler
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>200+ Binance Futures USDT-M çiftinin anlık izlenmesi</li>
                  <li>Çoklu yapay zeka strateji birleşimi: LSTM, Transformer, Gradient Boosting</li>
                  <li>Otomatik risk yönetimi ve pozisyon kontrolü</li>
                  <li>Sistem durumu: Aktif/Pasif mod kontrolü</li>
                  <li>24 saatlik performans takibi ve raporlama</li>
                  <li>Risk seviyesi analizi: Düşük/Orta/Yüksek</li>
                </ul>
              </div>

              {/* Section 4: Data Sources */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#EC4899', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  🔌 Veri Kaynakları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Binance Futures API: Tüm USDT-M çiftleri için fiyat, hacim ve değişim verileri</li>
                  <li>Quantum Pro Yapay Zeka API: Gelişmiş sinyal üretimi ve analiz</li>
                  <li>Risk Yönetim Motoru: Otomatik risk hesaplama ve pozisyon yönetimi</li>
                </ul>
              </div>

              {/* Section 5: Usage Tips */}
              <div>
                <h3 style={{ color: '#8B5CF6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  💡 Kullanım İpuçları
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Sistem Durumu bölümünden sistemin aktif olup olmadığını kontrol edin</li>
                  <li>Aktif Sinyaller sekmesinden anlık işlem fırsatlarını inceleyin</li>
                  <li>İzlenen Kripto Paralar sekmesinden tüm takip edilen çiftleri görüntüleyin</li>
                  <li>24 Saat Performans ölçümünü izleyerek sistemin başarısını takip edin</li>
                  <li>Risk Seviyesi göstergesine dikkat ederek piyasa riskini değerlendirin</li>
                  <li>BU BİR DEMO SİSTEMDİR - gerçek para ile işlem yapmaz, sadece eğitim amaçlıdır</li>
                </ul>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
