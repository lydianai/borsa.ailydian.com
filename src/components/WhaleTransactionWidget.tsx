'use client';

import { useEffect, useState } from 'react';
import type { WhaleTrackerAPIResponse, WhaleTransaction, Blockchain, TransactionSignificance } from '@/types/whale-tracker';

type BlockchainFilter = 'BTC' | 'ETH' | 'AVAX' | 'TÜMÜ';

export function WhaleTransactionWidget() {
  const [data, setData] = useState<WhaleTrackerAPIResponse['data'] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<BlockchainFilter>('TÜMÜ');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchData = async () => {
    try {
      const response = await fetch('/api/whale-tracker');
      const result: WhaleTrackerAPIResponse = await response.json();

      if (result.success && result.data) {
        setData(result.data);
        setLastUpdate(new Date());
        setError(null);
      } else {
        setError(result.error || 'Veri alınamadı');
      }
    } catch (err) {
      setError('API bağlantı hatası');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // 30 seconds
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number): string => {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(2)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(2)}M`;
    if (num >= 1_000) return `${(num / 1_000).toFixed(2)}K`;
    return num.toFixed(2);
  };

  const truncateAddress = (address: string): string => {
    if (address.length <= 10) return address;
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const getRelativeTime = (timestamp: number): string => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Az önce';
    if (minutes < 60) return `${minutes} dakika önce`;
    if (hours < 24) return `${hours} saat önce`;
    return `${days} gün önce`;
  };

  const getBlockchainColor = (blockchain: Blockchain) => {
    switch (blockchain) {
      case 'BTC':
        return { bg: 'rgba(247, 147, 26, 0.15)', border: 'rgba(247, 147, 26, 0.5)', text: '#F7931A' };
      case 'ETH':
        return { bg: 'rgba(98, 126, 234, 0.15)', border: 'rgba(98, 126, 234, 0.5)', text: '#627EEA' };
      case 'AVAX':
        return { bg: 'rgba(232, 65, 66, 0.15)', border: 'rgba(232, 65, 66, 0.5)', text: '#E84142' };
    }
  };

  const getSignificanceBadge = (significance: TransactionSignificance) => {
    switch (significance) {
      case 'CRITICAL':
        return { emoji: '🚨', text: 'KRİTİK', color: '#EF4444' };
      case 'HIGH':
        return { emoji: '⚠️', text: 'YÜKSEK', color: '#F59E0B' };
      case 'MEDIUM':
        return { emoji: '📊', text: 'ORTA', color: '#3B82F6' };
      case 'LOW':
        return { emoji: 'ℹ️', text: 'DÜŞÜK', color: '#6B7280' };
    }
  };

  const filteredTransactions = data?.transactions.filter(tx => {
    if (filter === 'TÜMÜ') return true;
    return tx.blockchain === filter;
  }) || [];

  if (loading) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.1) 100%)',
        backdropFilter: 'blur(20px)',
        border: '2px solid rgba(59, 130, 246, 0.4)',
        borderRadius: '16px',
        padding: '32px',
        marginTop: '24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '400px'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🐋</div>
          <div style={{ fontSize: '18px', color: '#FFFFFF', fontWeight: '600' }}>Balina hareketleri yükleniyor...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.1) 100%)',
        backdropFilter: 'blur(20px)',
        border: '2px solid rgba(239, 68, 68, 0.4)',
        borderRadius: '16px',
        padding: '32px',
        marginTop: '24px',
        textAlign: 'center'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
        <div style={{ fontSize: '18px', color: '#FFFFFF', fontWeight: '600', marginBottom: '8px' }}>Hata</div>
        <div style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>{error}</div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.1) 100%)',
      backdropFilter: 'blur(20px)',
      border: '2px solid rgba(59, 130, 246, 0.4)',
      borderRadius: '16px',
      padding: '32px',
      marginTop: '24px',
      boxShadow: '0 8px 32px rgba(59, 130, 246, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.1)',
      transition: 'all 0.3s ease'
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '56px',
            height: '56px',
            background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
            borderRadius: '14px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 6px 20px rgba(59, 130, 246, 0.4)',
            fontSize: '28px'
          }}>
            🐋
          </div>
          <div>
            <h2 style={{ fontSize: '26px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
              🐋 Balina Hareketleri (Canlı)
            </h2>
            <p style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Büyük kripto para transferlerini anlık takip et
            </p>
          </div>
        </div>

        {/* Blockchain Filter */}
        <div style={{ display: 'flex', gap: '8px' }}>
          {(['TÜMÜ', 'BTC', 'ETH', 'AVAX'] as BlockchainFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '8px 16px',
                background: filter === f
                  ? 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)'
                  : 'rgba(0, 0, 0, 0.4)',
                border: filter === f
                  ? '2px solid rgba(59, 130, 246, 0.6)'
                  : '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                backdropFilter: 'blur(8px)'
              }}
              onMouseEnter={(e) => {
                if (filter !== f) {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)';
                  e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.4)';
                }
              }}
              onMouseLeave={(e) => {
                if (filter !== f) {
                  e.currentTarget.style.background = 'rgba(0, 0, 0, 0.4)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                }
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Stats Bar */}
      {data?.stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '32px' }}>
          {/* Transaction Count */}
          <div style={{
            padding: '20px',
            background: 'rgba(0, 0, 0, 0.4)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '12px',
            backdropFilter: 'blur(12px)'
          }}>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
              SON 24 SAAT - İŞLEM SAYISI
            </div>
            <div style={{ fontSize: '28px', fontWeight: '700', color: '#3B82F6' }}>
              {data.stats.last24h.transactionCount}
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
              Büyük transfer tespit edildi
            </div>
          </div>

          {/* Total Volume */}
          <div style={{
            padding: '20px',
            background: 'rgba(0, 0, 0, 0.4)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: '12px',
            backdropFilter: 'blur(12px)'
          }}>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
              TOPLAM HACIM (24S)
            </div>
            <div style={{ fontSize: '28px', fontWeight: '700', color: '#10B981' }}>
              ${formatNumber(data.stats.last24h.totalVolumeUSD)}
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
              Transfer hacmi
            </div>
          </div>

          {/* Largest Transaction */}
          <div style={{
            padding: '20px',
            background: 'rgba(0, 0, 0, 0.4)',
            border: '1px solid rgba(234, 179, 8, 0.3)',
            borderRadius: '12px',
            backdropFilter: 'blur(12px)'
          }}>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
              EN BÜYÜK İŞLEM
            </div>
            <div style={{ fontSize: '28px', fontWeight: '700', color: '#EAB308' }}>
              ${formatNumber(data.stats.last24h.largestTransaction.amountUSD)}
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
              {data.stats.last24h.largestTransaction.blockchain} - {formatNumber(data.stats.last24h.largestTransaction.amount)}
            </div>
          </div>
        </div>
      )}

      {/* Transaction Feed */}
      <div style={{
        background: 'rgba(0, 0, 0, 0.4)',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '20px',
        backdropFilter: 'blur(12px)'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
          📡 Son Balina İşlemleri ({filteredTransactions.length})
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxHeight: '600px', overflowY: 'auto' }}>
          {filteredTransactions.slice(0, 20).map((tx, index) => {
            const blockchainColors = getBlockchainColor(tx.blockchain);
            const significanceBadge = getSignificanceBadge(tx.significance);

            return (
              <div
                key={`${tx.hash}-${index}`}
                style={{
                  background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '10px',
                  padding: '16px',
                  transition: 'all 0.2s ease',
                  cursor: 'pointer'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.04) 100%)';
                  e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.4)';
                  e.currentTarget.style.transform = 'translateX(4px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                  e.currentTarget.style.transform = 'translateX(0)';
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                  {/* Left: Blockchain + Amount */}
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    {/* Blockchain Badge */}
                    <div style={{
                      padding: '6px 12px',
                      background: blockchainColors.bg,
                      border: `1px solid ${blockchainColors.border}`,
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: '700',
                      color: blockchainColors.text
                    }}>
                      {tx.blockchain}
                    </div>

                    {/* Amount */}
                    <div>
                      <div style={{ fontSize: '18px', fontWeight: '700', color: '#FFFFFF' }}>
                        {formatNumber(tx.amount)} {tx.blockchain}
                      </div>
                      <div style={{ fontSize: '13px', color: '#10B981', fontWeight: '600' }}>
                        ${formatNumber(tx.amountUSD)}
                      </div>
                    </div>
                  </div>

                  {/* Right: Significance + Time */}
                  <div style={{ textAlign: 'right' }}>
                    <div style={{
                      padding: '4px 10px',
                      background: `${significanceBadge.color}20`,
                      border: `1px solid ${significanceBadge.color}`,
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontWeight: '700',
                      color: significanceBadge.color,
                      marginBottom: '6px',
                      display: 'inline-block'
                    }}>
                      {significanceBadge.emoji} {significanceBadge.text}
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                      {getRelativeTime(tx.timestamp)}
                    </div>
                  </div>
                </div>

                {/* Middle: Description */}
                <div style={{
                  fontSize: '14px',
                  color: 'rgba(255, 255, 255, 0.8)',
                  marginBottom: '12px',
                  lineHeight: '1.5'
                }}>
                  {tx.turkishDescription}
                </div>

                {/* Bottom: Addresses */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '12px', alignItems: 'center' }}>
                  {/* From */}
                  <div style={{
                    padding: '10px',
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    borderRadius: '8px'
                  }}>
                    <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '4px', fontWeight: '600' }}>
                      GÖNDEREN
                    </div>
                    <div style={{ fontSize: '12px', color: '#FFFFFF', fontFamily: 'monospace', marginBottom: '4px' }}>
                      {truncateAddress(tx.from)}
                    </div>
                    {tx.fromLabel && (
                      <div style={{ fontSize: '11px', color: '#EF4444', fontWeight: '600' }}>
                        {tx.fromLabel}
                      </div>
                    )}
                  </div>

                  {/* Arrow */}
                  <div style={{ fontSize: '20px', color: 'rgba(255, 255, 255, 0.4)' }}>
                    →
                  </div>

                  {/* To */}
                  <div style={{
                    padding: '10px',
                    background: 'rgba(16, 185, 129, 0.1)',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                    borderRadius: '8px'
                  }}>
                    <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '4px', fontWeight: '600' }}>
                      ALAN
                    </div>
                    <div style={{ fontSize: '12px', color: '#FFFFFF', fontFamily: 'monospace', marginBottom: '4px' }}>
                      {truncateAddress(tx.to)}
                    </div>
                    {tx.toLabel && (
                      <div style={{ fontSize: '11px', color: '#10B981', fontWeight: '600' }}>
                        {tx.toLabel}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {filteredTransactions.length === 0 && (
            <div style={{
              padding: '40px',
              textAlign: 'center',
              color: 'rgba(255, 255, 255, 0.5)',
              fontSize: '14px'
            }}>
              Bu blockchain için henüz işlem bulunamadı
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px 20px',
        background: 'rgba(0, 0, 0, 0.3)',
        borderRadius: '10px',
        backdropFilter: 'blur(8px)'
      }}>
        <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>
          Son güncelleme: {lastUpdate.toLocaleTimeString('tr-TR')}
        </div>
        <button
          onClick={() => fetchData()}
          style={{
            padding: '8px 16px',
            background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
            border: '1px solid rgba(59, 130, 246, 0.5)',
            borderRadius: '8px',
            color: '#FFFFFF',
            fontSize: '12px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'scale(1.05)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          🔄 Yenile
        </button>
      </div>
    </div>
  );
}
