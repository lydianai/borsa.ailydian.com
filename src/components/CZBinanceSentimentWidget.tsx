/**
 * CZ BINANCE SENTIMENT WIDGET
 * Tracks Changpeng Zhao (CZ) and Binance-related market sentiment
 * Beyaz Şapka: Educational purpose only - sentiment analysis for research
 */

'use client';

import { useState, useEffect } from 'react';
import { Icons } from '@/components/Icons';

interface CZSentimentData {
  overallSentiment: 'BULLISH' | 'NEUTRAL' | 'BEARISH';
  sentimentScore: number;
  binanceNewsImpact: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
  recentEvents: Array<{
    title: string;
    impact: 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE';
    timestamp: string;
    importance: number;
  }>;
  marketReaction: {
    btcChange24h: number;
    bnbChange24h: number;
    totalVolumeChange: number;
  };
  czInfluenceIndex: number;
  lastUpdate: string;
}

export function CZBinanceSentimentWidget() {
  const [sentimentData, setSentimentData] = useState<CZSentimentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [countdown, setCountdown] = useState(60);

  const fetchSentimentData = async () => {
    try {
      setLoading(true);

      const mockData: CZSentimentData = {
        overallSentiment: 'BULLISH',
        sentimentScore: 72,
        binanceNewsImpact: 'POSITIVE',
        recentEvents: [
          {
            title: 'Binance Yeni Regülasyon Onayları Aldı',
            impact: 'POSITIVE',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            importance: 8
          },
          {
            title: 'CZ: "Kripto Endüstrisi Büyümeye Devam Edecek"',
            impact: 'POSITIVE',
            timestamp: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
            importance: 7
          },
          {
            title: 'Binance Trading Volume Rekor Kırdı',
            impact: 'POSITIVE',
            timestamp: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
            importance: 9
          }
        ],
        marketReaction: {
          btcChange24h: 2.3,
          bnbChange24h: 4.7,
          totalVolumeChange: 15.2
        },
        czInfluenceIndex: 85,
        lastUpdate: new Date().toISOString()
      };

      setSentimentData(mockData);
    } catch (error) {
      console.error('[CZ Sentiment] Error:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSentimentData();

    const interval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          fetchSentimentData();
          return 60;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (loading && !sentimentData) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.08) 100%)',
        backdropFilter: 'blur(20px)',
        border: '2px solid rgba(245, 158, 11, 0.3)',
        borderRadius: '12px',
        padding: '24px',
        textAlign: 'center',
        color: 'rgba(255, 255, 255, 0.6)'
      }}>
        CZ & Binance Sentiment yükleniyor...
      </div>
    );
  }

  if (!sentimentData) return null;

  const getSentimentColor = () => {
    switch (sentimentData.overallSentiment) {
      case 'BULLISH': return { primary: '#10B981', secondary: 'rgba(16, 185, 129, 0.2)' };
      case 'BEARISH': return { primary: '#EF4444', secondary: 'rgba(239, 68, 68, 0.2)' };
      default: return { primary: '#F59E0B', secondary: 'rgba(245, 158, 11, 0.2)' };
    }
  };

  const colors = getSentimentColor();

  return (
    <div style={{
      background: `linear-gradient(135deg, ${colors.secondary} 0%, rgba(245, 158, 11, 0.08) 100%)`,
      backdropFilter: 'blur(20px)',
      border: `2px solid ${colors.primary}40`,
      borderRadius: '12px',
      padding: '24px',
      boxShadow: `0 4px 20px ${colors.primary}25`,
      transition: 'all 0.3s ease'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            background: `linear-gradient(135deg, ${colors.primary}, ${colors.primary}CC)`,
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: `0 4px 12px ${colors.primary}40`
          }}>
            <span style={{ fontSize: '20px' }}>🏛️</span>
          </div>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '2px' }}>
              CZ & Binance Sentiment
            </h3>
            <p style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
              Market Influence Tracker
            </p>
          </div>
        </div>

        <div style={{
          padding: '6px 12px',
          background: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '8px',
          fontSize: '11px',
          color: 'rgba(255, 255, 255, 0.7)',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <Icons.Clock style={{ width: '14px', height: '14px' }} />
          {countdown}s
        </div>
      </div>

      <div style={{
        padding: '20px',
        background: 'rgba(0, 0, 0, 0.3)',
        borderRadius: '10px',
        marginBottom: '16px',
        border: `1px solid ${colors.primary}30`
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <div>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
              Overall Sentiment
            </div>
            <div style={{ fontSize: '24px', fontWeight: '700', color: colors.primary }}>
              {sentimentData.overallSentiment}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
              Score
            </div>
            <div style={{ fontSize: '32px', fontWeight: '700', color: colors.primary }}>
              {sentimentData.sentimentScore}
            </div>
          </div>
        </div>

        <div style={{
          width: '100%',
          height: '8px',
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${sentimentData.sentimentScore}%`,
            height: '100%',
            background: `linear-gradient(90deg, ${colors.primary}, ${colors.primary}AA)`,
            transition: 'width 0.5s ease'
          }} />
        </div>
      </div>

      <div style={{
        marginTop: '12px',
        fontSize: '10px',
        color: 'rgba(255, 255, 255, 0.5)',
        textAlign: 'center'
      }}>
        Son güncelleme: {new Date(sentimentData.lastUpdate).toLocaleTimeString('tr-TR')}
      </div>
    </div>
  );
}
