'use client';

/**
 * 📰 KRİPTO HABERLER SAYFASI - YAPIM AŞAMASINDA
 * Gerçek fiyatlar ve anlık haberler yakında eklenecektir.
 */

import { SharedSidebar } from '@/components/SharedSidebar';
import { PWAProvider } from '@/components/PWAProvider';
import { COLORS } from '@/lib/colors';
import { Icons } from '@/components/Icons';

export default function HaberlerPage() {
  return (
    <PWAProvider>
      <div className="dashboard-container">
        <SharedSidebar currentPage="haberler" />

        <div className="dashboard-main">
          <main className="dashboard-content" style={{
            padding: '24px',
            paddingTop: '80px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh'
          }}>
            <div style={{
              textAlign: 'center',
              maxWidth: '600px',
              padding: '40px',
              background: COLORS.bg.primary,
              border: `2px solid ${COLORS.border.default}`,
              borderRadius: '16px',
            }}>
              {/* Construction Icon */}
              <div style={{
                fontSize: '80px',
                marginBottom: '24px',
                animation: 'pulse 2s ease-in-out infinite',
              }}>
                🚧
              </div>

              {/* Title */}
              <h1 className="neon-text" style={{
                fontSize: '2rem',
                marginBottom: '16px',
                color: COLORS.premium,
              }}>
                Yapım Aşamasında
              </h1>

              {/* Subtitle */}
              <h2 style={{
                fontSize: '1.25rem',
                marginBottom: '24px',
                color: COLORS.text.primary,
              }}>
                📰 Kripto Haberler
              </h2>

              {/* Description */}
              <div style={{
                color: COLORS.text.secondary,
                fontSize: '16px',
                lineHeight: '1.8',
                marginBottom: '32px',
              }}>
                <p style={{ marginBottom: '16px' }}>
                  Bu sayfa şu anda geliştirilme aşamasındadır.
                </p>
                <div style={{
                  background: COLORS.bg.secondary,
                  padding: '20px',
                  borderRadius: '12px',
                  border: `1px solid ${COLORS.border.default}`,
                  marginTop: '24px',
                }}>
                  <p style={{
                    fontWeight: '600',
                    color: COLORS.text.primary,
                    marginBottom: '12px',
                  }}>
                    ✨ Yakında Eklenecek Özellikler:
                  </p>
                  <ul style={{
                    textAlign: 'left',
                    paddingLeft: '20px',
                    color: COLORS.text.secondary,
                  }}>
                    <li style={{ marginBottom: '8px' }}>📊 Gerçek zamanlı kripto fiyatları</li>
                    <li style={{ marginBottom: '8px' }}>📰 Anlık kripto haberleri</li>
                    <li style={{ marginBottom: '8px' }}>🌐 Türkçe çeviri desteği</li>
                    <li style={{ marginBottom: '8px' }}>🔔 Önemli haber bildirimleri</li>
                    <li style={{ marginBottom: '8px' }}>📈 Piyasa etkisi analizi</li>
                  </ul>
                </div>
              </div>

              {/* Icons */}
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                gap: '16px',
                marginTop: '32px',
              }}>
                <Icons.Fire style={{ width: '32px', height: '32px', color: COLORS.warning }} />
                <Icons.TrendingUp style={{ width: '32px', height: '32px', color: COLORS.success }} />
                <Icons.Bot style={{ width: '32px', height: '32px', color: COLORS.premium }} />
              </div>

              {/* Footer Note */}
              <p style={{
                marginTop: '32px',
                fontSize: '14px',
                color: COLORS.text.muted,
                fontStyle: 'italic',
              }}>
                Bu sayfanın tamamlanması için çalışmalar devam etmektedir.
              </p>
            </div>
          </main>
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.8;
            transform: scale(1.05);
          }
        }
      `}</style>
    </PWAProvider>
  );
}
