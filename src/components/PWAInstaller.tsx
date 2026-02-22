'use client';

/**
 * PWA Installer Component
 * Registers Service Worker and provides install prompt
 */

import { useEffect, useState } from 'react';

export function PWAInstaller() {
  const [deferredPrompt, setDeferredPrompt] = useState<any>(null);
  const [showInstallButton, setShowInstallButton] = useState(false);

  useEffect(() => {
    // ONLY register Service Worker in production (NOT in development)
    if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
      window.addEventListener('load', () => {
        navigator.serviceWorker
          .register('/sw.js')
          .then((registration) => {
            console.log('[PWA] Service Worker kaydedildi:', registration.scope);

            // Her saat başı güncelleme kontrolü
            setInterval(() => {
              registration.update();
            }, 60 * 60 * 1000);
          })
          .catch((error) => {
            console.error('[PWA] Service Worker kaydı başarısız:', error);
          });
      });
    } else if (process.env.NODE_ENV === 'development') {
      console.log('[PWA] Service Worker geliştirme modunda KAPALI');
    }

    // Yükleme istemini yönet
    const handleBeforeInstallPrompt = (e: Event) => {
      console.log('[PWA] Yükleme istemi tetiklendi');
      e.preventDefault();
      setDeferredPrompt(e);
      setShowInstallButton(true);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Başarılı yüklemeyi yönet
    window.addEventListener('appinstalled', () => {
      console.log('[PWA] Uygulama başarıyla yüklendi');
      setDeferredPrompt(null);
      setShowInstallButton(false);

      // Analitik olayı gönder
      if (typeof window !== 'undefined' && (window as any).gtag) {
        (window as any).gtag('event', 'pwa_install', {
          event_category: 'engagement',
          event_label: 'PWA Yüklendi',
        });
      }
    });

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const handleInstallClick = async () => {
    if (!deferredPrompt) return;

    deferredPrompt.prompt();

    const { outcome } = await deferredPrompt.userChoice;
    console.log('[PWA] Kullanıcı seçimi:', outcome);

    if (outcome === 'accepted') {
      console.log('[PWA] Kullanıcı yükleme istemini kabul etti');
    } else {
      console.log('[PWA] Kullanıcı yükleme istemini reddetti');
    }

    setDeferredPrompt(null);
    setShowInstallButton(false);
  };

  if (!showInstallButton) return null;

  return (
    <div
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        zIndex: 9999,
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        border: '2px solid #00ff00',
        borderRadius: '12px',
        padding: '16px 24px',
        boxShadow: '0 0 20px rgba(0, 255, 0, 0.5)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        maxWidth: '320px',
      }}
    >
      <div style={{ color: '#ffffff', fontSize: '14px', fontWeight: '600' }}>
        📱 LyTrade'i Yükle
      </div>
      <div style={{ color: '#8b8b8b', fontSize: '12px' }}>
        Uygulamayı ana ekranınıza ekleyerek daha hızlı erişim sağlayın
      </div>
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={handleInstallClick}
          style={{
            flex: 1,
            background: '#00ff00',
            color: '#0a0a0a',
            border: 'none',
            borderRadius: '6px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: '700',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseOver={(e) => {
            (e.target as HTMLButtonElement).style.background = '#00cc00';
          }}
          onMouseOut={(e) => {
            (e.target as HTMLButtonElement).style.background = '#00ff00';
          }}
        >
          Yükle
        </button>
        <button
          onClick={() => setShowInstallButton(false)}
          style={{
            background: 'transparent',
            color: '#8b8b8b',
            border: '1px solid #444',
            borderRadius: '6px',
            padding: '8px 16px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseOver={(e) => {
            (e.target as HTMLButtonElement).style.borderColor = '#00ff00';
            (e.target as HTMLButtonElement).style.color = '#00ff00';
          }}
          onMouseOut={(e) => {
            (e.target as HTMLButtonElement).style.borderColor = '#444';
            (e.target as HTMLButtonElement).style.color = '#8b8b8b';
          }}
        >
          Şimdi Değil
        </button>
      </div>
    </div>
  );
}
