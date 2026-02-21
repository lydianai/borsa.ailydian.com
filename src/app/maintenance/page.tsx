/**
 * 🚧 MAINTENANCE MODE PAGE
 *
 * Profesyonel yapım aşaması sayfası
 * Sistem arkada çalışmaya devam ediyor (API'ler, Telegram, Python servisleri)
 */

'use client';

import { useState } from 'react';

export default function MaintenancePage() {
  const [showLogicModal, setShowLogicModal] = useState(false);
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        {/* Main Card */}
        <div className="bg-slate-800/50 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-2xl p-8 md:p-12">
          {/* Icon */}
          <div className="flex justify-center mb-8">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500 blur-2xl opacity-20 rounded-full"></div>
              <div className="relative bg-slate-700/50 rounded-full p-6 border border-slate-600/50">
                <svg
                  className="w-16 h-16 text-blue-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
              </div>
            </div>
          </div>

          {/* Title */}
          <div className="text-center mb-6">
            <div className="flex items-center justify-center gap-4 mb-4">
              <h1 className="text-4xl md:text-5xl font-bold text-white">
                Sistem Güncelleniyor
              </h1>

              {/* MANTIK Button - Responsive */}
              <div>
                <style>{`
                  @media (max-width: 768px) {
                    .mantik-button-maintenance {
                      padding: 10px 20px !important;
                      fontSize: 13px !important;
                      height: 42px !important;
                    }
                    .mantik-button-maintenance span {
                      fontSize: 18px !important;
                    }
                  }
                  @media (max-width: 480px) {
                    .mantik-button-maintenance {
                      padding: 8px 16px !important;
                      fontSize: 12px !important;
                      height: 40px !important;
                    }
                    .mantik-button-maintenance span {
                      fontSize: 16px !important;
                    }
                  }
                `}</style>
                <button
                  onClick={() => setShowLogicModal(true)}
                  className="mantik-button-maintenance"
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
                  <span style={{ fontSize: '20px' }}>🧠</span>
                  MANTIK
                </button>
              </div>
            </div>
            <p className="text-xl text-slate-300 mb-2">
              Daha iyi hizmet için çalışıyoruz
            </p>
          </div>

          {/* Description */}
          <div className="bg-slate-700/30 rounded-xl p-6 mb-8 border border-slate-600/30">
            <p className="text-slate-300 text-center leading-relaxed">
              LyDian AI Trading platformu şu anda bakım modunda.
              Sistemimizi geliştiriyoruz ve çok yakında daha güçlü özelliklerle
              geri döneceğiz.
            </p>
          </div>

          {/* Features List */}
          <div className="space-y-4 mb-8">
            <div className="flex items-center gap-3 text-slate-300">
              <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span>AI sinyal sistemleri aktif çalışıyor</span>
            </div>

            <div className="flex items-center gap-3 text-slate-300">
              <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span>Telegram bildirimleri kesintisiz</span>
            </div>

            <div className="flex items-center gap-3 text-slate-300">
              <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span>Python analiz servisleri çalışıyor</span>
            </div>

            <div className="flex items-center gap-3 text-slate-300">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span>Tahmini süre: Birkaç saat</span>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center justify-center gap-3 pt-6 border-t border-slate-700/50">
            <div className="relative">
              <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
              <div className="absolute inset-0 w-3 h-3 bg-blue-500 rounded-full animate-ping"></div>
            </div>
            <span className="text-slate-400 text-sm">
              Sistem arka planda çalışıyor
            </span>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-slate-500 text-sm">
            Sorularınız için:{' '}
            <a
              href="https://t.me/ailydian"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              @ailydian
            </a>
          </p>
        </div>
      </div>

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
                Bakım Modu MANTIK
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

            {/* Content */}
            <div style={{ color: 'rgba(255, 255, 255, 0.9)', lineHeight: '1.8' }}>

              {/* Section 1: Purpose */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#10B981', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  📌 Sayfa Amacı
                </h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0 }}>
                  Bakım modu sayfası, sistem güncellemeleri sırasında kullanıcıları bilgilendirmek için tasarlanmıştır.
                  Arka planda tüm servisler (API'ler, Telegram, Python) çalışmaya devam eder.
                </p>
              </div>

              {/* Section 2: How It Works */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#3B82F6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ⚙️ Nasıl Çalışır?
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Frontend güncellemeleri sırasında gösterilir</li>
                  <li>Statik bilgilendirme sayfası - API çağrısı yapmaz</li>
                  <li>Backend servisleri kesintisiz çalışmaya devam eder</li>
                  <li>Kullanıcı dostu animasyonlu durum göstergesi</li>
                </ul>
              </div>

              {/* Section 3: Key Features */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#F59E0B', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  ✨ Önemli Özellikler
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Temiz ve profesyonel glassmorphism tasarım</li>
                  <li>Tahmini geri dönüş süresi bilgisi</li>
                  <li>Aktif servislerin durumu (✅ işaretli liste)</li>
                  <li>Telegram destek linki</li>
                  <li>Animasyonlu durum göstergesi</li>
                </ul>
              </div>

              {/* Section 4: System Status */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#EC4899', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  🔌 Sistem Durumu
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>✅ AI sinyal sistemleri: Aktif</li>
                  <li>✅ Telegram bildirimleri: Kesintisiz</li>
                  <li>✅ Python analiz servisleri: Çalışıyor</li>
                  <li>🔵 Frontend: Bakım modunda</li>
                </ul>
              </div>

              {/* Section 5: Usage Tips */}
              <div>
                <h3 style={{ color: '#8B5CF6', fontSize: '20px', fontWeight: '600', marginBottom: '12px' }}>
                  💡 Kullanıcı Bilgilendirmesi
                </h3>
                <ul style={{ margin: 0, paddingLeft: '20px', color: 'rgba(255, 255, 255, 0.8)' }}>
                  <li>Bakım süresi genellikle birkaç saat sürer</li>
                  <li>Telegram kanalından anlık güncellemeler alabilirsiniz</li>
                  <li>Backend servisleri kesintisiz çalıştığı için sinyal kaybı olmaz</li>
                  <li>Sayfa otomatik olarak güncellendiğinde erişilebilir olacak</li>
                </ul>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
