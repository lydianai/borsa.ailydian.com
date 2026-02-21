/**
 * Ailydian Ecosystem Footer Component - UPDATED
 *
 * KULLANIM TALİMATLARI:
 * 1. Bu component'i subdomain projesine kopyalayın
 * 2. currentDomain parametresini subdomain'inize göre ayarlayın
 * 3. Mevcut footer'dan ÖNCE veya SONRA ekleyin
 * 4. Mevcut kodlara HİÇ dokunmayın!
 *
 * ÖNEMLİ GÜVENLİK:
 * - Bu component hata olursa kendini kapatır (silent fail)
 * - Mevcut sistemlere 0 etki garantisi
 * - Standalone çalışır, bağımlılık yok
 */

'use client';

import { useState, useEffect } from 'react';

interface SubdomainInfo {
  name: string;
  url: string;
  svgIcon: string; // Premium SVG icon
  description: string;
  shortDesc: string;
  color: string;
  gradient: string;
}

const ALL_AILYDIAN_SUBDOMAINS: SubdomainInfo[] = [
  {
    name: "Travel AI",
    url: "https://travel.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/></svg>',
    description: "Akıllı Seyahat & Transfer Platformu",
    shortDesc: "AI Transfer",
    color: "#3b82f6",
    gradient: "from-blue-500 to-cyan-500"
  },
  {
    name: "Voice AI",
    url: "https://voice.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>',
    description: "Ses Teknolojisi - TTS & STT",
    shortDesc: "Türkçe Ses",
    color: "#8b5cf6",
    gradient: "from-purple-500 to-pink-500"
  },
  {
    name: "Oto AI",
    url: "https://otoai.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>',
    description: "Otonom Araç AI Sistemleri",
    shortDesc: "Otonom Araç",
    color: "#10b981",
    gradient: "from-green-500 to-emerald-500"
  },
  {
    name: "Mimar AI",
    url: "https://mimar.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/></svg>',
    description: "AI Destekli Mimari Tasarım",
    shortDesc: "Akıllı Mimari",
    color: "#14b8a6",
    gradient: "from-teal-500 to-cyan-500"
  },
  {
    name: "Tarım AI",
    url: "https://tarim.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    description: "Akıllı Tarım Çözümleri",
    shortDesc: "Precision Ag",
    color: "#22c55e",
    gradient: "from-lime-500 to-green-500"
  },
  {
    name: "Turkey AI",
    url: "https://turkey.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>',
    description: "Seyahat Deneyiminin Geleceği - Sadece Konuşun",
    shortDesc: "Sadece Konuşun",
    color: "#ef4444",
    gradient: "from-red-500 to-rose-500"
  },
  {
    name: "Borsa AI",
    url: "https://borsa.ailydian.com",
    svgIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    description: "AI Destekli Kripto Analizi",
    shortDesc: "Kripto Signal",
    color: "#6366f1",
    gradient: "from-indigo-500 to-blue-500"
  }
];

interface AilydianEcosystemFooterProps {
  currentDomain: string;
  position?: 'above-footer' | 'below-footer';
  theme?: 'dark' | 'light';
}

export default function AilydianEcosystemFooter({
  currentDomain,
  position = 'above-footer',
  theme = 'dark'
}: AilydianEcosystemFooterProps) {
  const [isClient, setIsClient] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    try {
      setIsClient(true);
    } catch (error) {
      console.error('[Ailydian Ecosystem] Init error:', error);
      setHasError(true);
    }
  }, []);

  // Hata varsa veya client-side değilse hiçbir şey render etme (silent fail)
  if (!isClient || hasError) {
    return null;
  }

  // Current domain'i filtrele
  const otherDomains = ALL_AILYDIAN_SUBDOMAINS.filter(
    domain => !domain.url.includes(currentDomain)
  );

  const bgColor = theme === 'dark' ? 'bg-gradient-to-b from-black/50 to-black' : 'bg-gradient-to-b from-white/50 to-white';
  const textColor = theme === 'dark' ? 'text-white' : 'text-black';
  const borderColor = theme === 'dark' ? 'border-white/10' : 'border-black/10';

  return (
    <div
      className={`w-full ${bgColor} ${textColor} border-t ${borderColor} py-12 px-4 sm:px-6 lg:px-8`}
      style={{
        fontFamily: 'system-ui, -apple-system, sans-serif',
        position: 'relative',
        zIndex: 50,
        marginTop: '0'
      }}
    >
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Ailydian Ecosystem
          </h3>
          <p className={`mt-2 text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
            AI-Powered Solutions Across Industries
          </p>
        </div>

        {/* Subdomain Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          {otherDomains.map((domain, index) => (
            <a
              key={index}
              href={domain.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`
                group relative overflow-hidden rounded-xl border ${borderColor}
                ${theme === 'dark' ? 'bg-white/5 hover:bg-white/10' : 'bg-black/5 hover:bg-black/10'}
                p-4 transition-all duration-300 hover:scale-105 hover:shadow-xl
              `}
              style={{
                borderColor: domain.color + '40'
              }}
            >
              {/* Gradient Background on Hover */}
              <div
                className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-300"
                style={{
                  background: `linear-gradient(135deg, ${domain.color}20, ${domain.color}00)`
                }}
              />

              {/* Content */}
              <div className="relative z-10">
                {/* SVG Icon */}
                <div
                  className="mb-3 flex items-center justify-center"
                  style={{ color: domain.color }}
                  dangerouslySetInnerHTML={{ __html: domain.svgIcon }}
                />

                {/* Name */}
                <h4 className="font-semibold text-sm mb-1 text-center">
                  {domain.name}
                </h4>

                {/* Short Description */}
                <p className={`text-xs text-center ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>
                  {domain.shortDesc}
                </p>
              </div>

              {/* Hover Arrow */}
              <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                  stroke="currentColor"
                  className={theme === 'dark' ? 'text-white/60' : 'text-black/60'}
                >
                  <path d="M5 3l6 6-6 6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
            </a>
          ))}
        </div>

        {/* Main Site Link */}
        <div className="text-center">
          <a
            href="https://ailydian.com"
            target="_blank"
            rel="noopener noreferrer"
            className={`
              inline-flex items-center gap-2 px-6 py-3 rounded-full
              ${theme === 'dark' ? 'bg-white/10 hover:bg-white/20' : 'bg-black/10 hover:bg-black/20'}
              transition-all duration-300 font-medium hover:scale-105
            `}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
              <polyline points="9 22 9 12 15 12 15 22"/>
            </svg>
            <span>Ailydian Ana Site</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/>
              <line x1="10" x2="21" y1="14" y2="3"/>
            </svg>
          </a>
        </div>

        {/* Footer Text */}
        <div className={`text-center mt-6 text-xs ${theme === 'dark' ? 'text-gray-500' : 'text-gray-400'}`}>
          <p>Powered by Ailydian - AI-Driven Innovation</p>
        </div>
      </div>
    </div>
  );
}
