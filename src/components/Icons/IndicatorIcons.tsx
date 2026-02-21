/**
 * Premium SVG İkonlar - İndikatörler İçin
 * Glassmorphism ve modern tasarım ile uyumlu
 */

import React from 'react';

interface IconProps {
  className?: string;
  size?: number;
}

// Bollinger Bantları İkonu
export const BollingerIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 12C3 12 5 8 8 8C11 8 13 12 16 12C19 12 21 8 21 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M3 16C3 16 5 12 8 12C11 12 13 16 16 16C19 16 21 12 21 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.5"/>
    <path d="M3 8C3 8 5 4 8 4C11 4 13 8 16 8C19 8 21 4 21 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.5"/>
  </svg>
);

// Hareketli Ortalama İkonu
export const MovingAverageIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 18C3 18 6 12 9 12C12 12 15 18 18 18C21 18 24 12 24 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="9" cy="12" r="2" fill="currentColor"/>
    <circle cx="18" cy="18" r="2" fill="currentColor"/>
  </svg>
);

// VWAP İkonu
export const VWAPIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 20L8 10L13 15L21 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <rect x="7" y="18" width="2" height="4" fill="currentColor" opacity="0.6"/>
    <rect x="12" y="14" width="2" height="8" fill="currentColor" opacity="0.6"/>
    <rect x="19" y="10" width="2" height="12" fill="currentColor" opacity="0.6"/>
  </svg>
);

// FVG / İmbalans İkonu
export const FVGIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="4" y="8" width="6" height="10" stroke="currentColor" strokeWidth="2" fill="none"/>
    <rect x="14" y="6" width="6" height="14" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M10 13H14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeDasharray="2 2"/>
  </svg>
);

// Emir Blokları İkonu
export const OrderBlockIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="4" y="6" width="7" height="12" fill="currentColor" opacity="0.3" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="13" y="10" width="7" height="8" fill="currentColor" opacity="0.5" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M8 2L8 22" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 2"/>
  </svg>
);

// Destek/Direnç İkonu
export const SupportResistanceIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <line x1="2" y1="8" x2="22" y2="8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <line x1="2" y1="16" x2="22" y2="16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M6 12L10 8L14 12L18 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6"/>
  </svg>
);

// Fibonacci İkonu
export const FibonacciIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 20C3 20 6 15 10 12C14 9 17 5 21 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <line x1="3" y1="6" x2="21" y2="6" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
    <line x1="3" y1="10" x2="21" y2="10" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
    <line x1="3" y1="14" x2="21" y2="14" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
    <line x1="3" y1="18" x2="21" y2="18" stroke="currentColor" strokeWidth="1" opacity="0.3"/>
  </svg>
);

// RSI İkonu
export const RSIIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 12H21" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" opacity="0.3"/>
    <path d="M3 18L6 12L9 15L12 9L15 14L18 8L21 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <rect x="2" y="4" width="20" height="16" stroke="currentColor" strokeWidth="1.5" fill="none" rx="2"/>
  </svg>
);

// MFI İkonu
export const MFIIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 6V12L16 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <circle cx="12" cy="12" r="2" fill="currentColor"/>
  </svg>
);

// Hacim İkonu
export const VolumeIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="3" y="14" width="3" height="7" fill="currentColor" opacity="0.5"/>
    <rect x="7" y="10" width="3" height="11" fill="currentColor" opacity="0.7"/>
    <rect x="11" y="6" width="3" height="15" fill="currentColor"/>
    <rect x="15" y="12" width="3" height="9" fill="currentColor" opacity="0.7"/>
    <rect x="19" y="16" width="3" height="5" fill="currentColor" opacity="0.5"/>
  </svg>
);

// Delta İkonu
export const DeltaIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M12 3L21 21H3L12 3Z" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 10V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M9 14L12 16L15 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Likidite Havuzları İkonu
export const LiquidityIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" fill="none"/>
    <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.5"/>
    <circle cx="12" cy="12" r="2" fill="currentColor"/>
    <path d="M12 3V6" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M12 18V21" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M3 12H6" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M18 12H21" stroke="currentColor" strokeWidth="1.5"/>
  </svg>
);

// Piyasa Yapısı İkonu
export const MarketStructureIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <path d="M3 18L7 14L11 16L15 10L19 12L23 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="7" cy="14" r="2" fill="currentColor"/>
    <circle cx="15" cy="10" r="2" fill="currentColor"/>
    <circle cx="23" cy="6" r="2" fill="currentColor"/>
  </svg>
);

// Premium/İndirim Bölgesi İkonu
export const PremiumDiscountIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="4" y="4" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" rx="2"/>
    <path d="M4 12H20" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 3"/>
    <path d="M4 8H20" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
    <path d="M4 16H20" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
  </svg>
);

// POC (Kontrol Noktası) İkonu
export const POCIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none"/>
    <circle cx="12" cy="12" r="4" fill="currentColor"/>
    <path d="M12 2V7" stroke="currentColor" strokeWidth="1.5"/>
    <path d="M12 17V22" stroke="currentColor" strokeWidth="1.5"/>
  </svg>
);

// Değer Alanı İkonu
export const ValueAreaIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <rect x="4" y="6" width="3" height="12" fill="currentColor" opacity="0.3"/>
    <rect x="8" y="4" width="3" height="16" fill="currentColor" opacity="0.5"/>
    <rect x="12" y="2" width="3" height="20" fill="currentColor" opacity="0.8"/>
    <rect x="16" y="5" width="3" height="14" fill="currentColor" opacity="0.5"/>
    <rect x="20" y="8" width="3" height="8" fill="currentColor" opacity="0.3"/>
    <line x1="2" y1="8" x2="22" y2="8" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2"/>
    <line x1="2" y1="16" x2="22" y2="16" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2"/>
  </svg>
);

// Seans Seviyeleri İkonu
export const SessionIcon = ({ className = "", size = 16 }: IconProps) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className}>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" fill="none"/>
    <path d="M12 12L16 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M12 12L12 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <circle cx="12" cy="12" r="1.5" fill="currentColor"/>
  </svg>
);
