'use client';

/**
 * 🔐 CONDITIONAL AUTH WRAPPER
 * Sadece production'da authentication aktif eder
 * Localhost'ta çalışmaz - sistem çalışırlığına 0 etki
 */

import { ReactNode } from 'react';
import { AuthProvider } from './AuthProvider';

interface ConditionalAuthWrapperProps {
  children: ReactNode;
}

export function ConditionalAuthWrapper({ children }: ConditionalAuthWrapperProps) {
  // Production environment check
  const isProduction = process.env.NODE_ENV === 'production';

  // Localhost bypass - SADECE development için
  const isLocalhost = typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' ||
     window.location.hostname === '127.0.0.1');

  // Production'da VE localhost değilse auth aktif
  const authEnabled = isProduction && !isLocalhost;

  if (!authEnabled) {
    // Development veya localhost - direkt children döndür
    return <>{children}</>;
  }

  // Production - Auth provider ile wrap et
  return <AuthProvider>{children}</AuthProvider>;
}
