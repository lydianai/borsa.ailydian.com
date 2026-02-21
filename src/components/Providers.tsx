'use client';

/**
 * Client-side Providers Wrapper
 *
 * White-hat compliance: Wraps the app with necessary providers
 */

import { ReactNode } from 'react';

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  // SessionProvider disabled - not using authentication for now
  return <>{children}</>;
}
