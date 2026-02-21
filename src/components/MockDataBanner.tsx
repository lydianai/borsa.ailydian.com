'use client';

/**
 * Mock Data Warning Banner
 *
 * White-hat compliance: Clearly warn users when viewing demo/mock data
 * to prevent confusion and misuse
 */

import React, { useState, useEffect } from 'react';
import { MOCK_DATA_BANNER_TEXT, shouldUseMockData } from '@/lib/utils/mock-detection';

export function MockDataBanner(): JSX.Element | null {
  const [isVisible, setIsVisible] = useState(false);
  const [isDismissed, setIsDismissed] = useState(false);

  useEffect(() => {
    // Check if we should show the banner
    const useMock = shouldUseMockData();
    setIsVisible(useMock);

    // Check if user previously dismissed it (session storage)
    const dismissed = sessionStorage.getItem('mockDataBannerDismissed') === 'true';
    setIsDismissed(dismissed);
  }, []);

  if (!isVisible || isDismissed) {
    return null;
  }

  const handleDismiss = () => {
    setIsDismissed(true);
    sessionStorage.setItem('mockDataBannerDismissed', 'true');
  };

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-yellow-500 text-black border-b-2 border-yellow-600">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 flex-1">
            {/* Warning Icon */}
            <svg
              className="w-6 h-6 flex-shrink-0 mt-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>

            {/* Content */}
            <div className="flex-1">
              <h3 className="font-bold text-sm">{MOCK_DATA_BANNER_TEXT.title}</h3>
              <p className="text-xs mt-1 opacity-90">
                {MOCK_DATA_BANNER_TEXT.description}
              </p>
              <p className="text-xs mt-1 font-medium">
                💡 {MOCK_DATA_BANNER_TEXT.action}
              </p>
            </div>
          </div>

          {/* Dismiss Button */}
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 p-1 hover:bg-yellow-600 rounded transition-colors"
            aria-label="Dismiss banner"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Inline Mock Data Badge (for individual components)
 */
export function MockDataBadge(): JSX.Element {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-800 text-xs font-medium rounded border border-yellow-300">
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      DEMO DATA
    </span>
  );
}
