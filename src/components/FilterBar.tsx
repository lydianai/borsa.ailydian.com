'use client';

/**
 * 🎯 FILTER BAR - GLOBAL FILTERING UI
 *
 * Premium TradingView-style filter bar with:
 * - Timeframe selection (1H, 4H, 1D, 1W)
 * - SortBy selection (Volume, Change, Price, Name)
 * - Synchronized across all pages via useGlobalFilters hook
 * - Mobile responsive
 * - Türkçe labels
 */

import { useGlobalFilters, type Timeframe, type SortBy } from '@/hooks/useGlobalFilters';
import { COLORS } from '@/lib/colors';

const TIMEFRAME_OPTIONS: { value: Timeframe; label: string }[] = [
  { value: '1H', label: '1S' }, // 1 Saat
  { value: '4H', label: '4S' },
  { value: '1D', label: '1G' }, // 1 Gün
  { value: '1W', label: '1H' }, // 1 Hafta
];

const SORTBY_OPTIONS: { value: SortBy; label: string }[] = [
  { value: 'volume', label: 'Hacim' },
  { value: 'change', label: 'Değişim' },
  { value: 'price', label: 'Fiyat' },
  { value: 'name', label: 'İsim' },
];

export function FilterBar() {
  const { timeframe, sortBy, setTimeframe, setSortBy } = useGlobalFilters();

  return (
    <>
      <style>{`
        @media (max-width: 768px) {
          .filter-bar-container {
            flex-direction: column !important;
            gap: 12px !important;
            padding: 12px !important;
          }
          .filter-bar-section {
            width: 100% !important;
            justify-content: space-between !important;
          }
          .filter-bar-divider {
            display: none !important;
          }
          .filter-bar-button {
            min-height: 44px !important;
            min-width: 44px !important;
            padding: 8px 14px !important;
          }
          .filter-bar-label {
            font-size: 10px !important;
          }
        }
      `}</style>
      <div
        className="filter-bar-container"
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '16px',
          padding: '12px 16px',
          background: COLORS.bg.primary,
          border: `1px solid ${COLORS.border.default}`,
          borderRadius: '8px',
          alignItems: 'center',
        }}
      >
      {/* Timeframe Section */}
      <div
        className="filter-bar-section"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <span
          className="filter-bar-label"
          style={{
            fontSize: '11px',
            fontWeight: '600',
            color: COLORS.text.muted,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            whiteSpace: 'nowrap',
          }}
        >
          ZAMAN DİLİMİ:
        </span>
        <div
          style={{
            display: 'flex',
            gap: '4px',
            background: COLORS.bg.hover,
            borderRadius: '6px',
            padding: '2px',
          }}
        >
          {TIMEFRAME_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => setTimeframe(option.value)}
              className="filter-bar-button"
              style={{
                background:
                  timeframe === option.value
                    ? COLORS.premium
                    : 'transparent',
                color:
                  timeframe === option.value
                    ? '#000'
                    : COLORS.text.primary,
                border: 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: timeframe === option.value ? '700' : '500',
                cursor: 'pointer',
                transition: 'all 0.2s',
                minHeight: '36px',
                minWidth: '36px',
              }}
              onMouseEnter={(e) => {
                if (timeframe !== option.value) {
                  e.currentTarget.style.background = COLORS.bg.card;
                }
              }}
              onMouseLeave={(e) => {
                if (timeframe !== option.value) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* Divider - Hidden on mobile */}
      <div
        className="filter-bar-divider"
        style={{
          width: '1px',
          height: '24px',
          background: COLORS.border.default,
        }}
      />

      {/* SortBy Section */}
      <div
        className="filter-bar-section"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}
      >
        <span
          className="filter-bar-label"
          style={{
            fontSize: '11px',
            fontWeight: '600',
            color: COLORS.text.muted,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            whiteSpace: 'nowrap',
          }}
        >
          SIRALA:
        </span>
        <div
          style={{
            display: 'flex',
            gap: '4px',
            background: COLORS.bg.hover,
            borderRadius: '6px',
            padding: '2px',
          }}
        >
          {SORTBY_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => setSortBy(option.value)}
              className="filter-bar-button"
              style={{
                background:
                  sortBy === option.value
                    ? COLORS.premium
                    : 'transparent',
                color:
                  sortBy === option.value
                    ? '#000'
                    : COLORS.text.primary,
                border: 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '12px',
                fontWeight: sortBy === option.value ? '700' : '500',
                cursor: 'pointer',
                transition: 'all 0.2s',
                minHeight: '36px',
                minWidth: '36px',
              }}
              onMouseEnter={(e) => {
                if (sortBy !== option.value) {
                  e.currentTarget.style.background = COLORS.bg.card;
                }
              }}
              onMouseLeave={(e) => {
                if (sortBy !== option.value) {
                  e.currentTarget.style.background = 'transparent';
                }
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
    </>
  );
}
