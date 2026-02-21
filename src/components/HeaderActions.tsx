'use client';

/**
 * 🎯 HEADER ACTIONS - Modern Premium UI
 * AI Asistan ve Ayarlar ikonları header'da sağ tarafta
 *
 * Özellikler:
 * - Sleek, modern tasarım
 * - Hover efektleri
 * - Glassmorphism stil
 * - Responsive
 */

import { Icons } from '@/components/Icons';
import Link from 'next/link';

interface HeaderActionsProps {
  onAiAssistantOpen: () => void;
}

export function HeaderActions({ onAiAssistantOpen }: HeaderActionsProps) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '12px',
        alignItems: 'center',
        marginLeft: '16px'
      }}
    >
      {/* AI Assistant Button */}
      <button
        onClick={onAiAssistantOpen}
        className="header-action-btn"
        style={{
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '44px',
          height: '44px',
          background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(99, 102, 241, 0.1))',
          border: '1px solid rgba(124, 58, 237, 0.3)',
          borderRadius: '12px',
          cursor: 'pointer',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          backdropFilter: 'blur(10px)',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(99, 102, 241, 0.2))';
          e.currentTarget.style.borderColor = 'rgba(124, 58, 237, 0.6)';
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 8px 24px rgba(124, 58, 237, 0.3)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(99, 102, 241, 0.1))';
          e.currentTarget.style.borderColor = 'rgba(124, 58, 237, 0.3)';
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = 'none';
        }}
        aria-label="AI Asistan"
        title="AI Asistan"
      >
        <Icons.Bot
          style={{
            width: '22px',
            height: '22px',
            color: '#a78bfa',
            filter: 'drop-shadow(0 0 8px rgba(167, 139, 250, 0.5))'
          }}
        />
        {/* Pulse animation */}
        <span
          style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            width: '8px',
            height: '8px',
            background: '#8b5cf6',
            borderRadius: '50%',
            animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            boxShadow: '0 0 0 0 rgba(139, 92, 246, 0.7)',
          }}
        />
      </button>

      {/* News Button */}
      <Link href="/haberler" style={{ textDecoration: 'none' }}>
        <div
          className="header-action-btn"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '44px',
            height: '44px',
            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.1))',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '12px',
            cursor: 'pointer',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backdropFilter: 'blur(10px)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(251, 191, 36, 0.2))';
            e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.6)';
            e.currentTarget.style.transform = 'translateY(-2px)';
            e.currentTarget.style.boxShadow = '0 8px 24px rgba(245, 158, 11, 0.3)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.1))';
            e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.3)';
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
          role="button"
          aria-label="Kripto Haberler"
          title="Kripto Haberler"
        >
          <Icons.Newspaper
            style={{
              width: '22px',
              height: '22px',
              color: '#fbbf24',
              filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.5))'
            }}
          />
        </div>
      </Link>

      {/* Settings Button */}
      <Link href="/settings">
        <button
          className="header-action-btn"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '44px',
            height: '44px',
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(14, 165, 233, 0.1))',
            border: '1px solid rgba(59, 130, 246, 0.3)',
            borderRadius: '12px',
            cursor: 'pointer',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backdropFilter: 'blur(10px)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(14, 165, 233, 0.2))';
            e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.6)';
            e.currentTarget.style.transform = 'translateY(-2px) rotate(90deg)';
            e.currentTarget.style.boxShadow = '0 8px 24px rgba(59, 130, 246, 0.3)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(14, 165, 233, 0.1))';
            e.currentTarget.style.borderColor = 'rgba(59, 130, 246, 0.3)';
            e.currentTarget.style.transform = 'translateY(0) rotate(0deg)';
            e.currentTarget.style.boxShadow = 'none';
          }}
          aria-label="Ayarlar"
          title="Ayarlar"
        >
          <Icons.Eye
            style={{
              width: '22px',
              height: '22px',
              color: '#60a5fa',
              filter: 'drop-shadow(0 0 8px rgba(96, 165, 250, 0.5))'
            }}
          />
        </button>
      </Link>

      {/* Inline CSS for animations */}
      <style jsx global>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
            box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.7);
          }
          50% {
            opacity: 0.5;
            box-shadow: 0 0 0 8px rgba(139, 92, 246, 0);
          }
        }

        .header-action-btn {
          position: relative;
          overflow: hidden;
        }

        .header-action-btn::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 0;
          height: 0;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.1);
          transform: translate(-50%, -50%);
          transition: width 0.6s, height 0.6s;
        }

        .header-action-btn:active::before {
          width: 100px;
          height: 100px;
        }

        /* Mobile responsive */
        @media (max-width: 768px) {
          .header-action-btn {
            width: 40px;
            height: 40px;
          }
        }
      `}</style>
    </div>
  );
}
