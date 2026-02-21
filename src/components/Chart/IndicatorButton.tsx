import React, { ReactNode } from 'react';

interface IndicatorButtonProps {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  label: string;
  description: string;
  gradientFrom: string;
  gradientTo: string;
  glowColor: string;
}

export const IndicatorButton = ({
  active,
  onClick,
  icon,
  label,
  description,
  gradientFrom,
  gradientTo,
  glowColor
}: IndicatorButtonProps) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <button
        onClick={onClick}
        style={{
          width: '100%',
          background: active
            ? `linear-gradient(135deg, ${gradientFrom} 0%, ${gradientTo} 100%)`
            : 'linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)',
          backdropFilter: 'blur(20px) saturate(150%)',
          border: active
            ? `1px solid ${gradientFrom}80`
            : '1px solid rgba(255, 255, 255, 0.08)',
          boxShadow: active
            ? `0 8px 32px ${glowColor}, 0 0 0 1px ${gradientFrom}40, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
            : '0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)',
          color: active ? '#ffffff' : 'rgba(255, 255, 255, 0.65)',
          padding: '8px 12px',
          borderRadius: '12px',
          fontSize: '12px',
          fontWeight: '600',
          cursor: 'pointer',
          transition: 'all 0.35s cubic-bezier(0.4, 0, 0.2, 1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
          minHeight: '38px',
          position: 'relative',
          overflow: 'hidden'
        }}
        onMouseEnter={(e) => {
          if (!active) {
            e.currentTarget.style.background = `linear-gradient(135deg, ${gradientFrom}15 0%, ${gradientTo}10 100%)`;
            e.currentTarget.style.borderColor = `${gradientFrom}40`;
          }
          e.currentTarget.style.transform = 'translateY(-2px) scale(1.01)';
          e.currentTarget.style.boxShadow = active
            ? `0 12px 40px ${glowColor}, 0 0 0 1px ${gradientFrom}60, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
            : `0 8px 24px rgba(0, 0, 0, 0.4), 0 0 0 1px ${gradientFrom}30, inset 0 1px 0 rgba(255, 255, 255, 0.1)`;
        }}
        onMouseLeave={(e) => {
          if (!active) {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)';
            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
            e.currentTarget.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05)';
          }
          e.currentTarget.style.transform = 'translateY(0) scale(1)';
        }}
      >
        {/* Subtle shimmer overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '200%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%)',
            animation: active ? 'shimmer 3s infinite' : 'none',
            pointerEvents: 'none'
          }}
        />
        <span style={{ display: 'flex', alignItems: 'center', fontSize: '18px', zIndex: 1 }}>{icon}</span>
        <span style={{ textAlign: 'center', lineHeight: '1.3', zIndex: 1, letterSpacing: '0.3px' }}>{label}</span>
      </button>

      {/* Shimmer keyframes */}
      <style jsx>{`
        @keyframes shimmer {
          0% { transform: translateX(0); }
          100% { transform: translateX(50%); }
        }
      `}</style>

      {/* Sürekli Görünen Açıklama */}
      <div
        style={{
          padding: '8px 12px',
          borderRadius: '8px',
          fontSize: '10px',
          lineHeight: '1.5',
          color: 'rgba(255, 255, 255, 0.6)',
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.01) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.05)',
          backdropFilter: 'blur(10px)',
          transition: 'all 0.3s ease'
        }}
      >
        {description}
      </div>
    </div>
  );
};
