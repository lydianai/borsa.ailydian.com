'use client';

/**
 * 🌟 ULTRA PREMIUM LYDIAN LOADING SCREEN
 *
 * Features:
 * - Particle.js benzeri animasyon (pure CSS)
 * - Lydian markası için özel tasarım
 * - Responsive (mobile + desktop)
 * - 60fps performance
 * - Benzersiz cyber/fintech vibe
 */

import { useEffect, useState } from 'react';

interface PremiumLoaderProps {
  text?: string;
  subtext?: string;
}

export function PremiumLoader({
  text = 'CRYPTO SIGNALS',
  subtext = 'by Lydian'
}: PremiumLoaderProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  // Generate particles (80 particles for ultra density)
  const particles = Array.from({ length: 80 }, (_, i) => {
    const randomX = Math.random() * 100;
    const randomY = Math.random() * 100;
    const randomSize = Math.random() * 4 + 0.5;
    const randomDuration = Math.random() * 20 + 8;
    const randomDelay = Math.random() * 8;
    const randomOpacity = Math.random() * 0.6 + 0.4;

    return {
      id: i,
      x: randomX,
      y: randomY,
      size: randomSize,
      duration: randomDuration,
      delay: randomDelay,
      opacity: randomOpacity,
    };
  });

  return (
    <div
      suppressHydrationWarning
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        overflow: 'hidden',
      }}
    >
      {/* Animated Grid Background */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundImage: `
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'gridMove 20s linear infinite',
          pointerEvents: 'none',
        }}
      />

      {/* Particles - Ultra Dense Network */}
      {particles.map((particle) => (
        <div
          key={particle.id}
          style={{
            position: 'absolute',
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            background: particle.size > 2.5
              ? 'linear-gradient(135deg, #00D4FF, #FFD700, #0EA5E9)'
              : particle.size > 1.5
              ? 'radial-gradient(circle, #00D4FF, rgba(0, 212, 255, 0.4))'
              : 'rgba(0, 212, 255, 0.7)',
            borderRadius: '50%',
            boxShadow: particle.size > 2.5
              ? '0 0 15px rgba(0, 212, 255, 1), 0 0 30px rgba(0, 212, 255, 0.6), 0 0 45px rgba(255, 215, 0, 0.4)'
              : particle.size > 1.5
              ? '0 0 8px rgba(0, 212, 255, 0.8)'
              : '0 0 4px rgba(0, 212, 255, 0.5)',
            animation: `particleFloat ${particle.duration}s ease-in-out ${particle.delay}s infinite, particleGlow ${particle.duration * 0.7}s ease-in-out ${particle.delay}s infinite`,
            opacity: particle.opacity,
            pointerEvents: 'none',
          }}
        />
      ))}

      {/* Connecting Lines - Enhanced Network Effect */}
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          opacity: 0.25,
        }}
      >
        <defs>
          <linearGradient id="lineGradient1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00D4FF" stopOpacity="0" />
            <stop offset="50%" stopColor="#00D4FF" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#00D4FF" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="lineGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#FFD700" stopOpacity="0" />
            <stop offset="50%" stopColor="#FFD700" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#FFD700" stopOpacity="0" />
          </linearGradient>
        </defs>
        {particles.slice(0, 16).map((p1, i) => {
          const p2 = particles[(i + 4) % particles.length];
          const p3 = particles[(i + 8) % particles.length];
          return (
            <g key={`lines-${i}`}>
              <line
                x1={`${p1.x}%`}
                y1={`${p1.y}%`}
                x2={`${p2.x}%`}
                y2={`${p2.y}%`}
                stroke="url(#lineGradient1)"
                strokeWidth="0.8"
                style={{
                  animation: `lineOpacity 5s ease-in-out ${i * 0.3}s infinite`,
                }}
              />
              {i % 2 === 0 && (
                <line
                  x1={`${p2.x}%`}
                  y1={`${p2.y}%`}
                  x2={`${p3.x}%`}
                  y2={`${p3.y}%`}
                  stroke="url(#lineGradient2)"
                  strokeWidth="0.5"
                  style={{
                    animation: `lineOpacity 6s ease-in-out ${i * 0.4}s infinite`,
                  }}
                />
              )}
            </g>
          );
        })}
      </svg>

      {/* Central Glow Effect - Enhanced with Rings */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '600px',
          height: '600px',
          background: 'radial-gradient(circle, rgba(0, 212, 255, 0.2) 0%, rgba(255, 215, 0, 0.1) 40%, transparent 70%)',
          animation: 'pulseGlow 4s ease-in-out infinite',
          pointerEvents: 'none',
        }}
      />

      {/* Rotating Rings */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '300px',
          height: '300px',
          border: '2px solid rgba(0, 212, 255, 0.3)',
          borderRadius: '50%',
          animation: 'rotateRing 8s linear infinite',
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '400px',
          height: '400px',
          border: '1px dashed rgba(255, 215, 0, 0.2)',
          borderRadius: '50%',
          animation: 'rotateRing 12s linear reverse infinite',
          pointerEvents: 'none',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '200px',
          height: '200px',
          border: '3px solid rgba(0, 212, 255, 0.4)',
          borderRadius: '50%',
          borderStyle: 'dotted',
          animation: 'rotateRing 6s linear infinite',
          pointerEvents: 'none',
        }}
      />

      {/* Main Content */}
      <div
        style={{
          position: 'relative',
          zIndex: 10,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '24px',
        }}
      >
        {/* Lydian Logo Text */}
        <div style={{ textAlign: 'center' }}>
          <div
            style={{
              fontSize: 'clamp(32px, 5vw, 48px)',
              fontWeight: '900',
              background: 'linear-gradient(135deg, #00D4FF 0%, #0EA5E9 50%, #00D4FF 100%)',
              backgroundSize: '200% auto',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '2px',
              animation: 'shimmer 3s linear infinite, fadeInScale 0.8s ease-out',
              textShadow: '0 0 40px rgba(0, 212, 255, 0.3)',
            }}
          >
            {text}
          </div>
          <div
            style={{
              fontSize: 'clamp(12px, 2vw, 16px)',
              fontWeight: '600',
              color: '#FFD700',
              letterSpacing: '4px',
              marginTop: '8px',
              animation: 'fadeIn 1s ease-out 0.3s both',
            }}
          >
            {subtext}
          </div>
        </div>

        {/* Loading Bar */}
        <div
          style={{
            width: '280px',
            height: '4px',
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '2px',
            overflow: 'hidden',
            position: 'relative',
            animation: 'fadeIn 1s ease-out 0.5s both',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, transparent, #00D4FF, transparent)',
              animation: 'loadingBar 2s ease-in-out infinite',
            }}
          />
        </div>

        {/* Loading Dots */}
        <div
          style={{
            display: 'flex',
            gap: '8px',
            animation: 'fadeIn 1s ease-out 0.7s both',
          }}
        >
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: '8px',
                height: '8px',
                background: 'linear-gradient(135deg, #00D4FF, #0EA5E9)',
                borderRadius: '50%',
                animation: `dotPulse 1.5s ease-in-out ${i * 0.2}s infinite`,
                boxShadow: '0 0 10px rgba(0, 212, 255, 0.6)',
              }}
            />
          ))}
        </div>

        {/* Loading Text */}
        <div
          style={{
            fontSize: '14px',
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.5)',
            letterSpacing: '2px',
            animation: 'fadeIn 1s ease-out 0.9s both, textPulse 2s ease-in-out 1s infinite',
          }}
        >
          Yükleniyor
        </div>
      </div>

      {/* CSS Animations */}
      <style jsx global suppressHydrationWarning>{`
        @keyframes particleFloat {
          0%, 100% {
            transform: translate(0, 0) scale(1) rotate(0deg);
          }
          25% {
            transform: translate(30px, -40px) scale(1.3) rotate(90deg);
          }
          50% {
            transform: translate(-20px, -60px) scale(0.7) rotate(180deg);
          }
          75% {
            transform: translate(35px, -25px) scale(1.2) rotate(270deg);
          }
        }

        @keyframes particleGlow {
          0%, 100% {
            filter: brightness(1) blur(0px);
          }
          50% {
            filter: brightness(1.8) blur(2px);
          }
        }

        @keyframes gridMove {
          0% {
            transform: translateY(0) translateX(0);
          }
          100% {
            transform: translateY(50px) translateX(50px);
          }
        }

        @keyframes pulseGlow {
          0%, 100% {
            opacity: 0.3;
            transform: translate(-50%, -50%) scale(1);
          }
          50% {
            opacity: 0.6;
            transform: translate(-50%, -50%) scale(1.1);
          }
        }

        @keyframes shimmer {
          0% {
            background-position: 0% 50%;
          }
          100% {
            background-position: 200% 50%;
          }
        }

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes fadeInScale {
          from {
            opacity: 0;
            transform: scale(0.8);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }

        @keyframes loadingBar {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(200%);
          }
        }

        @keyframes dotPulse {
          0%, 100% {
            transform: scale(1);
            opacity: 1;
          }
          50% {
            transform: scale(1.5);
            opacity: 0.5;
          }
        }

        @keyframes textPulse {
          0%, 100% {
            opacity: 0.5;
          }
          50% {
            opacity: 1;
          }
        }

        @keyframes lineOpacity {
          0%, 100% {
            opacity: 0.1;
          }
          50% {
            opacity: 0.4;
          }
        }

        @keyframes rotateRing {
          0% {
            transform: translate(-50%, -50%) rotate(0deg);
          }
          100% {
            transform: translate(-50%, -50%) rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
