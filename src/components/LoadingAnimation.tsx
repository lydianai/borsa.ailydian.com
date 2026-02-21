'use client';

import React, { useEffect, useState } from 'react';

export function LoadingAnimation() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Prevent hydration mismatch by only rendering on client
  if (!mounted) {
    return null;
  }

  return (
    <div
      suppressHydrationWarning
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '32px',
        padding: '40px',
      }}
    >
      {/* Premium LyDian DNA Helix Loading Icon */}
      <div
        suppressHydrationWarning
        style={{
          position: 'relative',
          width: '200px',
          height: '200px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <svg
          suppressHydrationWarning
          width="200"
          height="200"
          viewBox="0 0 200 200"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          style={{
            filter: 'drop-shadow(0 0 40px rgba(139, 92, 246, 0.6))',
          }}
        >
          <defs>
            {/* Premium Purple-Cyan Gradient */}
            <linearGradient id="premiumGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#8B5CF6">
                <animate attributeName="stop-color" values="#8B5CF6;#A78BFA;#8B5CF6" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="50%" stopColor="#06B6D4">
                <animate attributeName="stop-color" values="#06B6D4;#22D3EE;#06B6D4" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="#3B82F6">
                <animate attributeName="stop-color" values="#3B82F6;#60A5FA;#3B82F6" dur="3s" repeatCount="indefinite" />
              </stop>
            </linearGradient>

            {/* Bright Accent Gradient */}
            <linearGradient id="accentGlow" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#22D3EE" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.3" />
            </linearGradient>

            {/* Radial Glow for Depth */}
            <radialGradient id="centerGlow">
              <stop offset="0%" stopColor="#A78BFA" stopOpacity="0.6" />
              <stop offset="50%" stopColor="#8B5CF6" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
            </radialGradient>

            {/* Glow Filter */}
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Outer Breathing Ring */}
          <circle
            cx="100"
            cy="100"
            r="85"
            stroke="url(#premiumGradient)"
            strokeWidth="2"
            fill="none"
            opacity="0.4"
          >
            <animate
              attributeName="r"
              values="85;90;85"
              dur="4s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="stroke-width"
              values="2;4;2"
              dur="4s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.4;0.7;0.4"
              dur="4s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Central Background Glow */}
          <circle
            cx="100"
            cy="100"
            r="60"
            fill="url(#centerGlow)"
          >
            <animate
              attributeName="r"
              values="55;65;55"
              dur="3s"
              repeatCount="indefinite"
            />
          </circle>

          {/* DNA Helix - Left Strand */}
          <g>
            {[0, 1, 2, 3, 4].map((i) => {
              const y = 50 + (i * 100 / 4);
              const offset = Math.sin((i * Math.PI) / 2) * 25;
              return (
                <circle
                  key={`left-${i}`}
                  cx={100 + offset}
                  cy={y}
                  r="4"
                  fill="#8B5CF6"
                  filter="url(#glow)"
                >
                  <animate
                    attributeName="cy"
                    values={`${y};${y + 150};${y}`}
                    dur="3s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="cx"
                    values={`${100 + offset};${100 - offset};${100 + offset}`}
                    dur="3s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="r"
                    values="4;6;4"
                    dur="1.5s"
                    repeatCount="indefinite"
                    begin={`${i * 0.3}s`}
                  />
                </circle>
              );
            })}
          </g>

          {/* DNA Helix - Right Strand */}
          <g>
            {[0, 1, 2, 3, 4].map((i) => {
              const y = 50 + (i * 100 / 4);
              const offset = Math.sin((i * Math.PI) / 2) * 25;
              return (
                <circle
                  key={`right-${i}`}
                  cx={100 - offset}
                  cy={y}
                  r="4"
                  fill="#22D3EE"
                  filter="url(#glow)"
                >
                  <animate
                    attributeName="cy"
                    values={`${y};${y + 150};${y}`}
                    dur="3s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="cx"
                    values={`${100 - offset};${100 + offset};${100 - offset}`}
                    dur="3s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="r"
                    values="4;6;4"
                    dur="1.5s"
                    repeatCount="indefinite"
                    begin={`${i * 0.3 + 0.15}s`}
                  />
                </circle>
              );
            })}
          </g>

          {/* Rotating Orbit Rings */}
          <circle
            cx="100"
            cy="100"
            r="70"
            stroke="url(#accentGlow)"
            strokeWidth="1.5"
            fill="none"
            strokeDasharray="5 10"
            opacity="0.6"
          >
            <animateTransform
              attributeName="transform"
              type="rotate"
              from="0 100 100"
              to="360 100 100"
              dur="10s"
              repeatCount="indefinite"
            />
          </circle>

          <circle
            cx="100"
            cy="100"
            r="75"
            stroke="url(#premiumGradient)"
            strokeWidth="1.5"
            fill="none"
            strokeDasharray="3 8"
            opacity="0.4"
          >
            <animateTransform
              attributeName="transform"
              type="rotate"
              from="360 100 100"
              to="0 100 100"
              dur="8s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Orbiting Energy Particles */}
          {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, index) => {
            const x = 100 + 65 * Math.cos((angle * Math.PI) / 180);
            const y = 100 + 65 * Math.sin((angle * Math.PI) / 180);
            return (
              <g key={index}>
                <circle
                  cx={x}
                  cy={y}
                  r="3"
                  fill={index % 2 === 0 ? '#8B5CF6' : '#22D3EE'}
                  filter="url(#glow)"
                >
                  <animateTransform
                    attributeName="transform"
                    type="rotate"
                    from={`${angle} 100 100`}
                    to={`${angle + 360} 100 100`}
                    dur="6s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="r"
                    values="3;5;3"
                    dur="2s"
                    repeatCount="indefinite"
                    begin={`${index * 0.25}s`}
                  />
                  <animate
                    attributeName="opacity"
                    values="0.5;1;0.5"
                    dur="2s"
                    repeatCount="indefinite"
                    begin={`${index * 0.25}s`}
                  />
                </circle>
              </g>
            );
          })}

          {/* Corner Sparkles */}
          {[
            { x: 25, y: 25, delay: 0 },
            { x: 175, y: 25, delay: 0.5 },
            { x: 25, y: 175, delay: 1 },
            { x: 175, y: 175, delay: 1.5 },
          ].map((point, i) => (
            <g key={i}>
              <circle cx={point.x} cy={point.y} r="2" fill="#A78BFA">
                <animate
                  attributeName="opacity"
                  values="0;1;0"
                  dur="3s"
                  repeatCount="indefinite"
                  begin={`${point.delay}s`}
                />
                <animate
                  attributeName="r"
                  values="2;4;2"
                  dur="3s"
                  repeatCount="indefinite"
                  begin={`${point.delay}s`}
                />
              </circle>
            </g>
          ))}
        </svg>
      </div>

      {/* Brand Name - LyDian */}
      <div
        suppressHydrationWarning
        style={{
          fontSize: '32px',
          fontWeight: '800',
          background: 'linear-gradient(135deg, #8B5CF6, #22D3EE, #3B82F6)',
          backgroundSize: '200% 200%',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          letterSpacing: '6px',
          animation: 'gradientShift 3s ease infinite, textPulse 2s ease-in-out infinite',
          textShadow: '0 0 20px rgba(139, 92, 246, 0.3)',
        }}
      >
        LyDian
      </div>

      {/* Premium Loading Bar */}
      <div
        suppressHydrationWarning
        style={{
          width: '240px',
          height: '6px',
          background: 'rgba(139, 92, 246, 0.2)',
          borderRadius: '3px',
          overflow: 'hidden',
          position: 'relative',
          border: '1px solid rgba(139, 92, 246, 0.3)',
        }}
      >
        <div
          suppressHydrationWarning
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, #8B5CF6, #22D3EE, #3B82F6, #22D3EE, #8B5CF6)',
            backgroundSize: '200% 100%',
            animation: 'liquidFlow 2s linear infinite',
            borderRadius: '3px',
            boxShadow: '0 0 15px rgba(139, 92, 246, 0.5)',
          }}
        />
      </div>

      {/* Animated Dots */}
      <div
        suppressHydrationWarning
        style={{
          display: 'flex',
          gap: '12px',
          alignItems: 'center',
        }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            suppressHydrationWarning
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #8B5CF6, #22D3EE)',
              animation: 'dotBounce 1.4s infinite ease-in-out both',
              animationDelay: `${-0.32 + i * 0.16}s`,
              boxShadow: '0 0 15px rgba(139, 92, 246, 0.6)',
            }}
          />
        ))}
      </div>

      {/* CSS Animations */}
      {mounted && (
        <style dangerouslySetInnerHTML={{
          __html: `
            @keyframes dotBounce {
              0%, 80%, 100% {
                transform: scale(0);
                opacity: 0.5;
              }
              40% {
                transform: scale(1.3);
                opacity: 1;
              }
            }

            @keyframes liquidFlow {
              0% {
                transform: translateX(-100%);
              }
              100% {
                transform: translateX(100%);
              }
            }

            @keyframes gradientShift {
              0% {
                background-position: 0% 50%;
              }
              50% {
                background-position: 100% 50%;
              }
              100% {
                background-position: 0% 50%;
              }
            }

            @keyframes textPulse {
              0%, 100% {
                opacity: 1;
                transform: scale(1);
              }
              50% {
                opacity: 0.8;
                transform: scale(1.02);
              }
            }
          `
        }} />
      )}
    </div>
  );
}
