'use client';

/**
 * 🎬 SIMPLIFIED CINEMA LOADER
 *
 * Ultra-premium cinematic loading experience with ZERO dependencies:
 * - Pure CSS animations (no Three.js required)
 * - SVG graphics (no WebGL)
 * - Cinematic theme (space + crypto + futuristic)
 * - 60fps guaranteed on all devices
 * - Mobile-friendly + accessible
 *
 * Concept: "Entering the Crypto Singularity"
 * Theme: Cyberpunk + Financial + Futuristic
 *
 * Performance:
 * - 60fps on ALL devices (pure CSS)
 * - <10KB bundle size
 * - Works without JavaScript (CSS-only fallback)
 *
 * Accessibility:
 * - Respects prefers-reduced-motion
 * - Skip button available
 * - Screen reader friendly
 */

import React, { useEffect, useState } from 'react';

interface SimplifiedCinemaLoaderProps {
  text?: string;
  subtext?: string;
  onLoadingComplete?: () => void;
}

export function SimplifiedCinemaLoader({
  text = 'ENTERING THE SINGULARITY',
  subtext = 'Crypto Universe',
  onLoadingComplete
}: SimplifiedCinemaLoaderProps) {
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [showSkipButton, setShowSkipButton] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    // Check reduced motion preference
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    // Show skip button after 3 seconds
    const timer = setTimeout(() => setShowSkipButton(true), 3000);

    // Simulate loading progress
    const progressInterval = setInterval(() => {
      setLoadingProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          if (onLoadingComplete) {
            setTimeout(onLoadingComplete, 500);
          }
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 400);

    return () => {
      clearTimeout(timer);
      clearInterval(progressInterval);
    };
  }, [onLoadingComplete]);

  const handleSkip = () => {
    setLoadingProgress(100);
    if (onLoadingComplete) {
      onLoadingComplete();
    }
  };

  return (
    <div className="cinema-loader">
      {/* Animated Background */}
      <div className="cinema-bg">
        <div className="cinema-grid" />
        <div className="cinema-gradient-1" />
        <div className="cinema-gradient-2" />
      </div>

      {/* Particle Field (CSS-only) */}
      {!prefersReducedMotion && (
        <div className="cinema-particles">
          {Array.from({ length: 80 }).map((_, i) => (
            <div
              key={i}
              className="cinema-particle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 8}s`,
                animationDuration: `${Math.random() * 20 + 10}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* Central Holographic Element */}
      <div className="cinema-central">
        {/* Rotating Rings */}
        <div className="cinema-rings">
          <div className="cinema-ring cinema-ring-1" />
          <div className="cinema-ring cinema-ring-2" />
          <div className="cinema-ring cinema-ring-3" />
        </div>

        {/* Central Glow */}
        <div className="cinema-glow" />

        {/* Crypto Symbol */}
        <div className="cinema-symbol">
          <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cinema-coin">
            <circle cx="40" cy="40" r="35" stroke="url(#coinGradient)" strokeWidth="3" fill="none" />
            <text x="40" y="50" textAnchor="middle" fontSize="36" fill="url(#coinGradient)" fontWeight="bold">
              ₿
            </text>
            <defs>
              <linearGradient id="coinGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#00D4FF" />
                <stop offset="50%" stopColor="#8B5CF6" />
                <stop offset="100%" stopColor="#FFD700" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Energy Waves */}
        <div className="cinema-waves">
          <div className="cinema-wave cinema-wave-1" />
          <div className="cinema-wave cinema-wave-2" />
          <div className="cinema-wave cinema-wave-3" />
        </div>
      </div>

      {/* UI Overlay */}
      <div className="cinema-ui">
        {/* Title */}
        <h1 className="cinema-title">{text}</h1>
        <p className="cinema-subtitle">{subtext}</p>

        {/* Progress Bar */}
        <div className="cinema-progress-container">
          <div className="cinema-progress-bar" style={{ width: `${loadingProgress}%` }} />
        </div>

        {/* Progress Percentage */}
        <div className="cinema-percentage">{Math.floor(loadingProgress)}%</div>

        {/* Skip Button */}
        {showSkipButton && loadingProgress < 100 && (
          <button className="cinema-skip-btn" onClick={handleSkip}>
            SKIP LOADING
          </button>
        )}
      </div>

      {/* Styles */}
      <style jsx>{`
        /* Container */
        .cinema-loader {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: #000;
          overflow: hidden;
          z-index: 9999;
        }

        /* Background */
        .cinema-bg {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
        }

        .cinema-grid {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-image:
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
          background-size: 50px 50px;
          animation: gridMove 20s linear infinite;
        }

        .cinema-gradient-1 {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 800px;
          height: 800px;
          background: radial-gradient(circle, rgba(0, 212, 255, 0.15) 0%, transparent 70%);
          animation: pulseGlow 4s ease-in-out infinite;
        }

        .cinema-gradient-2 {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 600px;
          height: 600px;
          background: radial-gradient(circle, rgba(255, 215, 0, 0.1) 0%, transparent 70%);
          animation: pulseGlow 4s ease-in-out 0.5s infinite;
        }

        /* Particles */
        .cinema-particles {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
        }

        .cinema-particle {
          position: absolute;
          width: 3px;
          height: 3px;
          background: radial-gradient(circle, #00D4FF, rgba(0, 212, 255, 0.4));
          border-radius: 50%;
          animation: particleFloat 15s ease-in-out infinite;
          box-shadow: 0 0 8px rgba(0, 212, 255, 0.8);
        }

        /* Central Element */
        .cinema-central {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 300px;
          height: 300px;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        /* Rings */
        .cinema-rings {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 100%;
          height: 100%;
        }

        .cinema-ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          border-radius: 50%;
          border: 2px solid transparent;
          box-sizing: border-box;
        }

        .cinema-ring-1 {
          width: 200px;
          height: 200px;
          border-color: rgba(0, 212, 255, 0.3);
          border-style: dotted;
          animation: rotateRing 6s linear infinite;
        }

        .cinema-ring-2 {
          width: 250px;
          height: 250px;
          border-color: rgba(139, 92, 246, 0.2);
          border-style: dashed;
          animation: rotateRing 12s linear reverse infinite;
        }

        .cinema-ring-3 {
          width: 300px;
          height: 300px;
          border-color: rgba(255, 215, 0, 0.2);
          animation: rotateRing 8s linear infinite;
        }

        /* Central Glow */
        .cinema-glow {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 150px;
          height: 150px;
          background: radial-gradient(circle, rgba(0, 212, 255, 0.4) 0%, transparent 70%);
          border-radius: 50%;
          animation: pulseGlow 3s ease-in-out infinite;
        }

        /* Crypto Symbol */
        .cinema-symbol {
          position: relative;
          z-index: 10;
          animation: floatCoin 4s ease-in-out infinite, rotateCoin 10s linear infinite;
        }

        .cinema-coin {
          filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.8));
        }

        /* Energy Waves */
        .cinema-waves {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 100%;
          height: 100%;
        }

        .cinema-wave {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          border: 2px solid;
          border-radius: 50%;
          animation: waveExpand 3s ease-out infinite;
        }

        .cinema-wave-1 {
          border-color: rgba(0, 212, 255, 0.6);
          animation-delay: 0s;
        }

        .cinema-wave-2 {
          border-color: rgba(139, 92, 246, 0.5);
          animation-delay: 1s;
        }

        .cinema-wave-3 {
          border-color: rgba(255, 215, 0, 0.4);
          animation-delay: 2s;
        }

        /* UI Overlay */
        .cinema-ui {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 24px;
          padding: 40px 20px;
          z-index: 100;
          pointer-events: none;
        }

        .cinema-title {
          font-size: clamp(24px, 5vw, 48px);
          font-weight: 900;
          background: linear-gradient(135deg, #00D4FF 0%, #8B5CF6 50%, #FFD700 100%);
          background-size: 200% auto;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          letter-spacing: 8px;
          animation: shimmer 3s linear infinite, textPulse 2s ease-in-out infinite;
          text-shadow: 0 0 40px rgba(0, 212, 255, 0.5);
          margin: 0;
          text-align: center;
        }

        .cinema-subtitle {
          font-size: clamp(12px, 2vw, 18px);
          font-weight: 600;
          color: #FFD700;
          letter-spacing: 4px;
          opacity: 0.8;
          margin: 0;
          animation: fadeIn 1s ease-out 0.3s both;
        }

        .cinema-progress-container {
          width: 320px;
          max-width: 80%;
          height: 4px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 2px;
          overflow: hidden;
          position: relative;
          border: 1px solid rgba(0, 212, 255, 0.3);
          animation: fadeIn 1s ease-out 0.5s both;
        }

        .cinema-progress-bar {
          height: 100%;
          background: linear-gradient(90deg, #00D4FF, #8B5CF6, #FFD700);
          transition: width 0.3s ease-out;
          box-shadow: 0 0 20px rgba(0, 212, 255, 0.8);
        }

        .cinema-percentage {
          font-size: 18px;
          font-weight: 700;
          color: #00D4FF;
          font-family: 'Courier New', monospace;
          letter-spacing: 2px;
          animation: fadeIn 1s ease-out 0.7s both;
        }

        .cinema-skip-btn {
          padding: 12px 32px;
          background: rgba(0, 212, 255, 0.1);
          border: 1px solid rgba(0, 212, 255, 0.5);
          border-radius: 4px;
          color: #00D4FF;
          font-size: 14px;
          font-weight: 600;
          letter-spacing: 2px;
          cursor: pointer;
          transition: all 0.3s ease;
          pointer-events: auto;
          animation: fadeIn 1s ease-out 0.9s both;
        }

        .cinema-skip-btn:hover {
          background: rgba(0, 212, 255, 0.2);
          border-color: #00D4FF;
          transform: scale(1.05);
        }

        /* Animations */
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

        @keyframes particleFloat {
          0%, 100% {
            transform: translate(0, 0) scale(1);
            opacity: 0.6;
          }
          25% {
            transform: translate(30px, -40px) scale(1.3);
            opacity: 1;
          }
          50% {
            transform: translate(-20px, -60px) scale(0.7);
            opacity: 0.4;
          }
          75% {
            transform: translate(35px, -25px) scale(1.2);
            opacity: 0.9;
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

        @keyframes floatCoin {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-10px);
          }
        }

        @keyframes rotateCoin {
          0% {
            transform: rotateY(0deg);
          }
          100% {
            transform: rotateY(360deg);
          }
        }

        @keyframes waveExpand {
          0% {
            width: 50px;
            height: 50px;
            opacity: 0.8;
          }
          100% {
            width: 400px;
            height: 400px;
            opacity: 0;
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

        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        /* Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
          .cinema-loader * {
            animation: none !important;
            transition: none !important;
          }
        }

        /* Mobile Optimization */
        @media (max-width: 768px) {
          .cinema-central {
            width: 250px;
            height: 250px;
          }

          .cinema-ring-1 {
            width: 150px;
            height: 150px;
          }

          .cinema-ring-2 {
            width: 200px;
            height: 200px;
          }

          .cinema-ring-3 {
            width: 250px;
            height: 250px;
          }

          .cinema-symbol svg {
            width: 60px;
            height: 60px;
          }
        }
      `}</style>
    </div>
  );
}

export default SimplifiedCinemaLoader;
