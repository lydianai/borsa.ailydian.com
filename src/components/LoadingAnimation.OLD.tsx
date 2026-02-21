'use client';

import React from 'react';

export function LoadingAnimation() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '20px',
        padding: '40px',
      }}
    >
      {/* Animated Monkey Character SVG */}
      <svg
        width="140"
        height="140"
        viewBox="0 0 140 140"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))',
        }}
      >
        {/* Outer Pulsing Ring - Natural brown */}
        <circle
          cx="70"
          cy="70"
          r="60"
          stroke="#8B5A3C"
          strokeWidth="3"
          fill="none"
          opacity="0.3"
        >
          <animate
            attributeName="r"
            values="55;65;55"
            dur="2s"
            repeatCount="indefinite"
          />
          <animate
            attributeName="opacity"
            values="0.3;0.1;0.3"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>

        {/* Monkey Character - Realistic Colors */}
        <g transform="translate(70, 70)">
          {/* Tail (Animated) - Brown */}
          <path
            d="M 20,10 Q 35,5 40,-5 Q 42,-15 38,-20"
            stroke="#8B5A3C"
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              type="rotate"
              values="0 20 10; 15 20 10; 0 20 10; -15 20 10; 0 20 10"
              dur="2s"
              repeatCount="indefinite"
            />
          </path>

          {/* Body (Bouncing) - Brown */}
          <ellipse
            cx="0"
            cy="5"
            rx="18"
            ry="22"
            fill="#8B5A3C"
            opacity="0.9"
          >
            <animate
              attributeName="ry"
              values="22;20;22;24;22"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </ellipse>

          {/* Head (Bouncing with body) - Brown */}
          <circle
            cx="0"
            cy="-18"
            r="16"
            fill="#8B5A3C"
          >
            <animate
              attributeName="cy"
              values="-18;-20;-18;-16;-18"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Left Ear - Darker brown */}
          <circle
            cx="-12"
            cy="-25"
            r="6"
            fill="#6B4423"
            opacity="0.9"
          >
            <animate
              attributeName="cy"
              values="-25;-27;-25;-23;-25"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Right Ear - Darker brown */}
          <circle
            cx="12"
            cy="-25"
            r="6"
            fill="#6B4423"
            opacity="0.9"
          >
            <animate
              attributeName="cy"
              values="-25;-27;-25;-23;-25"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Face Details - Beige/Tan */}
          <ellipse
            cx="0"
            cy="-15"
            rx="10"
            ry="8"
            fill="#D2B48C"
            opacity="0.9"
          >
            <animate
              attributeName="cy"
              values="-15;-17;-15;-13;-15"
              dur="1.5s"
              repeatCount="indefinite"
            />
          </ellipse>

          {/* Left Eye (Blinking) */}
          <circle
            cx="-5"
            cy="-20"
            r="2"
            fill="#000"
          >
            <animate
              attributeName="cy"
              values="-20;-22;-20;-18;-20"
              dur="1.5s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="r"
              values="2;2;0.5;2;2;2;2;2"
              dur="3s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Right Eye (Blinking) */}
          <circle
            cx="5"
            cy="-20"
            r="2"
            fill="#000"
          >
            <animate
              attributeName="cy"
              values="-20;-22;-20;-18;-20"
              dur="1.5s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="r"
              values="2;2;0.5;2;2;2;2;2"
              dur="3s"
              repeatCount="indefinite"
            />
          </circle>

          {/* Mouth (Smiling) */}
          <path
            d="M -4,-13 Q 0,-11 4,-13"
            stroke="#000"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
          >
            <animate
              attributeName="d"
              values="M -4,-13 Q 0,-11 4,-13; M -4,-13 Q 0,-10 4,-13; M -4,-13 Q 0,-11 4,-13"
              dur="2s"
              repeatCount="indefinite"
            />
          </path>

          {/* Left Arm (Waving) - Brown */}
          <ellipse
            cx="-16"
            cy="5"
            rx="4"
            ry="12"
            fill="#8B5A3C"
            opacity="0.9"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              type="rotate"
              values="-20 -16 5; -40 -16 5; -20 -16 5; 0 -16 5; -20 -16 5"
              dur="2s"
              repeatCount="indefinite"
            />
          </ellipse>

          {/* Right Arm (Waving opposite) - Brown */}
          <ellipse
            cx="16"
            cy="5"
            rx="4"
            ry="12"
            fill="#8B5A3C"
            opacity="0.9"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              type="rotate"
              values="20 16 5; 0 16 5; 20 16 5; 40 16 5; 20 16 5"
              dur="2s"
              repeatCount="indefinite"
            />
          </ellipse>

          {/* Left Leg - Brown */}
          <ellipse
            cx="-8"
            cy="22"
            rx="4"
            ry="10"
            fill="#8B5A3C"
            opacity="0.9"
          />

          {/* Right Leg - Brown */}
          <ellipse
            cx="8"
            cy="22"
            rx="4"
            ry="10"
            fill="#8B5A3C"
            opacity="0.9"
          />
        </g>

        {/* Orbiting Bananas (playful touch) */}
        <g>
          <path
            d="M 0,-3 Q 2,-5 3,-2 Q 2,1 0,0 Z"
            fill="#ffff00"
            opacity="0.8"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              type="rotate"
              from="0 70 70"
              to="360 70 70"
              dur="4s"
              repeatCount="indefinite"
            />
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              type="translate"
              values="70 20; 70 18; 70 20"
              dur="1s"
              repeatCount="indefinite"
              additive="sum"
            />
          </path>
        </g>

        {/* Sparkle Effects - Natural colors */}
        <g opacity="0.5">
          <circle cx="30" cy="30" r="2" fill="#8B5A3C">
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="1.5s"
              repeatCount="indefinite"
              begin="0s"
            />
          </circle>
          <circle cx="110" cy="30" r="2" fill="#8B5A3C">
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="1.5s"
              repeatCount="indefinite"
              begin="0.5s"
            />
          </circle>
          <circle cx="30" cy="110" r="2" fill="#8B5A3C">
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="1.5s"
              repeatCount="indefinite"
              begin="1s"
            />
          </circle>
          <circle cx="110" cy="110" r="2" fill="#8B5A3C">
            <animate
              attributeName="opacity"
              values="0;1;0"
              dur="1.5s"
              repeatCount="indefinite"
              begin="0.3s"
            />
          </circle>
        </g>
      </svg>

      {/* Loading Dots - Brown colors */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'center',
        }}
      >
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#8B5A3C',
            animation: 'bounce 1.4s infinite ease-in-out both',
            animationDelay: '-0.32s',
          }}
        />
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#8B5A3C',
            animation: 'bounce 1.4s infinite ease-in-out both',
            animationDelay: '-0.16s',
          }}
        />
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#8B5A3C',
            animation: 'bounce 1.4s infinite ease-in-out both',
          }}
        />
      </div>

      {/* CSS Animations */}
      <style jsx>{`
        @keyframes pulse-text {
          0%, 100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.7;
            transform: scale(1.05);
          }
        }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
          }
          40% {
            transform: scale(1);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}
