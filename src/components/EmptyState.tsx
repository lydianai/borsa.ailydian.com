'use client';

import React from 'react';
import { Icons } from './Icons';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: 'search' | 'chart' | 'signal' | 'alert';
  actionText?: string;
  onAction?: () => void;
}

export function EmptyState({
  title = 'Henüz Veri Yok',
  description = 'Şu anda gösterilebilecek veri bulunmuyor.',
  icon = 'chart',
  actionText,
  onAction,
}: EmptyStateProps) {
  const IconComponent = {
    search: Icons.Search,
    chart: Icons.BarChart,
    signal: Icons.TrendingUp,
    alert: Icons.Bell,
  }[icon];

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '80px 20px',
        textAlign: 'center',
        minHeight: '400px',
      }}
    >
      <div
        style={{
          marginBottom: '24px',
          opacity: 0.3,
        }}
      >
        <IconComponent style={{ width: '64px', height: '64px', color: '#666' }} />
      </div>
      <h3
        style={{
          color: '#ffffff',
          fontSize: '20px',
          fontWeight: '600',
          marginBottom: '12px',
        }}
      >
        {title}
      </h3>
      <p
        style={{
          color: '#666',
          fontSize: '14px',
          maxWidth: '400px',
          lineHeight: '1.6',
          marginBottom: actionText ? '24px' : '0',
        }}
      >
        {description}
      </p>
      {actionText && onAction && (
        <button
          onClick={onAction}
          style={{
            background: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid #ffffff',
            color: '#ffffff',
            padding: '10px 24px',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          {actionText}
        </button>
      )}
    </div>
  );
}
