'use client';

/**
 * 🔐 LOGIN MODAL COMPONENT
 * Güvenli popup login ekranı
 *
 * Özellikler:
 * - Modern popup design
 * - Session-based authentication
 * - Güvenli credential check
 * - Error handling
 * - Loading states
 */

import { useState } from 'react';
import { COLORS } from '@/lib/colors';

interface LoginModalProps {
  isOpen: boolean;
  onSuccess: () => void;
}

export function LoginModal({ isOpen, onSuccess }: LoginModalProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  if (!isOpen) return null;

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // Güvenli credential check
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (data.success) {
        // Session cookie otomatik set edilir
        onSuccess();
      } else {
        setError('Kullanıcı adı veya şifre hatalı!');
      }
    } catch (err) {
      setError('Giriş yapılırken bir hata oluştu.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'rgba(0, 0, 0, 0.85)',
        backdropFilter: 'blur(10px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        animation: 'fadeIn 0.3s ease',
      }}
    >
      <div
        style={{
          background: `linear-gradient(135deg, ${COLORS.bg.primary}, ${COLORS.bg.secondary})`,
          border: `2px solid ${COLORS.premium}`,
          borderRadius: '16px',
          padding: '40px',
          maxWidth: '450px',
          width: '90%',
          boxShadow: `0 20px 60px rgba(255, 0, 255, 0.3), 0 0 0 1px ${COLORS.premium}40`,
          animation: 'slideUp 0.4s ease',
        }}
      >
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            fontSize: '48px',
            marginBottom: '16px',
            animation: 'pulse 2s infinite',
          }}>
            🔐
          </div>
          <h2 style={{
            fontSize: '28px',
            fontWeight: 'bold',
            color: COLORS.text.primary,
            marginBottom: '8px',
            background: `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.cyan})`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Ailydian LYDIAN
          </h2>
          <p style={{ fontSize: '14px', color: COLORS.text.secondary }}>
            Sisteme giriş yapmak için lütfen kimlik bilgilerinizi girin
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleLogin}>
          {/* Username */}
          <div style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block',
              fontSize: '12px',
              color: COLORS.text.secondary,
              marginBottom: '8px',
              fontWeight: '500',
            }}>
              👤 Kullanıcı Adı
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Kullanıcı adınızı girin"
              required
              autoFocus
              style={{
                width: '100%',
                padding: '14px 16px',
                background: COLORS.bg.secondary,
                border: `2px solid ${COLORS.border.default}`,
                borderRadius: '8px',
                color: COLORS.text.primary,
                fontSize: '15px',
                outline: 'none',
                transition: 'all 0.2s ease',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = COLORS.premium;
                e.target.style.boxShadow = `0 0 0 3px ${COLORS.premium}20`;
              }}
              onBlur={(e) => {
                e.target.style.borderColor = COLORS.border.default;
                e.target.style.boxShadow = 'none';
              }}
            />
          </div>

          {/* Password */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{
              display: 'block',
              fontSize: '12px',
              color: COLORS.text.secondary,
              marginBottom: '8px',
              fontWeight: '500',
            }}>
              🔑 Şifre
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Şifrenizi girin"
                required
                style={{
                  width: '100%',
                  padding: '14px 48px 14px 16px',
                  background: COLORS.bg.secondary,
                  border: `2px solid ${COLORS.border.default}`,
                  borderRadius: '8px',
                  color: COLORS.text.primary,
                  fontSize: '15px',
                  outline: 'none',
                  transition: 'all 0.2s ease',
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = COLORS.premium;
                  e.target.style.boxShadow = `0 0 0 3px ${COLORS.premium}20`;
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = COLORS.border.default;
                  e.target.style.boxShadow = 'none';
                }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  position: 'absolute',
                  right: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'transparent',
                  border: 'none',
                  color: COLORS.text.secondary,
                  cursor: 'pointer',
                  fontSize: '20px',
                  padding: '4px',
                }}
              >
                {showPassword ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div style={{
              padding: '12px 16px',
              background: `${COLORS.danger}15`,
              border: `1px solid ${COLORS.danger}40`,
              borderRadius: '8px',
              color: COLORS.danger,
              fontSize: '13px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}>
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading || !username || !password}
            style={{
              width: '100%',
              padding: '16px',
              background: loading ? COLORS.bg.secondary : `linear-gradient(135deg, ${COLORS.premium}, ${COLORS.cyan})`,
              border: 'none',
              borderRadius: '8px',
              color: COLORS.text.primary,
              fontSize: '16px',
              fontWeight: 'bold',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'all 0.2s ease',
              opacity: loading || !username || !password ? 0.5 : 1,
            }}
            onMouseEnter={(e) => {
              if (!loading && username && password) {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = `0 8px 20px ${COLORS.premium}40`;
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {loading ? '🔄 Giriş yapılıyor...' : '🚀 Giriş Yap'}
          </button>
        </form>

        {/* Footer */}
        <div style={{
          marginTop: '24px',
          padding: '16px',
          background: `${COLORS.bg.secondary}80`,
          borderRadius: '8px',
          fontSize: '11px',
          color: COLORS.text.secondary,
          textAlign: 'center',
        }}>
          🔒 Güvenli bağlantı | 256-bit şifreleme
        </div>
      </div>

      {/* Animations */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        @keyframes pulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
}
