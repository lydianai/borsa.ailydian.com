'use client';

/**
 * 🔐 AUTH PROVIDER
 * Authentication context ve state management
 *
 * Özellikler:
 * - Session check
 * - Login modal management
 * - Protected routes
 * - Auto-refresh check
 */

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { LoginModal } from './LoginModal';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showLoginModal, setShowLoginModal] = useState(false);

  // Check auth status
  const checkAuth = async () => {
    try {
      const response = await fetch('/api/auth/check');
      const data = await response.json();
      setIsAuthenticated(data.authenticated);

      if (!data.authenticated) {
        setShowLoginModal(true);
      }
    } catch (error) {
      console.error('[Auth] Check failed:', error);
      setIsAuthenticated(false);
      setShowLoginModal(true);
    } finally {
      setIsLoading(false);
    }
  };

  // Initial auth check
  useEffect(() => {
    checkAuth();
  }, []);

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
    setShowLoginModal(false);
  };

  const login = () => {
    setShowLoginModal(true);
  };

  const logout = async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
      setIsAuthenticated(false);
      setShowLoginModal(true);
    } catch (error) {
      console.error('[Auth] Logout failed:', error);
    }
  };

  // Show loading screen while checking auth
  if (isLoading) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: '#0a0a0a',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: '20px',
      }}>
        <div style={{
          fontSize: '48px',
          animation: 'spin 2s linear infinite',
        }}>
          🔄
        </div>
        <div style={{ color: '#888', fontSize: '14px' }}>
          Sistem hazırlanıyor...
        </div>
        <style jsx>{`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, isLoading, login, logout }}>
      {showLoginModal && !isAuthenticated && (
        <LoginModal isOpen={true} onSuccess={handleLoginSuccess} />
      )}
      {isAuthenticated ? children : null}
    </AuthContext.Provider>
  );
}
