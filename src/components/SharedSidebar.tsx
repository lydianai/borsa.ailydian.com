'use client';

/**
 * 🎯 ULTRA PREMIUM UNIFIED HEADER - v4.0 FINAL
 *
 * ✅ NEW FIXES:
 * 1. Compact search bar added to header center (desktop)
 * 2. Optimized menu icon spacing and layout
 * 3. Completely redesigned filter bar (modern dropdowns)
 * 4. Mobile: Floating compact filter controls
 *
 * Responsive Breakpoints:
 * - 320px-1024px: Mobile/Tablet (hamburger + drawer)
 * - 1025px+: Desktop (horizontal menu icons + search)
 */

import { Icons } from '@/components/Icons';
import { useState, useEffect } from 'react';
import { NotificationPanel } from '@/components/NotificationPanel';
import { ActiveUsersIndicator } from '@/components/ActiveUsersIndicator';
import Link from 'next/link';
import { useGlobalFilters, type Timeframe, type SortBy } from '@/hooks/useGlobalFilters';
import { useNotifications } from '@/hooks/useNotifications';

interface SharedSidebarProps {
  currentPage: string;
  onAiAssistantOpen?: () => void;
  notificationCounts?: {
    market?: number;
    trading?: number;
    ai?: number;
    quantum?: number;
    conservative?: number;
    omnipotent?: number;
    correlation?: number;
    btceth?: number;
    traditional?: number;
    quantumLadder?: number;
  };
  coinCount?: number;
  countdown?: number;
  searchTerm?: string;
  onSearchChange?: (value: string) => void;
}

interface MenuCategory {
  id: string;
  title: string;
  icon: any;
  gradient: string;
  items: MenuItem[];
}

interface MenuItem {
  href: string;
  icon: any;
  label: string;
  key: string;
  color: string;
  notification?: number;
  requiresPayment?: boolean;
}

export function SharedSidebar({
  currentPage,
  onAiAssistantOpen,
  notificationCounts = {},
  coinCount = 0,
  countdown = 10,
  searchTerm = '',
  onSearchChange
}: SharedSidebarProps) {
  const {
    market = 0,
    trading = 0,
    ai = 0,
    quantum = 0,
    conservative = 0,
    omnipotent = 0,
    correlation = 0,
    btceth = 0,
    traditional = 0,
    quantumLadder = 0
  } = notificationCounts;

  const [mounted, setMounted] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);
  const [showNotifications, setShowNotifications] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [drawerSearchTerm, setDrawerSearchTerm] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<string[]>(['featured', 'analytics']);
  const [showTimeframeDropdown, setShowTimeframeDropdown] = useState(false);
  const [showSortByDropdown, setShowSortByDropdown] = useState(false);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);

  const { timeframe, sortBy, setTimeframe, setSortBy } = useGlobalFilters();
  const { unreadCount: globalUnreadCount } = useNotifications(true, 10000);

  const isLocalhost = typeof window !== 'undefined' && window.location.hostname === 'localhost';
  const isProduction = typeof window !== 'undefined' && window.location.hostname === 'borsa.ailydian.com';
  const isAdmin = typeof window !== 'undefined' && localStorage.getItem('user-role') === 'admin';
  const hasActivePayment = isLocalhost || isProduction || isAdmin;

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (mobileMenuOpen) setMobileMenuOpen(false);
        if (showSearchDropdown) setShowSearchDropdown(false);
        if (showNotifications) setShowNotifications(false);
        if (showTimeframeDropdown) setShowTimeframeDropdown(false);
        if (showSortByDropdown) setShowSortByDropdown(false);
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [mobileMenuOpen, showSearchDropdown, showNotifications, showTimeframeDropdown, showSortByDropdown]);

  useEffect(() => {
    const handleClickOutside = () => {
      setShowTimeframeDropdown(false);
      setShowSortByDropdown(false);
      setShowSearchDropdown(false);
    };
    if (showTimeframeDropdown || showSortByDropdown || showSearchDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showTimeframeDropdown, showSortByDropdown, showSearchDropdown]);

  if (!mounted) {
    return (
      <header suppressHydrationWarning style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '60px',
        width: '100%',
        background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.6)' }}>Yükleniyor...</div>
      </header>
    );
  }

  const TIMEFRAME_OPTIONS: { value: Timeframe; label: string }[] = [
    { value: '1H', label: '1 Saat' },
    { value: '4H', label: '4 Saat' },
    { value: '1D', label: '1 Gün' },
    { value: '1W', label: '1 Hafta' },
  ];

  const SORTBY_OPTIONS: { value: SortBy; label: string }[] = [
    { value: 'volume', label: 'Hacim' },
    { value: 'change', label: 'Değişim' },
    { value: 'price', label: 'Fiyat' },
    { value: 'name', label: 'İsim' },
  ];

  // All menu items for desktop horizontal display - COMPLETE LIST (23+ items)
  const allMenuItems: MenuItem[] = [
    { href: '/', icon: Icons.Dashboard, label: 'Kontrol Paneli', color: '#FFFFFF', key: 'home', requiresPayment: false },
    { href: '/nirvana', icon: Icons.Zap, label: 'Nirvana', color: '#00D4FF', key: 'nirvana', requiresPayment: false },
    { href: '/market-scanner', icon: Icons.Search, label: 'Piyasa Tarayıcı', color: '#14B8A6', key: 'market-scanner', notification: market, requiresPayment: false },
    { href: '/trading-signals', icon: Icons.TrendingUp, label: 'Trading', color: '#10B981', key: 'trading-signals', notification: trading, requiresPayment: false },
    { href: '/ai-signals', icon: Icons.Brain, label: 'AI', color: '#FFFFFF', key: 'ai-signals', notification: ai, requiresPayment: true },
    { href: '/quantum-signals', icon: Icons.Atom, label: 'Quantum', color: '#6366F1', key: 'quantum-signals', notification: quantum, requiresPayment: true },
    { href: '/quantum-pro', icon: Icons.Atom, label: 'Quantum Pro', color: '#7C3AED', key: 'quantum-pro', requiresPayment: true },
    { href: '/quantum-ladder', icon: Icons.Layers, label: 'Quantum Ladder', color: '#FF00E5', key: 'quantum-ladder', notification: quantumLadder, requiresPayment: true },
    { href: '/conservative-signals', icon: Icons.Shield, label: 'Conservative', color: '#F59E0B', key: 'conservative-signals', notification: conservative, requiresPayment: false },
    { href: '/omnipotent-futures', icon: Icons.Eye, label: 'Omnipotent', color: '#A855F7', key: 'omnipotent-futures', notification: omnipotent, requiresPayment: true },
    { href: '/market-correlation', icon: Icons.GitBranch, label: 'Korelasyon', color: '#3B82F6', key: 'market-correlation', notification: correlation, requiresPayment: true },
    { href: '/btc-eth-analysis', icon: Icons.BarChart, label: 'BTC/ETH', color: '#FFFFFF', key: 'btc-eth-analysis', notification: btceth, requiresPayment: false },
    { href: '/traditional-markets', icon: Icons.Globe, label: 'Traditional', color: '#0EA5E9', key: 'traditional-markets', notification: traditional, requiresPayment: false },
    { href: '/breakout-retest', icon: Icons.Target, label: 'Breakout', color: '#F97316', key: 'breakout-retest', requiresPayment: false },
    { href: '/alfabetik-pattern', icon: Icons.Layers, label: 'Alfabetik', color: '#8B5A3C', key: 'alfabetik-pattern', requiresPayment: false },
    { href: '/market-commentary', icon: Icons.FileText, label: 'Yorum', color: '#10B981', key: 'market-commentary', requiresPayment: false },
    { href: '/market-insights', icon: Icons.TrendingUp, label: 'İçgörü', color: '#FF00FF', key: 'market-insights', requiresPayment: true },
    { href: '/perpetual-hub', icon: Icons.Layers, label: 'Perpetual', color: '#FFD700', key: 'perpetual-hub', requiresPayment: true },
    { href: '/ai-learning-hub', icon: Icons.Sparkles, label: 'AI Learning', color: '#8B5CF6', key: 'ai-learning-hub', requiresPayment: true },
    { href: '/bot-analysis', icon: Icons.Bot, label: 'Bot', color: '#00D4FF', key: 'bot-analysis', requiresPayment: true },
    { href: '/azure-ai', icon: Icons.Cloud, label: 'Azure', color: '#0078D4', key: 'azure-ai', requiresPayment: true },
    { href: '/charts', icon: Icons.BarChart, label: 'Charts', color: '#00D4FF', key: 'charts', requiresPayment: true },
    { href: '/talib', icon: Icons.Activity, label: 'TA-Lib', color: '#FF6B6B', key: 'talib', requiresPayment: false },
    { href: '/auto-trading', icon: Icons.Bot, label: 'Auto Trade', color: '#10B981', key: 'auto-trading', requiresPayment: true },
    { href: '/haberler', icon: Icons.Newspaper, label: 'Haberler', color: '#fbbf24', key: 'haberler', requiresPayment: false },
  ];

  // Premium categorized menu for drawer
  const menuCategories: MenuCategory[] = [
    {
      id: 'featured',
      title: '⭐ ÖNE ÇIKANLAR',
      icon: Icons.Star,
      gradient: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
      items: [
        { href: '/', icon: Icons.Dashboard, label: 'Kontrol Paneli', color: '#FFFFFF', key: 'home', requiresPayment: false },
        { href: '/nirvana', icon: Icons.Zap, label: 'Nirvana Kontrol', color: '#00D4FF', key: 'nirvana', requiresPayment: false },
        { href: '/market-scanner', icon: Icons.Search, label: 'Piyasa Tarayıcı', color: '#14B8A6', key: 'market-scanner', notification: market, requiresPayment: false },
        { href: '/alfabetik-pattern', icon: Icons.Layers, label: 'Alfabetik Desen', color: '#8B5A3C', key: 'alfabetik-pattern', requiresPayment: false },
      ]
    },
    {
      id: 'analytics',
      title: '📈 ANALİZ & SİNYALLER',
      icon: Icons.TrendingUp,
      gradient: 'linear-gradient(135deg, #A855F7 0%, #7C3AED 100%)',
      items: [
        { href: '/trading-signals', icon: Icons.TrendingUp, label: 'Ticaret Sinyalleri', color: '#10B981', key: 'trading-signals', notification: trading, requiresPayment: false },
        { href: '/ai-signals', icon: Icons.Brain, label: 'YZ Sinyalleri', color: '#FFFFFF', key: 'ai-signals', notification: ai, requiresPayment: true },
        { href: '/quantum-signals', icon: Icons.Atom, label: 'Kuantum Sinyaller', color: '#6366F1', key: 'quantum-signals', notification: quantum, requiresPayment: true },
        { href: '/quantum-pro', icon: Icons.Atom, label: 'Kuantum Pro', color: '#6366F1', key: 'quantum-pro', requiresPayment: true },
        { href: '/quantum-ladder', icon: Icons.Layers, label: 'Kuantum Merdiven', color: '#FF00E5', key: 'quantum-ladder', notification: quantumLadder, requiresPayment: true },
        { href: '/conservative-signals', icon: Icons.Shield, label: 'Muhafazakar Sinyaller', color: '#F59E0B', key: 'conservative-signals', notification: conservative, requiresPayment: false },
        { href: '/breakout-retest', icon: Icons.Target, label: 'Kırılım Yeniden Test', color: '#F97316', key: 'breakout-retest', requiresPayment: false },
        { href: '/btc-eth-analysis', icon: Icons.BarChart, label: 'BTC/ETH Analiz', color: '#FFFFFF', key: 'btc-eth-analysis', notification: btceth, requiresPayment: false },
      ]
    },
    {
      id: 'markets',
      title: '🌍 PİYASALAR',
      icon: Icons.Globe,
      gradient: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)',
      items: [
        { href: '/market-correlation', icon: Icons.GitBranch, label: 'Piyasa Korelasyon', color: '#3B82F6', key: 'market-correlation', notification: correlation, requiresPayment: true },
        { href: '/traditional-markets', icon: Icons.Globe, label: 'Geleneksel Piyasalar', color: '#0EA5E9', key: 'traditional-markets', notification: traditional, requiresPayment: false },
        { href: '/market-commentary', icon: Icons.FileText, label: 'Piyasa Yorumu', color: '#10B981', key: 'market-commentary', requiresPayment: false },
        { href: '/market-insights', icon: Icons.TrendingUp, label: 'Piyasa İçgörüleri', color: '#FF00FF', key: 'market-insights', requiresPayment: true },
      ]
    },
    {
      id: 'tools',
      title: '🛠️ ARAÇLAR & ANALİZ',
      icon: Icons.Wrench,
      gradient: 'linear-gradient(135deg, #06B6D4 0%, #0891B2 100%)',
      items: [
        { href: '/omnipotent-futures', icon: Icons.Eye, label: 'Gelecek Matrisi', color: '#A855F7', key: 'omnipotent-futures', notification: omnipotent, requiresPayment: true },
        { href: '/bot-analysis', icon: Icons.Bot, label: 'Bot Analizi', color: '#00D4FF', key: 'bot-analysis', requiresPayment: true },
        { href: '/azure-ai', icon: Icons.Cloud, label: 'Azure YZ', color: '#0078D4', key: 'azure-ai', requiresPayment: true },
        { href: '/charts', icon: Icons.BarChart, label: 'Grafikler', color: '#00D4FF', key: 'charts', requiresPayment: true },
        { href: '/talib', icon: Icons.Activity, label: 'TA-Lib Analiz', color: '#FF6B6B', key: 'talib', requiresPayment: false },
        { href: '/auto-trading', icon: Icons.Bot, label: 'Otomatik Ticaret', color: '#10B981', key: 'auto-trading', requiresPayment: true },
      ]
    },
    {
      id: 'perpetual',
      title: '⚡ PERPETUAL HUB',
      icon: Icons.Layers,
      gradient: 'linear-gradient(135deg, #FFD700 0%, #F59E0B 100%)',
      items: [
        { href: '/perpetual-hub', icon: Icons.Layers, label: 'Ana Hub', color: '#FFD700', key: 'perpetual-hub', requiresPayment: true },
        { href: '/perpetual-hub/position-risk', icon: Icons.Shield, label: 'Pozisyon Risk', color: '#EF4444', key: 'position-risk', requiresPayment: true },
        { href: '/perpetual-hub/sentiment-hedge', icon: Icons.TrendingUp, label: 'Duygu Hedge', color: '#8B5CF6', key: 'sentiment-hedge', requiresPayment: true },
        { href: '/perpetual-hub/cross-chain', icon: Icons.GitBranch, label: 'Cross-Chain', color: '#06B6D4', key: 'cross-chain', requiresPayment: true },
        { href: '/perpetual-hub/orderbook-depth', icon: Icons.BarChart, label: 'Emir Defteri Derinlik', color: '#10B981', key: 'orderbook-depth', requiresPayment: true },
        { href: '/perpetual-hub/contract-scanner', icon: Icons.Search, label: 'Kontrat Tarayıcı', color: '#14B8A6', key: 'contract-scanner', requiresPayment: true },
        { href: '/perpetual-hub/whale-tracker', icon: Icons.Eye, label: 'Balina Takip', color: '#3B82F6', key: 'whale-tracker', requiresPayment: true },
        { href: '/perpetual-hub/market-microstructure', icon: Icons.Activity, label: 'Piyasa Mikroyapı', color: '#A855F7', key: 'market-microstructure', requiresPayment: true },
        { href: '/perpetual-hub/correlation-matrix', icon: Icons.GitBranch, label: 'Korelasyon Matrisi', color: '#EC4899', key: 'correlation-matrix', requiresPayment: true },
        { href: '/perpetual-hub/leverage-optimizer', icon: Icons.Target, label: 'Kaldıraç Optimize', color: '#F59E0B', key: 'leverage-optimizer', requiresPayment: true },
        { href: '/perpetual-hub/portfolio-rebalancer', icon: Icons.Layers, label: 'Portföy Dengeleme', color: '#10B981', key: 'portfolio-rebalancer', requiresPayment: true },
        { href: '/perpetual-hub/liquidity-flow', icon: Icons.TrendingUp, label: 'Likidite Akış', color: '#6366F1', key: 'liquidity-flow', requiresPayment: true },
      ]
    },
    {
      id: 'ai-learning',
      title: '🤖 YZ ÖĞRENME MERKEZİ',
      icon: Icons.Sparkles,
      gradient: 'linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%)',
      items: [
        { href: '/ai-learning-hub', icon: Icons.Sparkles, label: 'Ana Merkez', color: '#8B5CF6', key: 'ai-learning-hub', requiresPayment: true },
        { href: '/ai-learning-hub/automl', icon: Icons.Brain, label: 'AutoML', color: '#A855F7', key: 'automl', requiresPayment: true },
        { href: '/ai-learning-hub/nas', icon: Icons.Activity, label: 'Neural Architecture', color: '#7C3AED', key: 'nas', requiresPayment: true },
        { href: '/ai-learning-hub/meta-learning', icon: Icons.Sparkles, label: 'Meta Öğrenme', color: '#6366F1', key: 'meta-learning', requiresPayment: true },
        { href: '/ai-learning-hub/multi-agent', icon: Icons.Bot, label: 'Çoklu Ajan', color: '#8B5CF6', key: 'multi-agent', requiresPayment: true },
        { href: '/ai-learning-hub/rl-agent', icon: Icons.Target, label: 'RL Ajan', color: '#A855F7', key: 'rl-agent', requiresPayment: true },
        { href: '/ai-learning-hub/causal-ai', icon: Icons.GitBranch, label: 'Nedensel YZ', color: '#7C3AED', key: 'causal-ai', requiresPayment: true },
        { href: '/ai-learning-hub/regime-detection', icon: Icons.Eye, label: 'Rejim Tespit', color: '#6366F1', key: 'regime-detection', requiresPayment: true },
        { href: '/ai-learning-hub/online-learning', icon: Icons.TrendingUp, label: 'Online Öğrenme', color: '#8B5CF6', key: 'online-learning', requiresPayment: true },
        { href: '/ai-learning-hub/explainable-ai', icon: Icons.FileText, label: 'Açıklanabilir YZ', color: '#A855F7', key: 'explainable-ai', requiresPayment: true },
        { href: '/ai-learning-hub/federated', icon: Icons.Layers, label: 'Federe Öğrenme', color: '#7C3AED', key: 'federated', requiresPayment: true },
      ]
    },
    {
      id: 'news',
      title: '📰 HABER & BİLGİ',
      icon: Icons.Newspaper,
      gradient: 'linear-gradient(135deg, #F59E0B 0%, #EA580C 100%)',
      items: [
        { href: '/haberler', icon: Icons.Newspaper, label: 'Haberler', color: '#fbbf24', key: 'haberler', requiresPayment: false },
      ]
    },
  ];

  const visibleMenuItems = allMenuItems.filter(item => !item.requiresPayment || hasActivePayment);
  const filteredCategories = menuCategories.map(category => ({
    ...category,
    items: category.items.filter(item => !item.requiresPayment || hasActivePayment)
  })).filter(category => category.items.length > 0);

  const searchFilteredCategories = drawerSearchTerm
    ? filteredCategories.map(category => ({
        ...category,
        items: category.items.filter(item =>
          item.label.toLowerCase().includes(drawerSearchTerm.toLowerCase()) ||
          item.key.toLowerCase().includes(drawerSearchTerm.toLowerCase())
        )
      })).filter(category => category.items.length > 0)
    : filteredCategories;

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories(prev =>
      prev.includes(categoryId)
        ? prev.filter(id => id !== categoryId)
        : [...prev, categoryId]
    );
  };

  const formatBadge = (count: number) => {
    if (count === 0) return null;
    if (count <= 9) return count;
    return '9+';
  };

  const getCurrentTimeframeLabel = () => {
    return TIMEFRAME_OPTIONS.find(opt => opt.value === timeframe)?.label || '1 Saat';
  };

  const getCurrentSortByLabel = () => {
    return SORTBY_OPTIONS.find(opt => opt.value === sortBy)?.label || 'Hacim';
  };

  return (
    <>
      {/* ═══════════════════════════════════════════ */}
      {/* UNIFIED HEADER BAR (60px) - RESPONSIVE      */}
      {/* ═══════════════════════════════════════════ */}
      <header
        suppressHydrationWarning
        className="unified-header"
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: '60px',
          width: '100%',
          background: 'linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
          gap: '8px',
          zIndex: 1000,
          boxShadow: '0 4px 24px rgba(0, 0, 0, 0.4), 0 0 40px rgba(255, 255, 255, 0.1)',
        }}
      >
        {/* LEFT: Hamburger (Mobile) + Brand + Desktop Menu */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: '0 1 auto', minWidth: 0 }}>
          {/* Hamburger - Mobile/Tablet Only */}
          <button
            onClick={() => setMobileMenuOpen(true)}
            className="hamburger-btn"
            style={{
              display: 'none',
              width: '44px',
              height: '44px',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              borderRadius: '12px',
              cursor: 'pointer',
              transition: 'all 0.3s',
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
            }}
            aria-label="Menüyü Aç"
          >
            <Icons.Menu style={{ width: '24px', height: '24px', color: '#FFFFFF' }} />
          </button>

          {/* Brand */}
          <Link href="/" style={{ textDecoration: 'none', flexShrink: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', lineHeight: '1' }}>
              <span
                style={{
                  fontSize: '16px',
                  fontWeight: '800',
                  background: 'linear-gradient(135deg, #00D4FF 0%, #0EA5E9 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  letterSpacing: '0.5px',
                }}
              >
                CRYPTO
              </span>
              <span
                className="brand-subtitle"
                style={{
                  fontSize: '10px',
                  fontWeight: '600',
                  color: '#FFD700',
                  letterSpacing: '0.3px',
                  marginTop: '2px',
                }}
              >
                by Lydian
              </span>
            </div>
          </Link>

          {/* Desktop Horizontal Menu Icons (>1024px) - ALL MODULES */}
          <div
            className="desktop-menu-icons"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '1px',
              flex: '0 0 auto',
              overflow: 'hidden',
              maxWidth: '100%',
            }}
          >
            {visibleMenuItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = currentPage === item.key;
              const badge = formatBadge(item.notification || 0);

              return (
                <Link key={item.key} href={item.href} style={{ textDecoration: 'none' }}>
                  <div
                    onMouseEnter={() => setHoveredItem(item.key)}
                    onMouseLeave={() => setHoveredItem(null)}
                    style={{
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '32px',
                      height: '32px',
                      background: isActive
                        ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.15))'
                        : hoveredItem === item.key
                        ? 'rgba(255, 255, 255, 0.1)'
                        : 'transparent',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      color: isActive ? '#FFFFFF' : item.color,
                      filter: isActive ? 'drop-shadow(0 0 6px rgba(255, 255, 255, 0.4))' : 'none',
                    }}
                  >
                    <IconComponent style={{ width: '16px', height: '16px' }} />
                    {badge && (
                      <span
                        style={{
                          position: 'absolute',
                          top: '0px',
                          right: '0px',
                          background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                          color: '#fff',
                          fontSize: '7px',
                          fontWeight: '700',
                          padding: '1px 3px',
                          borderRadius: '4px',
                          minWidth: '10px',
                          height: '10px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          border: '1px solid rgba(0, 0, 0, 0.8)',
                        }}
                      >
                        {badge}
                      </span>
                    )}
                    {hoveredItem === item.key && (
                      <div
                        style={{
                          position: 'absolute',
                          bottom: '-30px',
                          left: '50%',
                          transform: 'translateX(-50%)',
                          whiteSpace: 'nowrap',
                          background: 'rgba(0, 0, 0, 0.95)',
                          color: '#fff',
                          fontSize: '9px',
                          fontWeight: '600',
                          padding: '4px 7px',
                          borderRadius: '5px',
                          pointerEvents: 'none',
                          zIndex: 10000,
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                        }}
                      >
                        {item.label}
                      </div>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        </div>

        {/* CENTER: Logo/Title - Spacer */}
        <div style={{ flex: '1 1 auto' }} />

        {/* RIGHT: Actions */}
        <div
          className="header-actions"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            flex: '0 0 auto',
          }}
        >
          {/* Search Icon - Toggle Dropdown */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowSearchDropdown(!showSearchDropdown);
            }}
            style={{
              width: '36px',
              height: '36px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: showSearchDropdown
                ? 'linear-gradient(135deg, rgba(0, 212, 255, 0.25), rgba(14, 165, 233, 0.25))'
                : 'rgba(255, 255, 255, 0.05)',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              position: 'relative',
            }}
            onMouseEnter={(e) => {
              if (!showSearchDropdown) {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
              }
            }}
            onMouseLeave={(e) => {
              if (!showSearchDropdown) {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
              }
            }}
            aria-label="Arama"
          >
            <Icons.Search style={{
              width: '18px',
              height: '18px',
              color: showSearchDropdown ? '#00D4FF' : '#FFFFFF'
            }} />
          </button>

          {/* Coin Count */}
          <div
            className="coin-count-badge"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '5px 8px',
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: '600',
              color: '#FFFFFF',
            }}
          >
            <Icons.TrendingUp style={{ width: '12px', height: '12px' }} />
            <span>{coinCount}</span>
          </div>

          {/* Countdown */}
          <div
            className="countdown-badge"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '5px 8px',
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: '600',
              color: '#FFFFFF',
            }}
          >
            <Icons.Clock style={{ width: '12px', height: '12px' }} />
            <span>{countdown}s</span>
          </div>

          {/* Notifications */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowNotifications(!showNotifications)}
              style={{
                position: 'relative',
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(255, 255, 255, 0.05)',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
              }}
              aria-label="Bildirimler"
            >
              <Icons.Bell style={{ width: '16px', height: '16px', color: '#FFFFFF' }} />
              {globalUnreadCount > 0 && (
                <span
                  style={{
                    position: 'absolute',
                    top: '2px',
                    right: '2px',
                    minWidth: '14px',
                    height: '14px',
                    background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '8px',
                    fontWeight: '700',
                    color: '#FFFFFF',
                    padding: '0 3px',
                    animation: 'pulse 2s infinite'
                  }}
                >
                  {globalUnreadCount > 9 ? '9+' : globalUnreadCount}
                </span>
              )}
            </button>
            <NotificationPanel
              isOpen={showNotifications}
              onClose={() => setShowNotifications(false)}
            />
          </div>

          {/* AI Assistant */}
          {(isLocalhost || isProduction) && (
            <button
              onClick={onAiAssistantOpen}
              style={{
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(99, 102, 241, 0.1))',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                position: 'relative',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(99, 102, 241, 0.2))';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(99, 102, 241, 0.1))';
              }}
              aria-label="AI Asistan"
            >
              <Icons.Bot style={{ width: '18px', height: '18px', color: '#a78bfa' }} />
            </button>
          )}

          {/* Settings */}
          <Link href="/settings" style={{ textDecoration: 'none' }}>
            <button
              className="settings-btn"
              style={{
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(14, 165, 233, 0.1))',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(14, 165, 233, 0.2))';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(14, 165, 233, 0.1))';
              }}
              aria-label="Ayarlar"
            >
              <Icons.Settings style={{ width: '18px', height: '18px', color: '#60a5fa' }} />
            </button>
          </Link>
        </div>
      </header>

      {/* ═══════════════════════════════════════════ */}
      {/* ULTRA PREMIUM SEARCH DROPDOWN               */}
      {/* ═══════════════════════════════════════════ */}
      {showSearchDropdown && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'fixed',
            top: '60px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 'calc(100% - 32px)',
            maxWidth: '600px',
            background: 'linear-gradient(135deg, rgba(20, 20, 20, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
            backdropFilter: 'blur(24px)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            borderRadius: '16px',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.8), 0 0 60px rgba(0, 212, 255, 0.15)',
            padding: '16px',
            zIndex: 1003,
            marginTop: '8px',
            animation: 'slideDown 0.3s ease-out',
          }}
        >
          {/* Search Input */}
          <div style={{ position: 'relative', marginBottom: '12px' }}>
            <Icons.Search
              style={{
                position: 'absolute',
                left: '16px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '18px',
                height: '18px',
                color: 'rgba(255, 255, 255, 0.4)',
                pointerEvents: 'none',
              }}
            />
            <input
              type="text"
              placeholder="Coin ara... (BTC, ETH, SOL)"
              value={searchTerm}
              onChange={(e) => onSearchChange?.(e.target.value)}
              autoFocus
              style={{
                width: '100%',
                height: '48px',
                background: 'rgba(255, 255, 255, 0.06)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                borderRadius: '12px',
                padding: '0 16px 0 48px',
                fontSize: '14px',
                fontWeight: '500',
                color: '#fff',
                outline: 'none',
                transition: 'all 0.2s',
              }}
              onFocus={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.5)';
                e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.2)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.12)';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />
            {searchTerm && (
              <button
                onClick={() => onSearchChange?.('')}
                style={{
                  position: 'absolute',
                  right: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '24px',
                  height: '24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                }}
                aria-label="Temizle"
              >
                <Icons.X style={{ width: '14px', height: '14px', color: '#fff' }} />
              </button>
            )}
          </div>

          {/* Search Info */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 12px',
              background: 'rgba(0, 212, 255, 0.08)',
              borderRadius: '10px',
              fontSize: '12px',
              color: 'rgba(255, 255, 255, 0.7)',
            }}
          >
            <Icons.Info style={{ width: '16px', height: '16px', color: '#00D4FF', flexShrink: 0 }} />
            <span>
              {searchTerm
                ? `"${searchTerm}" arıyor... Tabloda filtreleniyor.`
                : 'Coin sembolü veya isim girin (örn: BTC, ETH, SOL)'}
            </span>
          </div>

          {/* Close Hint */}
          <div
            style={{
              marginTop: '12px',
              textAlign: 'center',
              fontSize: '11px',
              color: 'rgba(255, 255, 255, 0.4)',
              fontWeight: '500',
            }}
          >
            ESC tuşu veya dışarı tıklayarak kapat
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════ */}
      {/* MODERN COMPACT FILTER BAR - DROPDOWNS       */}
      {/* ═══════════════════════════════════════════ */}
      {(isLocalhost || isProduction) && (
        <div
          suppressHydrationWarning
          className="modern-filter-bar"
          style={{
            position: 'fixed',
            top: '60px',
            left: 0,
            right: 0,
            height: 'auto',
            minHeight: '48px',
            width: '100%',
            background: 'linear-gradient(135deg, rgba(20, 20, 20, 0.95) 0%, rgba(10, 10, 10, 0.95) 100%)',
            backdropFilter: 'blur(16px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '8px 16px',
            gap: '12px',
            zIndex: 999,
            flexWrap: 'wrap',
            boxShadow: '0 2px 16px rgba(0, 0, 0, 0.3)',
          }}
        >
          {/* Timeframe Dropdown */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowTimeframeDropdown(!showTimeframeDropdown);
                setShowSortByDropdown(false);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '10px 14px',
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s',
                minWidth: '120px',
                minHeight: '44px',
                justifyContent: 'space-between',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
              }}
              aria-label="Zaman Dilimi"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Icons.Clock style={{ width: '14px', height: '14px' }} />
                <span>{getCurrentTimeframeLabel()}</span>
              </div>
              <Icons.ChevronDown style={{ width: '14px', height: '14px' }} />
            </button>
            {showTimeframeDropdown && (
              <div
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 4px)',
                  left: 0,
                  minWidth: '140px',
                  background: 'rgba(20, 20, 20, 0.98)',
                  backdropFilter: 'blur(16px)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '10px',
                  padding: '6px',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                  zIndex: 10000,
                  animation: 'slideDown 0.2s ease-out',
                }}
              >
                {TIMEFRAME_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={(e) => {
                      e.stopPropagation();
                      setTimeframe(opt.value);
                      setShowTimeframeDropdown(false);
                    }}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      background: timeframe === opt.value ? 'rgba(255, 255, 255, 0.15)' : 'transparent',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#fff',
                      fontSize: '13px',
                      fontWeight: timeframe === opt.value ? '700' : '500',
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      marginBottom: '2px',
                    }}
                    onMouseEnter={(e) => {
                      if (timeframe !== opt.value) {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (timeframe !== opt.value) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                  >
                    <span>{opt.label}</span>
                    {timeframe === opt.value && (
                      <span style={{ fontSize: '14px' }}>✓</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Divider */}
          <div style={{ width: '1px', height: '20px', background: 'rgba(255, 255, 255, 0.15)' }} />

          {/* SortBy Dropdown */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowSortByDropdown(!showSortByDropdown);
                setShowTimeframeDropdown(false);
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '10px 14px',
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '13px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.2s',
                minWidth: '120px',
                minHeight: '44px',
                justifyContent: 'space-between',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
              }}
              aria-label="Sıralama"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Icons.BarChart style={{ width: '14px', height: '14px' }} />
                <span>{getCurrentSortByLabel()}</span>
              </div>
              <Icons.ChevronDown style={{ width: '14px', height: '14px' }} />
            </button>
            {showSortByDropdown && (
              <div
                style={{
                  position: 'absolute',
                  top: 'calc(100% + 4px)',
                  left: 0,
                  minWidth: '140px',
                  background: 'rgba(20, 20, 20, 0.98)',
                  backdropFilter: 'blur(16px)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '10px',
                  padding: '6px',
                  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                  zIndex: 10000,
                  animation: 'slideDown 0.2s ease-out',
                }}
              >
                {SORTBY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={(e) => {
                      e.stopPropagation();
                      setSortBy(opt.value);
                      setShowSortByDropdown(false);
                    }}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      background: sortBy === opt.value ? 'rgba(255, 255, 255, 0.15)' : 'transparent',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#fff',
                      fontSize: '13px',
                      fontWeight: sortBy === opt.value ? '700' : '500',
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      marginBottom: '2px',
                    }}
                    onMouseEnter={(e) => {
                      if (sortBy !== opt.value) {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (sortBy !== opt.value) {
                        e.currentTarget.style.background = 'transparent';
                      }
                    }}
                  >
                    <span>{opt.label}</span>
                    {sortBy === opt.value && (
                      <span style={{ fontSize: '14px' }}>✓</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════ */}
      {/* PREMIUM ANIMATED DRAWER - GLASSMORPHISM      */}
      {/* ═══════════════════════════════════════════ */}

      {mobileMenuOpen && (
        <div
          onClick={() => setMobileMenuOpen(false)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.75)',
            backdropFilter: 'blur(8px)',
            zIndex: 9998,
            animation: 'fadeIn 0.25s ease-out',
          }}
        />
      )}

      <div
        suppressHydrationWarning
        className="premium-animated-drawer"
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          bottom: 0,
          width: '340px',
          maxWidth: '88vw',
          background: 'linear-gradient(135deg, rgba(20, 20, 20, 0.98) 0%, rgba(10, 10, 10, 0.98) 100%)',
          backdropFilter: 'blur(24px)',
          borderLeft: '1px solid rgba(255, 255, 255, 0.15)',
          boxShadow: '-6px 0 36px rgba(0, 0, 0, 0.6), inset 1px 0 0 rgba(255,255,255,0.05)',
          zIndex: 9999,
          transform: mobileMenuOpen ? 'translateX(0)' : 'translateX(100%)',
          transition: 'transform 0.35s cubic-bezier(0.23, 1, 0.32, 1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Drawer Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.12), rgba(14, 165, 233, 0.08))',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', lineHeight: '1' }}>
            <span
              style={{
                fontSize: '22px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #00D4FF 0%, #0EA5E9 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                letterSpacing: '0.8px',
              }}
            >
              CRYPTO
            </span>
            <span
              style={{
                fontSize: '13px',
                fontWeight: '700',
                color: '#FFD700',
                letterSpacing: '0.4px',
                marginTop: '4px',
              }}
            >
              by Lydian
            </span>
          </div>
          <button
            onClick={() => setMobileMenuOpen(false)}
            style={{
              width: '40px',
              height: '40px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'rgba(255, 255, 255, 0.1)',
              border: 'none',
              borderRadius: '10px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            aria-label="Kapat"
          >
            <Icons.X style={{ width: '20px', height: '20px', color: '#FFF' }} />
          </button>
        </div>

        {/* Search */}
        <div style={{ padding: '14px 16px', borderBottom: '1px solid rgba(255, 255, 255, 0.08)' }}>
          <div style={{ position: 'relative' }}>
            <Icons.Search
              style={{
                position: 'absolute',
                left: '14px',
                top: '50%',
                transform: 'translateY(-50%)',
                width: '16px',
                height: '16px',
                color: 'rgba(255, 255, 255, 0.4)',
                pointerEvents: 'none',
              }}
            />
            <input
              type="text"
              placeholder="Menüde ara..."
              value={drawerSearchTerm}
              onChange={(e) => setDrawerSearchTerm(e.target.value)}
              style={{
                width: '100%',
                height: '44px',
                background: 'rgba(255, 255, 255, 0.06)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '12px',
                padding: '0 16px 0 42px',
                fontSize: '14px',
                color: '#fff',
                outline: 'none',
                transition: 'all 0.2s',
              }}
              onFocus={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.2)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
              }}
            />
          </div>
        </div>

        {/* Stats */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-around',
            padding: '12px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.02)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: '#FFF' }}>
            <Icons.TrendingUp style={{ width: '16px', height: '16px', color: '#10B981' }} />
            <span>{coinCount}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: '#FFF' }}>
            <Icons.Clock style={{ width: '16px', height: '16px', color: '#F59E0B' }} />
            <span>{countdown}s</span>
          </div>
          <ActiveUsersIndicator />
        </div>

        {/* Categories */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          {searchFilteredCategories.map((cat) => {
            const isExp = expandedCategories.includes(cat.id);
            const CatIcon = cat.icon;

            return (
              <div key={cat.id} style={{ marginBottom: '8px' }}>
                <button
                  onClick={() => toggleCategory(cat.id)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '12px 14px',
                    background: cat.gradient,
                    border: 'none',
                    borderRadius: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.25s cubic-bezier(0.23, 1, 0.32, 1)',
                    marginBottom: '6px',
                    boxShadow: isExp ? '0 4px 16px rgba(0,0,0,0.3)' : 'none',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.02) translateX(2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                  aria-expanded={isExp}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <CatIcon style={{ width: '18px', height: '18px', color: '#FFF' }} />
                    <span style={{ fontSize: '13px', fontWeight: '800', color: '#FFF', letterSpacing: '0.5px' }}>
                      {cat.title}
                    </span>
                  </div>
                  {isExp ? (
                    <Icons.ChevronDown style={{ width: '16px', height: '16px', color: '#FFF' }} />
                  ) : (
                    <Icons.ChevronRight style={{ width: '16px', height: '16px', color: '#FFF' }} />
                  )}
                </button>

                {isExp && (
                  <div style={{ paddingLeft: '8px', animation: 'slideDown 0.3s ease-out' }}>
                    {cat.items.map((it) => {
                      const ItIcon = it.icon;
                      const isAct = currentPage === it.key;
                      const bdg = formatBadge(it.notification || 0);

                      return (
                        <Link key={it.key} href={it.href} style={{ textDecoration: 'none' }}>
                          <div
                            onClick={() => setMobileMenuOpen(false)}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '12px',
                              padding: '10px 12px',
                              marginBottom: '4px',
                              background: isAct
                                ? 'linear-gradient(135deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.1))'
                                : 'transparent',
                              border: isAct ? '1px solid rgba(255,255,255,0.15)' : 'none',
                              borderRadius: '10px',
                              cursor: 'pointer',
                              transition: 'all 0.2s',
                            }}
                            onMouseEnter={(e) => {
                              if (!isAct) {
                                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                                e.currentTarget.style.transform = 'translateX(4px)';
                              }
                            }}
                            onMouseLeave={(e) => {
                              if (!isAct) {
                                e.currentTarget.style.background = 'transparent';
                                e.currentTarget.style.transform = 'translateX(0)';
                              }
                            }}
                          >
                            <div
                              style={{
                                width: '34px',
                                height: '34px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                background: isAct ? 'rgba(255, 255, 255, 0.18)' : 'rgba(255, 255, 255, 0.06)',
                                borderRadius: '8px',
                                color: it.color,
                              }}
                            >
                              <ItIcon style={{ width: '18px', height: '18px' }} />
                            </div>
                            <span
                              style={{
                                fontSize: '14px',
                                fontWeight: isAct ? '700' : '500',
                                color: '#FFF',
                                flex: 1,
                              }}
                            >
                              {it.label}
                            </span>
                            {bdg && (
                              <span
                                style={{
                                  background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                                  color: '#fff',
                                  fontSize: '10px',
                                  fontWeight: '700',
                                  padding: '4px 7px',
                                  borderRadius: '8px',
                                  minWidth: '22px',
                                  textAlign: 'center',
                                }}
                              >
                                {bdg}
                              </span>
                            )}
                          </div>
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <style jsx global suppressHydrationWarning>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .premium-animated-drawer::-webkit-scrollbar {
          width: 5px;
        }
        .premium-animated-drawer::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.04);
        }
        .premium-animated-drawer::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
        }

        /* Tablet (768px - 1024px) */
        @media (max-width: 1024px) and (min-width: 768px) {
          .hamburger-btn {
            display: flex !important;
          }
          .desktop-menu-icons {
            display: none !important;
          }
          .header-search {
            display: none !important;
          }
          .modern-filter-bar {
            padding: 8px 16px !important;
            gap: 12px !important;
            justify-content: center !important;
          }
        }

        /* Mobile (< 768px) */
        @media (max-width: 767px) {
          .hamburger-btn {
            display: flex !important;
          }
          .desktop-menu-icons {
            display: none !important;
          }
          .header-search {
            display: none !important;
          }
          .coin-count-badge,
          .countdown-badge {
            font-size: 10px !important;
            padding: 4px 6px !important;
          }
          .modern-filter-bar {
            padding: 8px 12px !important;
            gap: 8px !important;
            height: auto !important;
            min-height: 56px !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
          }
          .modern-filter-bar > div {
            flex: 0 0 auto !important;
          }
          .modern-filter-bar button {
            min-width: 110px !important;
            min-height: 44px !important;
            padding: 10px 14px !important;
            font-size: 13px !important;
          }
        }

        /* Ultra Mobile (< 480px) */
        @media (max-width: 480px) {
          .brand-subtitle {
            display: block !important;
            font-size: 8px !important;
          }
          .coin-count-badge,
          .countdown-badge {
            display: none !important;
          }
          .settings-btn {
            display: flex !important;
          }
          /* Fix notification panel positioning on mobile */
          .notification-panel-dropdown {
            right: -8px !important;
            width: calc(100vw - 16px) !important;
            max-width: 320px !important;
          }
        }

        /* Desktop (> 1024px) */
        @media (min-width: 1025px) {
          .hamburger-btn {
            display: none !important;
          }
          .desktop-menu-icons {
            display: flex !important;
          }
          .header-search {
            display: flex !important;
          }
          .premium-animated-drawer {
            display: none !important;
          }
        }
      `}</style>
    </>
  );
}
