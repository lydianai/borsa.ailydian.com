'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, Home, TrendingUp, Bot, Settings, BarChart3, Bell } from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { label: 'Dashboard', href: '/', icon: Home },
  { label: 'Signals', href: '/conservative-signals', icon: TrendingUp },
  { label: 'Bot Analysis', href: '/bot-analysis', icon: Bot },
  { label: 'Market Insights', href: '/market-insights', icon: BarChart3 },
  { label: 'Alerts', href: '/settings', icon: Bell },
  { label: 'Settings', href: '/settings', icon: Settings },
];

/**
 * MobileNav - Responsive mobile navigation with hamburger menu
 *
 * Features:
 * - Hamburger menu for mobile (<768px)
 * - Slide-in drawer animation
 * - Active route highlighting
 * - Touch-friendly tap targets (min 44x44px)
 */
export function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <>
      {/* Mobile Menu Button (visible on mobile only) */}
      <button
        onClick={toggleMenu}
        className="md:hidden fixed top-4 right-4 z-50 p-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 transition-colors"
        aria-label="Toggle menu"
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <X className="w-6 h-6 text-white" />
        ) : (
          <Menu className="w-6 h-6 text-white" />
        )}
      </button>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden"
          onClick={closeMenu}
          aria-hidden="true"
        />
      )}

      {/* Mobile Drawer */}
      <nav
        className={cn(
          'fixed top-0 right-0 h-full w-64 bg-neutral-900 shadow-2xl z-40 md:hidden',
          'transform transition-transform duration-300 ease-in-out',
          isOpen ? 'translate-x-0' : 'translate-x-full'
        )}
        aria-label="Mobile navigation"
      >
        <div className="flex flex-col h-full pt-20 px-4 pb-6">
          {/* Navigation Links */}
          <ul className="space-y-2 flex-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={closeMenu}
                    className={cn(
                      'flex items-center gap-3 px-4 py-3 rounded-lg',
                      'transition-all duration-200',
                      'text-base font-medium',
                      'min-h-[44px]', // Touch-friendly tap target
                      isActive
                        ? 'bg-brand-600 text-white shadow-glow-brand'
                        : 'text-neutral-300 hover:bg-neutral-800 hover:text-white'
                    )}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    <span>{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* Footer */}
          <div className="pt-4 border-t border-neutral-800">
            <p className="text-xs text-neutral-500 text-center">
              AILYDIAN Trading Platform
            </p>
            <p className="text-xs text-neutral-600 text-center mt-1">
              v1.0.0 | © 2025
            </p>
          </div>
        </div>
      </nav>

      {/* Desktop Navigation (horizontal, hidden on mobile) */}
      <nav className="hidden md:block fixed top-0 left-0 right-0 z-40 bg-neutral-900/95 backdrop-blur-lg border-b border-neutral-800">
        <div className="max-w-7xl mx-auto px-6">
          <ul className="flex items-center gap-1 py-3">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2 rounded-lg',
                      'transition-all duration-200',
                      'text-sm font-medium',
                      isActive
                        ? 'bg-brand-600 text-white'
                        : 'text-neutral-300 hover:bg-neutral-800 hover:text-white'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      </nav>
    </>
  );
}
