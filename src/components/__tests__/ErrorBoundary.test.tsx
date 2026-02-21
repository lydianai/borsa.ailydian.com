/**
 * Error Boundary Tests
 *
 * White-hat testing: Ensures error boundaries catch and handle errors properly
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorBoundary, MinimalErrorBoundary } from '../ErrorBoundary';

// Component that throws an error
function ThrowError({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
}

// Component that throws during render
function AlwaysThrows() {
  throw new Error('Always throws error');
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // Suppress console.error for tests
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should render children when no error occurs', () => {
    render(
      <ErrorBoundary>
        <div>Test content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('should catch errors and render default fallback', () => {
    render(
      <ErrorBoundary>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText(/An unexpected error occurred/)).toBeInTheDocument();
  });

  it('should render custom fallback when provided', () => {
    const CustomFallback = <div>Custom error UI</div>;

    render(
      <ErrorBoundary fallback={CustomFallback}>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom error UI')).toBeInTheDocument();
  });

  it('should display error message in development mode', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';

    render(
      <ErrorBoundary showDetails={true}>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.getByText('Always throws error')).toBeInTheDocument();

    process.env.NODE_ENV = originalEnv;
  });

  it('should hide error details when showDetails is false', () => {
    render(
      <ErrorBoundary showDetails={false}>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.queryByText('Always throws error')).not.toBeInTheDocument();
  });

  it('should call onError callback when error occurs', () => {
    const onError = vi.fn();

    render(
      <ErrorBoundary onError={onError}>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
      })
    );
  });

  it('should render Try Again button in fallback UI', () => {
    render(
      <ErrorBoundary>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('should render Reload Page button in fallback UI', () => {
    render(
      <ErrorBoundary>
        <AlwaysThrows />
      </ErrorBoundary>
    );

    expect(screen.getByRole('button', { name: /reload page/i })).toBeInTheDocument();
  });
});

describe('MinimalErrorBoundary', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should render children when no error occurs', () => {
    render(
      <MinimalErrorBoundary>
        <div>Test content</div>
      </MinimalErrorBoundary>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('should render default fallback message on error', () => {
    render(
      <MinimalErrorBoundary>
        <AlwaysThrows />
      </MinimalErrorBoundary>
    );

    expect(screen.getByText('Failed to load this section')).toBeInTheDocument();
  });

  it('should render custom fallback message when provided', () => {
    render(
      <MinimalErrorBoundary fallbackMessage="Custom error message">
        <AlwaysThrows />
      </MinimalErrorBoundary>
    );

    expect(screen.getByText('Custom error message')).toBeInTheDocument();
  });

  it('should have appropriate styling classes', () => {
    render(
      <MinimalErrorBoundary>
        <AlwaysThrows />
      </MinimalErrorBoundary>
    );

    const fallback = screen.getByText('Failed to load this section');
    expect(fallback.className).toContain('bg-red-50');
    expect(fallback.className).toContain('border-red-200');
    expect(fallback.className).toContain('text-red-700');
  });
});
