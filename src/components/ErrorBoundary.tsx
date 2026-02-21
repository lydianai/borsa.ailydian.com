'use client';

/**
 * Error Boundary Component
 *
 * White-hat compliance: Prevents error information leakage and provides
 * graceful error recovery
 *
 * OWASP: https://owasp.org/www-community/Improper_Error_Handling
 */

import React, { Component, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  showDetails?: boolean;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

/**
 * Error Boundary for catching React component errors
 *
 * Usage:
 * <ErrorBoundary fallback={<CustomErrorUI />}>
 *   <YourComponent />
 * </ErrorBoundary>
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    // Log error to monitoring service
    this.logErrorToService(error, errorInfo);

    // Update state with error details
    this.setState({
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  private logErrorToService(error: Error, errorInfo: React.ErrorInfo): void {
    // In development, log to console
    if (process.env.NODE_ENV === 'development') {
      console.error('[ErrorBoundary] Component error:', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
      });
    }

    // Send to Sentry in production
    if (process.env.NODE_ENV === 'production') {
      try {
        // Dynamic import to avoid bundling in development
        import('@/lib/monitoring/sentry').then(({ captureException }) => {
          captureException(error, {
            react: {
              componentStack: errorInfo.componentStack,
            },
          });
        });
      } catch (sentryError) {
        console.error('[Sentry] Failed to log error:', sentryError);
      }
    }
  }

  private resetError = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <DefaultErrorFallback
          error={this.state.error}
          errorInfo={this.state.errorInfo}
          resetError={this.resetError}
          showDetails={this.props.showDetails ?? process.env.NODE_ENV === 'development'}
        />
      );
    }

    return this.props.children;
  }
}

/**
 * Default Error Fallback UI
 */
interface DefaultErrorFallbackProps {
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  resetError: () => void;
  showDetails: boolean;
}

function DefaultErrorFallback({
  error,
  errorInfo,
  resetError,
  showDetails,
}: DefaultErrorFallbackProps): JSX.Element {
  return (
    <div className="min-h-[400px] flex items-center justify-center p-6">
      <div className="max-w-lg w-full bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            <svg
              className="w-6 h-6 text-red-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>

          <div className="flex-1">
            <h3 className="text-lg font-semibold text-red-900 mb-2">
              Something went wrong
            </h3>

            <p className="text-sm text-red-700 mb-4">
              An unexpected error occurred while rendering this component. The error has been
              logged and our team has been notified.
            </p>

            {showDetails && error && (
              <div className="mb-4">
                <p className="text-xs font-mono text-red-800 bg-red-100 p-3 rounded border border-red-200 overflow-auto">
                  {error.message}
                </p>
                {errorInfo && (
                  <details className="mt-2">
                    <summary className="text-xs text-red-700 cursor-pointer hover:text-red-900">
                      Component Stack
                    </summary>
                    <pre className="text-xs font-mono text-red-800 bg-red-100 p-3 rounded border border-red-200 mt-2 overflow-auto">
                      {errorInfo.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={resetError}
                className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded hover:bg-red-700 transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-white text-red-600 text-sm font-medium rounded border border-red-300 hover:bg-red-50 transition-colors"
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Minimal Error Boundary for critical sections
 */
interface MinimalErrorBoundaryProps {
  children: ReactNode;
  fallbackMessage?: string;
}

export class MinimalErrorBoundary extends Component<
  MinimalErrorBoundaryProps,
  { hasError: boolean }
> {
  constructor(props: MinimalErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    if (process.env.NODE_ENV === 'development') {
      console.error('[MinimalErrorBoundary]', error, errorInfo);
    }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {this.props.fallbackMessage || 'Failed to load this section'}
        </div>
      );
    }

    return this.props.children;
  }
}
