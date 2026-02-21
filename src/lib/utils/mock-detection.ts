/**
 * Mock Data Detection Utilities
 *
 * White-hat compliance: Clearly mark mock/demo data to prevent user confusion
 * and ensure transparency
 */

/**
 * Add mock data warning to API response
 */
export function addMockDataWarning<T extends Record<string, unknown>>(
  response: T,
  warningMessage?: string
): T & { isMockData: true; mockDataWarning: string } {
  return {
    ...response,
    isMockData: true,
    mockDataWarning:
      warningMessage ||
      '⚠️ DEMO DATA: This is simulated data for development/testing purposes. NOT real market data.',
  };
}

/**
 * Check if response contains mock data
 */
export function isMockData(response: unknown): boolean {
  return (
    typeof response === 'object' &&
    response !== null &&
    'isMockData' in response &&
    response.isMockData === true
  );
}

/**
 * Environment-based mock data detection
 */
export function shouldUseMockData(): boolean {
  // Use mock data in test environment
  if (process.env.NODE_ENV === 'test') {
    return true;
  }

  // Check for explicit mock data flag
  if (process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true') {
    return true;
  }

  // System uses real Binance data by default
  // Only show warning if explicitly set to use mock data
  return false;
}

/**
 * Get mock data status message
 */
export function getMockDataStatusMessage(): string {
  if (process.env.NODE_ENV === 'test') {
    return 'Using mock data: Test environment';
  }

  if (process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true') {
    return 'Using mock data: Explicitly enabled via environment variable';
  }

  const requiredKeys = ['GROQ_API_KEY', 'ANTHROPIC_API_KEY'];
  const missingKeys = requiredKeys.filter((key) => !process.env[key]);

  if (missingKeys.length > 0) {
    return `Using mock data: Missing API keys (${missingKeys.join(', ')})`;
  }

  return 'Using real data: All API keys configured';
}

/**
 * Mock data warning banner text
 */
export const MOCK_DATA_BANNER_TEXT = {
  title: '⚠️ DEMO MODE',
  description:
    'You are viewing simulated data for demonstration purposes. This is NOT real market data and should not be used for actual trading decisions.',
  action: 'Configure API keys in .env.local to use real data',
} as const;

/**
 * Console warning for mock data usage
 */
export function logMockDataWarning(context: string): void {
  if (process.env.NODE_ENV !== 'production') {
    console.warn(
      `[Mock Data Warning] ${context}:`,
      getMockDataStatusMessage()
    );
  }
}
