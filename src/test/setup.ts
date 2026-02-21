/**
 * Vitest Test Setup
 *
 * Runs before all tests to configure the testing environment
 */

import '@testing-library/jest-dom';
import { beforeAll, afterEach, afterAll, vi } from 'vitest';

// Mock environment variables for tests
process.env.NODE_ENV = 'test';
process.env.NEXT_PUBLIC_APP_URL = 'http://localhost:3000';

// Global test setup
beforeAll(() => {
  // Setup before all tests
});

// Cleanup after each test
afterEach(() => {
  // Clear all mocks after each test
  vi.clearAllMocks();
});

// Global test teardown
afterAll(() => {
  // Cleanup after all tests
});

// Mock fetch globally (for API tests)
global.fetch = vi.fn();

// Suppress console errors in tests (optional)
const originalError = console.error;
beforeAll(() => {
  console.error = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('Warning: ReactDOM.render')
    ) {
      return;
    }
    originalError.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
});
