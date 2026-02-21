/**
 * API Health Check Tests
 *
 * Tests the basic health check endpoint to ensure API is functioning
 */

import { describe, it, expect, beforeEach } from 'vitest';

describe('Health Check API', () => {
  const baseUrl = 'http://localhost:3000';

  describe('GET /api/health', () => {
    it('should return 200 status code', async () => {
      const response = await fetch(`${baseUrl}/api/health`);
      expect(response.status).toBe(200);
    });

    it('should return correct response structure', async () => {
      const response = await fetch(`${baseUrl}/api/health`);
      const data = await response.json();

      expect(data).toHaveProperty('status');
      expect(data).toHaveProperty('message');
      expect(data).toHaveProperty('timestamp');
      expect(data).toHaveProperty('version');
    });

    it('should return status ok', async () => {
      const response = await fetch(`${baseUrl}/api/health`);
      const data = await response.json();

      expect(data.status).toBe('ok');
      expect(data.message).toBe('Backend API is running');
    });

    it('should return valid timestamp', async () => {
      const response = await fetch(`${baseUrl}/api/health`);
      const data = await response.json();

      const timestamp = new Date(data.timestamp);
      expect(timestamp).toBeInstanceOf(Date);
      expect(timestamp.getTime()).toBeLessThanOrEqual(Date.now());
    });

    it('should have valid version format', async () => {
      const response = await fetch(`${baseUrl}/api/health`);
      const data = await response.json();

      expect(data.version).toMatch(/^\d+\.\d+\.\d+/);
    });
  });
});
