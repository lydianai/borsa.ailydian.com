/**
 * Validation Schemas Tests
 *
 * White-hat testing: Ensures input validation prevents injection attacks
 * and data corruption
 */

import { describe, it, expect } from 'vitest';
import {
  SymbolSchema,
  TimeframeSchema,
  ConfidenceSchema,
  LimitSchema,
  PriceSchema,
  LeverageSchema,
  SignalRequestSchema,
  ConservativeSignalRequestSchema,
  BotControlRequestSchema,
  validateRequest,
  safeValidateRequest,
  formatZodError,
} from '../validation/schemas';

describe('SymbolSchema', () => {
  it('should accept valid USDT symbols', () => {
    expect(() => SymbolSchema.parse('BTCUSDT')).not.toThrow();
    expect(() => SymbolSchema.parse('ETHUSDT')).not.toThrow();
    expect(() => SymbolSchema.parse('SOLUSDT')).not.toThrow();
  });

  it('should reject invalid symbols', () => {
    expect(() => SymbolSchema.parse('BTC')).toThrow();
    expect(() => SymbolSchema.parse('btcusdt')).toThrow(); // lowercase
    expect(() => SymbolSchema.parse('BTC-USDT')).toThrow(); // dash
    expect(() => SymbolSchema.parse('BTCUSD')).toThrow(); // not USDT
  });

  it('should reject symbols that are too short or too long', () => {
    expect(() => SymbolSchema.parse('ABUSDT')).toThrow(); // too short
    expect(() => SymbolSchema.parse('VERYLONGSYMBOLUSDT')).toThrow(); // too long
  });
});

describe('TimeframeSchema', () => {
  it('should accept valid timeframes', () => {
    expect(() => TimeframeSchema.parse('1m')).not.toThrow();
    expect(() => TimeframeSchema.parse('1h')).not.toThrow();
    expect(() => TimeframeSchema.parse('1d')).not.toThrow();
    expect(() => TimeframeSchema.parse('1w')).not.toThrow();
  });

  it('should reject invalid timeframes', () => {
    expect(() => TimeframeSchema.parse('1s')).toThrow();
    expect(() => TimeframeSchema.parse('5h')).toThrow();
    expect(() => TimeframeSchema.parse('1D')).toThrow(); // uppercase
  });
});

describe('ConfidenceSchema', () => {
  it('should accept valid confidence values', () => {
    expect(() => ConfidenceSchema.parse(0)).not.toThrow();
    expect(() => ConfidenceSchema.parse(50)).not.toThrow();
    expect(() => ConfidenceSchema.parse(100)).not.toThrow();
  });

  it('should reject out of range values', () => {
    expect(() => ConfidenceSchema.parse(-1)).toThrow();
    expect(() => ConfidenceSchema.parse(101)).toThrow();
  });
});

describe('LimitSchema', () => {
  it('should accept valid limits', () => {
    expect(() => LimitSchema.parse(1)).not.toThrow();
    expect(() => LimitSchema.parse(50)).not.toThrow();
    expect(() => LimitSchema.parse(200)).not.toThrow();
  });

  it('should reject invalid limits', () => {
    expect(() => LimitSchema.parse(0)).toThrow();
    expect(() => LimitSchema.parse(201)).toThrow();
    expect(() => LimitSchema.parse(1.5)).toThrow(); // not integer
  });
});

describe('PriceSchema', () => {
  it('should accept valid prices', () => {
    expect(() => PriceSchema.parse(0.01)).not.toThrow();
    expect(() => PriceSchema.parse(50000)).not.toThrow();
    expect(() => PriceSchema.parse(0.12345678)).not.toThrow(); // 8 decimals OK
  });

  it('should reject invalid prices', () => {
    expect(() => PriceSchema.parse(0)).toThrow(); // not positive
    expect(() => PriceSchema.parse(-100)).toThrow(); // negative
    expect(() => PriceSchema.parse(Infinity)).toThrow(); // not finite
  });

  it('should reject prices with too many decimals', () => {
    expect(() => PriceSchema.parse(0.123456789)).toThrow(); // 9 decimals
  });
});

describe('LeverageSchema', () => {
  it('should accept valid leverage values', () => {
    expect(() => LeverageSchema.parse(1)).not.toThrow();
    expect(() => LeverageSchema.parse(10)).not.toThrow();
    expect(() => LeverageSchema.parse(125)).not.toThrow();
  });

  it('should reject invalid leverage', () => {
    expect(() => LeverageSchema.parse(0)).toThrow();
    expect(() => LeverageSchema.parse(126)).toThrow();
    expect(() => LeverageSchema.parse(1.5)).toThrow(); // not integer
  });
});

describe('SignalRequestSchema', () => {
  it('should accept valid signal request', () => {
    const result = SignalRequestSchema.parse({
      symbol: 'BTCUSDT',
      minConfidence: 70,
      limit: 50,
      timeframe: '1h',
    });
    expect(result).toBeDefined();
  });

  it('should apply defaults', () => {
    const result = SignalRequestSchema.parse({});
    expect(result.minConfidence).toBe(70);
    expect(result.limit).toBe(50);
  });

  it('should reject invalid data', () => {
    expect(() =>
      SignalRequestSchema.parse({
        symbol: 'INVALID',
        minConfidence: 150,
      })
    ).toThrow();
  });
});

describe('ConservativeSignalRequestSchema', () => {
  it('should accept valid conservative signal request', () => {
    const result = ConservativeSignalRequestSchema.parse({
      minConfidence: 80,
      limit: 20,
      minRiskReward: 2.5,
      maxLeverage: 5,
    });
    expect(result).toBeDefined();
    expect(result.minConfidence).toBe(80);
  });

  it('should apply conservative defaults', () => {
    const result = ConservativeSignalRequestSchema.parse({});
    expect(result.minConfidence).toBe(80); // Higher default
    expect(result.minRiskReward).toBe(2.5);
    expect(result.maxLeverage).toBe(5);
  });
});

describe('BotControlRequestSchema', () => {
  it('should accept valid bot control request', () => {
    const result = BotControlRequestSchema.parse({
      botId: '550e8400-e29b-41d4-a716-446655440000',
      action: 'start',
      symbol: 'BTCUSDT',
      leverage: 3,
    });
    expect(result).toBeDefined();
  });

  it('should reject invalid UUID', () => {
    expect(() =>
      BotControlRequestSchema.parse({
        botId: 'not-a-uuid',
        action: 'start',
      })
    ).toThrow();
  });

  it('should reject invalid action', () => {
    expect(() =>
      BotControlRequestSchema.parse({
        botId: '550e8400-e29b-41d4-a716-446655440000',
        action: 'invalid',
      })
    ).toThrow();
  });
});

describe('validateRequest', () => {
  it('should return validated data on success', () => {
    const result = validateRequest(SymbolSchema, 'BTCUSDT');
    expect(result).toBe('BTCUSDT');
  });

  it('should throw ZodError on failure', () => {
    expect(() => validateRequest(SymbolSchema, 'invalid')).toThrow();
  });
});

describe('safeValidateRequest', () => {
  it('should return success result for valid data', () => {
    const result = safeValidateRequest(SymbolSchema, 'BTCUSDT');
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).toBe('BTCUSDT');
    }
  });

  it('should return error result for invalid data', () => {
    const result = safeValidateRequest(SymbolSchema, 'invalid');
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error).toBeDefined();
    }
  });
});

describe('formatZodError', () => {
  it('should format Zod errors into readable messages', () => {
    const result = safeValidateRequest(SymbolSchema, 'btc');
    if (!result.success) {
      const messages = formatZodError(result.error);
      expect(messages).toBeInstanceOf(Array);
      expect(messages.length).toBeGreaterThan(0);
      expect(messages[0]).toContain('Invalid symbol format');
    }
  });
});
