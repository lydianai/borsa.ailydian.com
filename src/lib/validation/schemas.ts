/**
 * Input Validation Schemas using Zod
 *
 * White-hat security: Validates all user inputs to prevent injection attacks,
 * data corruption, and ensure data integrity
 *
 * OWASP: https://owasp.org/www-project-proactive-controls/v3/en/c5-validate-inputs
 */

import { z } from 'zod';

// ============================================
// COMMON SCHEMAS
// ============================================

/**
 * Trading symbol validation
 * Only allows valid Binance Futures USDT perpetual symbols
 */
export const SymbolSchema = z
  .string()
  .regex(/^[A-Z0-9]{3,10}USDT$/, 'Invalid symbol format. Must be uppercase and end with USDT')
  .min(6, 'Symbol too short')
  .max(14, 'Symbol too long');

/**
 * Timeframe validation
 */
export const TimeframeSchema = z.enum([
  '1m', '3m', '5m', '15m', '30m',
  '1h', '2h', '4h', '6h', '8h', '12h',
  '1d', '3d', '1w', '1M'
]);

/**
 * Confidence percentage (0-100)
 */
export const ConfidenceSchema = z
  .number()
  .min(0, 'Confidence cannot be negative')
  .max(100, 'Confidence cannot exceed 100');

/**
 * Limit for pagination (1-200)
 */
export const LimitSchema = z
  .number()
  .int('Limit must be an integer')
  .min(1, 'Limit must be at least 1')
  .max(200, 'Limit cannot exceed 200');

/**
 * Price validation (positive number with max 8 decimals)
 */
export const PriceSchema = z
  .number()
  .positive('Price must be positive')
  .finite('Price must be a finite number')
  .refine(
    (val) => {
      const decimals = val.toString().split('.')[1]?.length || 0;
      return decimals <= 8;
    },
    { message: 'Price cannot have more than 8 decimal places' }
  );

/**
 * Leverage validation (1-125x for Binance)
 */
export const LeverageSchema = z
  .number()
  .int('Leverage must be an integer')
  .min(1, 'Leverage must be at least 1x')
  .max(125, 'Leverage cannot exceed 125x');

// ============================================
// SIGNAL API SCHEMAS
// ============================================

/**
 * GET /api/signals query parameters
 */
export const SignalRequestSchema = z.object({
  symbol: SymbolSchema.optional(),
  minConfidence: ConfidenceSchema.optional().default(70),
  limit: LimitSchema.optional().default(50),
  timeframe: TimeframeSchema.optional(),
  strategy: z.string().max(50).optional(),
});

export type SignalRequest = z.infer<typeof SignalRequestSchema>;

/**
 * GET /api/conservative-signals query parameters
 */
export const ConservativeSignalRequestSchema = z.object({
  minConfidence: ConfidenceSchema.optional().default(80),
  limit: LimitSchema.optional().default(50),
  minRiskReward: z.number().min(1).max(10).optional().default(2.5),
  maxLeverage: LeverageSchema.optional().default(5),
});

export type ConservativeSignalRequest = z.infer<typeof ConservativeSignalRequestSchema>;

/**
 * GET /api/quantum-signals query parameters
 */
export const QuantumSignalRequestSchema = z.object({
  minConfidence: ConfidenceSchema.optional().default(60),
  limit: LimitSchema.optional().default(50),
  enableEnsemble: z.boolean().optional().default(true),
});

export type QuantumSignalRequest = z.infer<typeof QuantumSignalRequestSchema>;

// ============================================
// BOT MANAGEMENT SCHEMAS
// ============================================

/**
 * Bot ID validation (UUID v4)
 */
export const BotIdSchema = z
  .string()
  .regex(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    'Invalid bot ID format (must be UUID v4)'
  );

/**
 * Bot action validation
 */
export const BotActionSchema = z.enum(['start', 'stop', 'pause', 'resume']);

/**
 * POST /api/bot - Start/stop bot
 */
export const BotControlRequestSchema = z.object({
  botId: BotIdSchema,
  action: BotActionSchema,
  symbol: SymbolSchema.optional(),
  leverage: LeverageSchema.optional(),
  stopLoss: z.number().min(0).max(100).optional(), // Percentage
  takeProfit: z.number().min(0).max(1000).optional(), // Percentage
});

export type BotControlRequest = z.infer<typeof BotControlRequestSchema>;

/**
 * POST /api/bot/create - Create new bot
 */
export const CreateBotRequestSchema = z.object({
  name: z.string().min(1).max(100),
  strategy: z.string().min(1).max(50),
  symbol: SymbolSchema,
  leverage: LeverageSchema.default(1),
  stopLoss: z.number().min(0.1).max(50), // Percentage
  takeProfit: z.number().min(0.1).max(1000), // Percentage
  maxPositionSize: z.number().positive().max(10000), // USD
  enabled: z.boolean().default(false),
});

export type CreateBotRequest = z.infer<typeof CreateBotRequestSchema>;

// ============================================
// AUTO TRADING SCHEMAS
// ============================================

/**
 * POST /api/auto-trading/execute
 */
export const AutoTradingExecuteSchema = z.object({
  symbol: SymbolSchema,
  side: z.enum(['BUY', 'SELL']),
  quantity: z.number().positive().max(1000000),
  price: PriceSchema.optional(), // Market order if not provided
  leverage: LeverageSchema.optional().default(1),
  stopLoss: PriceSchema.optional(),
  takeProfit: PriceSchema.optional(),
  reduceOnly: z.boolean().optional().default(false),
});

export type AutoTradingExecute = z.infer<typeof AutoTradingExecuteSchema>;

// ============================================
// SETTINGS SCHEMAS
// ============================================

/**
 * PUT /api/settings/notifications
 */
export const NotificationSettingsSchema = z.object({
  enabled: z.boolean(),
  channels: z.object({
    push: z.boolean(),
    telegram: z.boolean(),
    email: z.boolean().optional(),
  }),
  filters: z.object({
    minConfidence: ConfidenceSchema.default(70),
    strategies: z.array(z.string().max(50)).max(20).optional(),
    symbols: z.array(SymbolSchema).max(50).optional(),
  }).optional(),
});

export type NotificationSettings = z.infer<typeof NotificationSettingsSchema>;

/**
 * PUT /api/settings/profile
 */
export const ProfileSettingsSchema = z.object({
  username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, hyphens and underscores'),
  email: z.string().email('Invalid email address').max(255).optional(),
  timezone: z.string().max(50).optional(),
  language: z.enum(['tr', 'en']).default('tr'),
});

export type ProfileSettings = z.infer<typeof ProfileSettingsSchema>;

// ============================================
// TELEGRAM SCHEMAS
// ============================================

/**
 * POST /api/telegram/send
 */
export const TelegramSendSchema = z.object({
  chatId: z.string().regex(/^-?\d+$/, 'Invalid Telegram chat ID'),
  message: z.string().min(1).max(4096), // Telegram message limit
  parseMode: z.enum(['Markdown', 'HTML', 'MarkdownV2']).optional(),
  disableNotification: z.boolean().optional().default(false),
});

export type TelegramSend = z.infer<typeof TelegramSendSchema>;

// ============================================
// PUSH NOTIFICATION SCHEMAS
// ============================================

/**
 * POST /api/push/subscribe
 */
export const PushSubscribeSchema = z.object({
  endpoint: z.string().url('Invalid push notification endpoint'),
  keys: z.object({
    p256dh: z.string().min(1),
    auth: z.string().min(1),
  }),
});

export type PushSubscribe = z.infer<typeof PushSubscribeSchema>;

/**
 * POST /api/push/send
 */
export const PushSendSchema = z.object({
  title: z.string().min(1).max(100),
  body: z.string().min(1).max(500),
  icon: z.string().url().optional(),
  badge: z.string().url().optional(),
  data: z.record(z.any()).optional(),
  tag: z.string().max(50).optional(),
});

export type PushSend = z.infer<typeof PushSendSchema>;

// ============================================
// MARKET DATA SCHEMAS
// ============================================

/**
 * GET /api/binance/klines
 */
export const KlinesRequestSchema = z.object({
  symbol: SymbolSchema,
  interval: TimeframeSchema,
  limit: z.number().int().min(1).max(1500).optional().default(500),
  startTime: z.number().int().positive().optional(),
  endTime: z.number().int().positive().optional(),
}).refine(
  (data) => {
    if (data.startTime && data.endTime) {
      return data.endTime > data.startTime;
    }
    return true;
  },
  { message: 'endTime must be greater than startTime' }
);

export type KlinesRequest = z.infer<typeof KlinesRequestSchema>;

// ============================================
// WATCHLIST SCHEMAS
// ============================================

/**
 * POST /api/watchlist/create
 */
export const CreateWatchlistSchema = z.object({
  name: z.string().min(1).max(100),
  symbols: z.array(SymbolSchema).min(1).max(50),
  description: z.string().max(500).optional(),
});

export type CreateWatchlist = z.infer<typeof CreateWatchlistSchema>;

/**
 * PUT /api/watchlist/[id]
 */
export const UpdateWatchlistSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  symbols: z.array(SymbolSchema).min(1).max(50).optional(),
  description: z.string().max(500).optional(),
});

export type UpdateWatchlist = z.infer<typeof UpdateWatchlistSchema>;

// ============================================
// ALERT SCHEMAS
// ============================================

/**
 * POST /api/alerts/create
 */
export const CreateAlertSchema = z.object({
  symbol: SymbolSchema,
  type: z.enum(['PRICE', 'RSI', 'VOLUME', 'CUSTOM']),
  condition: z.object({
    operator: z.enum(['>', '<', '>=', '<=', '==']),
    value: z.number(),
    indicator: z.string().max(50).optional(),
  }),
  message: z.string().min(1).max(500).optional(),
  enabled: z.boolean().default(true),
});

export type CreateAlert = z.infer<typeof CreateAlertSchema>;

// ============================================
// EXPORT HELPERS
// ============================================

/**
 * Validate request data with a schema
 * Throws ZodError with detailed error messages if validation fails
 */
export function validateRequest<T>(schema: z.ZodSchema<T>, data: unknown): T {
  return schema.parse(data);
}

/**
 * Safely validate request data
 * Returns { success: true, data } or { success: false, error }
 */
export function safeValidateRequest<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; error: z.ZodError } {
  const result = schema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: result.error };
}

/**
 * Format Zod errors for user-friendly messages
 */
export function formatZodError(error: z.ZodError): string[] {
  return error.errors.map((err) => {
    const path = err.path.join('.');
    return `${path ? `${path}: ` : ''}${err.message}`;
  });
}
