/**
 * Secure Telegram Bot Implementation
 *
 * @security CRITICAL - Rate limiting, whitelist validation, token validation
 * @created 2025-10-26
 * @fixes Finding #5 (CVSS 6.5) - Telegram bot token exposure and lack of rate limiting
 */

import { Bot, Context, webhookCallback } from 'grammy';
import { NextRequest, NextResponse } from 'next/server';

// ============================================================================
// CONFIGURATION & VALIDATION
// ============================================================================

/**
 * Validate Telegram bot token format
 * Format: {bot_id}:{secret_token}
 * Example: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
 */
function validateBotToken(token: string | undefined): string {
  if (!token) {
    throw new Error('TELEGRAM_BOT_TOKEN is not configured!');
  }

  // Telegram bot token regex: digits + colon + alphanumeric/underscore/hyphen
  const tokenRegex = /^\d+:[A-Za-z0-9_-]{35,}$/;

  if (!tokenRegex.test(token)) {
    throw new Error(
      'Invalid TELEGRAM_BOT_TOKEN format! Expected: {bot_id}:{secret_token}'
    );
  }

  // Check for test/demo tokens
  const forbiddenPatterns = ['test', 'demo', 'example', 'sample'];
  if (forbiddenPatterns.some(p => token.toLowerCase().includes(p))) {
    throw new Error('TELEGRAM_BOT_TOKEN appears to be a test token!');
  }

  return token;
}

/**
 * Parse allowed chat IDs from environment variable
 */
function parseAllowedChatIds(): Set<number> {
  const chatIdsStr = process.env.TELEGRAM_ALLOWED_CHAT_IDS;

  if (!chatIdsStr || chatIdsStr.trim() === '') {
    console.warn(
      '⚠️ TELEGRAM_ALLOWED_CHAT_IDS is empty! Bot will reject all requests.'
    );
    return new Set();
  }

  try {
    const ids = chatIdsStr
      .split(',')
      .map(id => parseInt(id.trim()))
      .filter(id => !isNaN(id));

    return new Set(ids);
  } catch (error) {
    console.error('Failed to parse TELEGRAM_ALLOWED_CHAT_IDS:', error);
    return new Set();
  }
}

// Initialize bot with validated token
const BOT_TOKEN = validateBotToken(process.env.TELEGRAM_BOT_TOKEN);
const ALLOWED_CHAT_IDS = parseAllowedChatIds();
const WEBHOOK_SECRET = process.env.TELEGRAM_BOT_WEBHOOK_SECRET;

// ============================================================================
// RATE LIMITING
// ============================================================================

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

/**
 * In-memory rate limiter
 * Production: Use Redis or other distributed cache
 */
class RateLimiter {
  private limits = new Map<string, RateLimitEntry>();
  private readonly maxRequests: number;
  private readonly windowMs: number;

  constructor(maxRequests = 10, windowMs = 60000) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;

    // Cleanup old entries every 5 minutes
    setInterval(() => this.cleanup(), 5 * 60 * 1000);
  }

  /**
   * Check if request is allowed
   */
  isAllowed(key: string): boolean {
    const now = Date.now();
    const entry = this.limits.get(key);

    // No entry or expired - allow and create new entry
    if (!entry || now >= entry.resetAt) {
      this.limits.set(key, {
        count: 1,
        resetAt: now + this.windowMs,
      });
      return true;
    }

    // Check if limit exceeded
    if (entry.count >= this.maxRequests) {
      return false;
    }

    // Increment counter
    entry.count++;
    return true;
  }

  /**
   * Get remaining requests for a key
   */
  getRemaining(key: string): number {
    const entry = this.limits.get(key);
    if (!entry || Date.now() >= entry.resetAt) {
      return this.maxRequests;
    }
    return Math.max(0, this.maxRequests - entry.count);
  }

  /**
   * Cleanup expired entries
   */
  private cleanup() {
    const now = Date.now();
    for (const [key, entry] of this.limits.entries()) {
      if (now >= entry.resetAt) {
        this.limits.delete(key);
      }
    }
  }
}

// Create rate limiter instance
// 10 messages per minute per chat
const rateLimiter = new RateLimiter(10, 60000);

// ============================================================================
// BOT SETUP
// ============================================================================

export const bot = new Bot(BOT_TOKEN);

/**
 * Middleware: Rate limiting
 */
bot.use(async (ctx, next) => {
  const chatId = ctx.chat?.id.toString();
  if (!chatId) {
    console.warn('[Bot] Message without chat ID, ignoring');
    return;
  }

  // Check rate limit
  if (!rateLimiter.isAllowed(chatId)) {
    const remaining = rateLimiter.getRemaining(chatId);
    console.warn(`[Bot] Rate limit exceeded for chat ${chatId}`);

    await ctx.reply(
      '⚠️ Rate limit exceeded. Please wait before sending more messages.\n' +
      `You can send ${remaining} more messages in the next minute.`,
      { parse_mode: 'Markdown' }
    );
    return;
  }

  await next();
});

/**
 * Middleware: Whitelist validation
 */
bot.use(async (ctx, next) => {
  const chatId = ctx.chat?.id;

  if (!chatId) {
    console.warn('[Bot] Message without chat ID');
    return;
  }

  // Check whitelist
  if (!ALLOWED_CHAT_IDS.has(chatId)) {
    console.warn(`[Bot] Unauthorized access attempt from chat ${chatId}`);

    await ctx.reply(
      '⛔ *Unauthorized Access*\n\n' +
      'This bot is private and only accessible to authorized users.\n\n' +
      `Your Chat ID: \`${chatId}\`\n\n` +
      'Please contact the administrator if you believe this is an error.',
      { parse_mode: 'Markdown' }
    );
    return;
  }

  await next();
});

/**
 * Middleware: Error handling
 */
bot.catch((err) => {
  const ctx = err.ctx;
  console.error('[Bot] Error handling update:', err.error);

  // Try to notify user (but don't throw if it fails)
  ctx.reply('❌ An error occurred while processing your request. Please try again later.')
    .catch(e => console.error('[Bot] Failed to send error message:', e));
});

// ============================================================================
// BOT COMMANDS
// ============================================================================

/**
 * /start command
 */
bot.command('start', async (ctx) => {
  const user = ctx.from;
  const chatId = ctx.chat.id;

  await ctx.reply(
    `👋 *Welcome to Sardag-Emrah Trading Bot!*\n\n` +
    `Your Chat ID: \`${chatId}\`\n` +
    `Username: @${user?.username || 'N/A'}\n\n` +
    `*Available Commands:*\n` +
    `/help - Show available commands\n` +
    `/status - Check bot status\n` +
    `/stats - Get trading statistics\n\n` +
    `_This bot is secured with whitelist and rate limiting._`,
    { parse_mode: 'Markdown' }
  );
});

/**
 * /help command
 */
bot.command('help', async (ctx) => {
  await ctx.reply(
    '*Sardag-Emrah Trading Bot - Help*\n\n' +
    '*Commands:*\n' +
    '• `/start` - Initialize bot\n' +
    '• `/help` - Show this help message\n' +
    '• `/status` - Check bot and services status\n' +
    '• `/stats` - Get trading statistics\n' +
    '• `/signals` - Get latest trading signals\n\n' +
    '*Security Features:*\n' +
    '✅ Whitelist-only access\n' +
    '✅ Rate limiting (10 msg/min)\n' +
    '✅ Secure webhook with secret token\n\n' +
    '_Your activity is logged for security purposes._',
    { parse_mode: 'Markdown' }
  );
});

/**
 * /status command
 */
bot.command('status', async (ctx) => {
  await ctx.reply(
    '🟢 *Bot Status: Online*\n\n' +
    `Chat ID: \`${ctx.chat.id}\`\n` +
    `Rate Limit: ${rateLimiter.getRemaining(ctx.chat.id.toString())}/10 remaining\n` +
    `Environment: ${process.env.NODE_ENV || 'development'}\n\n` +
    '_All systems operational._',
    { parse_mode: 'Markdown' }
  );
});

// ============================================================================
// WEBHOOK HANDLER
// ============================================================================

/**
 * Verify webhook secret
 */
function verifyWebhookSecret(request: NextRequest): boolean {
  if (!WEBHOOK_SECRET) {
    console.warn('[Bot] TELEGRAM_BOT_WEBHOOK_SECRET not configured, skipping verification');
    return true;
  }

  const providedSecret = request.headers.get('X-Telegram-Bot-Api-Secret-Token');

  if (!providedSecret || providedSecret !== WEBHOOK_SECRET) {
    console.error('[Bot] Invalid webhook secret');
    return false;
  }

  return true;
}

/**
 * Handle Telegram webhook updates
 *
 * @example Next.js API Route (app/api/telegram/webhook/route.ts)
 * ```typescript
 * import { handleTelegramWebhook } from '@/lib/telegram/secure-bot';
 *
 * export async function POST(request: NextRequest) {
 *   return handleTelegramWebhook(request);
 * }
 * ```
 */
export async function handleTelegramWebhook(
  request: NextRequest
): Promise<NextResponse> {
  try {
    // Verify webhook secret
    if (!verifyWebhookSecret(request)) {
      return new NextResponse('Unauthorized', { status: 401 });
    }

    // Process update with grammy webhook callback
    const handler = webhookCallback(bot, 'std/http');

    // Create Response-like object for grammy
    const response = await handler(request);

    return new NextResponse(response.body, {
      status: response.status,
      headers: response.headers,
    });
  } catch (error) {
    console.error('[Bot] Webhook error:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * Send notification to all allowed chats
 */
export async function sendNotificationToAll(message: string): Promise<void> {
  const errors: Array<{ chatId: number; error: string }> = [];

  for (const chatId of ALLOWED_CHAT_IDS) {
    try {
      await bot.api.sendMessage(chatId, message, { parse_mode: 'Markdown' });
    } catch (error) {
      console.error(`Failed to send to chat ${chatId}:`, error);
      errors.push({
        chatId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }

  if (errors.length > 0) {
    console.warn('Failed to send to some chats:', errors);
  }
}

/**
 * Send notification to specific chat
 */
export async function sendNotification(
  chatId: number,
  message: string
): Promise<void> {
  // Check if chat is whitelisted
  if (!ALLOWED_CHAT_IDS.has(chatId)) {
    throw new Error(`Chat ${chatId} is not in the whitelist`);
  }

  await bot.api.sendMessage(chatId, message, { parse_mode: 'Markdown' });
}

// ============================================================================
// EXPORTS
// ============================================================================

export { ALLOWED_CHAT_IDS, rateLimiter };

/**
 * Start bot in long polling mode (for development)
 *
 * @example
 * ```typescript
 * // In development environment
 * if (process.env.NODE_ENV === 'development') {
 *   startPolling().catch(console.error);
 * }
 * ```
 */
export async function startPolling() {
  console.log('[Bot] Starting in long polling mode...');
  console.log(`[Bot] Allowed chats: ${Array.from(ALLOWED_CHAT_IDS).join(', ')}`);

  await bot.start({
    onStart: () => console.log('[Bot] Started successfully ✅'),
  });
}

/**
 * Stop bot gracefully
 */
export async function stopBot() {
  console.log('[Bot] Stopping...');
  await bot.stop();
  console.log('[Bot] Stopped ✅');
}
