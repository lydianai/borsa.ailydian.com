/**
 * 🤖 TELEGRAM BOT INSTANCE
 * Grammy framework ile Telegram bot yönetimi
 *
 * Features:
 * - Grammy bot instance (singleton)
 * - Webhook callback handler
 * - Type-safe
 * - White-hat compliant
 *
 * Usage:
 * import { bot } from '@/lib/telegram/bot';
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational purposes only
 * - No trading operations
 * - Public data only
 * - User privacy protected
 */

import { Bot, webhookCallback } from 'grammy';

// ============================================================================
// TYPES
// ============================================================================

export interface BotConfig {
  token: string;
  webhookSecret?: string;
}

// ============================================================================
// SINGLETON BOT INSTANCE
// ============================================================================

let botInstance: Bot | null = null;

/**
 * Get or create bot instance
 */
export function getBot(): Bot {
  if (botInstance) {
    return botInstance;
  }

  const token = process.env.TELEGRAM_BOT_TOKEN;

  if (!token) {
    // During build time, token might not be available
    // Allow build without token (will fail at runtime if actually used)
    // This is safe because the webhook endpoint won't be called during build
    botInstance = new Bot('0:DUMMY_TOKEN_FOR_BUILD_TIME_ONLY');
    return botInstance;
  }

  botInstance = new Bot(token);

  return botInstance;
}

/**
 * Export bot instance (lazy initialization)
 */
export const bot = new Proxy({} as Bot, {
  get(target, prop) {
    const instance = getBot();
    return (instance as any)[prop];
  },
});

// ============================================================================
// WEBHOOK CALLBACK
// ============================================================================

/**
 * Create webhook callback for serverless functions
 * Compatible with Next.js API routes and Vercel
 *
 * Note: We handle updates manually for better Next.js compatibility
 */
export async function handleWebhookUpdate(update: any) {
  const instance = getBot();
  await instance.handleUpdate(update);
}

export default bot;
