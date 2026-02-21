/**
 * Backend Bot Connector
 *
 * White-hat compliance: Connects to backend trading bot services
 * This is a re-export for backward compatibility
 */

import botConnectorService, {
  BotConnectorService,
  type BotConfig,
  type BotMetrics,
} from '@/lib/bot-connector';

// Re-export everything from the main bot-connector
export default botConnectorService;
export { BotConnectorService };
export type { BotConfig, BotMetrics };
