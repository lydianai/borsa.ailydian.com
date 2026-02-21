/**
 * Bot Connector Service Tests
 *
 * White-hat compliance: Ensures bot connector works correctly
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { BotConnectorService, type BotConfig } from '../bot-connector';

describe('BotConnectorService', () => {
  let service: BotConnectorService;

  beforeEach(() => {
    service = new BotConnectorService();
  });

  describe('initializeBot', () => {
    it('should initialize a bot successfully', async () => {
      const config: BotConfig = {
        botId: 'test-bot-1',
        name: 'Test Bot',
        status: 'active',
        exchange: 'binance',
        strategy: 'momentum',
      };

      const result = await service.initializeBot(config);

      expect(result.success).toBe(true);
      expect(result.botId).toBe('test-bot-1');
    });

    it('should throw error for invalid config', async () => {
      const invalidConfig = {
        botId: '',
        name: '',
      } as BotConfig;

      await expect(service.initializeBot(invalidConfig)).rejects.toThrow('Invalid bot configuration');
    });

    it('should create metrics for initialized bot', async () => {
      const config: BotConfig = {
        botId: 'test-bot-2',
        name: 'Test Bot 2',
        status: 'active',
        exchange: 'binance',
        strategy: 'arbitrage',
      };

      await service.initializeBot(config);
      const metrics = service.getBotMetrics('test-bot-2');

      expect(metrics).not.toBeNull();
      expect(metrics?.totalTrades).toBe(0);
      expect(metrics?.winRate).toBe(0);
    });
  });

  describe('getBotStatus', () => {
    it('should return bot status', async () => {
      const config: BotConfig = {
        botId: 'status-test',
        name: 'Status Test Bot',
        status: 'active',
        exchange: 'binance',
        strategy: 'grid',
      };

      await service.initializeBot(config);
      const status = service.getBotStatus('status-test');

      expect(status).not.toBeNull();
      expect(status?.name).toBe('Status Test Bot');
      expect(status?.status).toBe('active');
    });

    it('should return null for non-existent bot', () => {
      const status = service.getBotStatus('non-existent');
      expect(status).toBeNull();
    });
  });

  describe('updateBotStatus', () => {
    it('should update bot status successfully', async () => {
      const config: BotConfig = {
        botId: 'update-test',
        name: 'Update Test Bot',
        status: 'active',
        exchange: 'binance',
        strategy: 'dca',
      };

      await service.initializeBot(config);
      const updated = service.updateBotStatus('update-test', 'paused');

      expect(updated).toBe(true);

      const status = service.getBotStatus('update-test');
      expect(status?.status).toBe('paused');
    });

    it('should return false for non-existent bot', () => {
      const updated = service.updateBotStatus('non-existent', 'stopped');
      expect(updated).toBe(false);
    });
  });

  describe('getActiveBots', () => {
    it('should return only active bots', async () => {
      await service.initializeBot({
        botId: 'active-1',
        name: 'Active Bot 1',
        status: 'active',
        exchange: 'binance',
        strategy: 'momentum',
      });

      await service.initializeBot({
        botId: 'paused-1',
        name: 'Paused Bot 1',
        status: 'paused',
        exchange: 'binance',
        strategy: 'grid',
      });

      await service.initializeBot({
        botId: 'active-2',
        name: 'Active Bot 2',
        status: 'active',
        exchange: 'binance',
        strategy: 'arbitrage',
      });

      const activeBots = service.getActiveBots();

      expect(activeBots).toHaveLength(2);
      expect(activeBots.every(bot => bot.status === 'active')).toBe(true);
    });
  });

  describe('disconnectBot', () => {
    it('should disconnect bot and remove data', async () => {
      await service.initializeBot({
        botId: 'disconnect-test',
        name: 'Disconnect Test Bot',
        status: 'active',
        exchange: 'binance',
        strategy: 'momentum',
      });

      const disconnected = service.disconnectBot('disconnect-test');
      expect(disconnected).toBe(true);

      const status = service.getBotStatus('disconnect-test');
      expect(status).toBeNull();

      const metrics = service.getBotMetrics('disconnect-test');
      expect(metrics).toBeNull();
    });

    it('should return false for non-existent bot', () => {
      const disconnected = service.disconnectBot('non-existent');
      expect(disconnected).toBe(false);
    });
  });
});
