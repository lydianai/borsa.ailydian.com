import { getBotIntegrationManager } from './services/orchestrator/BotIntegrationManager';

let isInitialized = false;

export async function initializeOrchestrator() {
  if (isInitialized) {
    console.log('[Init] Orchestrator already initialized');
    return;
  }

  try {
    console.log('='.repeat(60));
    console.log('🚀 UNIFIED ROBOT ORCHESTRATOR - INITIALIZING');
    console.log('='.repeat(60));

    const manager = getBotIntegrationManager();
    await manager.initialize();

    isInitialized = true;

    console.log('='.repeat(60));
    console.log('✅ UNIFIED ROBOT ORCHESTRATOR - READY');
    console.log('='.repeat(60));
    console.log('');
    console.log('📊 Status:');
    const status = manager.getStatus();
    console.log(`   - Total Bots: ${status.totalBots}`);
    console.log(`   - Active Bots: ${status.activeBots}`);
    console.log(`   - Cache Size: ${status.cacheSize}`);
    console.log('');
    console.log('🌐 API Endpoints:');
    console.log('   - GET  /api/orchestrator/status');
    console.log('   - GET  /api/orchestrator/bots');
    console.log('   - POST /api/orchestrator/health-check');
    console.log('   - POST /api/orchestrator/signal');
    console.log('   - POST /api/orchestrator/signals/batch');
    console.log('   - POST /api/orchestrator/control');
    console.log('='.repeat(60));
  } catch (error) {
    console.error('❌ Orchestrator initialization failed:', error);
    throw error;
  }
}

if (process.env.NODE_ENV !== 'production' && typeof window === 'undefined') {
  initializeOrchestrator().catch(console.error);
}
