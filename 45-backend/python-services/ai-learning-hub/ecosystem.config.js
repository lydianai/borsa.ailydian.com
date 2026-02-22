// PM2 Ecosystem Configuration for AI/ML Learning Hub
// Tüm Binance Futures USDT-M coins için sürekli öğrenme sistemi

module.exports = {
  apps: [
    // ========================================
    // MAIN ORCHESTRATOR
    // ========================================
    {
      name: 'ai-learning-orchestrator',
      script: 'orchestrator.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '2000M',
      env: {
        NODE_ENV: 'production',
        AI_LEARNING_MODE: 'continuous',
        LOG_LEVEL: 'INFO',
        BINANCE_FUTURES_MARKET: 'USDT-M',
        TOTAL_SYMBOLS: '538'
      },
      cron_restart: '0 4 * * *',  // Her gün 04:00'de restart (bakım)
      error_file: './logs/orchestrator-error.log',
      out_file: './logs/orchestrator-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    },

    // ========================================
    // AI/ML WORKERS (10 AI Systems)
    // ========================================

    // 1. Reinforcement Learning Agent Worker
    {
      name: 'rl-agent-worker',
      script: 'workers/rl_agent_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'rl_agent',
        TRAINING_INTERVAL: '300',  // 5 dakika
        CHECKPOINT_INTERVAL: '3600'  // 1 saat
      },
      error_file: './logs/rl-agent-error.log',
      out_file: './logs/rl-agent-out.log'
    },

    // 2. Online Learning Worker
    {
      name: 'online-learning-worker',
      script: 'workers/online_learning_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'online_learning',
        DRIFT_CHECK_INTERVAL: '1800',  // 30 dakika
        MODEL_UPDATE_INTERVAL: '600'  // 10 dakika
      },
      error_file: './logs/online-learning-error.log',
      out_file: './logs/online-learning-out.log'
    },

    // 3. Multi-Agent System Worker
    {
      name: 'multi-agent-worker',
      script: 'workers/multi_agent_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '2000M',
      env: {
        WORKER_TYPE: 'multi_agent',
        NUM_AGENTS: '5',
        ENSEMBLE_INTERVAL: '300'
      },
      error_file: './logs/multi-agent-error.log',
      out_file: './logs/multi-agent-out.log'
    },

    // 4. AutoML Optimizer Worker (Her 6 saatte bir çalışır)
    {
      name: 'automl-optimizer-worker',
      script: 'workers/automl_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '3000M',
      env: {
        WORKER_TYPE: 'automl',
        OPTIMIZATION_INTERVAL: '21600',  // 6 saat
        MAX_TRIALS: '100'
      },
      cron_restart: '0 */6 * * *',  // Her 6 saatte bir restart
      error_file: './logs/automl-error.log',
      out_file: './logs/automl-out.log'
    },

    // 5. Neural Architecture Search Worker (Günlük)
    {
      name: 'nas-worker',
      script: 'workers/nas_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '2500M',
      env: {
        WORKER_TYPE: 'nas',
        EVOLUTION_INTERVAL: '86400',  // 24 saat
        MAX_GENERATIONS: '50'
      },
      cron_restart: '0 2 * * *',  // Her gün 02:00'de restart
      error_file: './logs/nas-error.log',
      out_file: './logs/nas-out.log'
    },

    // 6. Meta-Learning Worker
    {
      name: 'meta-learning-worker',
      script: 'workers/meta_learning_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'meta_learning',
        ADAPTATION_INTERVAL: '3600',  // 1 saat
        FEW_SHOT_SAMPLES: '10'
      },
      error_file: './logs/meta-learning-error.log',
      out_file: './logs/meta-learning-out.log'
    },

    // 7. Federated Learning Worker
    {
      name: 'federated-learning-worker',
      script: 'workers/federated_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '2000M',
      env: {
        WORKER_TYPE: 'federated',
        ROUND_INTERVAL: '7200',  // 2 saat
        PRIVACY_BUDGET: '1.0'
      },
      error_file: './logs/federated-error.log',
      out_file: './logs/federated-out.log'
    },

    // 8. Causal AI Worker
    {
      name: 'causal-ai-worker',
      script: 'workers/causal_ai_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'causal_ai',
        GRAPH_UPDATE_INTERVAL: '3600'  // 1 saat
      },
      error_file: './logs/causal-ai-error.log',
      out_file: './logs/causal-ai-out.log'
    },

    // 9. Regime Detection Worker
    {
      name: 'regime-detection-worker',
      script: 'workers/regime_detection_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'regime_detection',
        DETECTION_INTERVAL: '300',  // 5 dakika
        REGIMES: 'Bull,Bear,Range,Volatile'
      },
      error_file: './logs/regime-detection-error.log',
      out_file: './logs/regime-detection-out.log'
    },

    // 10. Explainable AI Worker
    {
      name: 'explainable-ai-worker',
      script: 'workers/explainable_ai_worker.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1500M',
      env: {
        WORKER_TYPE: 'explainable_ai',
        EXPLANATION_INTERVAL: '600'  // 10 dakika
      },
      error_file: './logs/explainable-ai-error.log',
      out_file: './logs/explainable-ai-out.log'
    },

    // ========================================
    // MAIN API SERVER (Flask)
    // ========================================
    {
      name: 'ai-learning-api',
      script: 'app.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1000M',
      env: {
        FLASK_ENV: 'production',
        PORT: '5020',
        HOST: '0.0.0.0'
      },
      error_file: './logs/api-error.log',
      out_file: './logs/api-out.log'
    },

    // ========================================
    // DATA COLLECTOR (Binance Futures Data)
    // ========================================
    {
      name: 'data-collector',
      script: 'services/data_collector.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1000M',
      env: {
        COLLECTOR_TYPE: 'binance_futures',
        COLLECTION_INTERVAL: '60',  // 1 dakika
        SYMBOLS_PER_BATCH: '50',  // 50 coin aynı anda
        TOTAL_SYMBOLS: '538'
      },
      error_file: './logs/data-collector-error.log',
      out_file: './logs/data-collector-out.log'
    },

    // ========================================
    // SERVICE INTEGRATOR (Tüm Python servisleriyle entegrasyon)
    // ========================================
    {
      name: 'service-integrator',
      script: 'services/service_integrator.py',
      cwd: '/Users/lydian/Documents/ailydian-signal/45-backend/python-services/ai-learning-hub',
      interpreter: './venv/bin/python3',
      instances: 1,
      autorestart: true,
      max_memory_restart: '1000M',
      env: {
        TA_LIB_PORT: '5001',
        SIGNAL_GEN_PORT: '5002',
        RISK_MGMT_PORT: '5003',
        FEATURE_ENG_PORT: '5004',
        SMC_STRATEGY_PORT: '5005',
        INTEGRATION_INTERVAL: '300'  // 5 dakika
      },
      error_file: './logs/service-integrator-error.log',
      out_file: './logs/service-integrator-out.log'
    }
  ]
};
