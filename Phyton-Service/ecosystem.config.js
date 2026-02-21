/**
 * PM2 ECOSYSTEM CONFIG FOR PYTHON SERVICES
 * 11 Python microservices for AI/ML trading signals
 *
 * USAGE:
 * cd Phyton-Service
 * pm2 start ecosystem.config.js
 * pm2 save
 *
 * WHITE-HAT RULES:
 * - All services use real market data (no simulation)
 * - Transparent signal generation
 * - No market manipulation
 * - Educational & research purposes
 */

module.exports = {
  apps: [
    // ========================================================================
    // PRIORITY 1: CORE SERVICES (Must run first)
    // ========================================================================

    {
      name: 'ai-models',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/ai-models',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '2000M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5003',
      },
      error_file: './ai-models/logs/error.log',
      out_file: './ai-models/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
      min_uptime: 10000,
      max_restarts: 10,
      restart_delay: 5000,
    },

    {
      name: 'signal-generator',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/signal-generator',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1000M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5004',
        AI_SERVICE_URL: 'http://localhost:5003',
        DATA_SERVICE_URL: 'http://localhost:3000',
      },
      error_file: './signal-generator/logs/error.log',
      out_file: './signal-generator/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
      min_uptime: 10000,
      max_restarts: 10,
      restart_delay: 5000,
    },

    {
      name: 'risk-management',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/risk-management',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5006',
      },
      error_file: './risk-management/logs/error.log',
      out_file: './risk-management/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // PRIORITY 2: FEATURE SERVICES
    // ========================================================================

    {
      name: 'feature-engineering',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/feature-engineering',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1000M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5001',
      },
      error_file: './feature-engineering/logs/error.log',
      out_file: './feature-engineering/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'smc-strategy',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/smc-strategy',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5007',
      },
      error_file: './smc-strategy/logs/error.log',
      out_file: './smc-strategy/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'transformer-ai',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/transformer-ai',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5008',
      },
      error_file: './transformer-ai/logs/error.log',
      out_file: './transformer-ai/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // PRIORITY 3: ADVANCED SERVICES
    // ========================================================================

    {
      name: 'online-learning',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/online-learning',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1000M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5009',
      },
      error_file: './online-learning/logs/error.log',
      out_file: './online-learning/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'multi-timeframe',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/multi-timeframe',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '800M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5010',
      },
      error_file: './multi-timeframe/logs/error.log',
      out_file: './multi-timeframe/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'order-flow',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/order-flow',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '600M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5011',
      },
      error_file: './order-flow/logs/error.log',
      out_file: './order-flow/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'continuous-monitor',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/continuous-monitor',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5012',
      },
      error_file: './continuous-monitor/logs/error.log',
      out_file: './continuous-monitor/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // OMNIPOTENT FUTURES MATRIX v5.0 SERVICES
    // ========================================================================

    {
      name: 'liquidation-heatmap',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/liquidation-heatmap',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5013',
      },
      error_file: './liquidation-heatmap/logs/error.log',
      out_file: './liquidation-heatmap/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'funding-derivatives',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/funding-derivatives',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5014',
      },
      error_file: './funding-derivatives/logs/error.log',
      out_file: './funding-derivatives/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'whale-activity',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/whale-activity',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '600M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5015',
      },
      error_file: './whale-activity/logs/error.log',
      out_file: './whale-activity/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'macro-correlation',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/macro-correlation',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5016',
      },
      error_file: './macro-correlation/logs/error.log',
      out_file: './macro-correlation/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // FAZ 2: ADVANCED ANALYSIS SERVICES
    // ========================================================================

    {
      name: 'sentiment-analysis',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/sentiment-analysis',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5017',
      },
      error_file: './sentiment-analysis/logs/error.log',
      out_file: './sentiment-analysis/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'options-flow',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/options-flow',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '600M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5018',
      },
      error_file: './options-flow/logs/error.log',
      out_file: './options-flow/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // FAZ 3: ADVANCED CONFIRMATION & OPTIMIZATION
    // ========================================================================

    {
      name: 'confirmation-engine',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/confirmation-engine',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '600M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5019',
      },
      error_file: './confirmation-engine/logs/error.log',
      out_file: './confirmation-engine/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    // ========================================================================
    // FAZ 4: INFRASTRUCTURE SERVICES (NEW)
    // ========================================================================

    {
      name: 'database-service',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/database-service',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5020',
        SERVICE_NAME: 'Database Service',
        DB_ENABLED: 'false',
        REDIS_ENABLED: 'true',
        REDIS_HOST: 'localhost',
        REDIS_PORT: '6379',
        PROMETHEUS_ENABLED: 'true',
        LOG_LEVEL: 'INFO',
      },
      error_file: './database-service/logs/error.log',
      out_file: './database-service/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },

    {
      name: 'websocket-streaming',
      script: './venv/bin/python3',
      args: 'app.py',
      cwd: __dirname + '/websocket-streaming',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '600M',
      env: {
        FLASK_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PORT: '5021',
        SERVICE_NAME: 'WebSocket Streaming Service',
        REDIS_ENABLED: 'true',
        REDIS_HOST: 'localhost',
        REDIS_PORT: '6379',
        PROMETHEUS_ENABLED: 'true',
        LOG_LEVEL: 'INFO',
      },
      error_file: './websocket-streaming/logs/error.log',
      out_file: './websocket-streaming/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      merge_logs: true,
    },
  ],
};
