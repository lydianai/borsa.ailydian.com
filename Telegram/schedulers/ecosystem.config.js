module.exports = {
  apps: [{
    name: 'telegram-scheduler',
    script: './run-scheduler.js',
    instances: 1,
    exec_mode: 'fork',
    max_memory_restart: '500M',
    restart_delay: 4000,
    max_restarts: 10,
    min_uptime: '10s',
    autorestart: true,
    watch: false,
    env: {
      NODE_ENV: 'production'
    },
    error_file: './logs/telegram-scheduler-error.log',
    out_file: './logs/telegram-scheduler-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    kill_timeout: 5000,
    listen_timeout: 3000
  }]
};
