/**
 * 🛡️ SOC ROOM - 24/7 SECURITY MONITORING
 * Security Operations Center Configuration
 * borsa.ailydian.com - SARDAG PROTECTED
 */

module.exports = {
  // Real-time Threat Detection
  monitoring: {
    enabled: true,
    mode: '24/7',
    alertLevel: 'high',

    // Attack Detection Rules
    rules: {
      // SQL Injection Detection
      sqlInjection: {
        enabled: true,
        patterns: [
          /(\%27)|(\')|(\-\-)|(\%23)|(#)/i,
          /((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))/i,
          /\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))/i,
          /((\%27)|(\'))union/i
        ],
        action: 'block',
        alertLevel: 'critical'
      },

      // XSS Detection
      xss: {
        enabled: true,
        patterns: [
          /<script[^>]*>[\s\S]*?<\/script>/gi,
          /javascript:/gi,
          /on\w+\s*=\s*["'][^"']*["']/gi,
          /<iframe[^>]*>/gi
        ],
        action: 'block',
        alertLevel: 'critical'
      },

      // Path Traversal Detection
      pathTraversal: {
        enabled: true,
        patterns: [
          /\.\.\//g,
          /\.\.%2[fF]/g,
          /%2e%2e/gi
        ],
        action: 'block',
        alertLevel: 'high'
      },

      // Command Injection Detection
      commandInjection: {
        enabled: true,
        patterns: [
          /[;&|`$]/,
          /\$\{.*\}/,
          /`.*`/
        ],
        action: 'block',
        alertLevel: 'critical'
      },

      // Brute Force Detection
      bruteForce: {
        enabled: true,
        maxAttempts: 5,
        timeWindow: 300, // 5 minutes
        action: 'ban',
        banDuration: 3600, // 1 hour
        alertLevel: 'high'
      },

      // DDoS Detection
      ddos: {
        enabled: true,
        maxRequestsPerMinute: 100,
        maxRequestsPerIP: 60,
        action: 'rate-limit',
        alertLevel: 'critical'
      },

      // Suspicious User-Agent Detection
      userAgent: {
        enabled: true,
        blocklist: [
          'sqlmap',
          'nikto',
          'nmap',
          'masscan',
          'metasploit',
          'burp',
          'havij',
          'acunetix'
        ],
        action: 'block',
        alertLevel: 'high'
      },

      // Port Scanning Detection
      portScan: {
        enabled: true,
        maxPortsPerMinute: 10,
        action: 'block',
        alertLevel: 'high'
      }
    }
  },

  // Firewall Configuration
  firewall: {
    enabled: true,
    mode: 'auto',

    // IP Whitelist (allowed IPs)
    whitelist: [
      '127.0.0.1',
      '::1'
    ],

    // IP Blacklist (blocked IPs)
    blacklist: [],

    // Auto-ban rules
    autoBan: {
      enabled: true,
      conditions: {
        sqlInjection: { threshold: 1, duration: 86400 }, // 24 hours
        xss: { threshold: 1, duration: 86400 },
        bruteForce: { threshold: 5, duration: 3600 },
        ddos: { threshold: 10, duration: 7200 }
      }
    },

    // Geo-blocking (optional)
    geoBlock: {
      enabled: false,
      blockedCountries: []
    }
  },

  // Logging & Alerting
  logging: {
    enabled: true,
    level: 'info', // debug, info, warn, error, critical

    // Log destinations
    destinations: {
      file: {
        enabled: true,
        path: 'logs/security/',
        rotation: 'daily',
        retention: 30 // days
      },
      console: {
        enabled: true,
        colors: true
      },
      database: {
        enabled: true,
        table: 'security_events'
      }
    },

    // Event types to log
    events: [
      'attack_detected',
      'attack_blocked',
      'ip_banned',
      'suspicious_activity',
      'login_attempt',
      'rate_limit_exceeded',
      'unauthorized_access'
    ]
  },

  // Alert Notifications
  alerts: {
    enabled: true,

    // Alert channels
    channels: {
      email: {
        enabled: false, // Set to true with valid config
        recipients: []
      },
      webhook: {
        enabled: false, // Set to true with valid URL
        url: null
      },
      console: {
        enabled: true,
        colors: true
      }
    },

    // Alert thresholds
    thresholds: {
      critical: 1,  // Alert immediately
      high: 3,      // Alert after 3 events
      medium: 10,   // Alert after 10 events
      low: 50       // Alert after 50 events
    }
  },

  // Penetration Testing
  pentesting: {
    enabled: true,

    // Self-testing schedule
    schedule: {
      daily: true,
      weekly: true,
      monthly: true
    },

    // Tests to run
    tests: [
      'sql-injection',
      'xss',
      'csrf',
      'path-traversal',
      'authentication',
      'authorization',
      'rate-limiting',
      'header-security',
      'ssl-tls'
    ],

    // Report generation
    reporting: {
      enabled: true,
      path: 'security/reports/',
      format: 'json'
    }
  },

  // Intrusion Detection System (IDS)
  ids: {
    enabled: true,
    mode: 'active', // active or passive

    // Signature-based detection
    signatures: [
      {
        name: 'SQL Injection Attempt',
        pattern: /union.*select/i,
        severity: 'critical'
      },
      {
        name: 'Directory Traversal',
        pattern: /\.\.\/.*\/etc\/passwd/i,
        severity: 'critical'
      },
      {
        name: 'XSS Attack',
        pattern: /<script>.*alert/i,
        severity: 'high'
      }
    ],

    // Anomaly-based detection
    anomaly: {
      enabled: true,
      baseline: 'auto', // Learn normal behavior
      threshold: 0.8 // 80% similarity to normal
    }
  },

  // Security Headers
  headers: {
    enabled: true,
    headers: {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
      'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
      'X-Powered-By': 'CLASSIFIED' // Hide technology stack
    }
  },

  // White-Hat Compliance
  compliance: {
    whiteHat: true,
    defensiveOnly: true,
    noCredentialHarvesting: true,
    paperTradingDefault: true,
    riskManagementEnforced: true,
    ethicalGuidelines: [
      'Protect user data',
      'Block malicious activity',
      'Log security events',
      'Alert on threats',
      'Never offensive actions',
      'Defensive security only'
    ]
  }
};
