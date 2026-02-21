/**
 * 🔐 CODE PROTECTION & OBFUSCATION CONFIG
 * NIRVANA LEVEL SECURITY - WHITE HAT COMPLIANT
 * borsa.ailydian.com - SARDAG PROTECTED
 */

module.exports = {
  // JavaScript Obfuscation Settings
  obfuscation: {
    compact: true,
    controlFlowFlattening: true,
    controlFlowFlatteningThreshold: 1,
    deadCodeInjection: true,
    deadCodeInjectionThreshold: 1,
    debugProtection: true,
    debugProtectionInterval: 4000,
    disableConsoleOutput: true,
    identifierNamesGenerator: 'hexadecimal',
    log: false,
    numbersToExpressions: true,
    renameGlobals: true,
    selfDefending: true,
    simplify: true,
    splitStrings: true,
    splitStringsChunkLength: 5,
    stringArray: true,
    stringArrayCallsTransform: true,
    stringArrayEncoding: ['rc4'],
    stringArrayIndexShift: true,
    stringArrayRotate: true,
    stringArrayShuffle: true,
    stringArrayWrappersCount: 5,
    stringArrayWrappersChainedCalls: true,
    stringArrayWrappersParametersMaxCount: 5,
    stringArrayWrappersType: 'function',
    stringArrayThreshold: 1,
    transformObjectKeys: true,
    unicodeEscapeSequence: true
  },

  // Copyright & License Protection
  copyright: {
    banner: `
/**
 * 🔒 PROTECTED SOURCE CODE
 * © 2025 SARDAG - borsa.ailydian.com
 * ALL RIGHTS RESERVED - PROPRIETARY & CONFIDENTIAL
 *
 * This code is protected by international copyright law.
 * Unauthorized copying, modification, distribution, or use
 * is strictly prohibited and will be prosecuted.
 *
 * Technology Stack: CLASSIFIED
 * Algorithms: ENCRYPTED & PROTECTED
 * AI Models: PROPRIETARY
 *
 * Licensed to: SARDAG
 * White-Hat Security Certified
 * Penetration Tested & Hardened
 */
`,
    removeComments: true,
    removeConsole: true,
    removeDebugger: true
  },

  // Technology Stack Hiding
  fingerprint: {
    hideNextJs: true,
    hideReact: true,
    hideVercel: true,
    hideFrameworks: true,
    customHeaders: {
      'X-Powered-By': 'CLASSIFIED',
      'X-Framework': 'PROTECTED',
      'Server': 'SARDAG-SECURE'
    }
  },

  // Domain & Author Protection
  domainProtection: {
    allowedDomains: ['borsa.ailydian.com', 'localhost'],
    blockInvalidDomains: true,
    antiDebug: true,
    antiTamper: true
  },

  // File Protection Rules
  protect: {
    // AI Models - Maximum Protection
    aiModels: {
      level: 'maximum',
      encrypt: true,
      obfuscate: true,
      patterns: [
        'src/services/ai/**/*.ts',
        'src/services/AutoTradingEngine.ts',
        'python-services/**/*.py'
      ]
    },

    // Trading Algorithms - Maximum Protection
    algorithms: {
      level: 'maximum',
      encrypt: true,
      obfuscate: true,
      patterns: [
        'src/services/MarketDataService.ts',
        'src/services/AIBotSignalService.ts',
        'src/lib/**/*.ts'
      ]
    },

    // API Routes - High Protection
    api: {
      level: 'high',
      obfuscate: true,
      patterns: [
        'src/app/api/**/*.ts'
      ]
    },

    // Components - Medium Protection
    components: {
      level: 'medium',
      obfuscate: true,
      patterns: [
        'src/components/**/*.tsx'
      ]
    },

    // Public - No Protection
    public: {
      level: 'none',
      patterns: [
        'public/**/*'
      ]
    }
  },

  // Runtime Protection
  runtime: {
    antiDebugger: true,
    antiDevTools: true,
    domainLock: true,
    sourceMapDisable: true,
    consoleDisable: true
  }
};
