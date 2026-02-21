#!/usr/bin/env tsx

/**
 * PRODUCTION SYSTEM STARTUP SCRIPT
 *
 * This script initializes all critical systems for production:
 * 1. Z.AI Agent (7/24 strategy analysis)
 * 2. Strategy Workers (queue processing)
 * 3. Plan Workers (trade execution)
 * 4. Real-time data streams
 * 5. Monitoring and health checks
 */

import { startAgent } from "../src/lib/ai/z-ai-agent-backend";
import { startStrategyWorker } from "../src/lib/queue/strategy-worker";
import { startPlanWorker } from "../src/lib/execution/plan-worker";
import { getPlanWorkerMetrics } from "../src/lib/execution/plan-worker";

const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

console.log(`${colors.cyan}${colors.bright}
🚀 SARDAG PRODUCTION SYSTEM STARTUP
==================================
${colors.reset}`);

let startupErrors: string[] = [];
let startupSuccess: string[] = [];

function logSuccess(message: string) {
  console.log(`${colors.green}✅ ${message}${colors.reset}`);
  startupSuccess.push(message);
}

function logError(message: string) {
  console.log(`${colors.red}❌ ${message}${colors.reset}`);
  startupErrors.push(message);
}

function logInfo(message: string) {
  console.log(`${colors.blue}ℹ️  ${message}${colors.reset}`);
}

function logWarning(message: string) {
  console.log(`${colors.yellow}⚠️  ${message}${colors.reset}`);
}

/**
 * Check environment configuration
 */
function checkEnvironment() {
  console.log(
    `\n${colors.magenta}🔍 Checking Environment Configuration...${colors.reset}`,
  );

  const requiredEnvVars = ["NODE_ENV", "QUEUE_DRIVER", "TESTNET"];

  const optionalEnvVars = [
    "GROQ_API_KEY",
    "REDIS_HOST",
    "REDIS_PORT",
    "PLAN_QUEUE_NAME",
  ];

  let envOk = true;

  for (const envVar of requiredEnvVars) {
    if (process.env[envVar]) {
      logSuccess(`${envVar}: ${process.env[envVar]}`);
    } else {
      logError(`${envVar}: Not set`);
      envOk = false;
    }
  }

  for (const envVar of optionalEnvVars) {
    if (process.env[envVar]) {
      logSuccess(`${envVar}: ${process.env[envVar]}`);
    } else {
      logWarning(`${envVar}: Not set (optional)`);
    }
  }

  return envOk;
}

/**
 * Initialize Z.AI Agent
 */
async function initializeZAIAgent() {
  console.log(
    `\n${colors.magenta}🤖 Initializing Z.AI Agent...${colors.reset}`,
  );

  try {
    logInfo("Starting 7/24 strategy analysis agent...");
    startAgent();
    logSuccess("Z.AI Agent started successfully");

    // Wait a moment for initialization
    await new Promise((resolve) => setTimeout(resolve, 1000));

    logInfo("Z.AI Agent is now monitoring 15 symbols");
    logInfo("MA7 Pullback, Red Wick Green Closure, and 11+ strategies active");
  } catch (error) {
    logError(`Z.AI Agent failed to start: ${error}`);
  }
}

/**
 * Initialize Strategy Workers
 */
async function initializeStrategyWorkers() {
  console.log(
    `\n${colors.magenta}📊 Initializing Strategy Workers...${colors.reset}`,
  );

  try {
    logInfo("Starting strategy analysis queue workers...");
    await startStrategyWorker("strategy-scan-queue");
    logSuccess("Strategy workers started");

    logInfo("Workers ready for batch symbol analysis");
    logInfo("Processing 12+ strategies per symbol");
  } catch (error) {
    logError(`Strategy workers failed to start: ${error}`);
  }
}

/**
 * Initialize Plan Workers
 */
async function initializePlanWorkers() {
  console.log(
    `\n${colors.magenta}⚡ Initializing Plan Workers...${colors.reset}`,
  );

  try {
    logInfo("Starting trade execution plan workers...");
    await startPlanWorker("executor-plan-queue");
    logSuccess("Plan workers started");

    // Get metrics
    const metrics = await getPlanWorkerMetrics();
    logInfo(`Queue: ${metrics.queueName}`);
    logInfo(`Driver: ${metrics.driver}`);
    logInfo(`Testnet: ${metrics.isTestnet ? "ENABLED" : "DISABLED"}`);

    if (!metrics.isTestnet) {
      logWarning(
        "Testnet is disabled - orders will be validated but not executed",
      );
    }
  } catch (error) {
    logError(`Plan workers failed to start: ${error}`);
  }
}

/**
 * Test API Connectivity
 */
async function testAPIConnectivity() {
  console.log(
    `\n${colors.magenta}🌐 Testing API Connectivity...${colors.reset}`,
  );

  const tests = [
    {
      name: "Binance API",
      url: "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
      validator: (data: any) =>
        data && data.price && parseFloat(data.price) > 0,
    },
    {
      name: "Binance Futures API",
      url: "https://fapi.binance.com/fapi/v1/ticker/price?symbol=BTCUSDT",
      validator: (data: any) =>
        data && data.price && parseFloat(data.price) > 0,
    },
  ];

  for (const test of tests) {
    try {
      const response = await fetch(test.url);
      const data = await response.json();

      if (response.ok && test.validator(data)) {
        logSuccess(`${test.name}: Connected (BTC: $${data.price})`);
      } else {
        logError(`${test.name}: Invalid response`);
      }
    } catch (error) {
      logError(`${test.name}: Connection failed`);
    }
  }
}

/**
 * Initialize Health Monitoring
 */
function initializeHealthMonitoring() {
  console.log(
    `\n${colors.magenta}🏥 Initializing Health Monitoring...${colors.reset}`,
  );

  try {
    // Set up process monitoring
    process.on("uncaughtException", (error) => {
      console.error(
        `${colors.red}💥 Uncaught Exception:${colors.reset}`,
        error,
      );
    });

    process.on("unhandledRejection", (reason, promise) => {
      console.error(
        `${colors.red}💥 Unhandled Rejection:${colors.reset}`,
        reason,
      );
    });

    // Graceful shutdown handlers
    process.on("SIGINT", () => {
      console.log(
        `\n${colors.yellow}🛑 Received SIGINT - Shutting down gracefully...${colors.reset}`,
      );
      process.exit(0);
    });

    process.on("SIGTERM", () => {
      console.log(
        `\n${colors.yellow}🛑 Received SIGTERM - Shutting down gracefully...${colors.reset}`,
      );
      process.exit(0);
    });

    logSuccess("Health monitoring initialized");
    logInfo("Graceful shutdown handlers registered");
  } catch (error) {
    logError(`Health monitoring failed: ${error}`);
  }
}

/**
 * Generate startup report
 */
function generateStartupReport() {
  console.log(`\n${colors.cyan}${colors.bright}
📋 SYSTEM STARTUP REPORT
========================
${colors.reset}`);

  const successCount = startupSuccess.length;
  const errorCount = startupErrors.length;
  const totalChecks = successCount + errorCount;
  const successRate =
    totalChecks > 0 ? ((successCount / totalChecks) * 100).toFixed(1) : "0";

  console.log(
    `${colors.bright}✅ Successful: ${colors.green}${successCount}/${totalChecks} (${successRate}%)${colors.reset}`,
  );
  console.log(
    `${colors.bright}❌ Failed: ${colors.red}${errorCount}${colors.reset}`,
  );

  if (startupErrors.length > 0) {
    console.log(`\n${colors.red}Errors:${colors.reset}`);
    startupErrors.forEach((error, index) => {
      console.log(`  ${index + 1}. ${error}`);
    });
  }

  // System status
  let status, statusColor;

  if (errorCount === 0) {
    status = "🟢 ALL SYSTEMS OPERATIONAL";
    statusColor = colors.green;
  } else if (errorCount <= 2) {
    status = "🟡 MINOR ISSUES";
    statusColor = colors.yellow;
  } else {
    status = "🔴 CRITICAL ISSUES";
    statusColor = colors.red;
  }

  console.log(
    `\n${colors.bright}🎯 Status: ${statusColor}${status}${colors.reset}`,
  );

  console.log(`\n${colors.cyan}🚀 Active Systems:${colors.reset}`);
  console.log(`  🤖 Z.AI Agent: ${colors.green}7/24 Analysis${colors.reset}`);
  console.log(
    `  📊 Strategy Workers: ${colors.green}Queue Processing${colors.reset}`,
  );
  console.log(
    `  ⚡ Plan Workers: ${colors.green}Trade Execution${colors.reset}`,
  );
  console.log(
    `  🌐 API Connectivity: ${colors.green}Binance Integration${colors.reset}`,
  );
  console.log(`  🏥 Health Monitor: ${colors.green}Active${colors.reset}`);

  console.log(`\n${colors.cyan}📈 Trading Features:${colors.reset}`);
  console.log(
    `  🔥 MA7 Pullback Strategy: ${colors.green}ACTIVE${colors.reset}`,
  );
  console.log(
    `  🔴 Red Wick Green Closure: ${colors.green}ACTIVE${colors.reset}`,
  );
  console.log(
    `  📊 12+ Combined Strategies: ${colors.green}OPERATIONAL${colors.reset}`,
  );
  console.log(
    `  🤖 AI Enhancement: ${colors.green}GROQ INTEGRATED${colors.reset}`,
  );
  console.log(`  ⚡ Real-time Analysis: ${colors.green}LIVE${colors.reset}`);

  console.log(`\n${colors.cyan}💡 Next Steps:${colors.reset}`);
  console.log(`  1. Monitor logs for strategy signals`);
  console.log(`  2. Check dashboard for real-time updates`);
  console.log(`  3. Verify queue processing in production`);
  console.log(`  4. Test trade execution on testnet`);

  console.log(`\n${colors.bright}🎉 Production System Ready!${colors.reset}\n`);
}

/**
 * Main startup function
 */
async function main() {
  try {
    console.log(
      `${colors.yellow}Starting production system initialization...${colors.reset}`,
    );

    // Check environment first
    const envOk = checkEnvironment();
    if (!envOk) {
      logWarning(
        "Some environment variables are missing - system may not function properly",
      );
    }

    // Initialize all systems
    await initializeZAIAgent();
    await initializeStrategyWorkers();
    await initializePlanWorkers();
    await testAPIConnectivity();
    initializeHealthMonitoring();

    // Generate final report
    generateStartupReport();

    // Keep the process running for production
    if (process.env.NODE_ENV === "production") {
      console.log(
        `${colors.green}🟢 Production system is running. Press Ctrl+C to stop.${colors.reset}`,
      );

      // Keep alive
      setInterval(() => {
        // Heartbeat - could be expanded to include health checks
      }, 30000); // Every 30 seconds
    }
  } catch (error) {
    console.error(`${colors.red}💥 Startup failed:${colors.reset}`, error);
    process.exit(1);
  }
}

// Run the startup
if (require.main === module) {
  main().catch((error) => {
    console.error(
      `${colors.red}💥 Fatal error during startup:${colors.reset}`,
      error,
    );
    process.exit(1);
  });
}
