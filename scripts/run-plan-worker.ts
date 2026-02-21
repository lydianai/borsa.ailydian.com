import "dotenv/config";
import pino from "pino";
import { startPlanWorker } from "../src/lib/execution/plan-worker";

const log = pino({ name: "plan-worker-cli" });

async function main() {
  await startPlanWorker();

  log.info("Plan worker running. Press Ctrl+C to exit.");

  process.once("SIGINT", () => {
    log.info("Plan worker shutting down (SIGINT)");
    process.exit(0);
  });
  process.once("SIGTERM", () => {
    log.info("Plan worker shutting down (SIGTERM)");
    process.exit(0);
  });
}

main().catch((error) => {
  log.error({ error }, "Plan worker failed to start");
  process.exit(1);
});

