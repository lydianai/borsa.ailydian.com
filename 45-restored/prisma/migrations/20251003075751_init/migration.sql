-- CreateTable
CREATE TABLE "Trade" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "symbol" TEXT NOT NULL,
    "side" TEXT NOT NULL,
    "entryPrice" REAL NOT NULL,
    "exitPrice" REAL,
    "quantity" REAL NOT NULL,
    "leverage" INTEGER NOT NULL,
    "pnl" REAL,
    "pnlPercent" REAL,
    "status" TEXT NOT NULL,
    "entryTime" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "exitTime" DATETIME,
    "stopLoss" REAL,
    "takeProfit" REAL,
    "strategyId" TEXT,
    "botId" TEXT NOT NULL,
    CONSTRAINT "Trade_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Bot" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "leverage" INTEGER NOT NULL,
    "status" TEXT NOT NULL,
    "maxPositionSize" REAL NOT NULL,
    "stopLossPercent" REAL NOT NULL,
    "takeProfitPercent" REAL NOT NULL,
    "maxDailyLoss" REAL NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "DailyMetrics" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "botId" TEXT NOT NULL,
    "date" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "totalTrades" INTEGER NOT NULL,
    "winningTrades" INTEGER NOT NULL,
    "losingTrades" INTEGER NOT NULL,
    "winRate" REAL NOT NULL,
    "totalPnL" REAL NOT NULL,
    "dailyPnL" REAL NOT NULL,
    "sharpeRatio" REAL NOT NULL,
    "maxDrawdown" REAL NOT NULL,
    "currentDrawdown" REAL NOT NULL,
    "totalVolume" REAL NOT NULL,
    "avgTradeSize" REAL NOT NULL,
    CONSTRAINT "DailyMetrics_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "Alert" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "botId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "severity" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "data" JSONB,
    "acknowledged" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "Alert_botId_fkey" FOREIGN KEY ("botId") REFERENCES "Bot" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "PerformanceMetric" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "botId" TEXT NOT NULL,
    "timestamp" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "totalPnL" REAL NOT NULL,
    "openPositions" INTEGER NOT NULL,
    "winRate" REAL NOT NULL,
    "sharpeRatio" REAL NOT NULL,
    "drawdown" REAL NOT NULL
);

-- CreateTable
CREATE TABLE "MarketData" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "symbol" TEXT NOT NULL,
    "timestamp" DATETIME NOT NULL,
    "open" REAL NOT NULL,
    "high" REAL NOT NULL,
    "low" REAL NOT NULL,
    "close" REAL NOT NULL,
    "volume" REAL NOT NULL,
    "timeframe" TEXT NOT NULL
);

-- CreateIndex
CREATE INDEX "Trade_symbol_idx" ON "Trade"("symbol");

-- CreateIndex
CREATE INDEX "Trade_botId_idx" ON "Trade"("botId");

-- CreateIndex
CREATE INDEX "Trade_entryTime_idx" ON "Trade"("entryTime");

-- CreateIndex
CREATE INDEX "Bot_symbol_idx" ON "Bot"("symbol");

-- CreateIndex
CREATE INDEX "Bot_status_idx" ON "Bot"("status");

-- CreateIndex
CREATE INDEX "DailyMetrics_botId_idx" ON "DailyMetrics"("botId");

-- CreateIndex
CREATE INDEX "DailyMetrics_date_idx" ON "DailyMetrics"("date");

-- CreateIndex
CREATE UNIQUE INDEX "DailyMetrics_botId_date_key" ON "DailyMetrics"("botId", "date");

-- CreateIndex
CREATE INDEX "Alert_botId_idx" ON "Alert"("botId");

-- CreateIndex
CREATE INDEX "Alert_type_idx" ON "Alert"("type");

-- CreateIndex
CREATE INDEX "Alert_severity_idx" ON "Alert"("severity");

-- CreateIndex
CREATE INDEX "Alert_createdAt_idx" ON "Alert"("createdAt");

-- CreateIndex
CREATE INDEX "PerformanceMetric_botId_idx" ON "PerformanceMetric"("botId");

-- CreateIndex
CREATE INDEX "PerformanceMetric_timestamp_idx" ON "PerformanceMetric"("timestamp");

-- CreateIndex
CREATE INDEX "MarketData_symbol_idx" ON "MarketData"("symbol");

-- CreateIndex
CREATE INDEX "MarketData_timestamp_idx" ON "MarketData"("timestamp");

-- CreateIndex
CREATE UNIQUE INDEX "MarketData_symbol_timestamp_timeframe_key" ON "MarketData"("symbol", "timestamp", "timeframe");
