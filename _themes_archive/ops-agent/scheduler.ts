/**
 * 🧠 LyTrade OPS AGENT - Main Scheduler
 * 
 * 7×24 çalışan otonom operasyon ajanı
 * Her 5 dakikada bir sistem sağlığını kontrol eder, otomatik düzeltme yapar
 */

import cron from 'node-cron';
import { Logger } from '@lytrade/observability';
import { HealthChecker } from './checkers';
import { ChangePlanner } from './planners';
import { SafeFixer } from './fixers';
import { CanaryExecutor } from './executors';
import { OpsReporter } from './reporters';
import { PolicyGuard } from './guards';

interface OpsMetrics {
  p99: number;
  errorRate: number;
  lag_ms: number;
  reconnects_per_min: number;
  nan_inf_count: number;
  strategy_agreement: number;
}

interface OpsContext {
  timestamp: string;
  change_id: string;
  metrics: OpsMetrics;
  error_budget_used: number;
  rollforwards_today: number;
}

export class OpsAgent {
  private logger = Logger.child({ component: 'OpsAgent' });
  private checker = new HealthChecker();
  private planner = new ChangePlanner();
  private fixer = new SafeFixer();
  private executor = new CanaryExecutor();
  private reporter = new OpsReporter();
  private guard = new PolicyGuard();

  private rollforwardsToday = 0;
  private errorBudgetUsed = 0;

  constructor() {
    this.setupGracefulShutdown();
  }

  /**
   * Ana operasyon döngüsü
   * - Her 5 dakikada bir çalışır
   * - Kritik event'lerde anında tetiklenir
   */
  async start(): Promise<void> {
    this.logger.info('🧠 LyTrade Otonom OPS Agent başlatılıyor...');

    // Ana scheduler - her 5 dakikada bir
    cron.schedule('*/5 * * * *', async () => {
      await this.runOpsCycle();
    });

    // Kritik event listener - anında müdahale
    this.setupCriticalEventListener();

    // Başlangıç döngüsü
    await this.runOpsCycle();

    this.logger.info('✅ OPS Agent aktif - 7×24 monitoring başladı');
  }

  private async runOpsCycle(): Promise<void> {
    const change_id = `chg_${Date.now()}`;
    const startTime = Date.now();

    try {
      // 1) SAĞLIK KONTROLÜ
      this.logger.info({ change_id }, '🔍 Sağlık kontrolü başlatılıyor...');
      const health = await this.checker.runFullCheck();
      
      const context: OpsContext = {
        timestamp: new Date().toISOString(),
        change_id,
        metrics: health.metrics,
        error_budget_used: this.errorBudgetUsed,
        rollforwards_today: this.rollforwardsToday
      };

      // 2) POLİKA KONTROLÜ
      if (!this.guard.canProceed(context)) {
        this.logger.warn({ change_id }, '⚠️ Politika engeli - sadece teşhis modu');
        await this.reporter.report({
          ...context,
          phase: 'POLICY_BLOCK',
          result: 'ABORT',
          action: 'HOLD',
          notes: ['Error budget exceeded or approval required']
        });
        return;
      }

      // 3) DEĞİŞİKLİK PLANLAMA
      this.logger.info({ change_id }, '📋 Değişiklik planlanıyor...');
      const plan = await this.planner.planChanges(health);

      if (plan.action === 'noop') {
        this.logger.info({ change_id }, '✅ Sistem stabil - değişiklik gerekmiyor');
        await this.reporter.report({
          ...context,
          phase: 'OBSERVE',
          result: 'PASS',
          action: 'HOLD',
          notes: ['System stable']
        });
        return;
      }

      // 4) GÜVENLİ DÜZELTME
      this.logger.info({ change_id, plan: plan.action }, '🔧 Güvenli düzeltme uygulanıyor...');
      const fix = await this.fixer.createSafeFix(plan);

      // 5) CANARY DEPLOY
      this.logger.info({ change_id }, '🚀 Canary deploy başlatılıyor...');
      const canaryResult = await this.executor.deployCanary(fix);

      // 6) SLO DEĞERLENDİRMESİ
      if (this.evaluateSLO(canaryResult.metrics)) {
        // Başarılı - gradual rollout
        this.logger.info({ change_id }, '✅ Canary başarılı - gradual rollout başlatılıyor...');
        await this.executor.gradualRollout(fix);
        this.rollforwardsToday++;
      } else {
        // Başarısız - otomatik rollback
        this.logger.warn({ change_id }, '❌ Canary başarısız - otomatik rollback...');
        await this.executor.rollback(fix);
        this.errorBudgetUsed += 0.5;
      }

      // 7) RAPORLAMA
      await this.reporter.report({
        ...context,
        phase: 'COMPLETE',
        result: canaryResult.success ? 'PASS' : 'FAIL',
        action: plan.action,
        metrics: canaryResult.metrics,
        notes: plan.reasons
      });

      const duration = Date.now() - startTime;
      this.logger.info({ change_id, duration }, '🏁 Ops döngüsü tamamlandı');

    } catch (error) {
      this.logger.error({ change_id, error }, '💥 Ops döngüsü başarısız');
      
      await this.reporter.report({
        timestamp: new Date().toISOString(),
        change_id,
        phase: 'ERROR',
        result: 'FAIL',
        action: 'ABORT',
        metrics: { p99: 0, errorRate: 100, lag_ms: 0, reconnects_per_min: 0, nan_inf_count: 0, strategy_agreement: 0 },
        notes: [`Error: ${error.message}`]
      });
    }
  }

  /**
   * SLO (Service Level Objective) değerlendirmesi
   */
  private evaluateSLO(metrics: OpsMetrics): boolean {
    const SLO_THRESHOLDS = {
      p99_max: 600,
      errorRate_max: 1.5,
      lag_ms_max: 900,
      nan_inf_max: 0,
      agreement_min: 62
    };

    return (
      metrics.p99 <= SLO_THRESHOLDS.p99_max &&
      metrics.errorRate <= SLO_THRESHOLDS.errorRate_max &&
      metrics.lag_ms <= SLO_THRESHOLDS.lag_ms_max &&
      metrics.nan_inf_count <= SLO_THRESHOLDS.nan_inf_max &&
      metrics.strategy_agreement >= SLO_THRESHOLDS.agreement_min
    );
  }

  /**
   * Kritik event listener - anında müdahale
   */
  private setupCriticalEventListener(): void {
    process.on('uncaughtException', async (error) => {
      this.logger.error({ error }, '🚨 Kritik hata - anında müdahale');
      await this.runOpsCycle();
    });

    process.on('unhandledRejection', async (reason) => {
      this.logger.error({ reason }, '🚨 Kritik rejection - anında müdahale');
      await this.runOpsCycle();
    });
  }

  /**
   * Graceful shutdown
   */
  private setupGracefulShutdown(): void {
    const shutdown = () => {
      this.logger.info('🛑 OPS Agent kapatılıyor...');
      process.exit(0);
    };

    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);
  }
}

// CLI entry point
if (require.main === module) {
  const agent = new OpsAgent();
  agent.start().catch(console.error);
}