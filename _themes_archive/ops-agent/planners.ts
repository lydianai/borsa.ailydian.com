/**
 * 📋 LyTrade OPS AGENT - Change Planner
 * 
 * Sağlık sorunlarını analiz eder ve güvenli değişiklik planları oluşturur:
 * - Issues/alerts → plan → küçük değişiklik paketleri (change-set)
 * - Risk seviyesine göre kategorize eder
 * - Manuel onay gerektiren değişiklikleri belirler
 */

import { Logger } from '@lytrade/observability';
import { HealthCheckResult, HealthIssue } from './checkers';

export interface ChangePlan {
  action: 'noop' | 'config' | 'strategy' | 'observability' | 'rollback';
  priority: 'low' | 'medium' | 'high' | 'critical';
  category: 'ui' | 'data' | 'backend' | 'config' | 'strategy' | 'observability';
  description: string;
  reasons: string[];
  changes: ChangeItem[];
  requiresApproval: boolean;
  estimatedImpact: 'minimal' | 'low' | 'medium' | 'high';
  rollbackPlan: string;
}

export interface ChangeItem {
  type: 'env' | 'config' | 'code' | 'strategy_param' | 'documentation';
  file: string;
  key?: string;
  oldValue?: any;
  newValue?: any;
  description: string;
  riskLevel: 'low' | 'medium' | 'high';
}

export class ChangePlanner {
  private logger = Logger.child({ component: 'ChangePlanner' });

  async planChanges(health: HealthCheckResult): Promise<ChangePlan> {
    this.logger.info({ status: health.status, issuesCount: health.issues.length }, '📋 Değişiklik planlanıyor...');

    // Sistem stabil ise değişiklik gerekmiyor
    if (health.status === 'healthy' && health.issues.length === 0) {
      return {
        action: 'noop',
        priority: 'low',
        category: 'observability',
        description: 'System is stable - no changes needed',
        reasons: ['All health checks passed'],
        changes: [],
        requiresApproval: false,
        estimatedImpact: 'minimal',
        rollbackPlan: 'No changes to rollback'
      };
    }

    // Issues'ları kategorize et ve önceliklendir
    const criticalIssues = health.issues.filter(i => i.severity === 'critical');
    const highIssues = health.issues.filter(i => i.severity === 'high');
    const mediumIssues = health.issues.filter(i => i.severity === 'medium');

    // Critical issues için acil plan
    if (criticalIssues.length > 0) {
      return this.createCriticalPlan(criticalIssues, health);
    }

    // High issues için öncelikli plan
    if (highIssues.length > 0) {
      return this.createHighPriorityPlan(highIssues, health);
    }

    // Medium issues için rutin plan
    if (mediumIssues.length > 0) {
      return this.createMediumPriorityPlan(mediumIssues, health);
    }

    return {
      action: 'noop',
      priority: 'low',
      category: 'observability',
      description: 'No actionable issues found',
      reasons: ['Issues require manual intervention'],
      changes: [],
      requiresApproval: false,
      estimatedImpact: 'minimal',
      rollbackPlan: 'No changes to rollback'
    };
  }

  /**
   * Critical issues için acil eylem planı
   */
  private createCriticalPlan(issues: HealthIssue[], health: HealthCheckResult): ChangePlan {
    this.logger.warn({ issuesCount: issues.length }, '🚨 Critical issues için acil plan oluşturuluyor');

    const changes: ChangeItem[] = [];
    const reasons: string[] = [];

    for (const issue of issues) {
      switch (issue.category) {
        case 'data':
          if (issue.description.includes('WebSocket')) {
            changes.push({
              type: 'env',
              file: '.env.ops',
              key: 'WS_RECONNECT_JITTER',
              oldValue: '250',
              newValue: '500',
              description: 'Increase WebSocket reconnect jitter for stability',
              riskLevel: 'low'
            });
            reasons.push('WebSocket connection instability detected');
          }
          break;

        case 'backend':
          if (issue.description.includes('HTTP')) {
            changes.push({
              type: 'env',
              file: '.env.ops',
              key: 'API_RATE_LIMIT',
              oldValue: '1000',
              newValue: '800',
              description: 'Reduce API rate limit to prevent overload',
              riskLevel: 'medium'
            });
            reasons.push('Backend HTTP endpoints failing');
          }
          break;

        case 'ui':
          changes.push({
            type: 'env',
            file: '.env.ops',
            key: 'STREAM_BUFFER_SIZE',
            oldValue: '200',
            newValue: '150',
            description: 'Reduce stream buffer to improve UI responsiveness',
            riskLevel: 'low'
          });
          reasons.push('Frontend SSR/CSR issues detected');
          break;
      }
    }

    return {
      action: 'config',
      priority: 'critical',
      category: issues[0].category,
      description: `Critical fix for ${issues[0].category} issues`,
      reasons,
      changes,
      requiresApproval: false, // Critical issues için otomatik onay
      estimatedImpact: 'medium',
      rollbackPlan: 'Restore previous environment values from backup'
    };
  }

  /**
   * High priority issues için plan
   */
  private createHighPriorityPlan(issues: HealthIssue[], health: HealthCheckResult): ChangePlan {
    this.logger.info({ issuesCount: issues.length }, '⚡ High priority issues için plan oluşturuluyor');

    const changes: ChangeItem[] = [];
    const reasons: string[] = [];

    for (const issue of issues) {
      switch (issue.category) {
        case 'strategy':
          if (issue.metric === 'strategy_agreement' && issue.value && issue.value < 62) {
            changes.push({
              type: 'strategy_param',
              file: 'src/lib/strategy-aggregator.ts',
              key: 'CONSENSUS_MIN_CONFIDENCE',
              oldValue: '0.62',
              newValue: '0.58',
              description: 'Temporarily reduce consensus threshold during market volatility',
              riskLevel: 'medium'
            });
            reasons.push('Low strategy consensus during volatile market');
          }
          break;

        case 'data':
          if (issue.metric === 'lag_ms' && issue.value && issue.value > 900) {
            changes.push({
              type: 'env',
              file: '.env.ops',
              key: 'MAX_WS_SHARDS',
              oldValue: '12',
              newValue: '8',
              description: 'Reduce WebSocket shards to lower data lag',
              riskLevel: 'medium'
            });
            reasons.push('High data lag detected');
          }
          break;

        case 'observability':
          changes.push({
            type: 'config',
            file: 'packages/observability/metrics.ts',
            key: 'sampleRate',
            oldValue: '0.1',
            newValue: '0.05',
            description: 'Reduce tracing sample rate to improve performance',
            riskLevel: 'low'
          });
          reasons.push('High observability overhead detected');
          break;
      }
    }

    return {
      action: 'config',
      priority: 'high',
      category: issues[0].category,
      description: `High priority fix for ${issues[0].category} optimization`,
      reasons,
      changes,
      requiresApproval: false,
      estimatedImpact: 'low',
      rollbackPlan: 'Restore previous configuration values'
    };
  }

  /**
   * Medium priority issues için plan
   */
  private createMediumPriorityPlan(issues: HealthIssue[], health: HealthCheckResult): ChangePlan {
    this.logger.info({ issuesCount: issues.length }, '🔧 Medium priority issues için plan oluşturuluyor');

    const changes: ChangeItem[] = [];
    const reasons: string[] = [];

    for (const issue of issues) {
      switch (issue.category) {
        case 'config':
          changes.push({
            type: 'documentation',
            file: 'docs/ENV_MATRIX.md',
            description: 'Update environment matrix documentation with current values',
            riskLevel: 'low'
          });
          reasons.push('Configuration drift detected');
          break;

        case 'strategy':
          changes.push({
            type: 'strategy_param',
            file: 'src/lib/signals/ma7-pullback-strategy.ts',
            key: 'pullbackThreshold',
            oldValue: '0.02',
            newValue: '0.018',
            description: 'Fine-tune MA7 pullback threshold for better signal quality',
            riskLevel: 'medium'
          });
          reasons.push('Strategy performance optimization needed');
          break;

        case 'observability':
          changes.push({
            type: 'code',
            file: 'packages/observability/logger.ts',
            description: 'Add structured logging for better error tracking',
            riskLevel: 'low'
          });
          reasons.push('Improve observability and debugging capabilities');
          break;
      }
    }

    return {
      action: 'config',
      priority: 'medium',
      category: issues[0].category,
      description: `Medium priority optimization for ${issues[0].category}`,
      reasons,
      changes,
      requiresApproval: true, // Medium değişiklikler için onay gerekli
      estimatedImpact: 'minimal',
      rollbackPlan: 'Revert configuration changes using git'
    };
  }

  /**
   * Değişiklik riskini değerlendir
   */
  private assessChangeRisk(changes: ChangeItem[]): 'minimal' | 'low' | 'medium' | 'high' {
    if (changes.length === 0) return 'minimal';
    
    const highRiskChanges = changes.filter(c => c.riskLevel === 'high').length;
    const mediumRiskChanges = changes.filter(c => c.riskLevel === 'medium').length;
    
    if (highRiskChanges > 0) return 'high';
    if (mediumRiskChanges > 2) return 'medium';
    if (mediumRiskChanges > 0) return 'low';
    
    return 'minimal';
  }

  /**
   * Manuel onay gerektiren değişiklikleri kontrol et
   */
  private requiresManualApproval(changes: ChangeItem[]): boolean {
    // Strateji mantığı değişiklikleri
    if (changes.some(c => c.type === 'code' && c.file.includes('strategy'))) {
      return true;
    }

    // Database schema değişiklikleri
    if (changes.some(c => c.description.includes('migration') || c.description.includes('schema'))) {
      return true;
    }

    // Yüksek riskli değişiklikler
    if (changes.some(c => c.riskLevel === 'high')) {
      return true;
    }

    // Production API kontratını etkileyen değişiklikler
    if (changes.some(c => c.file.includes('api') && c.type === 'code')) {
      return true;
    }

    return false;
  }
}