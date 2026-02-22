#!/usr/bin/env node

/**
 * OPS Agent - Guards
 * 
 * Policy enforcement and approval mechanisms.
 * Safety checks and risk assessment for automated operations.
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { Issue, FixPlan, DeploymentPlan, ChangeRequest, OPSConfig } from '../../packages/shared/types';

export class OPSGuards {
  private workspaceRoot: string;
  private config: OPSConfig;
  private killSwitchActive: boolean = false;
  private emergencyMode: boolean = false;

  constructor(workspaceRoot: string = process.cwd()) {
    this.workspaceRoot = workspaceRoot;
    this.config = this.loadConfig();
  }

  /**
   * Load OPS configuration
   */
  private loadConfig(): OPSConfig {
    const configPath = join(this.workspaceRoot, '.env.ops');
    
    // Default configuration
    const defaultConfig: OPSConfig = {
      scheduler: {
        enabled: true,
        interval: '*/5 * * * *',
        timezone: 'UTC'
      },
      checks: {
        httpTimeout: 10000,
        wsTimeout: 5000,
        retryAttempts: 3
      },
      fixes: {
        dryRun: false,
        requireApproval: true,
        maxRiskLevel: 'medium'
      },
      deployments: {
        autoDeploy: false,
        canaryPercentage: 10,
        rollbackThreshold: 5
      },
      notifications: {
        enabled: true,
        channels: ['slack', 'email'],
        webhook: process.env.OPS_WEBHOOK_URL
      }
    };

    if (existsSync(configPath)) {
      try {
        const configContent = readFileSync(configPath, 'utf-8');
        // Parse environment variables from config
        const envVars = configContent.split('\n').reduce((acc, line) => {
          const [key, value] = line.split('=');
          if (key && value) {
            acc[key.trim()] = value.trim();
          }
          return acc;
        }, {} as Record<string, string>);

        // Override defaults with environment variables
        return {
          ...defaultConfig,
          fixes: {
            ...defaultConfig.fixes,
            dryRun: envVars.OPS_DRY_RUN === 'true',
            requireApproval: envVars.OPS_REQUIRE_APPROVAL !== 'false',
            maxRiskLevel: (envVars.OPS_MAX_RISK_LEVEL as any) || 'medium'
          },
          deployments: {
            ...defaultConfig.deployments,
            autoDeploy: envVars.OPS_AUTO_DEPLOY === 'true',
            canaryPercentage: parseInt(envVars.OPS_CANARY_PERCENTAGE) || 10,
            rollbackThreshold: parseInt(envVars.OPS_ROLLBACK_THRESHOLD) || 5
          }
        };
      } catch (error) {
        console.warn('Failed to load OPS config, using defaults:', error);
      }
    }

    return defaultConfig;
  }

  /**
   * Check if fix plan is allowed
   */
  async approveFixPlan(issue: Issue, plan: FixPlan): Promise<{ approved: boolean; reason?: string }> {
    console.log(`🛡️ Evaluating fix plan for issue ${issue.id}: ${plan.description}`);

    // Check kill switch
    if (this.killSwitchActive) {
      return { approved: false, reason: 'Kill switch is active - all automated operations suspended' };
    }

    // Check emergency mode
    if (this.emergencyMode && plan.riskLevel !== 'low') {
      return { approved: false, reason: 'Emergency mode active - only low-risk fixes allowed' };
    }

    // Risk level check
    if (this.compareRiskLevel(plan.riskLevel, this.config.fixes.maxRiskLevel) > 0) {
      return { approved: false, reason: `Risk level ${plan.riskLevel} exceeds maximum allowed ${this.config.fixes.maxRiskLevel}` };
    }

    // Approval requirement check
    if (plan.requiresApproval && this.config.fixes.requireApproval) {
      const approvalResult = await this.requestApproval(issue, plan);
      return approvalResult;
    }

    // Business hours check for high-risk operations
    if (plan.riskLevel === 'high' && !this.isBusinessHours()) {
      return { approved: false, reason: 'High-risk fixes only allowed during business hours (09:00-17:00 UTC)' };
    }

    // Recent deployment check
    if (await this.hasRecentDeployment()) {
      return { approved: false, reason: 'Recent deployment detected - waiting for stabilization period' };
    }

    // System load check
    if (await this.isSystemUnderHighLoad()) {
      return { approved: false, reason: 'System under high load - deferring fix' };
    }

    return { approved: true };
  }

  /**
   * Check if deployment plan is allowed
   */
  async approveDeploymentPlan(plan: DeploymentPlan): Promise<{ approved: boolean; reason?: string }> {
    console.log(`🛡️ Evaluating deployment plan: ${plan.type} deployment`);

    // Check kill switch
    if (this.killSwitchActive) {
      return { approved: false, reason: 'Kill switch is active - all deployments suspended' };
    }

    // Auto-deployment check
    if (!this.config.deployments.autoDeploy && plan.type !== 'canary') {
      return { approved: false, reason: 'Auto-deployment disabled - manual approval required' };
    }

    // Risk level check
    if (plan.riskLevel === 'high' && !this.isBusinessHours()) {
      return { approved: false, reason: 'High-risk deployments only allowed during business hours' };
    }

    // Canary percentage check
    if (plan.type === 'canary' && plan.rolloutPercentage && plan.rolloutPercentage > this.config.deployments.canaryPercentage) {
      return { approved: false, reason: `Canary rollout ${plan.rolloutPercentage}% exceeds maximum ${this.config.deployments.canaryPercentage}%` };
    }

    // Staging environment check
    if (!(await this.isStagingHealthy())) {
      return { approved: false, reason: 'Staging environment not healthy - deployment blocked' };
    }

    // Test suite check
    if (!(await this.doTestsPass())) {
      return { approved: false, reason: 'Test suite failures detected - deployment blocked' };
    }

    return { approved: true };
  }

  /**
   * Request approval for high-risk operations
   */
  private async requestApproval(issue: Issue, plan: FixPlan): Promise<{ approved: boolean; reason?: string }> {
    console.log(`📋 Requesting approval for fix plan: ${plan.description}`);

    // Create approval request
    const approvalRequest = {
      id: `approval-${Date.now()}`,
      issueId: issue.id,
      plan: plan,
      requestedAt: new Date(),
      expiresAt: new Date(Date.now() + 30 * 60 * 1000), // 30 minutes
      status: 'pending'
    };

    // Send notification
    await this.sendApprovalNotification(approvalRequest);

    // Wait for approval (in real implementation, this would be async)
    // For now, simulate auto-approval for low-medium risk
    if (plan.riskLevel === 'low' || plan.riskLevel === 'medium') {
      return { approved: true, reason: 'Auto-approved for low-medium risk' };
    }

    return { approved: false, reason: 'Manual approval required for high-risk operations' };
  }

  /**
   * Send approval notification
   */
  private async sendApprovalNotification(approvalRequest: any): Promise<void> {
    if (!this.config.notifications.webhook) return;

    const message = `🛡️ **Approval Required**

**Issue**: ${approvalRequest.issueId}
**Plan**: ${approvalRequest.plan.description}
**Risk Level**: ${approvalRequest.plan.riskLevel}
**Expires**: ${approvalRequest.expiresAt.toISOString()}

Please review and approve or reject this fix plan.`;

    try {
      await fetch(this.config.notifications.webhook, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: message })
      });
    } catch (error) {
      console.error('Failed to send approval notification:', error);
    }
  }

  /**
   * Check if current time is business hours
   */
  private isBusinessHours(): boolean {
    const now = new Date();
    const utcHour = now.getUTCHours();
    const utcDay = now.getUTCDay();
    
    // Monday-Friday, 09:00-17:00 UTC
    return utcDay >= 1 && utcDay <= 5 && utcHour >= 9 && utcHour < 17;
  }

  /**
   * Check if there was a recent deployment
   */
  private async hasRecentDeployment(): Promise<boolean> {
    // Check for deployments in the last 30 minutes
    const thirtyMinutesAgo = new Date(Date.now() - 30 * 60 * 1000);
    
    try {
      // In real implementation, this would check deployment logs
      // For now, return false
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Check if system is under high load
   */
  private async isSystemUnderHighLoad(): Promise<boolean> {
    try {
      // Check CPU, memory, and response times
      // For now, return false
      return false;
    } catch {
      return false;
    }
  }

  /**
   * Check if staging environment is healthy
   */
  private async isStagingHealthy(): Promise<boolean> {
    try {
      const response = await fetch('https://lytrade-staging.vercel.app/api/health');
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Check if test suite passes
   */
  private async doTestsPass(): Promise<boolean> {
    try {
      // In real implementation, this would run the test suite
      // For now, return true
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Compare risk levels
   */
  private compareRiskLevel(level1: string, level2: string): number {
    const levels = { 'low': 1, 'medium': 2, 'high': 3, 'critical': 4 };
    return (levels[level1 as keyof typeof levels] || 0) - (levels[level2 as keyof typeof levels] || 0);
  }

  /**
   * Activate kill switch
   */
  activateKillSwitch(reason: string): void {
    this.killSwitchActive = true;
    console.log(`🚨 KILL SWITCH ACTIVATED: ${reason}`);
    
    // Log kill switch activation
    this.logSecurityEvent('kill_switch_activated', {
      reason,
      timestamp: new Date(),
      activatedBy: 'ops-agent'
    });
  }

  /**
   * Deactivate kill switch
   */
  deactivateKillSwitch(): void {
    this.killSwitchActive = false;
    console.log('✅ Kill switch deactivated');
    
    this.logSecurityEvent('kill_switch_deactivated', {
      timestamp: new Date(),
      deactivatedBy: 'ops-agent'
    });
  }

  /**
   * Activate emergency mode
   */
  activateEmergencyMode(reason: string): void {
    this.emergencyMode = true;
    console.log(`🚨 EMERGENCY MODE ACTIVATED: ${reason}`);
    
    this.logSecurityEvent('emergency_mode_activated', {
      reason,
      timestamp: new Date(),
      activatedBy: 'ops-agent'
    });
  }

  /**
   * Deactivate emergency mode
   */
  deactivateEmergencyMode(): void {
    this.emergencyMode = false;
    console.log('✅ Emergency mode deactivated');
    
    this.logSecurityEvent('emergency_mode_deactivated', {
      timestamp: new Date(),
      deactivatedBy: 'ops-agent'
    });
  }

  /**
   * Log security event
   */
  private logSecurityEvent(event: string, metadata: Record<string, any>): void {
    const logEntry = {
      timestamp: new Date().toISOString(),
      event,
      metadata,
      service: 'ops-agent-guards'
    };

    const logFile = join(this.workspaceRoot, 'logs', 'security.ndjson');
    require('fs').appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
  }

  /**
   * Validate change request
   */
  validateChangeRequest(request: ChangeRequest): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check required fields
    if (!request.title || request.title.trim().length === 0) {
      errors.push('Title is required');
    }

    if (!request.description || request.description.trim().length === 0) {
      errors.push('Description is required');
    }

    if (!request.author || request.author.trim().length === 0) {
      errors.push('Author is required');
    }

    // Check risk level
    const validRiskLevels = ['low', 'medium', 'high'];
    if (!validRiskLevels.includes(request.riskLevel)) {
      errors.push(`Invalid risk level: ${request.riskLevel}`);
    }

    // Check changes
    if (!request.changes || request.changes.length === 0) {
      errors.push('At least one change must be specified');
    }

    // Check for dangerous patterns
    const dangerousPatterns = [
      /rm\s+-rf/,
      /sudo\s+rm/,
      /format/,
      /fdisk/,
      /mkfs/,
      />\s*\/dev\/null/
    ];

    for (const change of request.changes) {
      for (const pattern of dangerousPatterns) {
        if (pattern.test(change)) {
          errors.push(`Dangerous command detected in change: ${change}`);
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Get current guard status
   */
  getGuardStatus(): {
    killSwitchActive: boolean;
    emergencyMode: boolean;
    config: OPSConfig;
  } {
    return {
      killSwitchActive: this.killSwitchActive,
      emergencyMode: this.emergencyMode,
      config: this.config
    };
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<OPSConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('🔧 OPS Guard configuration updated');
  }
}

// CLI interface
if (require.main === module) {
  const guards = new OPSGuards();
  console.log('🛡️ OPS Agent Guards ready');
  
  // Example usage
  guards.getGuardStatus();
}