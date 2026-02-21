#!/usr/bin/env node

/**
 * OPS Agent - Executors
 * 
 * GitHub PR/commit creation and canary deployment execution.
 * Safe deployment with automatic rollback capabilities.
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';
import { FixPlan, DeploymentPlan, ChangeRequest } from '../../packages/shared/types';

export class DeploymentExecutors {
  private workspaceRoot: string;
  private dryRun: boolean;
  private githubToken?: string;

  constructor(workspaceRoot: string = process.cwd(), dryRun: boolean = false) {
    this.workspaceRoot = workspaceRoot;
    this.dryRun = dryRun;
    this.githubToken = process.env.GITHUB_TOKEN;
  }

  /**
   * Execute deployment plan
   */
  async executeDeployment(plan: DeploymentPlan): Promise<{ success: boolean; deploymentId?: string; error?: string }> {
    console.log(`🚀 Executing ${plan.type} deployment: ${plan.id}`);

    try {
      switch (plan.type) {
        case 'canary':
          return await this.executeCanaryDeployment(plan);
        case 'blue_green':
          return await this.executeBlueGreenDeployment(plan);
        case 'rolling':
          return await this.executeRollingDeployment(plan);
        default:
          return { success: false, error: `Unknown deployment type: ${plan.type}` };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Deployment failed'
      };
    }
  }

  /**
   * Execute fix plan with commit/PR
   */
  async executeFixPlan(plan: FixPlan): Promise<{ success: boolean; prUrl?: string; commit?: string; error?: string }> {
    console.log(`🔧 Executing fix plan: ${plan.description}`);

    try {
      // Create branch for fix
      const branchName = `ops/fix/${plan.issueId}-${Date.now()}`;
      if (!this.dryRun) {
        execSync(`git checkout -b ${branchName}`, { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Apply fix steps
      for (const step of plan.steps) {
        console.log(`  📝 Applying step: ${step}`);
        // Fix execution logic would be handled by fixers
      }

      // Commit changes
      const commitMessage = `fix(ops): ${plan.description}\n\nFixes issue ${plan.issueId}\nRisk level: ${plan.riskLevel}`;
      let commitHash: string;

      if (!this.dryRun) {
        execSync('git add .', { cwd: this.workspaceRoot, stdio: 'pipe' });
        execSync(`git commit -m "${commitMessage}"`, { cwd: this.workspaceRoot, stdio: 'pipe' });
        commitHash = execSync('git rev-parse HEAD', { cwd: this.workspaceRoot }).toString().trim();
      } else {
        commitHash = 'dry-run-commit-hash';
      }

      // Create PR if approval required
      if (plan.requiresApproval && this.githubToken) {
        const prUrl = await this.createPullRequest(branchName, commitMessage, plan);
        return { success: true, prUrl, commit: commitHash };
      } else {
        // Auto-merge if no approval required
        if (!this.dryRun) {
          execSync('git checkout main', { cwd: this.workspaceRoot, stdio: 'pipe' });
          execSync(`git merge ${branchName}`, { cwd: this.workspaceRoot, stdio: 'pipe' });
          execSync(`git branch -D ${branchName}`, { cwd: this.workspaceRoot, stdio: 'pipe' });
        }
        return { success: true, commit: commitHash };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Fix execution failed'
      };
    }
  }

  /**
   * Canary deployment execution
   */
  private async executeCanaryDeployment(plan: DeploymentPlan): Promise<{ success: boolean; deploymentId?: string; error?: string }> {
    const deploymentId = `canary-${Date.now()}`;
    const percentage = plan.rolloutPercentage || 10;

    console.log(`🕊️ Starting canary deployment with ${percentage}% traffic`);

    try {
      // Build application
      if (!this.dryRun) {
        console.log('  🔨 Building application...');
        execSync('pnpm build', { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Deploy to canary environment
      if (!this.dryRun) {
        console.log(`  🚀 Deploying to canary environment...`);
        // Vercel canary deployment
        execSync(`vercel --name sardag-emrah-canary --confirm`, { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Monitor deployment health
      const healthCheck = await this.monitorCanaryHealth(deploymentId);
      if (!healthCheck) {
        await this.rollbackDeployment(deploymentId);
        return { success: false, error: 'Canary deployment failed health checks' };
      }

      // Gradual rollout
      for (let current = percentage; current <= 100; current += 10) {
        console.log(`  📈 Rolling out to ${current}% traffic...`);
        await this.sleep(30000); // Wait 30 seconds between rollouts
        
        const health = await this.checkDeploymentHealth(deploymentId);
        if (!health) {
          await this.rollbackDeployment(deploymentId);
          return { success: false, error: `Health check failed at ${current}% rollout` };
        }
      }

      console.log('✅ Canary deployment completed successfully');
      return { success: true, deploymentId };
    } catch (error) {
      await this.rollbackDeployment(deploymentId);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Canary deployment failed',
        deploymentId
      };
    }
  }

  /**
   * Blue-green deployment execution
   */
  private async executeBlueGreenDeployment(plan: DeploymentPlan): Promise<{ success: boolean; deploymentId?: string; error?: string }> {
    const deploymentId = `blue-green-${Date.now()}`;

    console.log('🔵🟢 Starting blue-green deployment');

    try {
      // Build green environment
      if (!this.dryRun) {
        console.log('  🔨 Building green environment...');
        execSync('pnpm build', { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Deploy to green environment
      if (!this.dryRun) {
        console.log('  🚀 Deploying to green environment...');
        execSync('vercel --name sardag-emrah-green --confirm', { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Health check green environment
      const greenHealth = await this.checkGreenEnvironmentHealth();
      if (!greenHealth) {
        return { success: false, error: 'Green environment health check failed' };
      }

      // Switch traffic to green
      if (!this.dryRun) {
        console.log('  🔄 Switching traffic to green environment...');
        // DNS or load balancer switch logic
      }

      console.log('✅ Blue-green deployment completed successfully');
      return { success: true, deploymentId };
    } catch (error) {
      await this.rollbackDeployment(deploymentId);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Blue-green deployment failed',
        deploymentId
      };
    }
  }

  /**
   * Rolling deployment execution
   */
  private async executeRollingDeployment(plan: DeploymentPlan): Promise<{ success: boolean; deploymentId?: string; error?: string }> {
    const deploymentId = `rolling-${Date.now()}`;

    console.log('🔄 Starting rolling deployment');

    try {
      // Build application
      if (!this.dryRun) {
        console.log('  🔨 Building application...');
        execSync('pnpm build', { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      // Rolling update across instances
      const instances = await this.getDeploymentInstances();
      for (let i = 0; i < instances.length; i++) {
        const instance = instances[i];
        console.log(`  🔄 Updating instance ${i + 1}/${instances.length}: ${instance}`);
        
        // Update single instance
        if (!this.dryRun) {
          await this.updateInstance(instance);
        }

        // Health check after each update
        const health = await this.checkInstanceHealth(instance);
        if (!health) {
          await this.rollbackDeployment(deploymentId);
          return { success: false, error: `Instance ${instance} failed health check` };
        }

        // Wait between updates
        await this.sleep(10000);
      }

      console.log('✅ Rolling deployment completed successfully');
      return { success: true, deploymentId };
    } catch (error) {
      await this.rollbackDeployment(deploymentId);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Rolling deployment failed',
        deploymentId
      };
    }
  }

  /**
   * Create GitHub pull request
   */
  private async createPullRequest(branchName: string, title: string, plan: FixPlan): Promise<string> {
    if (!this.githubToken) {
      throw new Error('GitHub token required for PR creation');
    }

    const prBody = `
## Automated Fix Request

**Issue ID**: ${plan.issueId}
**Risk Level**: ${plan.riskLevel}
**Estimated Duration**: ${plan.estimatedDuration} minutes

### Changes
${plan.steps.map(step => `- ${step}`).join('\n')}

### Rollback Plan
${plan.rollbackPlan ? `- ${plan.rollbackPlan.type}: ${plan.rollbackPlan.target}` : 'No rollback plan'}

---
*This PR was created automatically by the OPS Agent*
`;

    // GitHub API call to create PR
    const prData = {
      title,
      body: prBody,
      head: branchName,
      base: 'main'
    };

    if (!this.dryRun) {
      const response = await fetch('https://api.github.com/repos/sardag-emrah/sardag-emrah/pulls', {
        method: 'POST',
        headers: {
          'Authorization': `token ${this.githubToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(prData)
      });

      if (!response.ok) {
        throw new Error(`Failed to create PR: ${response.statusText}`);
      }

      const pr = await response.json();
      return pr.html_url;
    }

    return 'https://github.com/sardag-emrah/sardag-emrah/pull/dry-run';
  }

  /**
   * Monitor canary deployment health
   */
  private async monitorCanaryHealth(deploymentId: string): Promise<boolean> {
    console.log('  🏥 Monitoring canary health...');
    
    // Check multiple health endpoints
    const checks = [
      this.checkEndpoint('https://sardag-emrah-canary.vercel.app/api/health'),
      this.checkEndpoint('https://sardag-emrah-canary.vercel.app/api/market/overview'),
      this.checkEndpoint('https://sardag-emrah-canary.vercel.app/api/symbols')
    ];

    const results = await Promise.allSettled(checks);
    const passed = results.filter(r => r.status === 'fulfilled').length;
    
    return passed >= checks.length * 0.8; // 80% pass rate
  }

  /**
   * Check deployment health
   */
  private async checkDeploymentHealth(deploymentId: string): Promise<boolean> {
    try {
      const response = await fetch('https://sardag-emrah-canary.vercel.app/api/health', {
        timeout: 10000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Check green environment health
   */
  private async checkGreenEnvironmentHealth(): Promise<boolean> {
    try {
      const response = await fetch('https://sardag-emrah-green.vercel.app/api/health', {
        timeout: 10000
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get deployment instances
   */
  private async getDeploymentInstances(): Promise<string[]> {
    // Mock instances - in real implementation, this would query cloud provider
    return ['instance-1', 'instance-2', 'instance-3'];
  }

  /**
   * Update single instance
   */
  private async updateInstance(instance: string): Promise<void> {
    console.log(`    🔄 Updating ${instance}...`);
    // Instance update logic
  }

  /**
   * Check instance health
   */
  private async checkInstanceHealth(instance: string): Promise<boolean> {
    try {
      // Health check for specific instance
      return true; // Mock
    } catch {
      return false;
    }
  }

  /**
   * Check endpoint health
   */
  private async checkEndpoint(url: string): Promise<boolean> {
    try {
      const response = await fetch(url, { timeout: 5000 });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Rollback deployment
   */
  async rollbackDeployment(deploymentId: string): Promise<boolean> {
    console.log(`🔄 Rolling back deployment: ${deploymentId}`);

    try {
      if (!this.dryRun) {
        // Git rollback
        execSync('git checkout main', { cwd: this.workspaceRoot, stdio: 'pipe' });
        execSync('git reset --hard HEAD~1', { cwd: this.workspaceRoot, stdio: 'pipe' });
        
        // Redeploy previous version
        execSync('vercel --prod --confirm', { cwd: this.workspaceRoot, stdio: 'pipe' });
      }

      console.log('✅ Rollback completed successfully');
      return true;
    } catch (error) {
      console.error('❌ Rollback failed:', error);
      return false;
    }
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// CLI interface
if (require.main === module) {
  const executors = new DeploymentExecutors();
  console.log('🚀 OPS Agent Deployment Executors ready');
}