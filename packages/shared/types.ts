/**
 * OPS Agent Shared Types
 * 
 * Common types used across the OPS agent system
 */

export interface Issue {
  id: string;
  type: 'config' | 'strategy' | 'docs' | 'dependency' | 'performance' | 'security' | 'api';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  component?: string;
  metadata?: Record<string, any>;
  detectedAt: Date;
  resolvedAt?: Date;
}

export interface HealthCheck {
  component: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime?: number;
  error?: string;
  metadata?: Record<string, any>;
  checkedAt: Date;
}

export interface FixPlan {
  issueId: string;
  type: 'automatic' | 'manual' | 'investigate';
  priority: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  steps: string[];
  riskLevel: 'low' | 'medium' | 'high';
  estimatedDuration: number; // in minutes
  requiresApproval: boolean;
  rollbackPlan?: RollbackPlan;
}

export interface FixResult {
  success: boolean;
  appliedChanges?: string[];
  error?: string;
  rollbackPlan?: RollbackPlan;
}

export interface RollbackPlan {
  type: 'restore_file' | 'git_reset' | 'config_revert';
  target: string;
  backup?: string;
  commit?: string;
}

export interface SLO {
  name: string;
  target: number; // percentage
  current: number;
  window: string; // time window
  status: 'met' | 'breached' | 'at_risk';
}

export interface DeploymentPlan {
  id: string;
  type: 'canary' | 'blue_green' | 'rolling';
  changes: string[];
  riskLevel: 'low' | 'medium' | 'high';
  requiresApproval: boolean;
  rolloutPercentage?: number;
  rollbackPlan: RollbackPlan;
}

export interface OPSMetrics {
  timestamp: Date;
  issues: {
    total: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  fixes: {
    applied: number;
    failed: number;
    rolledBack: number;
  };
  deployments: {
    total: number;
    successful: number;
    failed: number;
    rolledBack: number;
  };
  slo: {
    availability: number;
    performance: number;
    errorRate: number;
  };
}

export interface OPSConfig {
  scheduler: {
    enabled: boolean;
    interval: string; // cron expression
    timezone: string;
  };
  checks: {
    httpTimeout: number;
    wsTimeout: number;
    retryAttempts: number;
  };
  fixes: {
    dryRun: boolean;
    requireApproval: boolean;
    maxRiskLevel: 'low' | 'medium' | 'high';
  };
  deployments: {
    autoDeploy: boolean;
    canaryPercentage: number;
    rollbackThreshold: number;
  };
  notifications: {
    enabled: boolean;
    channels: string[];
    webhook?: string;
  };
}

export interface ChangeRequest {
  id: string;
  type: 'fix' | 'feature' | 'config' | 'deployment';
  title: string;
  description: string;
  author: string;
  status: 'pending' | 'approved' | 'rejected' | 'applied' | 'rolled_back';
  riskLevel: 'low' | 'medium' | 'high';
  changes: string[];
  approvals: string[];
  createdAt: Date;
  appliedAt?: Date;
}

export interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  component?: string;
  metadata?: Record<string, any>;
  createdAt: Date;
  resolvedAt?: Date;
}

export interface OPSReport {
  id: string;
  period: {
    start: Date;
    end: Date;
  };
  summary: {
    totalIssues: number;
    resolvedIssues: number;
    successfulDeployments: number;
    sloMet: number;
    uptime: number;
  };
  details: {
    issues: Issue[];
    deployments: DeploymentPlan[];
    metrics: OPSMetrics[];
    alerts: Alert[];
  };
  generatedAt: Date;
}