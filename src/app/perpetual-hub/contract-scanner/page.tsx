'use client';

/**
 * 🔒 SMART CONTRACT RISK SCANNER
 *
 * DeFi perpetual protocol security and risk assessment
 *
 * Features:
 * - Smart contract audit score analysis
 * - TVL and liquidity risk assessment
 * - Rug pull probability detection
 * - Oracle manipulation risk
 * - Protocol insurance coverage
 * - Historical exploit tracking
 */

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { SharedSidebar } from '@/components/SharedSidebar';
import { AIAssistantFullScreen } from '@/components/AIAssistantFullScreen';
import { PWAProvider } from '@/components/PWAProvider';
import { Icons } from '@/components/Icons';

interface ContractRisk {
  protocol: string;
  chain: string;
  contractAddress: string;
  auditScore: number; // 0-100
  auditedBy: string[];
  tvl: number;
  dailyVolume: number;
  rugPullRisk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risks: {
    category: string;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    description: string;
  }[];
  insurance: {
    covered: boolean;
    provider?: string;
    coverage?: number;
  };
  exploitHistory: {
    date: string;
    amount: number;
    resolved: boolean;
  }[];
  deployedDays: number;
  adminPrivileges: 'NONE' | 'LIMITED' | 'EXTENSIVE';
  timeLockHours: number;
  oracleType: string;
  liquidationMechanism: string;
}

export default function ContractScanner() {
  const [aiAssistantOpen, setAiAssistantOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [selectedChain, setSelectedChain] = useState('ALL');
  const [minAuditScore, setMinAuditScore] = useState('70');
  const [showOnlyInsured, setShowOnlyInsured] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Mock data - in production, this would integrate with security APIs
  const protocols: ContractRisk[] = [
    {
      protocol: 'GMX',
      chain: 'Arbitrum',
      contractAddress: '0x489ee077994B6658eAfA855C308275EAd8097C4A',
      auditScore: 95,
      auditedBy: ['PeckShield', 'OpenZeppelin', 'Trail of Bits'],
      tvl: 450000000,
      dailyVolume: 85000000,
      rugPullRisk: 'LOW',
      risks: [
        { category: 'Oracle Risk', severity: 'LOW', description: 'Chainlink oracles with backup feeds' },
        { category: 'Liquidity Risk', severity: 'LOW', description: 'Deep liquidity pool (>$450M TVL)' },
      ],
      insurance: {
        covered: true,
        provider: 'Nexus Mutual',
        coverage: 5000000,
      },
      exploitHistory: [],
      deployedDays: 850,
      adminPrivileges: 'LIMITED',
      timeLockHours: 48,
      oracleType: 'Chainlink + Backup',
      liquidationMechanism: 'GLP Pool',
    },
    {
      protocol: 'dYdX v4',
      chain: 'dYdX Chain',
      contractAddress: '0x...(Native Chain)',
      auditScore: 92,
      auditedBy: ['Trail of Bits', 'Consensys Diligence'],
      tvl: 380000000,
      dailyVolume: 950000000,
      rugPullRisk: 'LOW',
      risks: [
        { category: 'Chain Risk', severity: 'MEDIUM', description: 'App-specific chain, less battle-tested' },
        { category: 'Oracle Risk', severity: 'LOW', description: 'Native oracle with validator consensus' },
      ],
      insurance: {
        covered: true,
        provider: 'Unslashed Finance',
        coverage: 10000000,
      },
      exploitHistory: [],
      deployedDays: 180,
      adminPrivileges: 'NONE',
      timeLockHours: 0,
      oracleType: 'Validator Consensus',
      liquidationMechanism: 'Insurance Fund',
    },
    {
      protocol: 'Vertex Protocol',
      chain: 'Arbitrum',
      contractAddress: '0x59715...',
      auditScore: 88,
      auditedBy: ['Zellic', 'OpenZeppelin'],
      tvl: 125000000,
      dailyVolume: 42000000,
      rugPullRisk: 'LOW',
      risks: [
        { category: 'Centralization', severity: 'MEDIUM', description: 'Sequencer centralization risk' },
        { category: 'Oracle Risk', severity: 'LOW', description: 'Pyth Network with Chainlink backup' },
      ],
      insurance: {
        covered: false,
      },
      exploitHistory: [],
      deployedDays: 420,
      adminPrivileges: 'LIMITED',
      timeLockHours: 24,
      oracleType: 'Pyth + Chainlink',
      liquidationMechanism: 'Automated Market Maker',
    },
    {
      protocol: 'Gains Network',
      chain: 'Polygon',
      contractAddress: '0x5f0...',
      auditScore: 85,
      auditedBy: ['PeckShield'],
      tvl: 65000000,
      dailyVolume: 28000000,
      rugPullRisk: 'MEDIUM',
      risks: [
        { category: 'Oracle Risk', severity: 'MEDIUM', description: 'Single oracle dependency (Chainlink)' },
        { category: 'Liquidity Risk', severity: 'MEDIUM', description: 'DAI vault liquidity constraints' },
        { category: 'Admin Risk', severity: 'MEDIUM', description: 'Multisig with limited timelock' },
      ],
      insurance: {
        covered: true,
        provider: 'InsurAce',
        coverage: 2000000,
      },
      exploitHistory: [
        { date: '2023-04-12', amount: 1200000, resolved: true },
      ],
      deployedDays: 950,
      adminPrivileges: 'EXTENSIVE',
      timeLockHours: 12,
      oracleType: 'Chainlink Only',
      liquidationMechanism: 'DAI Vault',
    },
    {
      protocol: 'Level Finance',
      chain: 'BNB Chain',
      contractAddress: '0x98a...',
      auditScore: 68,
      auditedBy: ['HashEx'],
      tvl: 18000000,
      dailyVolume: 8500000,
      rugPullRisk: 'HIGH',
      risks: [
        { category: 'Audit Quality', severity: 'HIGH', description: 'Single audit by tier-2 firm' },
        { category: 'Liquidity Risk', severity: 'HIGH', description: 'Low TVL relative to open interest' },
        { category: 'Admin Risk', severity: 'CRITICAL', description: 'No timelock, single admin wallet' },
        { category: 'Oracle Risk', severity: 'MEDIUM', description: 'BNB Chain oracle risks' },
      ],
      insurance: {
        covered: false,
      },
      exploitHistory: [
        { date: '2024-01-05', amount: 2800000, resolved: false },
      ],
      deployedDays: 280,
      adminPrivileges: 'EXTENSIVE',
      timeLockHours: 0,
      oracleType: 'Binance Oracle',
      liquidationMechanism: 'LLP Pool',
    },
  ];

  const filteredProtocols = protocols.filter((p) => {
    if (selectedChain !== 'ALL' && p.chain !== selectedChain) return false;
    if (p.auditScore < parseInt(minAuditScore)) return false;
    if (showOnlyInsured && !p.insurance.covered) return false;
    return true;
  });

  const getRiskColor = (risk: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'): string => {
    switch (risk) {
      case 'LOW': return '#10B981';
      case 'MEDIUM': return '#F59E0B';
      case 'HIGH': return '#EF4444';
      case 'CRITICAL': return '#991B1B';
    }
  };

  const getAuditScoreColor = (score: number): string => {
    if (score >= 90) return '#10B981';
    if (score >= 80) return '#3B82F6';
    if (score >= 70) return '#F59E0B';
    return '#EF4444';
  };

  if (!mounted) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.6)' }}>Yükleniyor...</div>
      </div>
    );
  }

  return (
    <PWAProvider>
      <div
        suppressHydrationWarning
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
          paddingTop: '60px',
        }}
      >
        <SharedSidebar
          currentPage="perpetual-hub"
          onAiAssistantOpen={() => setAiAssistantOpen(true)}
        />

        <main style={{ maxWidth: '1600px', margin: '0 auto', padding: '40px 24px', paddingTop: '80px' }}>
          <div style={{ marginBottom: '32px' }}>
            <Link
              href="/perpetual-hub"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                color: 'rgba(255, 255, 255, 0.6)',
                textDecoration: 'none',
                fontSize: '14px',
                marginBottom: '12px',
              }}
            >
              <Icons.ArrowLeft style={{ width: '16px', height: '16px' }} />
              Perpetual Hub'a Dön
            </Link>

            <h1
              style={{
                fontSize: '40px',
                fontWeight: '900',
                background: 'linear-gradient(135deg, #DC2626 0%, #991B1B 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '8px',
              }}
            >
              Smart Contract Risk Scanner
            </h1>

            <p style={{ fontSize: '16px', color: 'rgba(255, 255, 255, 0.6)' }}>
              DeFi perpetual protokollerinin akıllı kontrat güvenlik analizi ve risk değerlendirmesi
            </p>
          </div>

          {/* Filters */}
          <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', flexWrap: 'wrap', alignItems: 'center' }}>
            <select
              value={selectedChain}
              onChange={(e) => setSelectedChain(e.target.value)}
              style={{
                padding: '12px 16px',
                background: 'rgba(255, 255, 255, 0.05)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '8px',
                color: '#FFFFFF',
                fontSize: '14px',
                fontWeight: '600',
              }}
            >
              <option value="ALL">Tüm Zincirler</option>
              <option value="Arbitrum">Arbitrum</option>
              <option value="Polygon">Polygon</option>
              <option value="BNB Chain">BNB Chain</option>
              <option value="dYdX Chain">dYdX Chain</option>
            </select>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Min Audit Score:
              </label>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={minAuditScore}
                onChange={(e) => setMinAuditScore(e.target.value)}
                style={{ width: '150px' }}
              />
              <span style={{ fontSize: '14px', fontWeight: '700', color: '#FFFFFF', width: '30px' }}>
                {minAuditScore}
              </span>
            </div>

            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={showOnlyInsured}
                onChange={(e) => setShowOnlyInsured(e.target.checked)}
                style={{ width: '18px', height: '18px' }}
              />
              <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>
                Sadece Sigortalı Protokoller
              </span>
            </label>

            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Sonuç: <span style={{ fontWeight: '700', color: '#FFFFFF' }}>{filteredProtocols.length}</span> protokol
              </div>
            </div>
          </div>

          {/* Risk Summary Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px', marginBottom: '32px' }}>
            <div style={{
              padding: '20px',
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                ✅ DÜŞÜK RİSK
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#10B981' }}>
                {filteredProtocols.filter(p => p.rugPullRisk === 'LOW').length}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Audit Score &gt; 85, Sigortalı
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(245, 158, 11, 0.1)',
              border: '1px solid rgba(245, 158, 11, 0.3)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                ⚠️ ORTA RİSK
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#F59E0B' }}>
                {filteredProtocols.filter(p => p.rugPullRisk === 'MEDIUM').length}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Dikkatli kullanım önerilir
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                🚨 YÜKSEK RİSK
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#EF4444' }}>
                {filteredProtocols.filter(p => p.rugPullRisk === 'HIGH' || p.rugPullRisk === 'CRITICAL').length}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Kullanım önerilmez
              </div>
            </div>

            <div style={{
              padding: '20px',
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: '12px',
            }}>
              <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                🛡️ SİGORTALI
              </div>
              <div style={{ fontSize: '32px', fontWeight: '900', color: '#3B82F6' }}>
                {filteredProtocols.filter(p => p.insurance.covered).length}
              </div>
              <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                Toplam kapsam: $
                {(filteredProtocols.reduce((sum, p) => sum + (p.insurance.coverage || 0), 0) / 1000000).toFixed(1)}M
              </div>
            </div>
          </div>

          {/* Protocol Cards */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {filteredProtocols.map((protocol, index) => (
              <div
                key={index}
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: `2px solid ${getRiskColor(protocol.rugPullRisk)}`,
                  borderRadius: '16px',
                  padding: '24px',
                }}
              >
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: '20px' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <h3 style={{ fontSize: '24px', fontWeight: '900', color: '#FFFFFF' }}>
                        {protocol.protocol}
                      </h3>
                      <span style={{
                        padding: '4px 12px',
                        background: 'rgba(255, 255, 255, 0.1)',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '600',
                        color: 'rgba(255, 255, 255, 0.8)',
                      }}>
                        {protocol.chain}
                      </span>
                      {protocol.insurance.covered && (
                        <span style={{
                          padding: '4px 12px',
                          background: 'rgba(59, 130, 246, 0.2)',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: '600',
                          color: '#3B82F6',
                        }}>
                          🛡️ Sigortalı
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', fontFamily: 'monospace' }}>
                      {protocol.contractAddress}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                      AUDIT SCORE
                    </div>
                    <div style={{ fontSize: '48px', fontWeight: '900', color: getAuditScoreColor(protocol.auditScore), lineHeight: '1' }}>
                      {protocol.auditScore}
                    </div>
                    <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
                      /100
                    </div>
                  </div>
                </div>

                {/* Stats Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px', marginBottom: '20px' }}>
                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>TVL</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                      ${(protocol.tvl / 1000000).toFixed(1)}M
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>24h Volume</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                      ${(protocol.dailyVolume / 1000000).toFixed(1)}M
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Rug Pull Risk</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: getRiskColor(protocol.rugPullRisk) }}>
                      {protocol.rugPullRisk}
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Deployed</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF' }}>
                      {protocol.deployedDays} days
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>TimeLock</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: protocol.timeLockHours >= 24 ? '#10B981' : '#EF4444' }}>
                      {protocol.timeLockHours}h
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.5)', marginBottom: '4px' }}>Admin Rights</div>
                    <div style={{ fontSize: '16px', fontWeight: '700', color: protocol.adminPrivileges === 'NONE' ? '#10B981' : protocol.adminPrivileges === 'LIMITED' ? '#F59E0B' : '#EF4444' }}>
                      {protocol.adminPrivileges}
                    </div>
                  </div>
                </div>

                {/* Audited By */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                    Audited By:
                  </div>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {protocol.auditedBy.map((auditor, i) => (
                      <span key={i} style={{
                        padding: '6px 12px',
                        background: 'rgba(16, 185, 129, 0.1)',
                        border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '600',
                        color: '#10B981',
                      }}>
                        {auditor}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Risks */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                    Identified Risks:
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {protocol.risks.map((risk, i) => (
                      <div key={i} style={{
                        padding: '12px',
                        background: 'rgba(0, 0, 0, 0.2)',
                        borderLeft: `3px solid ${getRiskColor(risk.severity)}`,
                        borderRadius: '6px',
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ fontSize: '13px', fontWeight: '700', color: '#FFFFFF' }}>
                            {risk.category}
                          </span>
                          <span style={{ fontSize: '11px', fontWeight: '600', color: getRiskColor(risk.severity) }}>
                            {risk.severity}
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)' }}>
                          {risk.description}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Insurance & Exploits */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                  {protocol.insurance.covered && (
                    <div style={{
                      padding: '16px',
                      background: 'rgba(59, 130, 246, 0.1)',
                      border: '1px solid rgba(59, 130, 246, 0.3)',
                      borderRadius: '8px',
                    }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                        🛡️ Insurance Coverage
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: '700', color: '#3B82F6', marginBottom: '4px' }}>
                        ${(protocol.insurance.coverage! / 1000000).toFixed(1)}M
                      </div>
                      <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                        Provider: {protocol.insurance.provider}
                      </div>
                    </div>
                  )}

                  {protocol.exploitHistory.length > 0 && (
                    <div style={{
                      padding: '16px',
                      background: 'rgba(239, 68, 68, 0.1)',
                      border: '1px solid rgba(239, 68, 68, 0.3)',
                      borderRadius: '8px',
                    }}>
                      <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px' }}>
                        ⚠️ Exploit History
                      </div>
                      {protocol.exploitHistory.map((exploit, i) => (
                        <div key={i} style={{ marginBottom: '8px' }}>
                          <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444' }}>
                            ${(exploit.amount / 1000000).toFixed(2)}M
                          </div>
                          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)' }}>
                            {exploit.date} - {exploit.resolved ? '✅ Resolved' : '❌ Unresolved'}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </main>

        {aiAssistantOpen && (
          <AIAssistantFullScreen isOpen={aiAssistantOpen} onClose={() => setAiAssistantOpen(false)} />
        )}
      </div>
    </PWAProvider>
  );
}
