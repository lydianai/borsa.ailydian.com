# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| < Latest | No       |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability, please report it responsibly.

**DO NOT** open a public issue for security vulnerabilities.

### How to Report

1. Open a **private security advisory** on GitHub: [Report a vulnerability](https://github.com/lydianai/borsa.ailydian.com/security/advisories/new)
2. Or email the maintainers directly (see GitHub profile)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | 24 hours |
| Initial Assessment | 48 hours |
| Status Update | 7 days |
| Resolution Target | 30 days |

### Safe Harbor

We support responsible disclosure. Security researchers acting in good faith will not face legal action, provided they:

- Do not access, modify, or delete user data
- Do not disrupt services
- Report findings exclusively to the maintainers
- Allow reasonable time for remediation before disclosure

## Security Measures

- All data encrypted at rest (AES-256-GCM) and in transit (TLS 1.3)
- Regular dependency audits via Dependabot
- Automated SAST scanning via CodeQL
- Strict access controls and least-privilege principles
- Comprehensive audit logging

## Contact

Report security issues via [GitHub Security Advisories](https://github.com/lydianai/borsa.ailydian.com/security/advisories/new).

---

*LyTrade Scanner is open-source software licensed under the MIT License.*
