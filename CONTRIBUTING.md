# Contributing to LyTrade Scanner

Thank you for your interest in contributing! This document provides guidelines for contributing to LyTrade Scanner.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/lytrade-scanner.git`
3. Install dependencies: `pnpm install`
4. Copy `.env.example` to `.env.local` and configure your API keys
5. Start the development server: `pnpm dev`

## Development Setup

### Prerequisites
- Node.js 20+
- pnpm 8+
- PostgreSQL (optional, for full features)
- Redis (optional, for caching)

### AI Provider Setup
The app works with any OpenAI-compatible API. Free options:
- **Groq** (recommended): Get a free key at https://console.groq.com
- **Ollama** (local): Run `ollama serve` with any model

## How to Contribute

### Reporting Bugs
- Use [GitHub Issues](https://github.com/lydianai/borsa.ailydian.com/issues)
- Include steps to reproduce, expected vs actual behavior
- Include browser/OS/Node.js version

### Suggesting Features
- Open a GitHub Issue with the "enhancement" label
- Describe the use case and expected behavior

### Submitting Pull Requests
1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure TypeScript compiles: `pnpm typecheck`
4. Test your changes locally
5. Submit a PR with a clear description

## Code Style

- TypeScript strict mode
- Functional components with hooks (React)
- Tailwind CSS for styling
- Meaningful variable/function names
- Comments for complex logic only

## Project Structure

```
src/
  app/          # Next.js App Router pages and API routes
  components/   # React components
  lib/          # Utility libraries and services
  types/        # TypeScript type definitions
```

## Commit Messages

Use conventional commit format:
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code refactoring
- `docs:` Documentation
- `chore:` Maintenance

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
