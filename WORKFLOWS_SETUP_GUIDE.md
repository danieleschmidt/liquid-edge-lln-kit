# GitHub Actions Workflows Setup Guide

## Overview

This repository includes 7 comprehensive GitHub Actions workflows that enhance the SDLC with advanced automation. Due to GitHub App permissions, these workflows need to be manually added to the repository.

## Workflow Files to Add

The following workflow files are ready in the `.github/workflows/` directory:

### 1. `ci.yml` - Comprehensive CI Pipeline
**Purpose**: Multi-platform testing, linting, security checks, and build validation

**Features**:
- Matrix testing across Python 3.10-3.12 and multiple OS
- Code quality checks (ruff, black, myspy)
- Security scanning with CodeQL
- Coverage reporting with Codecov
- Integration testing with Docker Compose

**Triggers**: Push to main/develop, pull requests to main

### 2. `security.yml` - Advanced Security Analysis
**Purpose**: Comprehensive security scanning and vulnerability detection

**Features**:
- Dependency vulnerability scanning (Safety, Bandit)
- Static application security testing (SAST) with CodeQL
- Container security scanning with Trivy
- Secrets detection with TruffleHog
- SBOM generation with CycloneDx
- OpenSSF Scorecard analysis

**Triggers**: Push to main, pull requests, weekly schedule, manual dispatch

### 3. `performance.yml` - Performance Testing & Monitoring  
**Purpose**: Automated performance testing and regression detection

**Features**:
- Benchmark execution with pytest-benchmark
- Memory leak detection
- Performance regression analysis
- Results comparison with baseline
- Performance history tracking

**Triggers**: Push to main (src changes), pull requests, daily schedule, manual dispatch

### 4. `dependency-update.yml` - Automated Dependency Management
**Purpose**: Automated dependency updates with security validation

**Features**:
- Automated dependency updates (patch/minor/major)
- Vulnerability scanning before updates
- License compliance checking
- Automated pull request creation
- Security advisory monitoring

**Triggers**: Weekly schedule, manual dispatch with update type selection

### 5. `release.yml` - Automated Release Management
**Purpose**: Comprehensive release workflow with validation

**Features**:
- Version validation and build testing
- Automated changelog generation
- GitHub release creation with artifacts
- PyPI publication with verification
- Docker image publishing
- Post-release automation

**Triggers**: Git tags (v*.*.*), manual dispatch

### 6. `docker.yml` - Container CI/CD
**Purpose**: Docker image building, testing, and security scanning

**Features**:
- Multi-architecture builds (AMD64, ARM64)
- Container functionality testing
- Security scanning with Trivy
- Automated image cleanup
- Registry publishing to GHCR

**Triggers**: Push to main/develop, pull requests, workflow dispatch

### 7. `docs.yml` - Documentation Pipeline
**Purpose**: Automated documentation building and deployment

**Features**:
- Sphinx documentation building
- API documentation generation
- Link checking and validation
- GitHub Pages deployment
- Example code validation
- Spelling and accessibility checks

**Triggers**: Push to main (docs changes), pull requests, manual dispatch

## Setup Instructions

### Step 1: Add Workflow Files
Copy all files from `.github/workflows/` to your repository's `.github/workflows/` directory:

```bash
# If you have the workflows locally
cp .github/workflows/*.yml /path/to/your/repo/.github/workflows/

# Or create them manually using the content provided
```

### Step 2: Configure Repository Secrets
Add the following secrets in your repository settings:

#### Required Secrets
- `PYPI_API_TOKEN` - For PyPI package publishing
- `CODECOV_TOKEN` - For coverage reporting (optional but recommended)

#### Optional Secrets  
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `TEAMS_WEBHOOK_URL` - For Microsoft Teams notifications

### Step 3: Configure Repository Settings

#### Branch Protection Rules
Enable branch protection for `main` branch with:
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Required status checks:
  - `Test Suite (ubuntu-latest, 3.10)`
  - `Code Quality` 
  - `Security Scan`
  - `Build Package`

#### GitHub Actions Permissions
Ensure GitHub Actions has the following permissions:
- **Actions**: Write (for workflow runs)
- **Contents**: Write (for creating releases)  
- **Packages**: Write (for container registry)
- **Security Events**: Write (for security scanning)
- **Pages**: Write (for documentation deployment)

### Step 4: Configure GitHub Pages
1. Go to Settings → Pages
2. Set Source to "GitHub Actions"
3. The `docs.yml` workflow will handle deployment

### Step 5: Enable Dependabot
The enhanced `.github/dependabot.yml` is already configured. Ensure Dependabot is enabled in repository settings.

## Workflow Dependencies

### External Services Integration

#### Codecov (Optional)
- Sign up at codecov.io
- Install Codecov GitHub App
- Add `CODECOV_TOKEN` secret

#### Container Registry
- GitHub Container Registry (GHCR) is used by default
- No additional setup required for public repositories

### Team Configuration

#### Code Owners
Update `.github/CODEOWNERS` to match your team structure:
```
# Global ownership
* @your-org/maintainers

# Core implementation  
/src/ @your-org/core-team

# CI/CD and infrastructure
/.github/ @your-org/devops-team
```

#### Issue Templates
The enhanced issue templates are already configured in `.github/ISSUE_TEMPLATE/`.

## Monitoring Integration

### Prometheus Metrics
The workflows expose metrics for:
- Build success/failure rates
- Test execution times  
- Security scan results
- Performance benchmarks

### Alerting
Configure alerts for:
- Failed builds on main branch
- High security vulnerability counts
- Performance regressions
- Dependency update failures

## Customization Options

### Environment-Specific Configuration
Each workflow supports customization through:
- Environment variables
- Workflow inputs
- Matrix strategies
- Conditional execution

### Organization-Specific Adaptations
Modify workflows for your organization:
- Change notification channels
- Adjust security thresholds
- Customize performance benchmarks
- Add organization-specific compliance checks

## Troubleshooting

### Common Issues

#### Workflow Permission Errors
```
Error: Resource not accessible by integration
```
**Solution**: Check repository permissions and ensure GitHub Actions has required access.

#### Secret Not Found Errors
```
Error: Secret PYPI_API_TOKEN not found
```
**Solution**: Add the required secret in repository settings.

#### Matrix Build Failures
```
Error: Matrix job failed for python-version 3.11
```
**Solution**: Check Python version compatibility and update matrix configuration.

### Debugging Workflows
1. Enable debug logging by setting `ACTIONS_STEP_DEBUG` secret to `true`
2. Review workflow run logs in the Actions tab
3. Test workflows with manual dispatch for debugging
4. Use workflow status badges in README for visibility

## Security Considerations

### Workflow Security
- All workflows use pinned action versions
- Secrets are properly scoped and masked
- Third-party actions are from trusted sources
- Dependency scanning prevents supply chain attacks

### Token Management
- Use fine-grained personal access tokens when possible
- Regularly rotate secrets and tokens
- Monitor token usage in audit logs
- Apply principle of least privilege

## Support and Maintenance

### Regular Updates
- Monthly review of action versions
- Quarterly security audit of workflows
- Update documentation with changes
- Test workflows after major updates

### Community Support
- Document workflow issues in repository
- Share improvements with the community
- Contribute back to action repositories
- Maintain workflow examples and templates

---

## Next Steps

1. **Add the workflow files** to your repository's `.github/workflows/` directory
2. **Configure required secrets** in repository settings
3. **Enable branch protection** with required status checks
4. **Test workflows** with a test pull request
5. **Monitor workflow performance** and adjust as needed

These workflows transform your repository from MATURING to ADVANCED maturity level with enterprise-grade automation and security practices.

For questions or support, please refer to the repository's issue tracker or documentation.