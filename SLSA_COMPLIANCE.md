# SLSA Compliance Documentation

This project implements SLSA (Supply-chain Levels for Software Artifacts) Level 3 compliance to ensure supply chain security.

## SLSA Level 3 Requirements Met

### Build Requirements
- ✅ **Scripted build**: Automated via GitHub Actions
- ✅ **Build service**: GitHub Actions hosted runners
- ✅ **Ephemeral environment**: Fresh containers for each build
- ✅ **Isolated**: Separate build environments per job

### Provenance Requirements  
- ✅ **Available**: Generated automatically via `slsa-github-generator`
- ✅ **Authenticated**: Signed with GitHub OIDC tokens
- ✅ **Service generated**: Created by trusted GitHub Actions
- ✅ **Non-falsifiable**: Cryptographically signed

### Common Requirements
- ✅ **Security**: Vulnerability scanning in CI/CD
- ✅ **Access**: Protected branches with required reviews
- ✅ **Superuser**: Admin access restricted and audited

## Implementation Details

### Provenance Generation
Our release workflow generates SLSA provenance using the official SLSA GitHub Generator:

```yaml
generate-provenance:
  needs: [build]
  permissions:
    actions: read
    id-token: write
    contents: write
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
```

### Verification
To verify package provenance:

```bash
# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify downloaded package
slsa-verifier verify-artifact \
  --provenance-path liquid-edge-0.1.0.intoto.jsonl \
  --source-uri github.com/your-org/liquid-edge \
  liquid-edge-0.1.0.tar.gz
```

### Security Scanning
- **Container scanning**: Trivy for vulnerability detection
- **Dependency scanning**: Safety and Bandit for Python security
- **Code analysis**: CodeQL for static analysis
- **SBOM generation**: Automated software bill of materials

## Repository Configuration

### Required Secrets
- `PYPI_API_TOKEN`: For publishing to PyPI (set as repository secret)

### Branch Protection
- Required status checks: CI, Security scanning
- Required reviews: 1+ approving reviews
- Admin override: Disabled for security

### Attestations
Release artifacts include cryptographic attestations for:
- Build provenance
- Vulnerability scan results  
- SBOM data
- Code signing information