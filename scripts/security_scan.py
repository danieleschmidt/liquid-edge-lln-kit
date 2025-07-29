#!/usr/bin/env python3
"""
Comprehensive security scanning script for Liquid Edge LLN Kit.
Integrates multiple security tools and generates unified reports.
"""

import json
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security finding from any scanner."""
    tool: str
    severity: str
    category: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass
class SecurityReport:
    """Consolidated security report."""
    timestamp: str
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    findings: List[SecurityFinding]
    tool_results: Dict[str, Any]


class SecurityScanner:
    """Orchestrates multiple security scanning tools."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.findings: List[SecurityFinding] = []
        self.tool_results: Dict[str, Any] = {}
    
    def run_bandit_scan(self) -> List[SecurityFinding]:
        """Run Bandit static analysis for Python security issues."""
        logger.info("Running Bandit security scan...")
        
        try:
            result = subprocess.run([
                'bandit', '-r', str(self.project_root / 'src'),
                '-f', 'json',
                '-o', '/tmp/bandit_results.json'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode not in [0, 1]:  # Bandit returns 1 when issues found
                logger.error(f"Bandit scan failed: {result.stderr}")
                return []
            
            with open('/tmp/bandit_results.json', 'r') as f:
                bandit_data = json.load(f)
            
            self.tool_results['bandit'] = bandit_data
            
            findings = []
            for issue in bandit_data.get('results', []):
                finding = SecurityFinding(
                    tool='bandit',
                    severity=issue['issue_severity'].lower(),
                    category='static_analysis',
                    title=issue['test_name'],
                    description=issue['issue_text'],
                    file_path=issue['filename'],
                    line_number=issue['line_number'],
                    cwe_id=issue.get('issue_cwe', {}).get('id')
                )
                findings.append(finding)
            
            logger.info(f"Bandit found {len(findings)} security issues")
            return findings
        
        except FileNotFoundError:
            logger.warning("Bandit not installed, skipping static analysis")
            return []
        except Exception as e:
            logger.error(f"Bandit scan error: {e}")
            return []
    
    def run_safety_scan(self) -> List[SecurityFinding]:
        """Run Safety scan for known vulnerable dependencies."""
        logger.info("Running Safety dependency scan...")
        
        try:
            result = subprocess.run([
                'safety', 'check', '--json',
                '--file', str(self.project_root / 'requirements.txt')
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info("Safety found no vulnerable dependencies")
                return []
            
            safety_data = json.loads(result.stdout)
            self.tool_results['safety'] = safety_data
            
            findings = []
            for vuln in safety_data:
                finding = SecurityFinding(
                    tool='safety',
                    severity='high',  # All Safety findings are high severity
                    category='dependency_vulnerability',
                    title=f"Vulnerable dependency: {vuln['package_name']}",
                    description=vuln['advisory'],
                    cve_id=vuln.get('cve'),
                    cvss_score=vuln.get('cvss_score')
                )
                findings.append(finding)
            
            logger.info(f"Safety found {len(findings)} vulnerable dependencies")
            return findings
        
        except FileNotFoundError:
            logger.warning("Safety not installed, skipping dependency scan")
            return []
        except Exception as e:
            logger.error(f"Safety scan error: {e}")
            return []
    
    def run_pip_audit_scan(self) -> List[SecurityFinding]:
        """Run pip-audit for comprehensive dependency vulnerability scanning."""
        logger.info("Running pip-audit dependency scan...")
        
        try:
            result = subprocess.run([
                'pip-audit', '--format=json', '--desc'
            ], capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info("pip-audit found no vulnerabilities")
                return []
            
            audit_data = json.loads(result.stdout)
            self.tool_results['pip_audit'] = audit_data
            
            findings = []
            for vuln in audit_data.get('vulnerabilities', []):
                finding = SecurityFinding(
                    tool='pip-audit',
                    severity=self._map_cvss_to_severity(vuln.get('fix_versions', [])),
                    category='dependency_vulnerability',
                    title=f"CVE in {vuln['package']}",
                    description=vuln.get('description', 'No description available'),
                    cve_id=vuln.get('id')
                )
                findings.append(finding)
            
            logger.info(f"pip-audit found {len(findings)} vulnerabilities")
            return findings
        
        except FileNotFoundError:
            logger.warning("pip-audit not installed, skipping enhanced dependency scan")
            return []
        except Exception as e:
            logger.error(f"pip-audit scan error: {e}")
            return []
    
    def run_semgrep_scan(self) -> List[SecurityFinding]:
        """Run Semgrep for advanced static analysis."""
        logger.info("Running Semgrep security analysis...")
        
        try:
            result = subprocess.run([
                'semgrep', '--config=auto',
                '--json',
                '--output=/tmp/semgrep_results.json',
                str(self.project_root / 'src')
            ], capture_output=True, text=True, check=False)
            
            with open('/tmp/semgrep_results.json', 'r') as f:
                semgrep_data = json.load(f)
            
            self.tool_results['semgrep'] = semgrep_data
            
            findings = []
            for result in semgrep_data.get('results', []):
                finding = SecurityFinding(
                    tool='semgrep',
                    severity=result.get('extra', {}).get('severity', 'medium').lower(),
                    category='static_analysis',
                    title=result.get('check_id', 'Unknown check'),
                    description=result.get('extra', {}).get('message', 'No description'),
                    file_path=result.get('path'),
                    line_number=result.get('start', {}).get('line')
                )
                findings.append(finding)
            
            logger.info(f"Semgrep found {len(findings)} security issues")
            return findings
        
        except FileNotFoundError:
            logger.warning("Semgrep not installed, skipping advanced static analysis")
            return []
        except Exception as e:
            logger.error(f"Semgrep scan error: {e}")
            return []
    
    def run_license_check(self) -> List[SecurityFinding]:
        """Check for problematic licenses in dependencies."""
        logger.info("Running license compatibility check...")
        
        # Define problematic licenses
        problematic_licenses = [
            'GPL-3.0', 'AGPL-3.0', 'CPAL-1.0', 'OSL-3.0'
        ]
        
        try:
            result = subprocess.run([
                'pip-licenses', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            licenses_data = json.loads(result.stdout)
            self.tool_results['licenses'] = licenses_data
            
            findings = []
            for package in licenses_data:
                license_name = package.get('License', 'Unknown')
                if any(prob in license_name for prob in problematic_licenses):
                    finding = SecurityFinding(
                        tool='license_check',
                        severity='medium',
                        category='license_compliance',
                        title=f"Problematic license: {package['Name']}",
                        description=f"Package {package['Name']} uses {license_name} which may have compliance issues"
                    )
                    findings.append(finding)
            
            logger.info(f"License check found {len(findings)} issues")
            return findings
        
        except FileNotFoundError:
            logger.warning("pip-licenses not installed, skipping license check")
            return []
        except Exception as e:
            logger.error(f"License check error: {e}")
            return []
    
    def run_secrets_scan(self) -> List[SecurityFinding]:
        """Scan for hardcoded secrets and credentials."""
        logger.info("Running secrets detection scan...")
        
        # Simple regex patterns for common secrets
        secret_patterns = {
            'api_key': r'(?i)api[_-]?key[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9]{20,}',
            'password': r'(?i)password[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9!@#$%^&*]{8,}',
            'private_key': r'-----BEGIN [A-Z ]+PRIVATE KEY-----',
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'jwt_token': r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*'
        }
        
        findings = []
        
        # Scan Python files
        for py_file in self.project_root.rglob('*.py'):
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for secret_type, pattern in secret_patterns.items():
                    import re
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        finding = SecurityFinding(
                            tool='secrets_scan',
                            severity='high',
                            category='secrets_exposure',
                            title=f"Potential {secret_type} exposure",
                            description=f"Potential hardcoded {secret_type} detected",
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=line_num
                        )
                        findings.append(finding)
            
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
        
        logger.info(f"Secrets scan found {len(findings)} potential exposures")
        return findings
    
    def _map_cvss_to_severity(self, cvss_score: Optional[float]) -> str:
        """Map CVSS score to severity level."""
        if cvss_score is None:
            return 'medium'
        
        if cvss_score >= 9.0:
            return 'critical'
        elif cvss_score >= 7.0:
            return 'high'
        elif cvss_score >= 4.0:
            return 'medium'
        else:
            return 'low'
    
    def generate_sarif_report(self, findings: List[SecurityFinding]) -> dict:
        """Generate SARIF format report for GitHub integration."""
        sarif_report = {
            "version": "2.1.0",
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "runs": []
        }
        
        # Group findings by tool
        tool_findings = {}
        for finding in findings:
            if finding.tool not in tool_findings:
                tool_findings[finding.tool] = []
            tool_findings[finding.tool].append(finding)
        
        for tool, tool_findings_list in tool_findings.items():
            run = {
                "tool": {
                    "driver": {
                        "name": tool,
                        "version": "latest"
                    }
                },
                "results": []
            }
            
            for finding in tool_findings_list:
                result = {
                    "ruleId": finding.category,
                    "message": {"text": finding.description},
                    "level": self._severity_to_sarif_level(finding.severity)
                }
                
                if finding.file_path and finding.line_number:
                    result["locations"] = [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.file_path},
                            "region": {"startLine": finding.line_number}
                        }
                    }]
                
                run["results"].append(result)
            
            sarif_report["runs"].append(run)
        
        return sarif_report
    
    def _severity_to_sarif_level(self, severity: str) -> str:
        """Convert severity to SARIF level."""
        mapping = {
            'critical': 'error',
            'high': 'error',
            'medium': 'warning',
            'low': 'note'
        }
        return mapping.get(severity, 'warning')
    
    def run_comprehensive_scan(self) -> SecurityReport:
        """Run all security scans and generate comprehensive report."""
        logger.info("Starting comprehensive security scan...")
        
        all_findings = []
        
        # Run all scanners
        all_findings.extend(self.run_bandit_scan())
        all_findings.extend(self.run_safety_scan())
        all_findings.extend(self.run_pip_audit_scan())
        all_findings.extend(self.run_semgrep_scan())
        all_findings.extend(self.run_license_check())
        all_findings.extend(self.run_secrets_scan())
        
        # Count findings by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for finding in all_findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        report = SecurityReport(
            timestamp=subprocess.run(['date', '-Iseconds'], capture_output=True, text=True).stdout.strip(),
            total_findings=len(all_findings),
            critical_count=severity_counts['critical'],
            high_count=severity_counts['high'],
            medium_count=severity_counts['medium'],
            low_count=severity_counts['low'],
            findings=all_findings,
            tool_results=self.tool_results
        )
        
        logger.info(f"Security scan complete: {len(all_findings)} total findings")
        return report
    
    def save_reports(self, report: SecurityReport, output_dir: Path):
        """Save security reports in multiple formats."""
        output_dir.mkdir(exist_ok=True)
        
        # JSON report
        json_report = asdict(report)
        with open(output_dir / 'security_report.json', 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # SARIF report for GitHub
        sarif_report = self.generate_sarif_report(report.findings)
        with open(output_dir / 'security_report.sarif', 'w') as f:
            json.dump(sarif_report, f, indent=2)
        
        # Human-readable summary
        with open(output_dir / 'security_summary.txt', 'w') as f:
            f.write(f"Security Scan Summary - {report.timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Findings: {report.total_findings}\n")
            f.write(f"Critical: {report.critical_count}\n")
            f.write(f"High: {report.high_count}\n")
            f.write(f"Medium: {report.medium_count}\n")
            f.write(f"Low: {report.low_count}\n\n")
            
            if report.critical_count > 0 or report.high_count > 0:
                f.write("CRITICAL AND HIGH SEVERITY FINDINGS:\n")
                f.write("-" * 40 + "\n")
                for finding in report.findings:
                    if finding.severity in ['critical', 'high']:
                        f.write(f"[{finding.severity.upper()}] {finding.title}\n")
                        f.write(f"Tool: {finding.tool}\n")
                        if finding.file_path:
                            f.write(f"File: {finding.file_path}:{finding.line_number or 'N/A'}\n")
                        f.write(f"Description: {finding.description}\n")
                        f.write("\n")
        
        logger.info(f"Security reports saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive security scanning for Liquid Edge LLN Kit')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Project root directory')
    parser.add_argument('--output-dir', type=Path, default=Path('security_reports'),
                       help='Output directory for reports')
    parser.add_argument('--fail-on-high', action='store_true',
                       help='Fail with non-zero exit code if high/critical findings')
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.project_root)
    report = scanner.run_comprehensive_scan()
    scanner.save_reports(report, args.output_dir)
    
    # Print summary
    print(f"\nSecurity Scan Results:")
    print(f"Critical: {report.critical_count}")
    print(f"High: {report.high_count}")
    print(f"Medium: {report.medium_count}")
    print(f"Low: {report.low_count}")
    print(f"Total: {report.total_findings}")
    
    # Exit with error code if high severity findings and fail-on-high is set
    if args.fail_on_high and (report.critical_count > 0 or report.high_count > 0):
        print("\nExiting with error due to high/critical security findings")
        sys.exit(1)
    
    print(f"\nReports saved to: {args.output_dir}")


if __name__ == '__main__':
    main()