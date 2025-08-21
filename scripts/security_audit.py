#!/usr/bin/env python3
"""Security audit script for Liquid Edge LLN Kit."""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import re
import hashlib


class SecurityAuditor:
    """Comprehensive security audit for the liquid neural network codebase."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings = []
        self.severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    
    def add_finding(self, severity: str, category: str, description: str, 
                   file_path: str = None, line_number: int = None):
        """Add a security finding."""
        finding = {
            "severity": severity,
            "category": category,
            "description": description,
            "file": file_path,
            "line": line_number,
            "timestamp": __import__('time').time()
        }
        
        self.findings.append(finding)
        self.severity_counts[severity] = self.severity_counts.get(severity, 0) + 1
    
    def audit_dependencies(self):
        """Audit Python dependencies for known vulnerabilities."""
        print("\n🔍 Auditing Dependencies...")
        
        # Check if safety is available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("   ✅ No known vulnerabilities in dependencies")
                self.add_finding("INFO", "Dependencies", "No known vulnerabilities found")
            else:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    self.add_finding(
                        "HIGH",
                        "Dependencies",
                        f"Vulnerable dependency: {vuln.get('package')} - {vuln.get('vulnerability')}"
                    )
                    
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            self.add_finding("INFO", "Dependencies", "Could not run dependency check (safety not available)")
            print("   ⚠️  Safety tool not available, skipping dependency audit")
    
    def audit_code_patterns(self):
        """Audit code for insecure patterns."""
        print("\n🔍 Auditing Code Patterns...")
        
        dangerous_patterns = {
            r'eval\s*\(': ("HIGH", "Code injection risk - eval() usage"),
            r'exec\s*\(': ("HIGH", "Code injection risk - exec() usage"),
            r'os\.system\s*\(': ("HIGH", "Command injection risk - os.system() usage"),
            r'subprocess\.\w+\s*\([^)]*shell=True': ("MEDIUM", "Command injection risk - shell=True"),
            r'pickle\.load\s*\(': ("MEDIUM", "Deserialization risk - pickle.load() usage"),
            r'input\s*\([^)]*\)': ("LOW", "Input validation needed"),
            r'print\s*\([^)]*password': ("MEDIUM", "Potential password exposure in logs"),
            r'print\s*\([^)]*secret': ("MEDIUM", "Potential secret exposure in logs"),
            r'print\s*\([^)]*key': ("MEDIUM", "Potential key exposure in logs"),
            r'random\.random\s*\(\)': ("LOW", "Weak random number generation"),
            r'tempfile\.mktemp\s*\(': ("MEDIUM", "Insecure temp file creation"),
            r'assert\s+': ("INFO", "Assertion usage (disabled in production)"),
        }
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip test files and examples for some checks
            is_test_file = "test" in str(file_path) or "example" in str(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for pattern, (severity, description) in dangerous_patterns.items():
                        # Skip some patterns in test files
                        if is_test_file and pattern in [r'assert\s+', r'input\s*\([^)]*\)']:
                            continue
                            
                        for line_num, line in enumerate(lines, 1):
                            if re.search(pattern, line):
                                self.add_finding(
                                    severity,
                                    "Code Patterns",
                                    f"{description}: {line.strip()}",
                                    str(file_path.relative_to(self.project_root)),
                                    line_num
                                )
                                
            except Exception as e:
                self.add_finding(
                    "LOW",
                    "File Access",
                    f"Could not read file {file_path}: {str(e)}"
                )
    
    def audit_file_permissions(self):
        """Audit file permissions for security issues."""
        print("\n🔍 Auditing File Permissions...")
        
        sensitive_files = [
            "*.key", "*.pem", "*.p12", "*.pfx", "*.crt", "*.cert",
            "config.py", "settings.py", ".env", "secrets.*"
        ]
        
        for pattern in sensitive_files:
            files = list(self.project_root.rglob(pattern))
            for file_path in files:
                try:
                    stat = file_path.stat()
                    # Check for world-readable files
                    if stat.st_mode & 0o004:
                        self.add_finding(
                            "HIGH",
                            "File Permissions", 
                            f"Sensitive file is world-readable: {file_path.name}",
                            str(file_path.relative_to(self.project_root))
                        )
                    
                    # Check for world-writable files
                    if stat.st_mode & 0o002:
                        self.add_finding(
                            "HIGH",
                            "File Permissions",
                            f"Sensitive file is world-writable: {file_path.name}",
                            str(file_path.relative_to(self.project_root))
                        )
                        
                except OSError:
                    continue
    
    def audit_configuration_security(self):
        """Audit configuration files for security issues."""
        print("\n🔍 Auditing Configuration Security...")
        
        config_files = list(self.project_root.rglob("*.py")) + list(self.project_root.rglob("*.json")) + list(self.project_root.rglob("*.yml")) + list(self.project_root.rglob("*.yaml"))
        
        insecure_configs = {
            r'debug\s*=\s*True': ("MEDIUM", "Debug mode enabled"),
            r'DEBUG\s*=\s*True': ("MEDIUM", "Debug mode enabled"),
            r'SECURE_SSL_REDIRECT\s*=\s*False': ("HIGH", "SSL redirect disabled"),
            r'ALLOWED_HOSTS\s*=\s*\[\s*["\'][*]["\']\s*\]': ("HIGH", "Wildcard allowed hosts"),
            r'SECRET_KEY\s*=\s*["\'][^"\']*["\']': ("HIGH", "Hardcoded secret key"),
            r'password\s*[=:]\s*["\'][^"\']*["\']': ("HIGH", "Hardcoded password"),
            r'api_key\s*[=:]\s*["\'][^"\']*["\']': ("HIGH", "Hardcoded API key"),
        }
        
        for file_path in config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern, (severity, description) in insecure_configs.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                self.add_finding(
                                    severity,
                                    "Configuration",
                                    f"{description}: {line.strip()}",
                                    str(file_path.relative_to(self.project_root)),
                                    line_num
                                )
                                
            except Exception:
                continue
    
    def audit_secrets_exposure(self):
        """Audit for potential secrets exposure."""
        print("\n🔍 Auditing Secrets Exposure...")
        
        secret_patterns = {
            r'["\']?[Aa]ccess_?[Kk]ey["\']?\s*[:=]\s*["\'][A-Z0-9]{16,}["\']': "AWS Access Key",
            r'["\']?[Ss]ecret_?[Kk]ey["\']?\s*[:=]\s*["\'][A-Za-z0-9/+=]{40,}["\']': "AWS Secret Key",
            r'["\']?[Aa]pi_?[Kk]ey["\']?\s*[:=]\s*["\'][A-Za-z0-9]{32,}["\']': "API Key",
            r'["\']?[Tt]oken["\']?\s*[:=]\s*["\'][A-Za-z0-9]{32,}["\']': "Token",
            r'["\']?[Pp]assword["\']?\s*[:=]\s*["\'][^"\']{8,}["\']': "Password",
            r'-----BEGIN [A-Z ]+-----': "Private Key",
            r'mongodb://[^\s]+:[^\s]+@': "MongoDB Connection String",
            r'mysql://[^\s]+:[^\s]+@': "MySQL Connection String",
            r'postgres://[^\s]+:[^\s]+@': "PostgreSQL Connection String",
        }
        
        all_files = list(self.project_root.rglob("*"))
        text_extensions = {'.py', '.js', '.json', '.yml', '.yaml', '.txt', '.md', '.env', '.conf', '.cfg'}
        
        for file_path in all_files:
            if file_path.is_file() and (file_path.suffix in text_extensions or file_path.name.startswith('.env')):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        for line_num, line in enumerate(lines, 1):
                            for pattern, secret_type in secret_patterns.items():
                                if re.search(pattern, line):
                                    # Don't flag obvious examples or test data
                                    if any(keyword in line.lower() for keyword in ['example', 'test', 'dummy', 'fake', 'placeholder']):
                                        continue
                                        
                                    self.add_finding(
                                        "HIGH",
                                        "Secrets Exposure",
                                        f"Potential {secret_type} exposure: {line[:50]}...",
                                        str(file_path.relative_to(self.project_root)),
                                        line_num
                                    )
                                    
                except Exception:
                    continue
    
    def audit_input_validation(self):
        """Audit input validation implementation."""
        print("\n🔍 Auditing Input Validation...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        validation_patterns = {
            r'def\s+\w*process\w*|def\s+\w*handle\w*': "Function that might process user input",
            r'request\.|input\(|sys\.argv': "Potential user input source",
            r'\.get\(|\[.*\]': "Data access without validation"
        }
        
        security_functions = {
            'validate', 'sanitize', 'escape', 'check', 'verify', 
            'isinstance', 'assert', 'raise', 'len(', 'range('
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    for line_num, line in enumerate(lines, 1):
                        # Look for functions that might process input
                        if re.search(r'def\s+\w*(process|handle|parse|decode)\w*', line):
                            # Check if validation is present in function
                            function_start = line_num
                            function_body = ''
                            
                            # Get next 20 lines to check for validation
                            for check_line in lines[line_num:line_num + 20]:
                                if check_line.strip() and not check_line.startswith(' ') and not check_line.startswith('\t'):
                                    break  # End of function
                                function_body += check_line.lower()
                            
                            # Check if any validation functions are used
                            has_validation = any(func in function_body for func in security_functions)
                            
                            if not has_validation:
                                self.add_finding(
                                    "MEDIUM",
                                    "Input Validation",
                                    f"Function may lack input validation: {line.strip()}",
                                    str(file_path.relative_to(self.project_root)),
                                    line_num
                                )
                                
            except Exception:
                continue
    
    def audit_error_handling(self):
        """Audit error handling for information disclosure."""
        print("\n🔍 Auditing Error Handling...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    in_except_block = False
                    
                    for line_num, line in enumerate(lines, 1):
                        stripped = line.strip()
                        
                        if stripped.startswith('except'):
                            in_except_block = True
                            
                            # Check for bare except clauses
                            if stripped == 'except:' or stripped.startswith('except:'):
                                self.add_finding(
                                    "MEDIUM",
                                    "Error Handling",
                                    "Bare except clause can hide errors",
                                    str(file_path.relative_to(self.project_root)),
                                    line_num
                                )
                        
                        elif in_except_block and (stripped.startswith('print(') or stripped.startswith('logging')):
                            # Check for potential information disclosure in error messages
                            if any(keyword in stripped.lower() for keyword in ['traceback', 'exception', 'error']):
                                self.add_finding(
                                    "LOW",
                                    "Error Handling",
                                    "Error details might be exposed in logs",
                                    str(file_path.relative_to(self.project_root)),
                                    line_num
                                )
                        
                        elif not stripped.startswith(' ') and not stripped.startswith('\t') and stripped:
                            in_except_block = False
                            
            except Exception:
                continue
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report."""
        print("\n📋 Generating Security Report...")
        
        # Sort findings by severity
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        sorted_findings = sorted(self.findings, key=lambda x: severity_order.get(x["severity"], 4))
        
        report = {
            "audit_summary": {
                "total_findings": len(self.findings),
                "severity_breakdown": self.severity_counts,
                "audit_timestamp": __import__('time').time(),
                "project_root": str(self.project_root)
            },
            "findings": sorted_findings,
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if self.severity_counts.get("HIGH", 0) > 0:
            recommendations.append("Address HIGH severity findings immediately - they pose significant security risks")
        
        if any(f["category"] == "Secrets Exposure" for f in self.findings):
            recommendations.append("Implement proper secrets management (environment variables, key vaults)")
        
        if any(f["category"] == "Dependencies" for f in self.findings):
            recommendations.append("Update vulnerable dependencies and implement dependency scanning in CI/CD")
        
        if any(f["category"] == "Input Validation" for f in self.findings):
            recommendations.append("Implement comprehensive input validation and sanitization")
        
        if any(f["category"] == "Code Patterns" for f in self.findings):
            recommendations.append("Review and secure dangerous code patterns identified in the audit")
        
        recommendations.extend([
            "Implement regular security auditing in CI/CD pipeline",
            "Enable static code analysis tools (bandit, semgrep)",
            "Implement proper logging and monitoring for security events",
            "Regular security training for development team",
            "Implement security code review process"
        ])
        
        return recommendations
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete security audit."""
        print("🔒 Running Comprehensive Security Audit...")
        print(f"Project: {self.project_root}")
        
        # Run all audit modules
        self.audit_dependencies()
        self.audit_code_patterns()
        self.audit_file_permissions()
        self.audit_configuration_security()
        self.audit_secrets_exposure()
        self.audit_input_validation()
        self.audit_error_handling()
        
        return self.generate_report()


def print_report(report: Dict[str, Any]):
    """Print formatted security audit report."""
    print("\n" + "="*60)
    print("🔒 SECURITY AUDIT REPORT")
    print("="*60)
    
    summary = report["audit_summary"]
    print(f"\n📈 Summary:")
    print(f"  Total Findings: {summary['total_findings']}")
    
    for severity, count in summary["severity_breakdown"].items():
        if count > 0:
            emoji = {
                "HIGH": "🚨",
                "MEDIUM": "⚠️", 
                "LOW": "🟡",
                "INFO": "ℹ️"
            }
            print(f"  {emoji.get(severity, '')} {severity}: {count}")
    
    # Print findings by category
    if report["findings"]:
        print(f"\n🔍 Detailed Findings:")
        
        by_category = {}
        for finding in report["findings"]:
            category = finding["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(finding)
        
        for category, findings in by_category.items():
            print(f"\n  📋 {category} ({len(findings)} findings):")
            
            for finding in findings[:3]:  # Show first 3 findings per category
                severity_emoji = {
                    "HIGH": "🚨",
                    "MEDIUM": "⚠️", 
                    "LOW": "🟡",
                    "INFO": "ℹ️"
                }
                
                print(f"    {severity_emoji.get(finding['severity'], '')} {finding['description']}")
                if finding.get('file'):
                    location = f"{finding['file']}"
                    if finding.get('line'):
                        location += f":{finding['line']}"
                    print(f"      📁 {location}")
            
            if len(findings) > 3:
                print(f"    ... and {len(findings) - 3} more findings")
    
    # Print recommendations
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"  {i}. {rec}")
    
    if len(report["recommendations"]) > 5:
        print(f"  ... and {len(report['recommendations']) - 5} more recommendations")
    
    print("\n" + "="*60)


def main():
    """Run security audit from command line."""
    project_root = Path(__file__).parent.parent
    
    auditor = SecurityAuditor(str(project_root))
    report = auditor.run_full_audit()
    
    # Print report to console
    print_report(report)
    
    # Save detailed report to file
    report_file = project_root / "results" / "security_audit_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved to: {report_file}")
    
    # Return exit code based on severity
    if report["audit_summary"]["severity_breakdown"].get("HIGH", 0) > 0:
        print("\n❌ Security audit failed: HIGH severity issues found")
        sys.exit(1)
    elif report["audit_summary"]["severity_breakdown"].get("MEDIUM", 0) > 5:
        print("\n⚠️  Security audit warning: Multiple MEDIUM severity issues found")
        sys.exit(0)  # Warning but don't fail
    else:
        print("\n✅ Security audit passed: No critical issues found")
        sys.exit(0)


if __name__ == "__main__":
    main()
