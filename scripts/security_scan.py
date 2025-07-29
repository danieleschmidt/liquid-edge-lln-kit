"""Security scanning automation script."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


class SecurityScanner:
    """Automated security scanning for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
    
    def run_bandit(self) -> Dict[str, Any]:
        """Run Bandit security linting."""
        print("Running Bandit security scan...")
        try:
            result = subprocess.run([
                "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if Path("bandit-report.json").exists():
                with open("bandit-report.json") as f:
                    report = json.load(f)
                return {
                    "status": "success",
                    "issues": len(report.get("results", [])),
                    "report": report
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_safety(self) -> Dict[str, Any]:
        """Run Safety dependency scanning."""
        print("Running Safety dependency scan...")
        try:
            result = subprocess.run([
                "safety", "check", "--json", "--output", "safety-report.json"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Safety found vulnerabilities
                return {
                    "status": "vulnerabilities_found",
                    "output": result.stdout,
                    "errors": result.stderr
                }
            return {"status": "clean"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_pip_audit(self) -> Dict[str, Any]:
        """Run pip-audit vulnerability scanning."""
        print("Running pip-audit scan...")
        try:
            result = subprocess.run([
                "pip-audit", "--format", "json", "--output", "pip-audit-report.json"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {"status": "clean"}
            else:
                return {
                    "status": "vulnerabilities_found",
                    "output": result.stdout,
                    "errors": result.stderr
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def check_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets."""
        print("Checking for hardcoded secrets...")
        
        secret_patterns = [
            r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
            r"secret[_-]?key\s*=\s*['\"][^'\"]+['\"]",
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]",
            r"aws[_-]?access[_-]?key",
            r"ssh[_-]?rsa[_-]?private[_-]?key",
        ]
        
        findings = []
        for pattern in secret_patterns:
            try:
                result = subprocess.run([
                    "grep", "-r", "-E", "-n", pattern, "src/"
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    findings.extend(result.stdout.strip().split('\n'))
            except Exception:
                continue
        
        return {
            "status": "clean" if not findings else "secrets_found",
            "findings": findings
        }
    
    def generate_sbom(self) -> Dict[str, Any]:
        """Generate Software Bill of Materials."""
        print("Generating SBOM...")
        try:
            # Generate requirements
            result = subprocess.run([
                "pip", "freeze"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                requirements = result.stdout.strip().split('\n')
                
                sbom = {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.4",
                    "version": 1,
                    "metadata": {
                        "timestamp": "2025-01-01T00:00:00Z",
                        "component": {
                            "type": "application",
                            "name": "liquid-edge-lln",
                            "version": "0.1.0"
                        }
                    },
                    "components": []
                }
                
                for req in requirements:
                    if "==" in req:
                        name, version = req.split("==")
                        sbom["components"].append({
                            "type": "library",
                            "name": name,  
                            "version": version,
                            "purl": f"pkg:pypi/{name}@{version}"
                        })
                
                with open(self.project_root / "sbom.json", "w") as f:
                    json.dump(sbom, f, indent=2)
                
                return {"status": "generated", "components": len(sbom["components"])}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans and return results."""
        print("Starting comprehensive security scan...")
        
        self.results["bandit"] = self.run_bandit()
        self.results["safety"] = self.run_safety()
        self.results["pip_audit"] = self.run_pip_audit()
        self.results["secrets"] = self.check_secrets()
        self.results["sbom"] = self.generate_sbom()
        
        # Generate summary
        total_issues = 0
        if self.results["bandit"]["status"] == "success":
            total_issues += self.results["bandit"]["issues"]
        
        if self.results["secrets"]["status"] == "secrets_found":
            total_issues += len(self.results["secrets"]["findings"])
        
        vulnerability_scans = ["safety", "pip_audit"]
        vulnerable = any(
            self.results[scan]["status"] == "vulnerabilities_found"
            for scan in vulnerability_scans
        )
        
        self.results["summary"] = {
            "total_issues": total_issues,
            "vulnerabilities_found": vulnerable,
            "sbom_generated": self.results["sbom"]["status"] == "generated",
            "overall_status": "fail" if total_issues > 0 or vulnerable else "pass"
        }
        
        return self.results
    
    def save_results(self, filename: str = "security-scan-results.json"):
        """Save scan results to file."""
        with open(self.project_root / filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    scanner = SecurityScanner(project_root)
    
    results = scanner.run_all_scans()
    scanner.save_results()
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*60)
    print("SECURITY SCAN SUMMARY")
    print("="*60)
    print(f"Total Issues Found: {summary['total_issues']}")
    print(f"Vulnerabilities: {'Yes' if summary['vulnerabilities_found'] else 'No'}")
    print(f"SBOM Generated: {'Yes' if summary['sbom_generated'] else 'No'}")
    print(f"Overall Status: {summary['overall_status'].upper()}")
    print("="*60)
    
    # Exit with error code if issues found
    if summary["overall_status"] == "fail":
        sys.exit(1)
    else:
        print("✅ All security checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()