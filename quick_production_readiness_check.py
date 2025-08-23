#!/usr/bin/env python3
"""
Quick Production Readiness Check
Fast validation script for production deployment readiness.
"""

import asyncio
import time
import json
from pathlib import Path
from datetime import datetime

from production_deployment_validator import (
    ProductionDeploymentValidator,
    DeploymentValidationConfig
)


async def quick_readiness_check():
    """Perform a quick production readiness assessment."""
    
    print("🚀 LIQUID EDGE LLN - QUICK PRODUCTION READINESS CHECK")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    # Quick configuration for essential checks only
    config = DeploymentValidationConfig(
        # Critical infrastructure checks
        validate_kubernetes=True,
        validate_docker=True,
        validate_monitoring=True,
        
        # Essential security checks
        validate_tls=True,
        validate_authentication=True,
        validate_container_security=True,
        
        # Core production readiness
        validate_health_checks=True,
        validate_graceful_shutdown=True,
        
        # Basic performance validation
        validate_load_testing=True,
        validate_latency=True,
        
        # Essential operations
        validate_rollback_procedures=True,
        validate_disaster_recovery=True,
        
        # Reduced test parameters for speed
        load_test_duration=10,
        concurrent_requests=20,
        max_acceptable_latency=250.0,
        min_acceptable_throughput=200,
        target_availability=99.5
    )
    
    print("🔍 Running essential production readiness checks...")
    print()
    
    # Initialize validator
    validator = ProductionDeploymentValidator(config)
    
    # Run validation
    report = await validator.run_comprehensive_validation()
    
    duration = time.time() - start_time
    
    # Display quick results
    print("📊 QUICK READINESS ASSESSMENT RESULTS")
    print("-" * 50)
    
    executive_summary = report.get("executive_summary", {})
    readiness_score = report.get("deployment_readiness_score", {})
    
    # Status indicator
    status = executive_summary.get("readiness_status", "Unknown")
    if "READY" in status:
        status_icon = "✅"
        status_color = "GREEN"
    elif "CONDITIONAL" in status:
        status_icon = "⚠️"
        status_color = "YELLOW" 
    else:
        status_icon = "❌"
        status_color = "RED"
    
    print(f"{status_icon} Status: {status}")
    print(f"⭐ Score: {executive_summary.get('overall_score', 'N/A')}")
    print(f"🏆 Grade: {readiness_score.get('grade', 'N/A')}")
    print(f"✅ Passed: {executive_summary.get('tests_passed', 0)}/{executive_summary.get('tests_executed', 0)}")
    print(f"⚠️  Issues: {executive_summary.get('critical_issues', 0)} critical")
    print(f"⏱️  Duration: {duration:.1f}s")
    
    # Quick recommendations
    print()
    if "READY" in status:
        print("🎉 READY FOR PRODUCTION!")
        print("✅ All essential checks passed")
        print("✅ No critical issues found")
        print("✅ Production deployment approved")
    else:
        print("⚠️  ACTION REQUIRED:")
        critical_issues = report.get("critical_issues", [])
        if critical_issues:
            for issue in critical_issues[:3]:
                print(f"   • {issue.get('name', 'Unknown issue')}")
        
        recommendations = report.get("recommendations", [])[:3]
        if recommendations:
            print("\n💡 Priority fixes:")
            for i, rec in enumerate(recommendations, 1):
                if rec.get("priority") == "critical":
                    print(f"   {i}. {rec.get('title', 'Unknown')}")
    
    print()
    print("=" * 60)
    
    return report


async def check_infrastructure_files():
    """Quick check of key infrastructure files."""
    
    print("📋 INFRASTRUCTURE FILES CHECK")
    print("-" * 40)
    
    # Key files to check
    key_files = {
        "Kubernetes Deployment": [
            "/root/repo/deployment/kubernetes/k8s-deployment.yaml",
            "/root/repo/k8s-deployment.yaml"
        ],
        "Kubernetes Service": [
            "/root/repo/k8s-service.yaml"
        ],
        "Kubernetes Ingress": [
            "/root/repo/k8s-ingress.yaml"
        ],
        "Docker Configuration": [
            "/root/repo/Dockerfile",
            "/root/repo/docker-compose.yml"
        ],
        "Monitoring Config": [
            "/root/repo/deployment/monitoring/prometheus.yml",
            "/root/repo/prometheus.yml"
        ],
        "Security Policies": [
            "/root/repo/deployment/security/network-policies.yaml"
        ]
    }
    
    for category, files in key_files.items():
        found_files = [f for f in files if Path(f).exists()]
        if found_files:
            print(f"✅ {category}: {len(found_files)} file(s) found")
        else:
            print(f"⚠️  {category}: No files found")
    
    print()


def check_system_requirements():
    """Check system requirements for deployment."""
    
    print("🔧 SYSTEM REQUIREMENTS CHECK")
    print("-" * 40)
    
    import subprocess
    import shutil
    
    requirements = {
        "Python": ["python3", "--version"],
        "Docker": ["docker", "--version"],
        "Kubectl": ["kubectl", "version", "--client"],
        "Git": ["git", "--version"]
    }
    
    for tool, cmd in requirements.items():
        try:
            if shutil.which(cmd[0]):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
                    print(f"✅ {tool}: {version}")
                else:
                    print(f"⚠️  {tool}: Available but version check failed")
            else:
                print(f"❌ {tool}: Not found in PATH")
        except Exception as e:
            print(f"❌ {tool}: Error checking - {str(e)}")
    
    print()


async def main():
    """Main function to run quick production readiness check."""
    
    # System requirements check
    check_system_requirements()
    
    # Infrastructure files check
    await check_infrastructure_files()
    
    # Quick readiness validation
    report = await quick_readiness_check()
    
    # Final summary
    executive_summary = report.get("executive_summary", {})
    status = executive_summary.get("readiness_status", "")
    
    print()
    if "READY" in status:
        print("🎯 CONCLUSION: System is READY for production deployment!")
        return 0
    else:
        print("⚠️  CONCLUSION: System requires improvements before production.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)