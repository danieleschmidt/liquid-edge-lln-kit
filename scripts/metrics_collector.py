#!/usr/bin/env python3

# Copyright (c) 2025 Liquid Edge LLN Kit Contributors
# SPDX-License-Identifier: MIT

"""Automated metrics collection script for Liquid Edge LLN Kit."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import requests


class MetricsCollector:
    """Collect and aggregate project metrics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.metrics: Dict[str, Any] = {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        self.metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": self._collect_project_info(),
            "code_quality": self._collect_code_quality_metrics(),
            "performance": self._collect_performance_metrics(),
            "development": self._collect_development_metrics(),
            "adoption": self._collect_adoption_metrics(),
            "hardware_support": self._collect_hardware_metrics(),
        }
        return self.metrics
    
    def _collect_project_info(self) -> Dict[str, Any]:
        """Collect basic project information."""
        try:
            # Read pyproject.toml for project info
            import tomllib
            with open(self.project_root / "pyproject.toml", "rb") as f:
                pyproject = tomllib.load(f)
            
            project = pyproject.get("project", {})
            return {
                "name": project.get("name", "unknown"),
                "version": project.get("version", "unknown"),
                "description": project.get("description", ""),
                "python_requires": project.get("requires-python", ""),
                "dependencies_count": len(project.get("dependencies", [])),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage
        try:
            result = subprocess.run(
                ["coverage", "report", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                metrics["test_coverage"] = {
                    "percent": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                    "lines_total": coverage_data.get("totals", {}).get("num_statements", 0),
                }
        except Exception:
            pass
        
        # Code complexity (using ruff)
        try:
            result = subprocess.run(
                ["ruff", "check", "src/", "--output-format=json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.stdout:
                ruff_data = json.loads(result.stdout)
                metrics["linting"] = {
                    "issues_count": len(ruff_data),
                    "categories": self._categorize_ruff_issues(ruff_data),
                }
        except Exception:
            pass
        
        # Security scanning
        try:
            result = subprocess.run(
                ["bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                metrics["security"] = {
                    "high_severity": len([
                        issue for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ]),
                    "medium_severity": len([
                        issue for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") == "MEDIUM"
                    ]),
                    "low_severity": len([
                        issue for issue in bandit_data.get("results", [])
                        if issue.get("issue_severity") == "LOW"
                    ]),
                }
        except Exception:
            pass
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance benchmarks."""
        metrics = {}
        
        # Try to load latest benchmark results
        benchmark_file = self.project_root / "benchmarks" / "latest.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    metrics["benchmarks"] = benchmark_data
            except Exception:
                pass
        
        # Package size
        try:
            result = subprocess.run(
                ["python", "-m", "build", "--wheel"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                dist_dir = self.project_root / "dist"
                wheel_files = list(dist_dir.glob("*.whl"))
                if wheel_files:
                    wheel_size = wheel_files[-1].stat().st_size
                    metrics["package_size_bytes"] = wheel_size
        except Exception:
            pass
        
        return metrics
    
    def _collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development process metrics."""
        metrics = {}
        
        # Git metrics
        try:
            # Commit frequency (last 30 days)
            result = subprocess.run(
                ["git", "log", "--since=30 days ago", "--oneline"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                commit_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                metrics["commits_last_30_days"] = commit_count
            
            # Contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                contributors = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                metrics["total_contributors"] = contributors
        except Exception:
            pass
        
        # CI/CD metrics from GitHub Actions (if available)
        github_token = self._get_github_token()
        if github_token:
            try:
                metrics["ci_cd"] = self._collect_github_actions_metrics(github_token)
            except Exception:
                pass
        
        return metrics
    
    def _collect_adoption_metrics(self) -> Dict[str, Any]:
        """Collect adoption and usage metrics."""
        metrics = {}
        
        # GitHub metrics
        github_token = self._get_github_token()
        if github_token:
            try:
                metrics["github"] = self._collect_github_metrics(github_token)
            except Exception:
                pass
        
        # PyPI download stats (if available)
        try:
            package_name = self._get_package_name()
            if package_name:
                response = requests.get(
                    f"https://pypistats.org/api/packages/{package_name}/recent",
                    timeout=10
                )
                if response.status_code == 200:
                    pypi_data = response.json()
                    metrics["pypi_downloads"] = pypi_data.get("data", {})
        except Exception:
            pass
        
        return metrics
    
    def _collect_hardware_metrics(self) -> Dict[str, Any]:
        """Collect hardware support and deployment metrics."""
        metrics = {}
        
        # Hardware test results
        hardware_results_file = self.project_root / "tests" / "hardware_results.json"
        if hardware_results_file.exists():
            try:
                with open(hardware_results_file) as f:
                    metrics["hardware_tests"] = json.load(f)
            except Exception:
                pass
        
        # Platform compatibility
        metrics["supported_platforms"] = [
            "stm32h743",
            "stm32f767", 
            "esp32s3",
            "esp32c3",
            "nrf52840",
            "simulation"
        ]
        
        return metrics
    
    def _categorize_ruff_issues(self, issues: List[Dict]) -> Dict[str, int]:
        """Categorize ruff linting issues."""
        categories = {}
        for issue in issues:
            code = issue.get("code", "unknown")
            category = code[0] if code else "unknown"
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _get_github_token(self) -> Optional[str]:
        """Get GitHub token from environment."""
        import os
        return os.getenv("GITHUB_TOKEN")
    
    def _get_package_name(self) -> Optional[str]:
        """Get package name from pyproject.toml."""
        try:
            import tomllib
            with open(self.project_root / "pyproject.toml", "rb") as f:
                pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("name")
        except Exception:
            return None
    
    def _collect_github_metrics(self, token: str) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        headers = {"Authorization": f"token {token}"}
        
        # Repository info
        response = requests.get(
            "https://api.github.com/repos/danieleschmidt/liquid-edge-lln-kit",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            repo_data = response.json()
            return {
                "stars": repo_data.get("stargazers_count", 0),
                "forks": repo_data.get("forks_count", 0),
                "watchers": repo_data.get("watchers_count", 0),
                "open_issues": repo_data.get("open_issues_count", 0),
                "size_kb": repo_data.get("size", 0),
                "created_at": repo_data.get("created_at"),
                "updated_at": repo_data.get("updated_at"),
            }
        
        return {}
    
    def _collect_github_actions_metrics(self, token: str) -> Dict[str, Any]:
        """Collect GitHub Actions workflow metrics."""
        headers = {"Authorization": f"token {token}"}
        
        # Workflow runs
        response = requests.get(
            "https://api.github.com/repos/danieleschmidt/liquid-edge-lln-kit/actions/runs",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            runs_data = response.json()
            runs = runs_data.get("workflow_runs", [])
            
            if runs:
                success_count = len([r for r in runs if r.get("conclusion") == "success"])
                total_count = len(runs)
                
                return {
                    "total_runs": total_count,
                    "success_rate": success_count / total_count if total_count > 0 else 0,
                    "latest_run_status": runs[0].get("conclusion") if runs else None,
                    "latest_run_date": runs[0].get("created_at") if runs else None,
                }
        
        return {}
    
    def save_metrics(self, output_file: Path) -> None:
        """Save collected metrics to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report."""
        if not self.metrics:
            return "No metrics collected."
        
        report = ["# Liquid Edge LLN Kit - Metrics Report", ""]
        report.append(f"Generated: {self.metrics.get('timestamp', 'unknown')}")
        report.append("")
        
        # Project info
        project = self.metrics.get("project", {})
        if project:
            report.append("## Project Information")
            report.append(f"- Name: {project.get('name', 'unknown')}")
            report.append(f"- Version: {project.get('version', 'unknown')}")
            report.append(f"- Dependencies: {project.get('dependencies_count', 0)}")
            report.append("")
        
        # Code quality
        code_quality = self.metrics.get("code_quality", {})
        if code_quality:
            report.append("## Code Quality")
            
            coverage = code_quality.get("test_coverage", {})
            if coverage:
                report.append(f"- Test Coverage: {coverage.get('percent', 0):.1f}%")
            
            linting = code_quality.get("linting", {})
            if linting:
                report.append(f"- Linting Issues: {linting.get('issues_count', 0)}")
            
            security = code_quality.get("security", {})
            if security:
                total_security_issues = (
                    security.get("high_severity", 0) +
                    security.get("medium_severity", 0) +
                    security.get("low_severity", 0)
                )
                report.append(f"- Security Issues: {total_security_issues}")
            
            report.append("")
        
        # Development metrics
        development = self.metrics.get("development", {})
        if development:
            report.append("## Development Activity")
            report.append(f"- Commits (30 days): {development.get('commits_last_30_days', 0)}")
            report.append(f"- Contributors: {development.get('total_contributors', 0)}")
            
            ci_cd = development.get("ci_cd", {})
            if ci_cd:
                report.append(f"- CI Success Rate: {ci_cd.get('success_rate', 0)*100:.1f}%")
            
            report.append("")
        
        # Adoption metrics
        adoption = self.metrics.get("adoption", {})
        if adoption:
            report.append("## Adoption Metrics")
            
            github = adoption.get("github", {})
            if github:
                report.append(f"- GitHub Stars: {github.get('stars', 0)}")
                report.append(f"- GitHub Forks: {github.get('forks', 0)}")
                report.append(f"- Open Issues: {github.get('open_issues', 0)}")
            
            pypi = adoption.get("pypi_downloads", {})
            if pypi:
                report.append(f"- Recent Downloads: {pypi.get('last_month', 0)}")
            
            report.append("")
        
        return "\n".join(report)


@click.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for metrics JSON")
@click.option("--report", "-r", is_flag=True, help="Generate human-readable report")
@click.option("--project-root", type=click.Path(exists=True), default=".", help="Project root directory")
def main(output: Optional[str], report: bool, project_root: str) -> None:
    """Collect project metrics for Liquid Edge LLN Kit."""
    
    root_path = Path(project_root).resolve()
    collector = MetricsCollector(root_path)
    
    print("Collecting metrics...")
    metrics = collector.collect_all_metrics()
    
    if output:
        output_path = Path(output)
        collector.save_metrics(output_path)
        print(f"Metrics saved to {output_path}")
    
    if report:
        print("\n" + collector.generate_report())
    
    if not output and not report:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()