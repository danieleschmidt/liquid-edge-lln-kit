"""Automated changelog generation from git history."""

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    TEST = "test"
    CHORE = "chore"
    BREAKING = "BREAKING"


@dataclass
class CommitInfo:
    """Information about a git commit."""
    hash: str
    message: str
    author: str
    date: str
    change_type: Optional[ChangeType] = None
    scope: Optional[str] = None
    description: str = ""
    breaking_change: bool = False


class ChangelogGenerator:
    """Generate changelog from git history using conventional commits."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.conventional_commit_pattern = re.compile(
            r'^(?P<type>\w+)(?:\((?P<scope>[\w\-_]+)\))?: (?P<description>.+?)(?:\n\n(?P<body>.*))?$',
            re.DOTALL
        )
    
    def get_git_commits(self, since_tag: Optional[str] = None) -> List[CommitInfo]:
        """Get git commits since a specific tag or all commits."""
        cmd = ["git", "log", "--pretty=format:%H|%s|%an|%ad", "--date=short"]
        
        if since_tag:
            cmd.append(f"{since_tag}..HEAD")
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.project_root, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        commits.append(CommitInfo(
                            hash=parts[0],
                            message=parts[1],
                            author=parts[2],
                            date=parts[3]
                        ))
            
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def parse_conventional_commit(self, commit: CommitInfo) -> CommitInfo:
        """Parse conventional commit format."""
        match = self.conventional_commit_pattern.match(commit.message)
        
        if match:
            commit.change_type = ChangeType(match.group('type').lower())
            commit.scope = match.group('scope')
            commit.description = match.group('description')
            
            # Check for breaking changes
            body = match.group('body') or ""
            if "BREAKING CHANGE" in body or commit.message.endswith('!'):
                commit.breaking_change = True
                commit.change_type = ChangeType.BREAKING
        else:
            # Fallback for non-conventional commits
            commit.description = commit.message
            commit.change_type = self._infer_change_type(commit.message)
        
        return commit
    
    def _infer_change_type(self, message: str) -> ChangeType:
        """Infer change type from commit message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['fix', 'bug', 'patch']):
            return ChangeType.FIX
        elif any(word in message_lower for word in ['feat', 'feature', 'add']):
            return ChangeType.FEAT
        elif any(word in message_lower for word in ['doc', 'readme']):
            return ChangeType.DOCS
        elif any(word in message_lower for word in ['test', 'spec']):
            return ChangeType.TEST
        elif any(word in message_lower for word in ['refactor', 'clean']):
            return ChangeType.REFACTOR
        else:
            return ChangeType.CHORE
    
    def get_latest_tag(self) -> Optional[str]:
        """Get the latest git tag."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def generate_changelog_section(
        self, 
        commits: List[CommitInfo], 
        version: str = "Unreleased"
    ) -> str:
        """Generate a changelog section for a version."""
        # Group commits by type
        grouped_commits: Dict[ChangeType, List[CommitInfo]] = {
            change_type: [] for change_type in ChangeType
        }
        
        for commit in commits:
            parsed_commit = self.parse_conventional_commit(commit)
            grouped_commits[parsed_commit.change_type].append(parsed_commit)
        
        # Generate changelog section
        changelog = [f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}", ""]
        
        # Breaking changes first
        if grouped_commits[ChangeType.BREAKING]:
            changelog.extend(["### ⚠ BREAKING CHANGES", ""])
            for commit in grouped_commits[ChangeType.BREAKING]:
                changelog.append(f"- {commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Features
        if grouped_commits[ChangeType.FEAT]:
            changelog.extend(["### ✨ Features", ""])
            for commit in grouped_commits[ChangeType.FEAT]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                changelog.append(f"- {scope_str}{commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Bug fixes
        if grouped_commits[ChangeType.FIX]:
            changelog.extend(["### 🐛 Bug Fixes", ""])
            for commit in grouped_commits[ChangeType.FIX]:
                scope_str = f"**{commit.scope}**: " if commit.scope else ""
                changelog.append(f"- {scope_str}{commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Documentation
        if grouped_commits[ChangeType.DOCS]:
            changelog.extend(["### 📚 Documentation", ""])
            for commit in grouped_commits[ChangeType.DOCS]:
                changelog.append(f"- {commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Refactoring
        if grouped_commits[ChangeType.REFACTOR]:
            changelog.extend(["### ♻️ Code Refactoring", ""])
            for commit in grouped_commits[ChangeType.REFACTOR]:
                changelog.append(f"- {commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Tests
        if grouped_commits[ChangeType.TEST]:
            changelog.extend(["### ✅ Tests", ""])
            for commit in grouped_commits[ChangeType.TEST]:
                changelog.append(f"- {commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        # Chores (build, CI, etc.)
        if grouped_commits[ChangeType.CHORE]:
            changelog.extend(["### 🔧 Chores", ""])
            for commit in grouped_commits[ChangeType.CHORE]:
                changelog.append(f"- {commit.description} ({commit.hash[:7]})")
            changelog.append("")
        
        return "\n".join(changelog)
    
    def update_changelog_file(self, new_section: str):
        """Update CHANGELOG.md file with new section."""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                existing_content = f.read()
            
            # Insert new section after header
            lines = existing_content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    header_end = i
                    break
            
            # Insert new section
            new_lines = lines[:header_end] + new_section.split('\n') + [''] + lines[header_end:]
            new_content = '\n'.join(new_lines)
        else:
            # Create new changelog
            header = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
            new_content = header + new_section
        
        with open(changelog_path, 'w') as f:
            f.write(new_content)
        
        return changelog_path
    
    def generate_release_notes(self, version: str, commits: List[CommitInfo]) -> str:
        """Generate release notes for GitHub releases."""
        parsed_commits = [self.parse_conventional_commit(c) for c in commits]
        
        # Count changes
        feat_count = len([c for c in parsed_commits if c.change_type == ChangeType.FEAT])
        fix_count = len([c for c in parsed_commits if c.change_type == ChangeType.FIX])
        breaking_count = len([c for c in parsed_commits if c.breaking_change])
        
        # Generate summary
        summary_parts = []
        if feat_count > 0:
            summary_parts.append(f"{feat_count} new feature{'s' if feat_count != 1 else ''}")
        if fix_count > 0:
            summary_parts.append(f"{fix_count} bug fix{'es' if fix_count != 1 else ''}")
        if breaking_count > 0:
            summary_parts.append(f"{breaking_count} breaking change{'s' if breaking_count != 1 else ''}")
        
        summary = "This release includes " + ", ".join(summary_parts) + "."
        
        # Generate full release notes
        release_notes = [
            f"# Release {version}",
            "",
            summary,
            "",
            self.generate_changelog_section(commits, version)
        ]
        
        return "\n".join(release_notes)


def main():
    """Main entry point for changelog generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate changelog from git history")
    parser.add_argument("--version", help="Version for the changelog entry")
    parser.add_argument("--since", help="Generate changelog since this tag")
    parser.add_argument("--output", help="Output file (default: CHANGELOG.md)")
    parser.add_argument("--release-notes", action="store_true", 
                       help="Generate release notes format")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    generator = ChangelogGenerator(project_root)
    
    # Get commits
    since_tag = args.since or generator.get_latest_tag()
    commits = generator.get_git_commits(since_tag)
    
    if not commits:
        print("No commits found since last tag")
        return
    
    version = args.version or "Unreleased"
    
    if args.release_notes:
        # Generate release notes
        notes = generator.generate_release_notes(version, commits)
        output_file = args.output or f"release_notes_{version}.md"
        with open(project_root / output_file, 'w') as f:
            f.write(notes)
        print(f"Release notes generated: {output_file}")
    else:
        # Generate changelog section
        changelog_section = generator.generate_changelog_section(commits, version)
        
        if args.output:
            with open(project_root / args.output, 'w') as f:
                f.write(changelog_section)
            print(f"Changelog section written to: {args.output}")
        else:
            # Update CHANGELOG.md
            changelog_path = generator.update_changelog_file(changelog_section)
            print(f"CHANGELOG.md updated: {changelog_path}")
    
    print(f"Processed {len(commits)} commits")


if __name__ == "__main__":
    main()