"""Tests for deny rules and policy enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp_guard.cli import main
from mcp_guard.models import (
    MCPCapability,
    MCPCapabilityType,
    MCPManifest,
    RiskLevel,
)
from mcp_guard.policy import DenyPolicy
from mcp_guard.scanner import Scanner


class TestDenyPolicy:
    """Test DenyPolicy parsing and matching logic."""

    def test_from_yaml_string_with_comments(self):
        yaml_content = """
        # Security Policy Configuration
        deny:
          # Malicious or untrusted servers
          servers:
            - "malicious-server" # CVE-2024-xxx
            - "github-*"
          # Restricted tools
          tools:
            - "github/delete_repo" # Critical tool
            - "filesystem/write"
            - "admin_*"
        """
        policy = DenyPolicy.from_yaml(yaml_content)
        assert policy.servers == ["malicious-server", "github-*"]
        assert policy.tools == ["github/delete_repo", "filesystem/write", "admin_*"]

    def test_from_yaml_file(self, tmp_path: Path):
        config_file = tmp_path / "policy.yaml"
        config_file.write_text(
            """
            deny:
              servers:
                - "blocked-server"
              tools:
                - "dangerous_tool"
            """,
            encoding="utf-8",
        )
        policy = DenyPolicy.from_yaml(config_file)
        assert policy.servers == ["blocked-server"]
        assert policy.tools == ["dangerous_tool"]

    def test_server_matching_exact_and_wildcard(self):
        policy = DenyPolicy(
            servers=["evil-server", "untrusted-*", "*-legacy"],
        )
        # Exact match
        denied, pat = policy.is_server_denied("evil-server")
        assert denied is True
        assert pat == "evil-server"

        # Wildcard prefix match
        denied, pat = policy.is_server_denied("untrusted-bot")
        assert denied is True
        assert pat == "untrusted-*"

        # Wildcard suffix match
        denied, pat = policy.is_server_denied("database-legacy")
        assert denied is True
        assert pat == "*-legacy"

        # Non-matching server
        denied, pat = policy.is_server_denied("safe-server")
        assert denied is False
        assert pat is None

    def test_tool_matching_exact_and_wildcard(self):
        policy = DenyPolicy(
            tools=["drop_table", "delete_*", "sys_*"],
        )
        assert policy.is_tool_denied("drop_table")[0] is True
        assert policy.is_tool_denied("delete_user")[0] is True
        assert policy.is_tool_denied("sys_exec")[0] is True
        assert policy.is_tool_denied("get_user")[0] is False

    def test_tool_matching_qualified_server(self):
        policy = DenyPolicy(
            tools=["github/delete_repo", "filesystem/*", "*-service/nuke"],
        )
        # Qualified match on server + tool
        assert policy.is_tool_denied("delete_repo", server_name="github")[0] is True
        assert policy.is_tool_denied("delete_repo", server_name="gitlab")[0] is False

        # Wildcard tool under server
        assert policy.is_tool_denied("write", server_name="filesystem")[0] is True
        assert policy.is_tool_denied("read", server_name="filesystem")[0] is True
        assert policy.is_tool_denied("write", server_name="memory")[0] is False

        # Wildcard server with specific tool
        assert policy.is_tool_denied("nuke", server_name="auth-service")[0] is True


class TestScannerDenyIntegration:
    """Test Scanner behavior with DenyPolicy."""

    def test_scanner_detects_denied_server(self):
        manifest = MCPManifest(
            name="github-experimental",
            capabilities=[
                MCPCapability(
                    name="get_status", type=MCPCapabilityType.TOOL, description="Get status"
                ),
            ],
        )
        policy = DenyPolicy(servers=["github-*"])
        scanner = Scanner(deny_policy=policy)
        result = scanner.scan(manifest)

        assert result.risk_score == RiskLevel.CRITICAL
        deny_findings = [f for f in result.findings if f.rule_id == "DENY001"]
        assert len(deny_findings) == 1
        assert "github-experimental" in deny_findings[0].message
        assert "github-*" in deny_findings[0].message

    def test_scanner_detects_denied_tool(self):
        manifest = MCPManifest(
            name="github",
            capabilities=[
                MCPCapability(
                    name="delete_repo",
                    type=MCPCapabilityType.TOOL,
                    description="Delete a repository",
                    is_destructive=True,
                    has_auth=True,
                    input_schema={"properties": {"confirm": {"type": "boolean"}}},
                ),
            ],
        )
        policy = DenyPolicy(tools=["github/delete_repo"])
        scanner = Scanner(deny_policy=policy)
        result = scanner.scan(manifest)

        assert result.risk_score == RiskLevel.CRITICAL
        deny_findings = [f for f in result.findings if f.rule_id == "DENY002"]
        assert len(deny_findings) == 1
        assert "delete_repo" in deny_findings[0].message

    def test_scanner_no_deny_findings_when_compliant(self):
        manifest = MCPManifest(
            name="allowed-server",
            capabilities=[
                MCPCapability(
                    name="get_data", type=MCPCapabilityType.TOOL, description="Fetch some data"
                ),
            ],
        )
        policy = DenyPolicy(servers=["evil-*"], tools=["drop_*"])
        scanner = Scanner(deny_policy=policy)
        result = scanner.scan(manifest)

        deny_findings = [f for f in result.findings if f.rule_id.startswith("DENY")]
        assert len(deny_findings) == 0


class TestCLIDenyOptions:
    """Test CLI commands and flags with deny rules."""

    @pytest.fixture
    def sample_manifest_file(self, tmp_path: Path):
        data = {
            "name": "untrusted-mcp",
            "version": "1.0.0",
            "description": "Test server for deny rules",
            "tools": [
                {
                    "name": "delete_all",
                    "description": "Delete all resources",
                },
                {
                    "name": "get_info",
                    "description": "Get server info",
                },
            ],
        }
        manifest_file = tmp_path / "mcp.json"
        manifest_file.write_text(json.dumps(data), encoding="utf-8")
        return manifest_file

    def test_cli_scan_with_config_yaml(self, sample_manifest_file: Path, tmp_path: Path):
        config_file = tmp_path / "mcp-guard.yaml"
        config_file.write_text(
            """
            # Custom security policy
            deny:
              servers:
                - "untrusted-*"
              tools:
                - "delete_*"
            """,
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["scan", str(sample_manifest_file), "--config", str(config_file), "--format", "json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        rule_ids = [f["rule_id"] for f in parsed["findings"]]
        assert "DENY001" in rule_ids
        assert "DENY002" in rule_ids

    def test_cli_scan_with_deny_flag_exits_error(self, sample_manifest_file: Path, tmp_path: Path):
        config_file = tmp_path / "policy.yaml"
        config_file.write_text(
            """
            deny:
              tools:
                - "delete_all"
            """,
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["scan", str(sample_manifest_file), "--config", str(config_file), "--deny"],
        )
        assert result.exit_code == 1

    def test_cli_scan_with_deny_flag_clean_exits_zero(self, tmp_path: Path):
        clean_manifest = tmp_path / "mcp.json"
        clean_manifest.write_text(
            json.dumps(
                {
                    "name": "clean-server",
                    "tools": [{"name": "get_status", "description": "Safe status check"}],
                }
            ),
            encoding="utf-8",
        )
        config_file = tmp_path / "policy.yaml"
        config_file.write_text(
            """
            deny:
              servers:
                - "bad-server"
              tools:
                - "nuke"
            """,
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["scan", str(clean_manifest), "--config", str(config_file), "--deny"],
        )
        assert result.exit_code == 0

    def test_cli_scan_with_inline_deny_options(self, sample_manifest_file: Path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "scan",
                str(sample_manifest_file),
                "--deny-server",
                "untrusted-*",
                "--deny",
            ],
        )
        assert result.exit_code == 1
