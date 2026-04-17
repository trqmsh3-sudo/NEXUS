"""Deployment configuration tests — TDD for Dockerfile, docker-compose, Caddyfile, and live health.

Coverage:
  1. Dockerfile — exists, correct base image, port, CMD
  2. docker-compose.yml — exists, nexus + caddy services, volume mount, env_file
  3. Caddyfile — exists, reverse_proxy to nexus service
  4. Live health check — GET /api/health returns 200 + valid JSON (skipped if offline)
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

import pytest

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # NEXUS/
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
CADDYFILE = PROJECT_ROOT / "Caddyfile"

SERVER_HOST = "159.65.124.135"
SERVER_PORT = 80
HEALTH_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/api/health"


def _server_reachable() -> bool:
    try:
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=5):
            return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  1. Dockerfile
# ══════════════════════════════════════════════════════════════════

class TestDockerfile:
    def test_dockerfile_exists(self):
        assert DOCKERFILE.exists(), "Dockerfile not found at project root"

    def test_uses_python_312_slim(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "python:3.12" in content, "Dockerfile must use python:3.12 base image"

    def test_sets_workdir_app(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "WORKDIR /app" in content

    def test_copies_requirements(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY requirements.txt" in content

    def test_runs_pip_install(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "pip install" in content and "requirements.txt" in content

    def test_copies_nexus_source(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY nexus/" in content

    def test_copies_data_directory(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "COPY data/" in content

    def test_exposes_port_8000(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "EXPOSE 8000" in content

    def test_cmd_runs_uvicorn(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "uvicorn" in content and "nexus.api:app" in content

    def test_cmd_binds_all_interfaces(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "0.0.0.0" in content

    def test_runs_as_non_root(self):
        content = DOCKERFILE.read_text(encoding="utf-8")
        assert "USER " in content, "Dockerfile must run as non-root user"


# ══════════════════════════════════════════════════════════════════
#  2. docker-compose.yml
# ══════════════════════════════════════════════════════════════════

class TestDockerCompose:
    def _content(self) -> str:
        assert COMPOSE_FILE.exists(), "docker-compose.yml not found at project root"
        return COMPOSE_FILE.read_text(encoding="utf-8")

    def test_compose_file_exists(self):
        assert COMPOSE_FILE.exists()

    def test_defines_nexus_service(self):
        assert "nexus:" in self._content()

    def test_defines_caddy_service(self):
        assert "caddy:" in self._content()

    def test_nexus_has_restart_policy(self):
        assert "restart:" in self._content()

    def test_nexus_mounts_data_volume(self):
        content = self._content()
        assert "./data:/app/data" in content, "data/ must be mounted as volume for persistence"

    def test_nexus_uses_env_file(self):
        content = self._content()
        assert "env_file" in content or ".env" in content

    def test_caddy_exposes_port_80(self):
        assert '"80:80"' in self._content() or "80:80" in self._content()

    def test_caddy_exposes_port_443(self):
        assert '"443:443"' in self._content() or "443:443" in self._content()

    def test_caddy_mounts_caddyfile(self):
        assert "Caddyfile" in self._content()

    def test_caddy_depends_on_nexus(self):
        content = self._content()
        assert "depends_on" in content

    def test_caddy_data_volume_defined(self):
        content = self._content()
        assert "caddy_data" in content


# ══════════════════════════════════════════════════════════════════
#  3. Caddyfile
# ══════════════════════════════════════════════════════════════════

class TestCaddyfile:
    def test_caddyfile_exists(self):
        assert CADDYFILE.exists(), "Caddyfile not found at project root"

    def test_has_reverse_proxy(self):
        content = CADDYFILE.read_text(encoding="utf-8")
        assert "reverse_proxy" in content

    def test_proxies_to_nexus_service(self):
        content = CADDYFILE.read_text(encoding="utf-8")
        assert "nexus:8000" in content, "Caddy must proxy to nexus service on port 8000"

    def test_listens_on_port_80(self):
        content = CADDYFILE.read_text(encoding="utf-8")
        assert ":80" in content


# ══════════════════════════════════════════════════════════════════
#  4. Live health check
# ══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _server_reachable(), reason=f"{SERVER_HOST}:{SERVER_PORT} not reachable")
class TestLiveHealthEndpoint:
    def test_health_returns_200(self):
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200, f"Expected 200, got {resp.status}"

    def test_health_returns_json(self):
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        assert isinstance(data, dict), "Health endpoint must return a JSON object"

    def test_health_has_total_beliefs_field(self):
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "total_beliefs" in data, f"Missing 'total_beliefs' in health response: {data}"

    def test_health_has_system_score(self):
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "system_score" in data

    def test_health_has_daily_cost(self):
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert "daily_cost" in data

    def test_root_returns_html(self):
        url = f"http://{SERVER_HOST}:{SERVER_PORT}/"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("content-type", "")
            assert "html" in content_type.lower()
