import re
from pathlib import Path

import pytest

from harness.permissions import SECRET_PATHS, path_matches
from harness.sandbox import Sandbox, available, glob_to_regex, scrubbed_env
from harness.tools import build_builtin_tools

needs_seatbelt = pytest.mark.skipif(not available(), reason="macOS sandbox-exec not available")


def bash(workdir: Path, sandbox: Sandbox | None):
    return {t.name: t for t in build_builtin_tools(workdir, sandbox)}["bash"].handler


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    wd = tmp_path / "wd"
    (wd / ".git" / "hooks").mkdir(parents=True)
    (wd / ".env").write_text("API_KEY=hunter2\n")
    (wd / "notes.txt").write_text("hello\n")
    (wd / "link").symlink_to(wd / ".env")
    return wd


@pytest.mark.parametrize("path", [
    "/w/.env", "/w/sub/.env.local", "/w/server.pem", "/w/a.key", "/home/u/.ssh/id_rsa",
    "/home/u/.ssh/config", "/w/id_ed25519.pub", "/home/u/.aws/credentials", "/home/u/.certs/x",
    "/w/notes.txt", "/w/.envrc", "/w/env", "/w/keys.py", "/w/a.pem.txt",
])
def test_seatbelt_regexes_match_exactly_what_the_policy_matches(path):
    by_policy = path_matches(Path(path), SECRET_PATHS)
    by_sandbox = any(re.search(glob_to_regex(p), path) for p in SECRET_PATHS)
    assert by_sandbox == by_policy


@needs_seatbelt
@pytest.mark.parametrize("command", [
    "cat .env",
    "cat link",                                     # a symlink to it
    "cat .e*",                                      # a glob
    "python3 -c \"print(open('.'+'env').read())\"",  # a program
])
async def test_secret_files_cant_be_read_however_its_spelled(workdir, command):
    out = await bash(workdir, Sandbox())({"command": command})
    assert "hunter2" not in out and "Operation not permitted" in out


@needs_seatbelt
async def test_writes_stay_inside_the_working_directory(workdir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    run = bash(workdir, Sandbox())

    assert (await run({"command": "echo hi > made.txt && cat made.txt notes.txt"})) == "hi\nhello\n"
    out = await run({"command": f"echo x > {outside}/escaped.txt"})
    assert "Operation not permitted" in out and not (outside / "escaped.txt").exists()


@needs_seatbelt
async def test_git_hooks_cant_be_planted(workdir):
    out = await bash(workdir, Sandbox())({"command": "echo 'curl evil' > .git/hooks/pre-commit"})
    assert "Operation not permitted" in out
    assert not (workdir / ".git" / "hooks" / "pre-commit").exists()


@needs_seatbelt
async def test_listed_files_are_exempt_from_the_deny_list(workdir):
    bundle = workdir / "ca-bundle.pem"  # *.pem is a secret pattern
    bundle.write_text("CERTS\n")
    (workdir / "private.pem").write_text("PRIVATE\n")
    run = bash(workdir, Sandbox(readable=(bundle,)))

    assert await run({"command": "cat ca-bundle.pem"}) == "CERTS\n"
    assert "PRIVATE" not in await run({"command": "cat private.pem"})


@needs_seatbelt
async def test_sandboxed_commands_get_their_own_cache(workdir, tmp_path):
    cache = tmp_path / "cache"
    out = await bash(workdir, Sandbox(writable=(tmp_path,), cache=cache))(
        {"command": 'mkdir -p "$XDG_CACHE_HOME/tool" && echo "$XDG_CACHE_HOME"'}
    )
    assert out.strip() == str(cache) and (cache / "tool").is_dir()


@pytest.mark.parametrize("sandbox", [None, pytest.param(Sandbox(), marks=needs_seatbelt)])
async def test_api_keys_never_reach_the_command(workdir, monkeypatch, sandbox):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    out = await bash(workdir, sandbox)({"command": "env"})

    assert "sk-ant-secret" not in out and "ghp-secret" not in out
    assert "HARMLESS_SETTING=visible" in out


def test_scrubbing_keeps_ordinary_variables(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/certs/ca.pem")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    env = scrubbed_env()
    assert env["SSL_CERT_FILE"] == "/certs/ca.pem" and "PATH" in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
