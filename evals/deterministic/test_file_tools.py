from pathlib import Path

import pytest

from harness.permissions import Policy
from harness.tools import ToolError, build_builtin_tools


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text("import os\nTIMEOUT = 30\n")
    (tmp_path / "src" / "pkg" / "util.py").write_text("def helper():\n    return TIMEOUT\n")
    (tmp_path / "README.md").write_text("TIMEOUT is in seconds\n")
    (tmp_path / ".env").write_text("TIMEOUT_SECRET=hunter2\n")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfeTIMEOUT\x00")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "dep.py").write_text("TIMEOUT = 1\n")
    return tmp_path


def tools(workdir: Path):
    return {t.name: t.handler for t in build_builtin_tools(workdir)}


async def test_list_files_walks_the_tree_but_skips_package_dirs(tree):
    listed = await tools(tree)["list_files"]({})
    assert listed.splitlines() == [".env", "README.md", "blob.bin", "src/app.py", "src/pkg/util.py"]


async def test_list_files_filters_by_name_and_directory(tree):
    assert (await tools(tree)["list_files"]({"pattern": "*.py"})).splitlines() == [
        "src/app.py", "src/pkg/util.py",
    ]
    assert await tools(tree)["list_files"]({"path": "src/pkg"}) == "src/pkg/util.py"


async def test_search_files_returns_path_line_and_text(tree):
    found = await tools(tree)["search_files"]({"regex": r"TIMEOUT\b"})
    assert found.splitlines() == [
        "README.md:1: TIMEOUT is in seconds",
        "src/app.py:2: TIMEOUT = 30",
        "src/pkg/util.py:2: return TIMEOUT",
    ]  # not .env (secret), not blob.bin (binary), not .venv


async def test_search_files_never_reads_secret_files(tree):
    assert await tools(tree)["search_files"]({"regex": "hunter2"}) == "(no matches)"


async def test_search_files_skips_links_that_leave_the_workdir(tree, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "notes.txt"
    outside.write_text("TIMEOUT from outside\n")
    (tree / "link.txt").symlink_to(outside)
    assert "outside" not in await tools(tree)["search_files"]({"regex": "TIMEOUT"})


async def test_bad_input_is_a_tool_error(tree):
    with pytest.raises(ToolError, match="Invalid regex"):
        await tools(tree)["search_files"]({"regex": "("})
    with pytest.raises(ToolError, match="escapes"):
        await tools(tree)["list_files"]({"path": ".."})
    with pytest.raises(ToolError, match="Not found"):
        await tools(tree)["list_files"]({"path": "nope"})


def test_policy_allows_both_but_not_on_secret_paths(tree):
    policy = Policy(tree)
    assert policy.check("list_files", {}).decision == "allow"
    assert policy.check("search_files", {"regex": "x"}).decision == "allow"
    assert policy.check("search_files", {"regex": "x", "path": ".env"}).decision == "deny"
