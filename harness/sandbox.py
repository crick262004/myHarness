"""OS sandbox for bash: the kernel, not the harness, decides what a command touches.

The permission policy checks a tool's input, but bash's input is a whole
shell program. `cat .env`, `cat .e*` and `python -c "open('.'+'env')"` all
read the same file, and no check on the command's text catches them all.
So on macOS, bash runs under Seatbelt (sandbox-exec), the sandbox macOS
uses for its own apps (Claude Code and Codex use it too). The kernel applies
the rules to every file the command, and anything it starts, opens:

  - secret files (the policy's SECRET_PATHS) can't be read or written,
    whatever the path is called (symlinks are resolved first);
  - writes go only to the working directory and the temp folders;
  - inside the working directory, .git/hooks and .git/config stay
    read-only: writing them would run code later, outside the sandbox.

Credential-like environment variables (API keys loaded from .env) are
removed from every bash command's environment, sandboxed or not.

Tools like uv want to write to a cache in your home folder. Opening that
up would let a command plant a tampered package your own tools later run
outside the sandbox, so sandboxed commands get a cache of their own
(XDG_CACHE_HOME pointed into the temp folder).

Network access is not restricted, and file names stay visible (`ls` works;
only contents are blocked).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .permissions import SECRET_PATHS

# Env vars that point HTTPS tools at a CA bundle. Behind a TLS-inspecting
# proxy (e.g. Zscaler) the bundle can sit in a folder the deny-list covers.
CA_BUNDLE_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")

# Variable names that look like credentials. Blocking .env is not enough:
# load_dotenv() has already copied its keys into this process's environment,
# and a child process inherits it, so `echo $ANTHROPIC_API_KEY` would work.
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


def scrubbed_env() -> dict[str, str]:
    """This process's environment minus anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def glob_to_regex(pattern: str) -> str:
    """A SECRET_PATHS glob as a Seatbelt regex, matching what the policy matches:
    a pattern without "/" matches the file name, one with "/" the whole path."""
    if "/" in pattern:
        body = "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern)
        return f"^{body}$"
    body = "".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in pattern)
    return f"/{body}$"


def _quote(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


@dataclass(frozen=True)
class Sandbox:
    writable: tuple[Path, ...] = ()  # besides the working directory
    readable: tuple[Path, ...] = ()  # exact files exempt from the secret deny-list
    cache: Path | None = None        # XDG_CACHE_HOME for sandboxed commands (must be writable)

    @classmethod
    def default(cls) -> Sandbox | None:
        """Temp folders writable, the configured CA bundle readable, a cache in
        temp. None where Seatbelt isn't available (bash then runs unsandboxed)."""
        if not available():
            return None
        temp = Path(tempfile.gettempdir()).resolve()
        bundles = {
            Path(os.environ[var]).expanduser().resolve()
            for var in CA_BUNDLE_VARS
            if os.environ.get(var) and Path(os.environ[var]).expanduser().is_file()
        }
        return cls(
            writable=tuple(sorted({temp, Path("/private/tmp")})),
            readable=tuple(sorted(bundles)),
            cache=temp / "myharness-sandbox-cache",
        )

    def env(self) -> dict[str, str]:
        """The environment for a sandboxed command."""
        env = scrubbed_env()
        if self.cache is not None:
            env["XDG_CACHE_HOME"] = str(self.cache)
        return env

    def profile(self, workdir: Path) -> str:
        """The Seatbelt rules. When several rules match, the LAST one wins."""
        workdir = workdir.resolve()
        writable = " ".join(f"(subpath {_quote(p)})" for p in (workdir, *self.writable))
        secrets = " ".join(f'(regex #"{glob_to_regex(p)}")' for p in SECRET_PATHS)
        lines = [
            "(version 1)",
            "(allow default)",
            "(deny file-write*)",
            f'(allow file-write* {writable} (subpath "/dev"))',
            f"(deny file-write* (subpath {_quote(workdir / '.git' / 'hooks')})"
            f" (literal {_quote(workdir / '.git' / 'config')}))",
            f"(deny file-read-data file-write* {secrets})",
        ]
        if self.readable:
            lines.append(f"(allow file-read-data {' '.join(f'(literal {_quote(p)})' for p in self.readable)})")
        return "\n".join(lines)

    def wrap(self, workdir: Path, command: str) -> list[str]:
        """argv that runs `command` in /bin/sh inside the sandbox."""
        return ["sandbox-exec", "-p", self.profile(workdir), "/bin/sh", "-c", command]
