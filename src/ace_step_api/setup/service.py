"""``--service``: run the server at login with launchd, then wait for it to answer."""

from __future__ import annotations

import os
import plistlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ace_step_api.setup.tools import SetupError, Tools

REPO = "ace-step-api"
LABEL = f"com.github.kabatin.{REPO}"
HEALTH_WAIT_S = 60.0
HEALTH_POLL_S = 2.0
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
_SUFFIX = re.compile(r"^[A-Za-z0-9-]{1,32}$")


@dataclass(frozen=True)
class Service:
    label: str
    plist: Path
    log: Path

    def as_json(self, *, healthy: bool) -> dict[str, Any]:
        return {"installed": self.plist.is_file(), "label": self.label, "healthy": healthy}


def check_suffix(suffix: str) -> None:
    if not _SUFFIX.fullmatch(suffix):
        raise SetupError("--label-suffix takes letters, digits and '-' (at most 32)")


def service_for(tools: Tools, suffix: str | None) -> Service:
    if suffix is not None:
        check_suffix(suffix)
    tail = f".{suffix}" if suffix else ""
    return Service(
        label=LABEL + tail,
        plist=tools.home / "Library" / "LaunchAgents" / f"{LABEL}{tail}.plist",
        log=tools.home / "Library" / "Logs" / f"{REPO}{tail}.log",
    )


def plist_bytes(service: Service, cli: str, config: Path | None) -> bytes:
    env = {"PATH": PATH} | ({"ACE_STEP_API_CONFIG": str(config)} if config else {})
    return plistlib.dumps(
        {
            "Label": service.label,
            "ProgramArguments": [cli, "serve"],
            "EnvironmentVariables": env,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(service.log),
            "StandardErrorPath": str(service.log),
        }
    )


def install(service: Service, tools: Tools, config: Path | None) -> None:
    """Write the plist and (re)load it. ``config``: a non-default config path, if any."""
    found = tools.which(REPO)
    if found is None:
        raise SetupError(
            f"{REPO} is not on PATH",
            hint="install it first",
            command=f"uv tool install git+https://github.com/kabatin/{REPO}",
        )
    cli = str(Path(found).absolute())  # not resolved: `uv tool upgrade` keeps this path working
    service.plist.parent.mkdir(parents=True, exist_ok=True)
    service.log.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".plist-", dir=service.plist.parent)
    with os.fdopen(fd, "wb") as handle:
        handle.write(plist_bytes(service, cli, config))
    Path(tmp).chmod(0o644)
    Path(tmp).replace(service.plist)
    domain = f"gui/{tools.uid}"
    tools.run(["launchctl", "bootout", f"{domain}/{service.label}"], None, None)  # not loaded: ok
    if tools.run(["launchctl", "bootstrap", domain, str(service.plist)], None, None) != 0:
        raise SetupError(
            f"launchctl bootstrap {service.label} failed",
            hint=f"see {service.log}",
            command=f"launchctl bootstrap {domain} {service.plist}",
        )
    tools.emit(f"service: {service.label} loaded ({service.plist})")


def wait_healthy(url: str, token: str | None, tools: Tools, wait_s: float = HEALTH_WAIT_S) -> bool:
    deadline = tools.clock() + wait_s
    while True:
        if tools.http_status(f"{url}/v1/health", token) == 200:
            return True
        if tools.clock() >= deadline:
            return False
        tools.sleep(HEALTH_POLL_S)
