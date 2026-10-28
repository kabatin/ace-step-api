"""``ace-step-api setup``: the uniform install command (see README "Setup on the Mac").

Exit codes: 0 ok, 1 failure, 2 bad arguments, 3 needs a confirmation (or a person).
Progress goes to stderr; with ``--json`` stdout carries exactly one JSON object.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ace_step_api import __version__
from ace_step_api.settings import (
    DEFAULT_PORT,
    Settings,
    SettingsError,
    load_settings,
    resolve_config_path,
)
from ace_step_api.setup import config_write, pins, service
from ace_step_api.setup.ace_step import AceStepComponent
from ace_step_api.setup.component import Component, Context, Plan
from ace_step_api.setup.machine import Machine, probe
from ace_step_api.setup.minimax import MiniMaxComponent
from ace_step_api.setup.tools import SetupError, Tools

COMPONENTS: tuple[Component, ...] = (AceStepComponent(), MiniMaxComponent())
DEFAULT_PREFIX = Path("~/.local/opt/ace-step-api")
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_CONFIRM = 0, 1, 2, 3


@dataclass(frozen=True)
class Options:
    models: tuple[str, ...]
    explicit: bool = False  # --models named them: one setup cannot install is a failure
    check: bool = False
    yes: bool = False
    lan: bool = False
    service: bool = False
    port: int | None = None
    prefix: Path | None = None
    label_suffix: str | None = None
    coordinate_with: tuple[str, ...] = ()
    config: Path | None = None


@dataclass(frozen=True)
class Outcome:
    id: str
    status: str  # installed | already | missing | unsupported | failed | needs_human
    message: str
    download_bytes: int = 0
    fix: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "downloadBytes": self.download_bytes,
        }
        return out | ({"fix": self.fix} if self.fix else {})


@dataclass(frozen=True)
class Report:
    action: str
    ok: bool
    config_path: Path
    machine: Machine
    service: dict[str, Any]
    components: tuple[Outcome, ...] = ()
    url: str = ""
    lan: bool = False
    token_set: bool = False
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "server": service.REPO,
            "version": __version__,
            "ok": self.ok,
            "action": self.action,
            "url": self.url,
            "lan": self.lan,
            "tokenSet": self.token_set,
            "configPath": str(self.config_path),
            "service": self.service,
            "machine": self.machine.as_json(),
            "components": [c.as_json() for c in self.components],
        }
        return out | ({"error": self.error} if self.error else {})


def parse_models(value: str | None) -> tuple[str, ...]:
    valid = [c.id for c in COMPONENTS]
    if value is None or value.strip() == "":
        return tuple(valid)
    wanted = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [m for m in wanted if m not in valid]
    if unknown:
        raise SetupError(f"unknown model(s) {', '.join(unknown)}; valid: {', '.join(valid)}")
    return wanted


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def run_setup(
    opts: Options,
    tools: Tools,
    *,
    env: Mapping[str, str] | None = None,
    confirm: Callable[[str], bool] | None = None,
    probe_machine: Callable[[Path, Tools], Machine] = probe,
) -> tuple[int, Report]:
    """Do (or with ``check`` only report) the setup. ``confirm`` is None when nobody can be
    asked (no TTY)."""
    environ = dict(os.environ if env is None else env)
    config_path = resolve_config_path(environ, opts.config)
    prefix = (opts.prefix or DEFAULT_PREFIX).expanduser().absolute()
    machine = probe_machine(prefix, tools)
    svc = service.service_for(tools, opts.label_suffix)
    action = "check" if opts.check else "install"
    base = Report(action, True, config_path, machine, svc.as_json(healthy=False))
    try:
        existing = config_write.read(config_path)
        settings = _settings(environ, config_path)
    except SetupError as exc:
        return EXIT_FAILED, replace(base, ok=False, error=str(exc))
    url = f"http://127.0.0.1:{opts.port or settings.port}"
    base = replace(base, url=url, lan=_is_lan(settings), token_set=settings.token is not None)
    ctx = Context(prefix, existing.get("backends", {}), tools)
    selected = [c for c in COMPONENTS if c.id in opts.models]
    plans = {c.id: _plan(c, ctx) for c in selected}

    if opts.check:
        healthy = tools.http_status(f"{url}/v1/health", _token(settings)) == 200
        outcomes = tuple(_checked(c, plans[c.id], machine) for c in selected)
        ok = _ok(outcomes, opts.explicit, check=True)
        report = replace(base, ok=ok, components=outcomes, service=svc.as_json(healthy=healthy))
        return (EXIT_OK if ok else EXIT_FAILED), report

    cannot = {c.id: why for c in selected if (why := _unsupported(plans[c.id], machine))}
    todo = [c for c in selected if _pending(plans[c.id]) and c.id not in cannot]
    gate = _gate(todo, plans, machine, opts, confirm)
    if gate is not None:
        code, message, fix = gate
        outcomes = tuple(
            Outcome(c.id, "missing", message, _bytes(plans[c.id]), fix)
            if c in todo
            else _done(c, plans[c.id], cannot)
            for c in selected
        )
        return code, replace(base, ok=False, components=outcomes, error=message)

    outcomes = tuple(
        _install(c, plans[c.id], ctx) if c in todo else _done(c, plans[c.id], cannot)
        for c in selected
    )
    try:
        changed = _write_config(config_path, existing, selected, outcomes, plans, opts, environ)
        if changed:
            tools.emit(f"config: updated {config_path}")
        settings = _settings(environ, config_path)
    except SetupError as exc:
        return EXIT_FAILED, replace(base, ok=False, components=outcomes, error=str(exc))
    base = replace(
        base,
        url=f"http://127.0.0.1:{settings.port}",
        lan=_is_lan(settings),
        token_set=settings.token is not None,
    )
    report = replace(base, ok=_ok(outcomes, opts.explicit), components=outcomes)
    if opts.service:
        report = _service(report, svc, settings, config_path, tools)
    if not report.ok:
        return EXIT_FAILED, report
    return EXIT_OK, report


# -- steps ---------------------------------------------------------------------------------------
def _settings(env: Mapping[str, str], path: Path) -> Settings:
    try:
        return load_settings(env=env, config_path=path)
    except SettingsError as exc:
        raise SetupError(str(exc)) from exc


def _token(settings: Settings) -> str | None:
    return settings.token.get_secret_value() if settings.token else None


def _is_lan(settings: Settings) -> bool:
    return settings.host not in ("127.0.0.1", "localhost", "::1")


def _plan(component: Component, ctx: Context) -> Plan | SetupError:
    try:
        return component.plan(ctx)
    except SetupError as exc:
        return exc


def _ok(outcomes: Sequence[Outcome], explicit: bool, *, check: bool = False) -> bool:
    """Unsupported is fine for the default selection, a failure when asked for by name;
    a check is fine with missing components (reporting them is its job)."""
    allowed = {"installed", "already"} | (set() if explicit else {"unsupported"})
    allowed |= {"missing"} if check else set()
    return all(o.status in allowed for o in outcomes)


def _unsupported(plan: Plan | SetupError, machine: Machine) -> str | None:
    if not isinstance(plan, Plan) or plan.working:
        return None
    return plan.reason if plan.unsupported else machine.supported_reason


def _pending(plan: Plan | SetupError) -> bool:
    return isinstance(plan, Plan) and not plan.working


def _bytes(plan: Plan | SetupError) -> int:
    return plan.download_bytes if isinstance(plan, Plan) else 0


def _fix(exc: SetupError) -> dict[str, Any] | None:
    if not (exc.hint or exc.command):
        return None
    return {"hint": exc.hint, "command": exc.command, "needsHuman": exc.human}


def _checked(component: Component, plan: Plan | SetupError, machine: Machine) -> Outcome:
    if isinstance(plan, SetupError):
        return Outcome(component.id, "failed", str(plan), fix=_fix(plan))
    if plan.working:
        note = "; setup will add it to the config" if plan.config else ""
        return Outcome(component.id, "installed", f"ready{note}")
    if (why := _unsupported(plan, machine)) is not None:
        return Outcome(component.id, "unsupported", why)
    return Outcome(
        component.id,
        "missing",
        f"{plan.reason}; would download ~{gb(plan.download_bytes)}",
        plan.download_bytes,
    )


def _done(
    component: Component, plan: Plan | SetupError, cannot: Mapping[str, str] | None = None
) -> Outcome:
    if isinstance(plan, SetupError):
        return Outcome(component.id, "failed", str(plan), fix=_fix(plan))
    if cannot and component.id in cannot:
        return Outcome(component.id, "unsupported", cannot[component.id])
    return Outcome(component.id, "already", "ready")


def _gate(
    todo: Sequence[Component],
    plans: Mapping[str, Plan | SetupError],
    machine: Machine,
    opts: Options,
    confirm: Callable[[str], bool] | None,
) -> tuple[int, str, dict[str, Any]] | None:
    """Disk and consent before anything is downloaded; None when setup may go on."""
    if not todo:
        return None
    todo_plans = [p for c in todo if isinstance(p := plans[c.id], Plan)]
    download = sum(p.download_bytes for p in todo_plans)
    need = int(sum(p.disk_bytes for p in todo_plans) * pins.DISK_MARGIN) + pins.DISK_RESERVE_BYTES
    if machine.free_bytes < need:
        message = (
            f"needs ~{gb(need)} free disk, {gb(machine.free_bytes)} free (download ~{gb(download)})"
        )
        fix = {
            "hint": "free disk space, or install one model at a time",
            "command": "",
            "needsHuman": True,
        }
        return EXIT_FAILED, message, fix
    if opts.yes:
        return None
    ids = ",".join(c.id for c in todo)
    prompt = "\n".join(
        [f"setup will download ~{gb(download)} for {ids}:", *(f"  {c.licence}" for c in todo)]
    )
    if confirm is not None and confirm(prompt + "\ncontinue? [y/N] "):
        return None
    fix = {
        "hint": "rerun with --yes",
        "command": f"ace-step-api setup --models {ids} --yes",
        "needsHuman": False,
    }
    reason = "not confirmed" if confirm is not None else "needs confirmation (no terminal)"
    return EXIT_CONFIRM, f"{reason}: would download ~{gb(download)}", fix


def _install(component: Component, plan: Plan | SetupError, ctx: Context) -> Outcome:
    if isinstance(plan, SetupError) or plan.working:
        return _done(component, plan)
    ctx.tools.emit(f"{component.id}: installing ({plan.reason})")
    try:
        component.install(ctx)
    except SetupError as exc:
        ctx.tools.emit(f"{component.id}: failed: {exc}")
        return Outcome(component.id, "failed", str(exc), plan.download_bytes, _fix(exc))
    ctx.tools.emit(f"{component.id}: installed")
    return Outcome(component.id, "installed", "ready", plan.download_bytes)


def _write_config(
    path: Path,
    existing: dict[str, Any],
    selected: Sequence[Component],
    outcomes: Sequence[Outcome],
    plans: Mapping[str, Plan | SetupError],
    opts: Options,
    env: Mapping[str, str],
) -> bool:
    ready = {o.id for o in outcomes if o.status in ("installed", "already")}
    backends = {
        c.backend: plan.config
        for c in selected
        if c.id in ready and isinstance(plan := plans[c.id], Plan) and plan.config
    }
    changes = config_write.Changes(
        backends=backends,
        lan=opts.lan,
        port=opts.port,
        peers=tuple(config_write.parse_peer(spec) for spec in opts.coordinate_with),
    )
    return config_write.write(path, config_write.merge(existing, changes, env))


def _service(
    report: Report,
    svc: service.Service,
    settings: Settings,
    config_path: Path,
    tools: Tools,
) -> Report:
    custom = config_path if config_path != resolve_config_path({}, None) else None
    try:
        service.install(svc, tools, custom)
    except SetupError as exc:
        return replace(report, ok=False, error=str(exc), service=svc.as_json(healthy=False))
    tools.emit(f"service: waiting for {report.url}/v1/health")
    healthy = service.wait_healthy(report.url, _token(settings), tools)
    error = None if healthy else f"the service did not answer within 60 s; see {svc.log}"
    return replace(
        report,
        ok=report.ok and healthy,
        error=report.error or error,
        service=svc.as_json(healthy=healthy),
    )


# -- command line --------------------------------------------------------------------------------
def add_parser(sub: Any) -> None:
    ids = ", ".join(c.id for c in COMPONENTS)
    parser = sub.add_parser("setup", help="install the engines, write the config, run at login")
    parser.add_argument("--models", help=f"comma list of {ids} (default: all)")
    parser.add_argument("--check", action="store_true", help="report only, change nothing")
    parser.add_argument("--yes", action="store_true", help="download without asking")
    parser.add_argument("--json", action="store_true", help="one JSON object on stdout")
    parser.add_argument("--lan", action="store_true", help="listen on the LAN (sets a token)")
    parser.add_argument("--service", action="store_true", help="run at login with launchd")
    parser.add_argument("--port", type=int)
    parser.add_argument("--prefix", help=f"where engines go (default {DEFAULT_PREFIX})")
    parser.add_argument("--label-suffix", help="a second launchd label beside the real one")
    parser.add_argument(
        "--coordinate-with",
        action="append",
        default=[],
        metavar="NAME=HEALTH_URL",
        help="another GPU server to wait for (repeatable)",
    )


def options_from(args: Any, env: Mapping[str, str]) -> Options:
    """Everything that can be wrong with the arguments, found before anything runs (exit 2)."""
    if args.port is not None and not 1 <= args.port <= 65535:
        raise SetupError("--port must be 1-65535")
    if args.label_suffix is not None:
        service.check_suffix(args.label_suffix)
        if args.port is None:
            raise SetupError("--label-suffix needs --port: the real service has the default one")
    if args.lan:
        config_write.check_lan_env(env)
    for spec in args.coordinate_with:
        config_write.check_peer(*config_write.parse_peer(spec), env)
    return Options(
        models=parse_models(args.models),
        explicit=bool(args.models and args.models.strip()),
        check=args.check,
        yes=args.yes,
        lan=args.lan,
        service=args.service,
        port=args.port,
        prefix=Path(args.prefix) if args.prefix else None,
        label_suffix=args.label_suffix,
        coordinate_with=tuple(args.coordinate_with),
        config=Path(args.config) if args.config else None,
    )


def usage_report(message: str, args: Any, tools: Tools, env: Mapping[str, str]) -> Report:
    """The complete report for an argument error (nothing was looked at but the machine)."""
    config = Path(args.config) if getattr(args, "config", None) else None
    prefix = DEFAULT_PREFIX.expanduser().absolute()
    return Report(
        action="check" if getattr(args, "check", False) else "install",
        ok=False,
        config_path=resolve_config_path(env, config),
        machine=probe(prefix, tools),
        service=service.service_for(tools, None).as_json(healthy=False),
        url=f"http://127.0.0.1:{DEFAULT_PORT}",
        error=message,
    )


def describe(report: Report) -> list[str]:
    m = report.machine
    lines = [
        f"{service.REPO} {__version__} setup ({report.action}): {'ok' if report.ok else 'not ok'}",
        f"  config   {report.config_path}",
        f"  url      {report.url} (lan: {report.lan}, token set: {report.token_set})",
        f"  machine  arm64={m.arm64} macOS={m.macos} RAM={m.ram_gb} GB free={gb(m.free_bytes)} "
        + " ".join(f"{k}={'yes' if v else 'no'}" for k, v in m.tools.items()),
        f"  service  {report.service['label']} installed={report.service['installed']} "
        f"healthy={report.service['healthy']}",
    ]
    for o in report.components:
        lines.append(f"  {o.id:<16} {o.status:<12} {o.message}")
        if o.fix:
            lines.append(f"  {'':<16} fix: {o.fix['hint']} {o.fix['command']}".rstrip())
    if report.error:
        lines.append(f"  error: {report.error}")
    return lines
