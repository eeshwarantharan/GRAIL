from __future__ import annotations

from pathlib import Path
import argparse
import json
from datetime import datetime
import webbrowser

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.core.validation import format_report, validate_config
from gnss_vim_sim.io.init_project import init_project
from gnss_vim_sim.ml.runtime import load_runtime_model
from gnss_vim_sim.planning.interactive import plan_mission_interactively, preview_mission
from gnss_vim_sim.sim.runner import SimulationRunner
from gnss_vim_sim.viz.dashboard import make_dashboard
from gnss_vim_sim.viz.planner_html import make_planner_html
from gnss_vim_sim.viz.player import make_flight_player
from gnss_vim_sim.viz.plots import make_plots, replay_live


def run_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    report = validate_config(cfg, require_mesh=args.require_mesh)
    if report.warnings:
        print(format_report(report))
    report.raise_for_errors()
    checkpoint_arg = args.model_checkpoint or args.checkpoint
    checkpoint = Path(checkpoint_arg).resolve() if checkpoint_arg else None
    model = load_runtime_model(checkpoint)
    out_dir = _resolve_run_dir(args.out)
    runner = SimulationRunner(cfg, model, out_dir)
    summary = runner.run()
    make_plots(out_dir / "flight_log.csv", out_dir, runner.scene)
    drone_icon = _find_drone_icon()
    if make_flight_player(out_dir / "flight_log.csv", out_dir / "flight_player.html", runner.scene, drone_icon=drone_icon):
        print(f"Wrote 30fps flight player to: {out_dir / 'flight_player.html'}")
    if args.dashboard and make_dashboard(out_dir / "flight_log.csv", out_dir / "dashboard.html", runner.scene):
        print(f"Wrote interactive dashboard to: {out_dir / 'dashboard.html'}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote run artifacts to: {out_dir}")


def _resolve_run_dir(out: str | None) -> Path:
    if out:
        return Path(out).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / "runs" / f"run_{stamp}").resolve()


def _find_drone_icon() -> Path | None:
    for path in [Path.cwd() / "drone.png", Path.cwd().parent / "drone.png"]:
        if path.exists():
            return path.resolve()
    return None


def inspect_scene_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    report = validate_config(cfg, require_mesh=False)
    if report.errors or report.warnings:
        print(format_report(report))
    from gnss_vim_sim.world.scene import MeshScene

    scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    print(json.dumps(scene.stats().__dict__, indent=2, sort_keys=True))


def plan_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    validate_config(cfg).raise_for_errors()
    plan_mission_interactively(cfg, Path(args.out).resolve())
    print(f"Wrote mission config to: {Path(args.out).resolve()}")


def preview_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    validate_config(cfg).raise_for_errors()
    out = Path(args.out).resolve() if args.out else None
    preview_mission(cfg, out)
    if out:
        print(f"Wrote mission preview to: {out}")


def replay_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config) if args.config else None
    scene = None
    if cfg is not None:
        from gnss_vim_sim.world.scene import MeshScene

        scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    replay_live(Path(args.log).resolve(), scene=scene, step=args.step)


def dashboard_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config) if args.config else None
    scene = None
    if cfg is not None:
        from gnss_vim_sim.world.scene import MeshScene

        validate_config(cfg).raise_for_errors()
        scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    ok = make_dashboard(Path(args.log).resolve(), Path(args.out).resolve(), scene=scene, frame_step=args.step)
    if not ok:
        raise SystemExit("Plotly is required for dashboard rendering. Install with: pip install -e .")
    print(f"Wrote interactive dashboard to: {Path(args.out).resolve()}")


def player_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config) if args.config else None
    scene = None
    if cfg is not None:
        from gnss_vim_sim.world.scene import MeshScene

        validate_config(cfg).raise_for_errors()
        scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    ok = make_flight_player(
        Path(args.log).resolve(),
        Path(args.out).resolve(),
        scene=scene,
        drone_icon=Path(args.drone_icon).resolve() if args.drone_icon else _find_drone_icon(),
        fps=args.fps,
    )
    if not ok:
        raise SystemExit("Could not build flight player")
    print(f"Wrote 30fps flight player to: {Path(args.out).resolve()}")


def planner_html_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    validate_config(cfg).raise_for_errors()
    make_planner_html(cfg, Path(args.out).resolve())
    print(f"Wrote WebGL mission studio to: {Path(args.out).resolve()}")


def validate_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    report = validate_config(cfg, require_mesh=args.require_mesh)
    print(format_report(report))
    report.raise_for_errors()


def init_command(args: argparse.Namespace) -> None:
    written = init_project(Path(args.out), force=args.force)
    print(f"Initialized GNSS-VIM-Sim demo project at: {Path(args.out).resolve()}")
    for path in written:
        print(f"- {path}")


def wizard_command(args: argparse.Namespace) -> None:
    cfg = SimConfig.load(args.config)
    validate_config(cfg).raise_for_errors()
    workspace = _resolve_run_dir(args.out)
    workspace.mkdir(parents=True, exist_ok=True)
    planner_path = workspace / "mission_planner.html"
    make_planner_html(cfg, planner_path)

    print("\nGNSS-VIM guided flight workflow")
    print("=" * 34)
    print(f"1. Mission planner: {planner_path}")
    print("2. In the browser: choose takeoff, waypoints, landing, then click Download Mission Config.")
    print("3. Save the downloaded JSON, then return here and paste its path.")
    if not args.no_open:
        webbrowser.open(planner_path.as_uri())

    default_download = Path.home() / "Downloads" / "mission_config.json"
    entered = input(f"\nMission config path [{default_download}]: ").strip()
    mission_config = Path(entered).expanduser() if entered else default_download
    if not mission_config.exists():
        raise SystemExit(f"Mission config not found: {mission_config}\nRun cancelled before simulation.")

    checkpoint_arg = args.model_checkpoint or args.checkpoint
    checkpoint = Path(checkpoint_arg).resolve() if checkpoint_arg else None
    model = load_runtime_model(checkpoint)
    run_dir = workspace / "flight"
    mission_cfg = SimConfig.load(mission_config)
    validate_config(mission_cfg).raise_for_errors()
    runner = SimulationRunner(mission_cfg, model, run_dir)
    summary = runner.run()
    make_plots(run_dir / "flight_log.csv", run_dir, runner.scene)
    drone_icon = _find_drone_icon()
    make_flight_player(run_dir / "flight_log.csv", run_dir / "flight_player.html", runner.scene, drone_icon=drone_icon)
    if args.dashboard:
        make_dashboard(run_dir / "flight_log.csv", run_dir / "dashboard.html", runner.scene)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nOpen the 30fps player: {run_dir / 'flight_player.html'}")
    print(f"Wrote run artifacts to: {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnss-vim-sim")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a runnable demo project with a tiny mesh")
    init.add_argument("--out", default="gnss_vim_demo", help="output project directory")
    init.add_argument("--force", action="store_true", help="overwrite bundled demo files in an existing directory")
    init.set_defaults(func=init_command)

    validate = sub.add_parser("validate", help="validate a simulator config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--require-mesh", action="store_true", help="treat missing/empty mesh directory as an error")
    validate.set_defaults(func=validate_command)

    run = sub.add_parser("run", help="run a UAV mesh simulation")
    run.add_argument("--config", required=True, help="path to simulation JSON config")
    run.add_argument("--checkpoint", default=None, help="backward-compatible alias for --model-checkpoint")
    run.add_argument("--model-checkpoint", default=None, help="optional runtime model checkpoint with predict_proba/predict")
    run.add_argument("--out", default=None, help="output directory; omit to create runs/run_YYYYmmdd_HHMMSS")
    run.add_argument("--dashboard", action="store_true", help="also build the heavier Plotly dashboard.html")
    run.add_argument("--require-mesh", action="store_true", help="fail if scene mesh files are missing")
    run.set_defaults(func=run_command)

    inspect = sub.add_parser("inspect-scene", help="inspect mesh alignment and bounds")
    inspect.add_argument("--config", required=True)
    inspect.set_defaults(func=inspect_scene_command)

    plan = sub.add_parser("plan-mission", help="click waypoints on the map and save a mission config")
    plan.add_argument("--config", required=True, help="base config to copy sensor/fusion settings from")
    plan.add_argument("--out", required=True, help="output mission config JSON")
    plan.set_defaults(func=plan_command)

    preview = sub.add_parser("preview-mission", help="render a waypoint/path preview")
    preview.add_argument("--config", required=True)
    preview.add_argument("--out", default=None, help="optional PNG output path; omit to show a window")
    preview.set_defaults(func=preview_command)

    replay = sub.add_parser("replay", help="live replay a flight_log.csv")
    replay.add_argument("--log", required=True)
    replay.add_argument("--config", default=None, help="optional config for mesh underlay")
    replay.add_argument("--step", type=int, default=30)
    replay.set_defaults(func=replay_command)

    dashboard = sub.add_parser("dashboard", help="build an interactive 3D HTML replay/dashboard")
    dashboard.add_argument("--log", required=True)
    dashboard.add_argument("--config", default=None, help="optional config for mesh underlay")
    dashboard.add_argument("--out", default="runs/dashboard.html")
    dashboard.add_argument("--step", type=int, default=25)
    dashboard.set_defaults(func=dashboard_command)

    player = sub.add_parser("player", help="build a lightweight 30fps browser flight player")
    player.add_argument("--log", required=True)
    player.add_argument("--config", default=None, help="optional config for mesh underlay")
    player.add_argument("--out", default="runs/flight_player.html")
    player.add_argument("--drone-icon", default=None, help="path to drone.png")
    player.add_argument("--fps", type=int, default=30)
    player.set_defaults(func=player_command)

    planner_html = sub.add_parser("planner-html", help="build a browser WebGL mission planning page")
    planner_html.add_argument("--config", required=True)
    planner_html.add_argument("--out", default="runs/planner.html")
    planner_html.set_defaults(func=planner_html_command)

    studio = sub.add_parser("studio", help="build a browser WebGL mission studio")
    studio.add_argument("--config", required=True)
    studio.add_argument("--out", default="runs/mission_studio.html")
    studio.set_defaults(func=planner_html_command)

    wizard = sub.add_parser("wizard", help="guided browser waypoint planning followed by a simulation run")
    wizard.add_argument("--config", required=True, help="base config used for scene, sensors, fusion, and energy")
    wizard.add_argument("--checkpoint", default=None, help="backward-compatible alias for --model-checkpoint")
    wizard.add_argument("--model-checkpoint", default=None, help="optional runtime model checkpoint with predict_proba/predict")
    wizard.add_argument("--out", default=None, help="workflow directory; omit to create runs/run_YYYYmmdd_HHMMSS")
    wizard.add_argument("--dashboard", action="store_true", help="also build the heavier Plotly dashboard.html")
    wizard.add_argument("--no-open", action="store_true", help="do not attempt to open the planner in a browser")
    wizard.set_defaults(func=wizard_command)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
