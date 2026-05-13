from __future__ import annotations

from importlib import resources
from pathlib import Path
import shutil


def init_project(out_dir: Path, *, force: bool = False) -> list[Path]:
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} is not empty. Pass --force to overwrite demo files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    config_dst = out_dir / "demo_config.json"
    mesh_dst = out_dir / "demo_mesh"
    runs_dst = out_dir / "runs"
    mesh_dst.mkdir(parents=True, exist_ok=True)
    runs_dst.mkdir(parents=True, exist_ok=True)

    assets = resources.files("gnss_vim_sim.assets")
    _copy_resource(assets / "demo_config.json", config_dst)
    written.append(config_dst)

    mesh_assets = assets / "demo_mesh"
    for item in mesh_assets.iterdir():
        if item.is_file():
            dst = mesh_dst / item.name
            _copy_resource(item, dst)
            written.append(dst)

    readme = out_dir / "README.md"
    readme.write_text(
        "# GNSS-VIM-Sim Demo Project\n\n"
        "Run this project with:\n\n"
        "```bash\n"
        "gnss-vim-sim inspect-scene --config demo_config.json\n"
        "gnss-vim-sim studio --config demo_config.json --out runs/mission_studio.html\n"
        "gnss-vim-sim run --config demo_config.json --out runs/demo_run\n"
        "```\n"
    )
    written.append(readme)
    return written


def _copy_resource(src, dst: Path) -> None:
    with resources.as_file(src) as path:
        shutil.copyfile(path, dst)
