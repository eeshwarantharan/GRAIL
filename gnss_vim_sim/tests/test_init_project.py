from pathlib import Path

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.core.validation import validate_config
from gnss_vim_sim.io.init_project import init_project


def test_init_project_writes_runnable_demo_config(tmp_path):
    out = tmp_path / "demo"

    written = init_project(out)

    assert out / "demo_config.json" in written
    assert (out / "demo_mesh" / "demo_scene.ply").exists()
    cfg = SimConfig.load(out / "demo_config.json")
    assert validate_config(cfg, require_mesh=True).ok


def test_init_project_refuses_non_empty_directory_without_force(tmp_path):
    out = tmp_path / "demo"
    out.mkdir()
    (out / "existing.txt").write_text("keep")

    try:
        init_project(out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("init_project should refuse non-empty directories without force=True")
