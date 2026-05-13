from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SceneStats:
    mesh_count: int
    vertex_count: int
    bounds_min: list[float] | None
    bounds_max: list[float] | None


class MeshScene:
    """Optional PLY-backed scene aligned to the simulator ENU frame."""

    def __init__(self, mesh_dir: Path | None, blend_file: Path | None = None):
        self.mesh_dir = mesh_dir
        self.blend_file = blend_file
        self.mesh = None
        self.mesh_parts = []
        self.vertices = np.empty((0, 3), dtype=float)
        self._load()

    def _load(self) -> None:
        if self.mesh_dir is None or not self.mesh_dir.exists():
            return
        try:
            import trimesh
        except Exception:
            self._load_vertices_only()
            return

        meshes = []
        for path in sorted(self.mesh_dir.glob("*.ply")):
            try:
                mesh = trimesh.load(path, force="mesh", process=False)
                meshes.append(mesh)
                self.mesh_parts.append((path.name, mesh))
            except Exception:
                continue
        if not meshes:
            return
        self.mesh = trimesh.util.concatenate(meshes)
        if hasattr(self.mesh, "vertices"):
            self.vertices = np.asarray(self.mesh.vertices, dtype=float)

    def _load_vertices_only(self) -> None:
        verts: list[list[float]] = []
        for path in sorted(self.mesh_dir.glob("*.ply")):
            try:
                lines = path.read_text(errors="ignore").splitlines()
            except Exception:
                continue
            vertex_count = 0
            header_end = None
            is_ascii = False
            for i, line in enumerate(lines):
                if line.startswith("format ascii"):
                    is_ascii = True
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                if line.strip() == "end_header":
                    header_end = i + 1
                    break
            if header_end is None or not is_ascii:
                continue
            for row in lines[header_end : header_end + vertex_count]:
                parts = row.split()
                if len(parts) >= 3:
                    try:
                        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
        if verts:
            self.vertices = np.asarray(verts, dtype=float)

    def stats(self) -> SceneStats:
        mesh_count = len(list(self.mesh_dir.glob("*.ply"))) if self.mesh_dir and self.mesh_dir.exists() else 0
        if len(self.vertices) == 0:
            return SceneStats(mesh_count, 0, None, None)
        return SceneStats(
            mesh_count=mesh_count,
            vertex_count=int(len(self.vertices)),
            bounds_min=self.vertices.min(axis=0).round(3).tolist(),
            bounds_max=self.vertices.max(axis=0).round(3).tolist(),
        )

    def building_boxes(self, max_boxes: int = 400) -> list[dict[str, float]]:
        boxes: list[dict[str, float]] = []
        if self.mesh_parts:
            for name, mesh in self.mesh_parts:
                if "building" not in name.lower() and "ground" in name.lower():
                    continue
                parts = []
                try:
                    parts = list(mesh.split(only_watertight=False))
                except Exception:
                    parts = [mesh]
                for part in parts:
                    verts = np.asarray(getattr(part, "vertices", []), dtype=float)
                    if len(verts) == 0:
                        continue
                    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
                    height = float(bmax[2] - bmin[2])
                    sx, sy = float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1])
                    if height < 1.5 or sx < 1.0 or sy < 1.0:
                        continue
                    boxes.append(
                        {
                            "x0": float(bmin[0]),
                            "x1": float(bmax[0]),
                            "y0": float(bmin[1]),
                            "y1": float(bmax[1]),
                            "z0": float(bmin[2]),
                            "z1": float(bmax[2]),
                        }
                    )
        if not boxes and len(self.vertices):
            boxes = self._coarse_vertex_boxes(max_boxes=max_boxes)
        boxes.sort(key=lambda b: (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]), reverse=True)
        return boxes[:max_boxes]

    def webgl_mesh(self, max_faces: int = 80_000) -> dict:
        """Return compact triangle data for the browser WebGL planner/player."""
        if self.mesh is not None and hasattr(self.mesh, "vertices") and hasattr(self.mesh, "faces"):
            verts = np.asarray(self.mesh.vertices, dtype=float)
            faces = np.asarray(self.mesh.faces, dtype=np.int64)
            if len(faces) > max_faces:
                step = int(np.ceil(len(faces) / max_faces))
                faces = faces[::step]
            try:
                normals = np.asarray(self.mesh.vertex_normals, dtype=float)
            except Exception:
                normals = np.zeros_like(verts)
            if normals.shape != verts.shape:
                normals = np.zeros_like(verts)
            return {
                "positions": verts.round(3).reshape(-1).tolist(),
                "normals": normals.round(4).reshape(-1).tolist(),
                "indices": faces.reshape(-1).astype(int).tolist(),
                "mode": "triangles",
            }
        if len(self.vertices):
            return {
                "positions": self.vertices.round(3).reshape(-1).tolist(),
                "normals": [],
                "indices": [],
                "mode": "points",
            }
        return {"positions": [], "normals": [], "indices": [], "mode": "empty"}

    def _coarse_vertex_boxes(self, max_boxes: int = 400) -> list[dict[str, float]]:
        high = self.vertices[self.vertices[:, 2] > 1.5]
        if len(high) == 0:
            return []
        cell = 14.0
        buckets: dict[tuple[int, int], list[np.ndarray]] = {}
        for v in high:
            key = (int(np.floor(v[0] / cell)), int(np.floor(v[1] / cell)))
            buckets.setdefault(key, []).append(v)
        boxes = []
        for pts in buckets.values():
            arr = np.asarray(pts)
            if len(arr) < 3:
                continue
            bmin, bmax = arr.min(axis=0), arr.max(axis=0)
            boxes.append(
                {
                    "x0": float(bmin[0]),
                    "x1": float(bmax[0]),
                    "y0": float(bmin[1]),
                    "y1": float(bmax[1]),
                    "z0": float(bmin[2]),
                    "z1": float(bmax[2]),
                }
            )
        return boxes[:max_boxes]

    def raycast_down(self, pos: np.ndarray) -> float:
        if self.mesh is not None:
            try:
                locs, _, _ = self.mesh.ray.intersects_location(
                    np.array([pos], dtype=float), np.array([[0.0, 0.0, -1.0]], dtype=float)
                )
                if len(locs):
                    return float(np.linalg.norm(locs - pos, axis=1).min())
            except Exception:
                pass
        return max(float(pos[2]), 0.0)

    def proximity_score(self, pos: np.ndarray, radius_m: float = 20.0) -> float:
        if len(self.vertices) == 0:
            return 0.0
        xy = self.vertices[:, :2] - pos[:2]
        dz_ok = np.abs(self.vertices[:, 2] - pos[2]) < 20.0
        if not np.any(dz_ok):
            return 0.0
        d = np.linalg.norm(xy[dz_ok], axis=1)
        nearest = float(np.min(d)) if len(d) else radius_m
        return float(np.clip(1.0 - nearest / radius_m, 0.0, 1.0))
