"""Fitted splat prior rows for objects that are already labeled.

Render an orbit of the mesh, run COLMAP, fit a splatfacto model, and pack
``prior_input.npy`` at (N, 912). Geometry columns 896–911 come from that
ply. Opacity is the fitted sigmoid opacity. DINOv2 ViT-S/14 and OpenCLIP
ViT-B-16 stay the feature columns. This module does not call Gemini and
does not download or label objects.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import subprocess
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from sceneagent.mlp.contract import (
    PriorContractError,
    geometry_from_ply,
    pack_prior_input,
    save_prior_input,
    write_gaussian_ply,
)
from sceneagent.mlp.represent import CLIP_SIZE, VIEW_SIZE, FeatureEncoders
from sceneagent.mlp.splat_from_mesh import (
    _camera_up,
    _fit_camera,
    _quaternion_from_matrices,
    _vertex_normals,
    load_textured_mesh,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "mlp"
LABELS_PATH = ROOT / "stratified_labels.jsonl"
REPORT_PATH = ROOT / "fitted_convert_report.json"
MARKER_NAME = "fitted_splat.json"
TMP_ROOT = ROOT / "_fitted_tmp"

ORBIT_AZIMUTHS = 8
ORBIT_ELEVATIONS = (18.0, 50.0)
SPLAT_ITERS = 600
MAX_GAUSSIANS = 8192
MIN_KEPT = 32
MIN_COLMAP_POINTS = 40
DISK_FLOOR = 40 * (1024 ** 3)
DISK_START = 42 * (1024 ** 3)
BACKGROUND = np.array([32, 32, 32], dtype=np.uint8)
SOURCES = ("ycb", "gso", "abo")
KNOWN_IDS = (
    "002_master_chef_can",
    "Weisshai_Great_White_Shark",
    "B079TXJCR5",
)


class DiskFloor(RuntimeError):
    """Free space would drop under the 40 GB floor."""


def free_bytes(path=ROOT):
    return int(shutil.disk_usage(path).free)


def _require_free(margin=DISK_START):
    free = free_bytes()
    if free < margin:
        raise DiskFloor(f"free disk {free / (1024 ** 3):.1f} GiB is under the floor")
    return free


def convert_labeled(limit=0, train=False):
    """Fit splats for status=ok labels that have a mesh, then optionally train.

    ``limit`` stops after that many new successes. Zero means no success cap.
    Objects that already have a fitted marker are counted as already done and
    are not rebuilt. Holdout is applied when training, not by skipping a pack.
    """

    if not LABELS_PATH.is_file():
        raise SystemExit(f"stratified labels are not at {LABELS_PATH}")
    if shutil.which("colmap") is None:
        raise SystemExit("colmap is not on PATH")
    labels = _read_ok_labels()
    not_labeled = _count_unlabeled_dirs(labels)
    work, missing = _work_list(labels)
    report = {
        "gemini_called": False,
        "acquire_called": False,
        "ok_labels": {source: sum(row["source"] == source for row in labels) for source in SOURCES},
        "already_fitted": {source: 0 for source in SOURCES},
        "converted": {source: 0 for source in SOURCES},
        "skipped_missing_mesh": missing,
        "skipped_not_labeled_dirs": not_labeled,
        "known_ids_in_ok_labels": [obj_id for obj_id in KNOWN_IDS if any(row["id"] == obj_id for row in labels)],
        "failed": [],
        "stopped_for_disk": False,
        "disk_free_start": free_bytes(),
    }
    pending = []
    for row in work:
        if _already_fitted(row["directory"]):
            report["already_fitted"][row["source"]] += 1
            continue
        pending.append(row)
    print(
        f"ok {sum(report['ok_labels'].values())} "
        f"pending {len(pending)} already {sum(report['already_fitted'].values())} "
        f"missing {missing} unlabeled_dirs {not_labeled} "
        f"free_gb {report['disk_free_start'] / (1024 ** 3):.1f}",
        flush=True,
    )
    encoders = None
    successes = 0
    try:
        if pending and (limit <= 0 or successes < limit):
            _require_tools()
            encoders = FeatureEncoders().load()
        for row in pending:
            if limit > 0 and successes >= limit:
                break
            try:
                _require_free()
            except DiskFloor as exc:
                report["stopped_for_disk"] = True
                report["disk_stop_reason"] = str(exc)
                break
            try:
                summary = _convert_one(row, encoders)
            except DiskFloor as exc:
                report["stopped_for_disk"] = True
                report["disk_stop_reason"] = str(exc)
                _delete_tree(row["work"])
                break
            except Exception as exc:
                _delete_tree(row["work"])
                reason = f"{type(exc).__name__}: {exc}"
                report["failed"].append({
                    "id": row["id"],
                    "source": row["source"],
                    "reason": reason[:400],
                })
                print(f"fail {row['source']} {row['id']} {reason[:240]}", flush=True)
                continue
            report["converted"][row["source"]] += 1
            successes += 1
            print(
                f"packed {row['source']} {row['id']} n={summary['n']} "
                f"opacity_std={summary['opacity_std']:.4f} free_gb {free_bytes() / (1024 ** 3):.1f}",
                flush=True,
            )
            _write_report(report)
    finally:
        _delete_tree(TMP_ROOT)
        report["disk_free_end"] = free_bytes()
        report["converted_total"] = sum(report["converted"].values())
        report["already_fitted_total"] = sum(report["already_fitted"].values())
        report["failed_total"] = len(report["failed"])
        _write_report(report)
    if train:
        from sceneagent.mlp.train_prior import train_fitted_and_report

        train_fitted_and_report()
    return report


def _read_ok_labels():
    rows = []
    seen = set()
    with LABELS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            source = record.get("source")
            obj_id = record.get("id")
            if source not in SOURCES or not obj_id or obj_id in seen:
                continue
            seen.add(obj_id)
            rows.append(record)
    return rows


def _count_unlabeled_dirs(labels):
    labeled = {(row["source"], row["id"]) for row in labels}
    count = 0
    for source in SOURCES:
        directory = ROOT / source
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_dir() and (source, child.name) not in labeled:
                count += 1
    return count


def _work_list(labels):
    missing = {source: 0 for source in SOURCES}
    work = []
    for record in labels:
        source = record["source"]
        obj_id = record["id"]
        directory = ROOT / source / obj_id
        if not directory.is_dir():
            missing[source] += 1
            continue
        mesh = _find_mesh(directory)
        if mesh is None:
            missing[source] += 1
            continue
        work.append({
            "id": obj_id,
            "source": source,
            "name": record.get("name") or obj_id,
            "directory": directory,
            "mesh": mesh,
            "work": TMP_ROOT / source / obj_id,
        })
    return work, missing


def _find_texture(directory):
    """Publisher texture next to the mesh. Views and thumbnails are not textures."""

    found = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if "views" in path.parts or "thumbnails" in path.parts or "_fitted_tmp" in path.parts:
            continue
        found.append(path)
    for path in found:
        if path.name.lower().startswith("texture"):
            return path
    return found[0] if found else None


def _find_mesh(directory):
    files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if "views" in path.parts or "_fitted_tmp" in path.parts:
            continue
        if path.suffix.lower() in {".sdf", ".glb", ".gltf", ".obj"}:
            files.append(path)
    for path in files:
        if path.name == "model.sdf":
            return path
    glbs = [path for path in files if path.suffix.lower() in {".glb", ".gltf"}]
    if glbs:
        return glbs[0]
    objs = [path for path in files if path.suffix.lower() == ".obj"]
    if objs:
        return objs[0]
    return None


def _already_fitted(directory):
    marker_path = directory / MARKER_NAME
    prior_path = directory / "prior_input.npy"
    if not marker_path.is_file() or not prior_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        shape = tuple(int(value) for value in np.load(prior_path, mmap_mode="r").shape)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        marker.get("kind") == "fitted"
        and not marker.get("opacity_constant_099")
        and float(marker.get("opacity_std") or 0.0) >= 1e-3
        and len(shape) == 2
        and shape[1] == 912
        and shape[0] >= MIN_KEPT
    )


def _convert_one(row, encoders):
    work = row["work"]
    _delete_tree(work)
    image_dir = work / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    _require_free()
    mesh, image, uv = load_textured_mesh(row["mesh"], _find_texture(row["directory"]))
    renders = _render_orbit(mesh, image, uv)
    del mesh, image, uv
    if len(renders) < 8:
        raise RuntimeError(f"only {len(renders)} orbit views rendered")
    grids = _encode_views(encoders, renders)
    _run_colmap(image_dir, renders, work)
    _require_free(DISK_FLOOR)
    means, log_scales, quats, opacity, f_dc, cameras = _run_splatfacto(work)
    _require_free(DISK_FLOOR)
    kept_idx = _cap_gaussians(opacity)
    ply_path = row["directory"] / "fitted_splat.ply"
    write_gaussian_ply(
        ply_path,
        xyz=means[kept_idx],
        log_scales=log_scales[kept_idx],
        opacity=opacity[kept_idx],
        rotation=quats[kept_idx],
        f_dc=f_dc[kept_idx],
    )
    geometry = geometry_from_ply(ply_path)
    dino, clip, visible = _associate(geometry.rows[:, 0:3], cameras, renders, grids)
    if int(visible.sum()) < MIN_KEPT:
        raise RuntimeError(f"only {int(visible.sum())} Gaussians received a feature")
    used = geometry.rows[visible]
    opacity_std = float(used[:, 6].std())
    opacity_mean = float(used[:, 6].mean())
    constant = opacity_std < 1e-3 or float(np.max(np.abs(used[:, 6] - 0.99))) < 1e-3
    if constant:
        raise RuntimeError(
            f"fitted opacity is constant (std={opacity_std:.6g}, mean={opacity_mean:.4f})"
        )
    packed = pack_prior_input(dino[visible], clip[visible], used)
    prior_path = row["directory"] / "prior_input.npy"
    partial = row["directory"] / "prior_input.partial.npy"
    save_prior_input(partial, packed)
    partial.replace(prior_path)
    summary = {
        "kind": "fitted",
        "id": row["id"],
        "source": row["source"],
        "prior_input": str(prior_path),
        "ply": str(ply_path),
        "n": int(packed.shape[0]),
        "width": int(packed.shape[1]),
        "gaussians_in_ply": int(len(kept_idx)),
        "gaussians_dropped_feature": int((~visible).sum()),
        "opacity_min": float(used[:, 6].min()),
        "opacity_max": float(used[:, 6].max()),
        "opacity_mean": opacity_mean,
        "opacity_std": opacity_std,
        "opacity_constant_099": False,
        "splat_iterations": SPLAT_ITERS,
        "orbit_views": len(renders),
        "dino": "vit_small_patch14_dinov2.lvd142m",
        "clip": "ViT-B-16 openai",
    }
    (row["directory"] / MARKER_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _delete_tree(work)
    return summary


def _render_orbit(mesh, image, uv):
    import numba

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_normals = np.asarray(_vertex_normals(vertices, faces), dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    texture = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    radius = max(float(np.linalg.norm(extent)) * 0.5, 1e-4)
    _compile_raster(numba)
    print(f"render faces {len(faces)}", flush=True)
    renders = []
    for index, direction in enumerate(_orbit_directions()):
        up = _camera_up(direction)
        camera = _fit_camera(vertices, center, direction, up, radius)
        rgb, depth = _rasterize_numba(
            vertices,
            faces,
            vertex_normals,
            uv,
            np.ascontiguousarray(texture),
            np.asarray(camera["eye"], dtype=np.float64),
            np.asarray(camera["rot"], dtype=np.float64),
            float(camera["focal"]),
            np.asarray(camera["light"], dtype=np.float64),
            int(VIEW_SIZE),
            int(VIEW_SIZE),
        )
        foreground = int(np.count_nonzero(np.any(rgb != BACKGROUND, axis=2)))
        if foreground < 400:
            continue
        name = f"{index:03d}.png"
        renders.append({
            "name": name,
            "rgb": rgb,
            "depth": depth,
            "eye": np.asarray(camera["eye"], dtype=np.float64),
            "rot": np.asarray(camera["rot"], dtype=np.float64),
            "focal": float(camera["focal"]),
            "foreground": foreground,
        })
    return renders


def _compile_raster(numba):
    if getattr(_compile_raster, "done", False):
        return
    tiny_v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    tiny_f = np.array([[0, 1, 2]], dtype=np.int64)
    tiny_n = np.ones_like(tiny_v)
    tiny_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    tiny_tex = np.zeros((2, 2, 3), dtype=np.float32)
    _rasterize_numba(
        tiny_v, tiny_f, tiny_n, tiny_uv, tiny_tex,
        np.zeros(3), np.eye(3), 100.0, np.array([0.0, 0.0, 1.0]), 8, 8,
    )
    _compile_raster.done = True


def _orbit_directions():
    directions = []
    for elev_deg in ORBIT_ELEVATIONS:
        elev = math.radians(elev_deg)
        for index in range(ORBIT_AZIMUTHS):
            azimuth = 2.0 * math.pi * index / ORBIT_AZIMUTHS
            directions.append(np.array([
                math.cos(elev) * math.cos(azimuth),
                math.cos(elev) * math.sin(azimuth),
                math.sin(elev),
            ], dtype=np.float64))
    return directions


def _rasterize_numba(
    vertices, faces, normals, uv, texture, eye, rot, focal, light, width, height,
):
    return _ensure_raster()(
        vertices, faces, normals, uv, texture, eye, rot, focal, light, width, height,
    )


def _ensure_raster():
    """Compile the raster helpers together so Numba can see the inner calls."""

    if _ensure_raster.fn is not None:
        return _ensure_raster.fn
    from numba import njit

    global _to_cam, _project_face, _texel, _raster_impl
    _to_cam = njit(_to_cam)
    _project_face = njit(_project_face)
    _texel = njit(_texel)
    _raster_impl = njit(_raster_impl)
    _ensure_raster.fn = _raster_impl
    return _raster_impl


_ensure_raster.fn = None


def _raster_impl(vertices, faces, normals, uv, texture, eye, rot, focal, light, width, height):
    color = np.empty((height, width, 3), dtype=np.float64)
    color[:, :, :] = 32.0 / 255.0
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    tex_h, tex_w = texture.shape[0], texture.shape[1]
    ambient = 0.78
    diffuse = 0.22
    for face_index in range(faces.shape[0]):
        i0 = faces[face_index, 0]
        i1 = faces[face_index, 1]
        i2 = faces[face_index, 2]
        z0, z1, z2, x0, y0, x1, y1, x2, y2, ok = _project_face(
            vertices[i0], vertices[i1], vertices[i2], eye, rot, focal, width, height,
        )
        if not ok:
            continue
        area = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)
        if area == 0.0 or abs(area) < 0.8:
            continue
        minx = int(math.floor(min(x0, x1, x2)))
        maxx = int(math.ceil(max(x0, x1, x2)))
        miny = int(math.floor(min(y0, y1, y2)))
        maxy = int(math.ceil(max(y0, y1, y2)))
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx > width - 1:
            maxx = width - 1
        if maxy > height - 1:
            maxy = height - 1
        if minx > maxx or miny > maxy:
            continue
        n0 = normals[i0]
        n1 = normals[i1]
        n2 = normals[i2]
        uv0 = uv[i0]
        uv1 = uv[i1]
        uv2 = uv[i2]
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                w0 = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
                w1 = (px - x2) * (y0 - y2) - (py - y2) * (x0 - x2)
                w2 = (px - x0) * (y1 - y0) - (py - y0) * (x1 - x0)
                if area < 0.0:
                    w0, w1, w2 = -w0, -w1, -w2
                    used = -area
                else:
                    used = area
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                b0 = w0 / used
                b1 = w1 / used
                b2 = w2 / used
                inv = b0 / z0 + b1 / z1 + b2 / z2
                if inv <= 1e-8:
                    continue
                su = ((b0 / z0) * uv0[0] + (b1 / z1) * uv1[0] + (b2 / z2) * uv2[0]) / inv
                sv = ((b0 / z0) * uv0[1] + (b1 / z1) * uv1[1] + (b2 / z2) * uv2[1]) / inv
                nx = ((b0 / z0) * n0[0] + (b1 / z1) * n1[0] + (b2 / z2) * n2[0]) / inv
                ny = ((b0 / z0) * n0[1] + (b1 / z1) * n1[1] + (b2 / z2) * n2[1]) / inv
                nz = ((b0 / z0) * n0[2] + (b1 / z1) * n1[2] + (b2 / z2) * n2[2]) / inv
                pixel_z = 1.0 / inv
                if pixel_z >= zbuf[py, px]:
                    continue
                albedo = _texel(texture, su, sv, tex_h, tex_w)
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                ndot = 0.65
                if length > 1e-8:
                    ndot = abs((nx * light[0] + ny * light[1] + nz * light[2]) / length)
                shade = ambient + diffuse * ndot
                zbuf[py, px] = pixel_z
                color[py, px, 0] = min(1.0, max(0.0, albedo[0] * shade))
                color[py, px, 1] = min(1.0, max(0.0, albedo[1] * shade))
                color[py, px, 2] = min(1.0, max(0.0, albedo[2] * shade))
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    for py in range(height):
        for px in range(width):
            rgb[py, px, 0] = int(color[py, px, 0] * 255.0 + 0.5)
            rgb[py, px, 1] = int(color[py, px, 1] * 255.0 + 0.5)
            rgb[py, px, 2] = int(color[py, px, 2] * 255.0 + 0.5)
    return rgb, zbuf


def _project_face(v0, v1, v2, eye, rot, focal, width, height):
    c0 = _to_cam(v0, eye, rot)
    c1 = _to_cam(v1, eye, rot)
    c2 = _to_cam(v2, eye, rot)
    if c0[2] <= 1e-6 or c1[2] <= 1e-6 or c2[2] <= 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    center = (width - 1) * 0.5
    x0 = focal * c0[0] / c0[2] + center
    y0 = center - focal * c0[1] / c0[2]
    x1 = focal * c1[0] / c1[2] + center
    y1 = center - focal * c1[1] / c1[2]
    x2 = focal * c2[0] / c2[2] + center
    y2 = center - focal * c2[1] / c2[2]
    return c0[2], c1[2], c2[2], x0, y0, x1, y1, x2, y2, True


def _to_cam(point, eye, rot):
    moved = point - eye
    return np.array([
        rot[0, 0] * moved[0] + rot[0, 1] * moved[1] + rot[0, 2] * moved[2],
        rot[1, 0] * moved[0] + rot[1, 1] * moved[1] + rot[1, 2] * moved[2],
        rot[2, 0] * moved[0] + rot[2, 1] * moved[1] + rot[2, 2] * moved[2],
    ])


def _texel(texture, u, v, tex_h, tex_w):
    u = u - math.floor(u)
    v = v - math.floor(v)
    x = u * (tex_w - 1)
    y = (1.0 - v) * (tex_h - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 > tex_w - 1:
        x0 = tex_w - 1
    if y0 > tex_h - 1:
        y0 = tex_h - 1
    x1 = x0 + 1 if x0 + 1 < tex_w else tex_w - 1
    y1 = y0 + 1 if y0 + 1 < tex_h else tex_h - 1
    wx = x - x0
    wy = y - y0
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    return (c00 * (1.0 - wx) + c10 * wx) * (1.0 - wy) + (c01 * (1.0 - wx) + c11 * wx) * wy


def _encode_views(encoders, renders):
    import torch

    _move_encoders(encoders, "cuda" if torch.cuda.is_available() else "cpu")
    grids = {}
    for view in renders:
        grids[view["name"]] = encoders.patches(view["rgb"])
    _move_encoders(encoders, "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return grids


def _move_encoders(encoders, device):
    if encoders._dino is not None:
        encoders._dino.to(device)
    if encoders._clip is not None:
        encoders._clip.to(device)


def _run_colmap(image_dir, renders, work):
    for view in renders:
        Image.fromarray(view["rgb"], mode="RGB").save(image_dir / view["name"])
    database = work / "colmap.db"
    sparse_in = work / "sparse_in"
    sparse_out = work / "sparse" / "0"
    sparse_in.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)
    focal = float(renders[0]["focal"])
    center = (VIEW_SIZE - 1) * 0.5
    params = f"{focal},{focal},{center},{center}"
    extractor = [
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.camera_params", params,
        "--SiftExtraction.max_image_size", str(VIEW_SIZE),
        "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.max_num_features", "8192",
    ]
    try:
        _colmap(extractor + ["--SiftExtraction.use_gpu", "1"])
        gpu_match = True
    except (RuntimeError, subprocess.TimeoutExpired):
        if database.exists():
            database.unlink()
        _colmap(extractor + ["--SiftExtraction.use_gpu", "0"])
        gpu_match = False
    try:
        _colmap([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database),
            *_sift_match_flags(gpu_match),
        ])
    except (RuntimeError, subprocess.TimeoutExpired):
        _colmap([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database),
            *_sift_match_flags(False),
        ])
    _write_known_model(database, renders, sparse_in, focal, center)
    triangulated = False
    if _colmap_supports("point_triangulator"):
        tri_cmd = [
            "colmap", "point_triangulator",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--input_path", str(sparse_in),
            "--output_path", str(sparse_out),
            *_triangulator_flags(bundle_adjustment=True),
        ]
        try:
            _colmap(tri_cmd)
            triangulated = _count_points(sparse_out) >= MIN_COLMAP_POINTS
        except (RuntimeError, subprocess.TimeoutExpired):
            triangulated = False
        if not triangulated:
            _delete_tree(sparse_out)
            sparse_out.mkdir(parents=True, exist_ok=True)
            try:
                _colmap([
                    "colmap", "point_triangulator",
                    "--database_path", str(database),
                    "--image_path", str(image_dir),
                    "--input_path", str(sparse_in),
                    "--output_path", str(sparse_out),
                    *_triangulator_flags(bundle_adjustment=False),
                ])
                triangulated = _count_points(sparse_out) >= MIN_COLMAP_POINTS
            except (RuntimeError, subprocess.TimeoutExpired):
                triangulated = False
    if not triangulated:
        _delete_tree(work / "sparse")
        (work / "sparse").mkdir(parents=True, exist_ok=True)
        _colmap([
            "colmap", "mapper",
            "--database_path", str(database),
            "--image_path", str(image_dir),
            "--output_path", str(work / "sparse"),
            "--Mapper.multiple_models", "0",
            "--Mapper.init_min_num_inliers", "12",
            "--Mapper.abs_pose_min_num_inliers", "8",
            "--Mapper.min_num_matches", "8",
            "--Mapper.tri_ignore_two_view_tracks", "0",
            "--Mapper.ba_refine_focal_length", "0",
        ], timeout=300)
        if not (sparse_out / "cameras.bin").is_file() and not (sparse_out / "cameras.txt").is_file():
            models = sorted(path for path in (work / "sparse").iterdir() if path.is_dir())
            if not models:
                raise RuntimeError("COLMAP mapper wrote no model")
            # nerfstudio reads sparse/0
            if models[0] != sparse_out:
                if sparse_out.exists():
                    shutil.rmtree(sparse_out)
                shutil.move(str(models[0]), str(sparse_out))
        if _count_points(sparse_out) < MIN_COLMAP_POINTS:
            raise RuntimeError("COLMAP reconstruction has too few points")
    for name in ("cameras", "images", "points3D"):
        text = sparse_out / f"{name}.txt"
        binary = sparse_out / f"{name}.bin"
        if binary.is_file() or not text.is_file():
            continue
        _colmap([
            "colmap", "model_converter",
            "--input_path", str(sparse_out),
            "--output_path", str(sparse_out),
            "--output_type", "BIN",
        ])
        break


def _sift_match_flags(use_gpu):
    """Smooth product renders share few corners, so the default ratio test matches nothing."""

    return [
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SiftMatching.max_ratio", "0.92",
        "--SiftMatching.max_distance", "0.9",
        "--SiftMatching.min_num_inliers", "8",
    ]


def _triangulator_flags(bundle_adjustment):
    """Keep the rendered poses and two-view tracks.

    COLMAP's mapper asks for 100 inliers before it will register a pair.
    These orbits often have about ten. Triangulating the known cameras is
    the reconstruction that succeeds. Two-view tracks are the common case
    when a face is only shared by neighboring orbit frames.
    """

    iterations = "20" if bundle_adjustment else "0"
    local = "10" if bundle_adjustment else "0"
    return [
        "--Mapper.fix_existing_images", "1",
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.min_num_matches", "8",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_global_max_num_iterations", iterations,
        "--Mapper.ba_local_max_num_iterations", local,
        "--Mapper.filter_min_tri_angle", "0.5",
        "--Mapper.tri_min_angle", "0.5",
    ]


def _colmap_supports(command):
    if _colmap_supports.commands is None:
        result = subprocess.run(["colmap", "help"], check=False, capture_output=True, text=True)
        _colmap_supports.commands = result.stdout + result.stderr
    return command in _colmap_supports.commands


_colmap_supports.commands = None


def _colmap(cmd, timeout=180):
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-600:].replace("\n", " ")
        raise RuntimeError(f"{cmd[1]} failed: {tail}")
    return result


def _write_known_model(database, renders, sparse_in, focal, center):
    names = _database_images(database)
    by_name = {view["name"]: view for view in renders}
    camera_lines = [f"1 PINHOLE {VIEW_SIZE} {VIEW_SIZE} {focal} {focal} {center} {center}"]
    image_lines = []
    for image_id, name, _camera_id in names:
        view = by_name.get(name)
        if view is None:
            raise RuntimeError(f"COLMAP image {name} is not an orbit view")
        rotation, translation = _colmap_w2c(view["rot"], view["eye"])
        quat = _quaternion_from_matrices(rotation[None, :, :])[0]
        image_lines.append(
            f"{image_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
            f"{translation[0]} {translation[1]} {translation[2]} 1 {name}"
        )
        image_lines.append("")
    (sparse_in / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: 1\n" + "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    (sparse_in / "images.txt").write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(names)}\n" + "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    (sparse_in / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0\n",
        encoding="utf-8",
    )


def _database_images(database):
    connection = sqlite3.connect(str(database))
    try:
        rows = connection.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("COLMAP database has no images")
    cleaned = []
    for image_id, name, camera_id in rows:
        cleaned.append((int(image_id), Path(name).name, int(camera_id)))
    return cleaned


def _colmap_w2c(rot, eye):
    rot = np.asarray(rot, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    candidates = (
        np.diag([1.0, -1.0, 1.0]) @ rot,
        np.diag([1.0, -1.0, -1.0]) @ rot,
    )
    rotation = None
    for candidate in candidates:
        fixed = _nearest_rotation(candidate)
        if rotation is None:
            rotation = fixed
        if abs(float(np.linalg.det(candidate))) > 0.5 and float(np.linalg.det(fixed)) > 0.0:
            rotation = fixed
            if float(np.linalg.det(candidate)) > 0.0:
                break
    translation = -rotation @ eye
    return rotation, translation


def _nearest_rotation(matrix):
    left, _values, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left = left.copy()
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _count_points(model_dir):
    text = model_dir / "points3D.txt"
    binary = model_dir / "points3D.bin"
    if not text.is_file() and binary.is_file():
        text_dir = model_dir / "_txt_count"
        text_dir.mkdir(parents=True, exist_ok=True)
        _colmap([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ])
        text = text_dir / "points3D.txt"
    if not text.is_file():
        return 0
    count = 0
    for line in text.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and not line.startswith("#"):
            count += 1
    return count


def _close_train_writers():
    """Drop Nerfstudio's process-global log writers before the work dir is removed.

    The writers live for the life of the process. The next object deletes the
    previous work directory, and a leftover writer then fails on that path.
    """

    try:
        import nerfstudio.utils.writer as ns_writer
    except Exception:
        return
    for event_writer in list(getattr(ns_writer, "EVENT_WRITERS", ())):
        for owner in (event_writer, getattr(event_writer, "writer", None)):
            close = getattr(owner, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                pass
    for name in ("EVENT_WRITERS", "EVENT_STORAGE"):
        bucket = getattr(ns_writer, name, None)
        if hasattr(bucket, "clear"):
            bucket.clear()
    buffer = getattr(ns_writer, "GLOBAL_BUFFER", None)
    if hasattr(buffer, "clear"):
        buffer.clear()


def _force_dataloader_workers():
    """Nerfstudio asks for ``world_size * 4`` workers and forks the CUDA process."""

    import torch.utils.data.dataloader as dataloader

    if getattr(dataloader.DataLoader, "_fitted_workers", False):
        return
    original = dataloader.DataLoader.__init__

    def init(self, *args, **kwargs):
        # num_workers is the 6th positional parameter. A keyword override
        # has to be the only form or Python raises.
        if len(args) > 5:
            args = list(args)
            args[5] = 0
            args = tuple(args)
        else:
            kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        return original(self, *args, **kwargs)

    dataloader.DataLoader.__init__ = init
    dataloader.DataLoader._fitted_workers = True


def _run_splatfacto(work):
    import copy
    import gc

    import torch

    _force_dataloader_workers()
    from nerfstudio.configs.method_configs import all_methods
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

    if "splatfacto" not in all_methods:
        raise RuntimeError("splatfacto is not registered")
    config = copy.deepcopy(all_methods["splatfacto"])
    data = work
    config.experiment_name = "fitted"
    config.method_name = "splatfacto"
    config.output_dir = work / "ns"
    config.max_num_iterations = SPLAT_ITERS
    config.steps_per_eval_batch = 10 ** 9
    config.steps_per_eval_image = 10 ** 9
    config.steps_per_save = SPLAT_ITERS
    config.save_only_latest_checkpoint = True
    config.load_dir = None
    config.load_config = None
    config.load_checkpoint = None
    # Tensorboard is closed in ``_close_train_writers`` before the work
    # directory is deleted. A viewer server would stay up for every object.
    config.vis = "tensorboard"
    if hasattr(config, "viewer"):
        config.viewer.quit_on_train_completion = True
    if hasattr(config.pipeline.datamanager, "num_workers"):
        config.pipeline.datamanager.num_workers = 0
    if hasattr(config.pipeline.datamanager, "max_thread_workers"):
        config.pipeline.datamanager.max_thread_workers = 1
    parser = ColmapDataParserConfig(data=data)
    for key, value in {
        "colmap_path": Path("sparse/0"),
        "orientation_method": "none",
        "center_method": "none",
        "auto_scale_poses": False,
        "assume_colmap_world_coordinate_convention": True,
        "eval_mode": "all",
        "downscale_factor": 1,
        "load_3D_points": True,
    }.items():
        if hasattr(parser, key):
            setattr(parser, key, value)
    config.pipeline.datamanager.dataparser = parser
    model = config.pipeline.model
    for key, value in {
        "sh_degree": 0,
        "warmup_length": 100,
        "refine_every": 50,
        "stop_split_at": 400,
        "densify_grad_thresh": 0.001,
    }.items():
        if hasattr(model, key):
            setattr(model, key, value)
    config.set_timestamp()
    _close_train_writers()
    trainer = config.setup()
    trainer.setup()
    try:
        trainer.train()
        outputs = _dataparser_outputs(trainer)
        cameras = _export_cameras(outputs)
        means, log_scales, quats, opacity, f_dc = _export_gaussians(trainer.pipeline.model)
    finally:
        del trainer
        _close_train_writers()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(cameras) < 4:
        raise RuntimeError(f"splatfacto exposed {len(cameras)} cameras")
    if means.shape[0] < MIN_KEPT:
        raise RuntimeError(f"splatfacto produced {means.shape[0]} Gaussians")
    return means, log_scales, quats, opacity, f_dc, cameras


def _dataparser_outputs(trainer):
    manager = trainer.pipeline.datamanager
    outputs = getattr(manager, "train_dataparser_outputs", None)
    if outputs is None and getattr(manager, "train_dataset", None) is not None:
        outputs = getattr(manager.train_dataset, "_dataparser_outputs", None)
    if outputs is None:
        raise RuntimeError("splatfacto datamanager has no training cameras")
    return outputs


def _export_cameras(outputs):
    cameras = outputs.cameras
    files = list(outputs.image_filenames)
    c2w = cameras.camera_to_worlds.detach().cpu().numpy()
    fx = np.asarray(cameras.fx.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    fy = np.asarray(cameras.fy.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    cx = np.asarray(cameras.cx.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    cy = np.asarray(cameras.cy.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    height = np.asarray(cameras.height.detach().cpu().numpy()).reshape(-1)
    width = np.asarray(cameras.width.detach().cpu().numpy()).reshape(-1)
    exported = []
    count = min(len(files), c2w.shape[0], fx.shape[0])
    for index in range(count):
        matrix = np.asarray(c2w[index], dtype=np.float64)
        if matrix.shape[0] == 3:
            rotation = matrix[:, :3]
            translation = matrix[:, 3]
        else:
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
        exported.append({
            "name": Path(str(files[index])).name,
            "rotation": rotation,
            "translation": translation,
            "fx": float(fx[index]),
            "fy": float(fy[index]),
            "cx": float(cx[index]),
            "cy": float(cy[index]),
            "width": int(width[index]),
            "height": int(height[index]),
        })
    return exported


def _export_gaussians(model):
    params = getattr(model, "gauss_params", None)
    if params is None:
        names = [name for name in dir(model) if "gauss" in name.lower() or name.endswith("params")]
        raise RuntimeError(f"splatfacto model has no gauss_params ({names})")

    def column(name):
        return params[name].detach().float().cpu().numpy()

    means = np.asarray(column("means"), dtype=np.float64)
    log_scales = np.asarray(column("scales"), dtype=np.float64)
    quats = np.asarray(column("quats"), dtype=np.float64)
    opacity = np.asarray(column("opacities"), dtype=np.float64).reshape(-1)
    features = np.asarray(column("features_dc"), dtype=np.float64)
    if features.ndim == 3:
        features = features[:, 0, :]
    features = features.reshape(means.shape[0], -1)[:, :3]
    count = means.shape[0]
    if log_scales.shape != (count, 3) or quats.shape != (count, 4) or features.shape != (count, 3):
        raise RuntimeError(
            f"unexpected splat tensors means {means.shape} scales {log_scales.shape} "
            f"quats {quats.shape} dc {features.shape}"
        )
    if opacity.shape != (count,):
        raise RuntimeError(f"unexpected opacity shape {opacity.shape}")
    finite = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(log_scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacity)
        & np.isfinite(features).all(axis=1)
    )
    if int(finite.sum()) < MIN_KEPT:
        raise RuntimeError("splatfacto Gaussians are not finite")
    return (
        means[finite],
        log_scales[finite],
        quats[finite],
        opacity[finite],
        features[finite],
    )


def _cap_gaussians(opacity):
    count = int(opacity.shape[0])
    if count <= MAX_GAUSSIANS:
        return np.arange(count)
    scores = 1.0 / (1.0 + np.exp(-np.clip(opacity, -40.0, 40.0)))
    chosen = np.argpartition(scores, -MAX_GAUSSIANS)[-MAX_GAUSSIANS:]
    chosen.sort()
    return chosen


def _associate(xyz, cameras, renders, grids):
    by_name = {view["name"]: view for view in renders}
    count = xyz.shape[0]
    dino_sum = np.zeros((count, 384), dtype=np.float64)
    clip_sum = np.zeros((count, 512), dtype=np.float64)
    hits = np.zeros(count, dtype=np.int32)
    variant = _best_variant(xyz, cameras, by_name)
    if variant is None:
        raise RuntimeError("fitted Gaussians do not project into the orbit")
    transpose, z_sign, y_sign = variant
    for camera in cameras:
        view = by_name.get(camera["name"])
        grid = grids.get(camera["name"])
        if view is None or grid is None:
            continue
        px, py, depth = _project_variant(xyz, camera, transpose, z_sign, y_sign)
        mask = _foreground_hits(px, py, depth, view["rgb"], camera)
        if int(mask.sum()) < 8:
            continue
        visible = _front_surface(px, py, depth, mask, camera)
        dino_grid, clip_grid = grid
        _accumulate(visible, px, py, camera, view["rgb"].shape, dino_grid, clip_grid, dino_sum, clip_sum, hits)
    keep = hits > 0
    if not np.any(keep):
        raise RuntimeError("no fitted Gaussian received a finite feature")
    dino = np.zeros_like(dino_sum)
    clip = np.zeros_like(clip_sum)
    dino[keep] = dino_sum[keep] / hits[keep, None]
    clip[keep] = clip_sum[keep] / hits[keep, None]
    usable = keep & np.isfinite(dino).all(axis=1) & np.isfinite(clip).all(axis=1)
    usable &= np.linalg.norm(dino, axis=1) > 0.0
    usable &= np.linalg.norm(clip, axis=1) > 0.0
    return dino, clip, usable


def _best_variant(xyz, cameras, by_name):
    best = None
    best_hits = -1
    sample = xyz
    if len(sample) > 4000:
        sample = xyz[:: max(len(xyz) // 4000, 1)]
    for transpose in (False, True):
        for z_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                hits = 0
                for camera in cameras:
                    view = by_name.get(camera["name"])
                    if view is None:
                        continue
                    px, py, depth = _project_variant(sample, camera, transpose, z_sign, y_sign)
                    hits += int(_foreground_hits(px, py, depth, view["rgb"], camera).sum())
                if hits > best_hits:
                    best_hits = hits
                    best = (transpose, z_sign, y_sign)
    if best_hits < MIN_KEPT:
        return None
    return best


def _project_variant(xyz, camera, transpose, z_sign, y_sign):
    rotation = camera["rotation"]
    if transpose:
        rotation = rotation.T
    translated = xyz - camera["translation"]
    local = translated @ rotation
    depth = z_sign * local[:, 2]
    safe = np.where(np.abs(depth) < 1e-8, 1e-8, depth)
    px = camera["fx"] * local[:, 0] / safe + camera["cx"]
    py = camera["fy"] * (y_sign * local[:, 1]) / safe + camera["cy"]
    if camera["width"] != VIEW_SIZE or camera["height"] != VIEW_SIZE:
        px = px * (VIEW_SIZE / max(camera["width"], 1))
        py = py * (VIEW_SIZE / max(camera["height"], 1))
    return px, py, depth


def _foreground_hits(px, py, depth, rgb, camera):
    del camera
    xi = np.rint(px).astype(np.int64)
    yi = np.rint(py).astype(np.int64)
    height, width = rgb.shape[:2]
    inside = (depth > 1e-4) & (xi >= 0) & (yi >= 0) & (xi < width) & (yi < height)
    hits = np.zeros(px.shape[0], dtype=bool)
    if not np.any(inside):
        return hits
    chosen = np.flatnonzero(inside)
    pixels = rgb[yi[chosen], xi[chosen]].astype(np.int16)
    delta = np.max(np.abs(pixels - BACKGROUND.astype(np.int16)), axis=1)
    hits[chosen[delta > 18]] = True
    return hits


def _front_surface(px, py, depth, mask, camera):
    del camera
    xi = np.rint(px).astype(np.int64)
    yi = np.rint(py).astype(np.int64)
    chosen = np.flatnonzero(mask)
    if chosen.size == 0:
        return mask
    order = chosen[np.argsort(depth[chosen])]
    nearest = {}
    for index in order:
        key = (int(yi[index]), int(xi[index]))
        if key not in nearest:
            nearest[key] = float(depth[index])
    span = float(depth[chosen].max() - depth[chosen].min())
    tolerance = max(span * 0.04, 1e-4)
    keep = np.zeros(mask.shape, dtype=bool)
    for index in chosen:
        key = (int(yi[index]), int(xi[index]))
        if abs(float(depth[index]) - nearest[key]) <= tolerance:
            keep[index] = True
    if int(keep.sum()) >= MIN_KEPT:
        return keep
    return mask


def _accumulate(visible, px, py, camera, image_shape, dino_grid, clip_grid, dino_sum, clip_sum, hits):
    del camera
    height, width = image_shape[:2]
    chosen = np.flatnonzero(visible)
    if chosen.size == 0:
        return
    dino_index = _patch_index(py[chosen], px[chosen], dino_grid.shape[0], dino_grid.shape[1], height, width)
    clip_py = py[chosen] * (CLIP_SIZE / height)
    clip_px = px[chosen] * (CLIP_SIZE / width)
    clip_index = _patch_index(clip_py, clip_px, clip_grid.shape[0], clip_grid.shape[1], CLIP_SIZE, CLIP_SIZE)
    dino_rows = dino_grid[dino_index[:, 0], dino_index[:, 1]]
    clip_rows = clip_grid[clip_index[:, 0], clip_index[:, 1]]
    usable = np.isfinite(dino_rows).all(axis=1) & np.isfinite(clip_rows).all(axis=1)
    picked = chosen[usable]
    dino_sum[picked] += dino_rows[usable]
    clip_sum[picked] += clip_rows[usable]
    hits[picked] += 1


def _patch_index(py, px, grid_h, grid_w, height, width):
    row = np.clip(np.floor(py / height * grid_h).astype(np.int64), 0, grid_h - 1)
    col = np.clip(np.floor(px / width * grid_w).astype(np.int64), 0, grid_w - 1)
    return np.stack([row, col], axis=1)


def _require_tools():
    import importlib

    if importlib.util.find_spec("numba") is None:
        raise SystemExit("numba is not installed")
    if importlib.util.find_spec("gsplat") is None:
        raise SystemExit("gsplat is not installed")
    if importlib.util.find_spec("nerfstudio") is None:
        raise SystemExit("nerfstudio is not installed")


def _delete_tree(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _write_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fit splats for already-labeled objects and optionally train.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many new fitted objects. 0 means all.")
    parser.add_argument("--train", action="store_true", help="Train physics_prior_splat.npz after conversion.")
    args = parser.parse_args()
    try:
        convert_labeled(limit=args.limit, train=args.train)
    except DiskFloor as exc:
        print(f"stopped for disk: {exc}", flush=True)
        raise SystemExit(2) from exc
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
