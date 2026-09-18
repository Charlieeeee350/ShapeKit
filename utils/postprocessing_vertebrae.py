"""ShapeKit-Geodesic: anatomy-aware cleanup with CT-guided geodesic refinement."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


VERTEBRA_NAMES: Tuple[str, ...] = (
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2",
    "vertebrae_L1", "vertebrae_T12", "vertebrae_T11", "vertebrae_T10",
    "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
    "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2",
    "vertebrae_T1", "vertebrae_C7", "vertebrae_C6", "vertebrae_C5",
    "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
)

# Anatomical groups shown in the reference: 7 cervical, 12 thoracic and
# 5 lumbar vertebrae.  Sacrum and coccyx are deliberately outside the label map.
ANATOMICAL_REGION: Dict[int, str] = {
    **{label: "lumbar" for label in range(1, 6)},
    **{label: "thoracic" for label in range(6, 18)},
    **{label: "cervical" for label in range(18, 25)},
}

VALID_LABEL_COUNT = 25
ARRAY_CHUNK_VOXELS = 8_000_000


def arrays_are_finite(array: np.ndarray) -> bool:
    """Check floating-point data in bounded-memory chunks."""
    if np.issubdtype(array.dtype, np.integer):
        return True
    flat = array.ravel(order="K")
    return all(
        bool(np.all(np.isfinite(flat[start:start + ARRAY_CHUNK_VOXELS])))
        for start in range(0, flat.size, ARRAY_CHUNK_VOXELS)
    )


def validate_prediction_labels(raw: np.ndarray, case: str) -> np.ndarray:
    """Validate labels without first performing a lossy or full-size conversion."""
    if not (
        np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.floating)
    ):
        raise ValueError(f"{case}: prediction must contain real numeric labels")
    if not arrays_are_finite(raw):
        raise ValueError(f"{case}: prediction contains NaN or infinite values")

    flat = raw.ravel(order="K")
    minimum = math.inf
    maximum = -math.inf
    for start in range(0, flat.size, ARRAY_CHUNK_VOXELS):
        chunk = flat[start:start + ARRAY_CHUNK_VOXELS]
        if np.issubdtype(raw.dtype, np.floating) and not np.all(chunk == np.rint(chunk)):
            raise ValueError(f"{case}: prediction contains non-integer label values")
        if chunk.size:
            minimum = min(minimum, float(np.min(chunk)))
            maximum = max(maximum, float(np.max(chunk)))
    if minimum < 0 or maximum >= VALID_LABEL_COUNT:
        raise ValueError(
            f"{case}: labels must be integers from 0 to {VALID_LABEL_COUNT - 1}; "
            f"observed range [{minimum:g}, {maximum:g}]"
        )
    return raw.astype(np.uint8, copy=False)


def label_transition_counts(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Count source-to-target label transitions with bounded temporary memory."""
    if source.shape != target.shape:
        raise ValueError(
            f"source and target shapes differ: {source.shape} != {target.shape}"
        )
    counts = np.zeros((VALID_LABEL_COUNT, VALID_LABEL_COUNT), dtype=np.int64)
    iterator = np.nditer(
        (source, target),
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=(("readonly",), ("readonly",)),
        order="C",
        buffersize=ARRAY_CHUNK_VOXELS,
    )
    for source_chunk, target_chunk in iterator:
        codes = (
            source_chunk.astype(np.int16, copy=False) * VALID_LABEL_COUNT
            + target_chunk.astype(np.int16, copy=False)
        )
        counts += np.bincount(
            codes, minlength=VALID_LABEL_COUNT * VALID_LABEL_COUNT
        ).reshape(VALID_LABEL_COUNT, VALID_LABEL_COUNT)
    return counts


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either path contains the other, including filesystem aliases."""
    if first == second or first in second.parents or second in first.parents:
        return True

    def same_existing_path(left: Path, right: Path) -> bool:
        try:
            return left.samefile(right)
        except FileNotFoundError:
            return False

    if same_existing_path(first, second):
        return True
    if any(same_existing_path(parent, second) for parent in first.parents):
        return True
    return any(same_existing_path(parent, first) for parent in second.parents)


def validate_case_ids(cases: Sequence[str]) -> None:
    """Require unique, single-directory case identifiers."""
    seen = set()
    for case in cases:
        path = Path(case)
        if (
            not case
            or case in {".", ".."}
            or path.is_absolute()
            or len(path.parts) != 1
            or "/" in case
            or "\\" in case
            or "\x00" in case
        ):
            raise ValueError(f"invalid case ID {case!r}: expected one directory name")
        if case in seen:
            raise ValueError(f"duplicate case ID: {case}")
        seen.add(case)


def anatomical_radius_mm(label: int, args: argparse.Namespace) -> float:
    """Maximum transverse distance from the fitted spinal centreline.

    Thoracic masks use a tighter corridor to suppress leakage into ribs.  Lumbar
    transverse processes are wider, while cervical vertebrae are smaller.  C1
    (atlas) and C2 (axis) receive a small allowance for their special shapes.
    """
    region = ANATOMICAL_REGION[label]
    radius = {
        "lumbar": args.lumbar_radius_mm,
        "thoracic": args.thoracic_radius_mm,
        "cervical": args.cervical_radius_mm,
    }[region]
    if label in (23, 24):  # C2 (axis), C1 (atlas)
        radius += args.upper_cervical_radius_allowance_mm
    return float(radius)


@dataclass
class Component:
    component_id: int
    voxel_count: int
    volume_mm3: float
    center_voxel: np.ndarray
    center_world: np.ndarray
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    bone_fraction: float
    below_min_hu_fraction: float
    coords: np.ndarray = field(repr=False)


def _world_center(affine: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center_voxel = coords.mean(axis=0, dtype=np.float64)
    return center_voxel, nib.affines.apply_affine(affine, center_voxel)


def _bbox_from_coords(coords: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    return tuple((int(low[d]), int(high[d])) for d in range(3))


def extract_components(
    label: int,
    prediction: np.ndarray,
    ct: np.ndarray,
    affine: np.ndarray,
    voxel_volume_mm3: float,
    min_hu: float,
) -> List[Component]:
    """Extract 26-connected components without labelling the full CT 24 times."""
    locations = np.where(prediction == label)
    if locations[0].size == 0:
        return []

    starts = np.array([int(axis.min()) for axis in locations], dtype=np.int32)
    stops = np.array([int(axis.max()) + 1 for axis in locations], dtype=np.int32)
    crop_slice = tuple(slice(int(starts[d]), int(stops[d])) for d in range(3))
    binary = prediction[crop_slice] == label
    cc, number = ndi.label(binary, structure=np.ones((3, 3, 3), dtype=bool))

    components: List[Component] = []
    for component_id in range(1, number + 1):
        local_coords = np.column_stack(np.where(cc == component_id)).astype(np.int32)
        if local_coords.size == 0:
            continue
        coords = local_coords + starts
        center_voxel, center_world = _world_center(affine, coords)
        values = np.asarray(ct[tuple(coords.T)])
        voxel_count = int(coords.shape[0])
        components.append(
            Component(
                component_id=component_id,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count * voxel_volume_mm3),
                center_voxel=center_voxel,
                center_world=center_world,
                bbox=_bbox_from_coords(coords),
                bone_fraction=float(np.mean(values >= 130.0)),
                below_min_hu_fraction=float(np.mean(values < min_hu)),
                coords=coords,
            )
        )
    return components


def isotonic_increasing(
    values: Sequence[float], weights: Sequence[float], offsets: Sequence[float]
) -> np.ndarray:
    """Weighted pool-adjacent-violators fit with caller-provided offsets."""
    y = np.asarray(values, dtype=np.float64) - np.asarray(offsets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    blocks: List[List[float]] = []
    for index, (value, weight) in enumerate(zip(y, w)):
        blocks.append([float(index), float(index + 1), float(weight), float(value * weight)])
        while len(blocks) >= 2:
            left_mean = blocks[-2][3] / blocks[-2][2]
            right_mean = blocks[-1][3] / blocks[-1][2]
            if left_mean <= right_mean:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            )
    fitted = np.empty_like(y)
    for start, stop, weight, weighted_sum in blocks:
        fitted[int(start):int(stop)] = weighted_sum / weight
    return fitted + np.asarray(offsets, dtype=np.float64)


def fit_axis(components: Sequence[Component]):
    """Robust piecewise-linear x(z), y(z) spinal centreline model.

    A global polynomial can cut across pronounced cervical or lumbar curvature.
    Median-smoothed vertebral anchors followed by local interpolation follows the
    actual lordosis/kyphosis while rejecting a single displaced component.
    """
    if not components:
        return lambda z: np.zeros(np.asarray(z).shape + (2,), dtype=np.float64)
    if len(components) == 1:
        xy = components[0].center_world[:2].copy()
        return lambda z: np.broadcast_to(xy, np.asarray(z).shape + (2,)).copy()

    ordered = sorted(components, key=lambda component: component.center_world[2])
    z = np.asarray([c.center_world[2] for c in ordered], dtype=np.float64)
    xy = np.asarray([c.center_world[:2] for c in ordered], dtype=np.float64)
    if len(ordered) >= 3:
        median_xy = np.column_stack(
            [ndi.median_filter(xy[:, axis], size=3, mode="nearest") for axis in range(2)]
        )
        # Preserve genuine endpoints such as the atlas while damping isolated
        # internal anchors caused by fragmented posterior elements.
        median_xy[0] = xy[0]
        median_xy[-1] = xy[-1]
        xy = 0.75 * median_xy + 0.25 * xy

    def interpolate(z_value):
        values = np.asarray(z_value, dtype=np.float64)
        x_value = np.interp(values, z, xy[:, 0])
        y_value = np.interp(values, z, xy[:, 1])
        return np.stack((x_value, y_value), axis=-1)

    return interpolate


def local_spacing(label: int, expected_z: Dict[int, float]) -> float:
    neighbours: List[float] = []
    labels = sorted(expected_z)
    position = labels.index(label)
    if position > 0:
        previous = labels[position - 1]
        neighbours.append((expected_z[label] - expected_z[previous]) / (label - previous))
    if position + 1 < len(labels):
        following = labels[position + 1]
        neighbours.append((expected_z[following] - expected_z[label]) / (following - label))
    positive = [value for value in neighbours if value > 0]
    return float(np.median(positive)) if positive else 20.0


def choose_primary_components(
    components_by_label: Dict[int, List[Component]], max_axis_distance_mm: float
) -> Tuple[Dict[int, Component], Dict[int, float], object]:
    labels = sorted(components_by_label)
    primary = {
        label: max(components_by_label[label], key=lambda component: component.voxel_count)
        for label in labels
    }

    for _ in range(2):
        current = [primary[label] for label in labels]
        offsets = [(label - labels[0]) * 2.0 for label in labels]
        fitted_z = isotonic_increasing(
            [component.center_world[2] for component in current],
            [max(component.voxel_count, 1) for component in current],
            offsets,
        )
        expected_z = dict(zip(labels, fitted_z))
        axis = fit_axis(current)
        updated: Dict[int, Component] = {}
        for label in labels:
            spacing = max(local_spacing(label, expected_z), 5.0)
            largest = max(c.voxel_count for c in components_by_label[label])

            def score(component: Component) -> float:
                size_term = 2.0 * np.log(max(component.voxel_count, 1) / largest)
                z_term = abs(component.center_world[2] - expected_z[label]) / spacing
                radial = np.linalg.norm(
                    component.center_world[:2] - axis(component.center_world[2])
                ) / max(max_axis_distance_mm, 1.0)
                return float(
                    size_term - 1.5 * z_term * z_term - 0.5 * radial * radial
                    + 0.25 * component.bone_fraction
                )

            updated[label] = max(components_by_label[label], key=score)
        primary = updated

    current = [primary[label] for label in labels]
    offsets = [(label - labels[0]) * 2.0 for label in labels]
    fitted_z = isotonic_increasing(
        [component.center_world[2] for component in current],
        [max(component.voxel_count, 1) for component in current],
        offsets,
    )
    return primary, dict(zip(labels, fitted_z)), fit_axis(current)


def bbox_gap_mm(
    first: Tuple[Tuple[int, int], ...],
    second: Tuple[Tuple[int, int], ...],
    spacing: np.ndarray,
) -> float:
    gaps = []
    for axis in range(3):
        gap_voxels = max(
            0,
            first[axis][0] - second[axis][1],
            second[axis][0] - first[axis][1],
        )
        gaps.append(gap_voxels * spacing[axis])
    return float(np.linalg.norm(gaps))


def ellipsoid_structure(radius_mm: float, spacing: np.ndarray) -> np.ndarray:
    if radius_mm <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    radii = np.ceil(radius_mm / spacing).astype(int)
    grids = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    distance = np.zeros(tuple(2 * radii + 1), dtype=np.float64)
    for axis, grid in enumerate(grids):
        distance += (grid * spacing[axis] / radius_mm) ** 2
    return distance <= 1.0


def refine_local_mask(
    selected: Sequence[Component],
    label: int,
    ct: np.ndarray,
    original: np.ndarray,
    output: np.ndarray,
    affine: np.ndarray,
    axis,
    expected_z: Dict[int, float],
    spacing: np.ndarray,
    transverse_radius_mm: float,
    terminal_guard_factor: float,
    min_hu: float,
    closing_radius_mm: float,
    max_hole_volume_mm3: float,
    voxel_volume_mm3: float,
) -> Dict[str, int]:
    """Apply voxel-level anatomical constraints and conservative gap repair."""
    all_coords = np.concatenate([component.coords for component in selected], axis=0)
    region = ANATOMICAL_REGION[label]
    # Cervical vertebrae are smaller and contain genuine foramina.  In particular,
    # preserve the ring-like atlas (C1) and the special anatomy of the axis (C2).
    effective_closing = closing_radius_mm * (2.0 / 3.0 if region == "cervical" else 1.0)
    effective_hole_volume = 0.0 if region == "cervical" else max_hole_volume_mm3
    padding = np.maximum(
        np.ceil(max(effective_closing, 1.0) / spacing).astype(int) + 2, 2
    )
    low = np.maximum(all_coords.min(axis=0) - padding, 0)
    high = np.minimum(all_coords.max(axis=0) + padding + 1, original.shape)
    crop_slice = tuple(slice(int(low[d]), int(high[d])) for d in range(3))
    shape = tuple(int(high[d] - low[d]) for d in range(3))
    core = np.zeros(shape, dtype=bool)
    for component in selected:
        local = component.coords - low
        core[tuple(local.T)] = True

    ct_crop = np.asarray(ct[crop_slice])
    raw_core_voxels = int(core.sum())
    core &= ct_crop >= min_hu
    hu_removed_voxels = raw_core_voxels - int(core.sum())

    # A connected vertebra prediction can leak into an attached rib.  Apply the
    # region-specific spinal corridor to individual voxels as well as components.
    core_local_coords = np.column_stack(np.where(core)).astype(np.int32)
    global_coords = core_local_coords + low
    world = nib.affines.apply_affine(affine, global_coords)
    radial = np.linalg.norm(world[:, :2] - axis(world[:, 2]), axis=1)
    transverse_ok = radial <= transverse_radius_mm
    corridor_removed_voxels = int(np.count_nonzero(~transverse_ok))

    # Sacrum/coccyx and skull are not among the 24 requested labels.  Guard only
    # the inferior edge of L5 and superior edge of C1; internal spinous processes
    # may legitimately cross an intervertebral midpoint.
    terminal_ok = np.ones_like(transverse_ok)
    level_spacing = max(local_spacing(label, expected_z), 5.0)
    if label == 1:
        terminal_ok &= world[:, 2] >= (
            expected_z[label] - terminal_guard_factor * level_spacing
        )
    elif label == 24:
        terminal_ok &= world[:, 2] <= (
            expected_z[label] + terminal_guard_factor * level_spacing
        )
    terminal_removed_voxels = int(np.count_nonzero(transverse_ok & ~terminal_ok))
    accepted = transverse_ok & terminal_ok
    rejected_local = core_local_coords[~accepted]
    if rejected_local.size:
        core[tuple(rejected_local.T)] = False

    refined = core.copy()
    if effective_closing > 0:
        closed = ndi.binary_closing(
            core, structure=ellipsoid_structure(effective_closing, spacing)
        )
        refined |= closed & (ct_crop >= min_hu)

    if effective_hole_volume > 0:
        holes = ndi.binary_fill_holes(refined) & ~refined
        hole_labels, hole_count = ndi.label(holes, structure=np.ones((3, 3, 3), bool))
        if hole_count:
            counts = np.bincount(hole_labels.ravel())
            allowed = np.where(counts * voxel_volume_mm3 <= effective_hole_volume)[0]
            allowed = allowed[allowed != 0]
            if allowed.size:
                refined |= np.isin(hole_labels, allowed) & (ct_crop >= min_hu)

    output_crop = output[crop_slice]
    original_crop = original[crop_slice]
    output_crop[core] = label
    additions = refined & ~core & (original_crop == 0) & (output_crop == 0)
    output_crop[additions] = label
    return {
        "assigned_original_voxels": int(core.sum()),
        "added_voxels": int(additions.sum()),
        "hu_removed_voxels": hu_removed_voxels,
        "corridor_removed_voxels": corridor_removed_voxels,
        "terminal_guard_removed_voxels": terminal_removed_voxels,
    }


def atomic_save_nifti(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".nii.gz", dir=str(path.parent), delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        image = nib.Nifti1Image(data.astype(np.uint8, copy=False), reference.affine, header)
        image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
        image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
        nib.save(image, str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_json(value: object, path: Path) -> None:
    """Write JSON completely before exposing it at the destination path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_case_directories(
    staged_root: Path,
    output_root: Path,
    cases: Sequence[str],
    overwrite: bool,
) -> None:
    """Publish an entire batch, rolling every case back if any rename fails."""
    output_root.mkdir(parents=True, exist_ok=True)
    destinations = {case: output_root / case for case in cases}
    for case, destination in destinations.items():
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise ValueError(f"{case}: output path is not a regular directory: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {destination}")
        staged = staged_root / case
        if not staged.is_dir():
            raise FileNotFoundError(f"{case}: staged output is missing: {staged}")

    backup_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.backup-", dir=str(output_root.parent))
    )
    backed_up: List[str] = []
    published: List[str] = []
    try:
        for case, destination in destinations.items():
            if destination.exists():
                os.replace(destination, backup_root / case)
                backed_up.append(case)
        for case, destination in destinations.items():
            os.replace(staged_root / case, destination)
            published.append(case)
    except BaseException as publish_error:
        rollback_errors: List[str] = []
        for case in reversed(published):
            destination = destinations[case]
            try:
                if destination.exists():
                    os.replace(destination, staged_root / case)
            except OSError as error:
                rollback_errors.append(f"remove new {case}: {error}")
        for case in reversed(backed_up):
            backup = backup_root / case
            try:
                if backup.exists():
                    os.replace(backup, destinations[case])
            except OSError as error:
                rollback_errors.append(f"restore old {case}: {error}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"batch publication failed and rollback was incomplete: {details}"
            ) from publish_error
        try:
            backup_root.rmdir()
        except OSError as cleanup_error:
            print(
                "WARNING: publication was rolled back, but the empty backup "
                f"directory could not be removed: {backup_root}: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    else:
        try:
            shutil.rmtree(backup_root)
        except OSError as cleanup_error:
            print(
                "WARNING: new outputs were published successfully, but the old "
                f"backup could not be removed: {backup_root}: {cleanup_error}",
                file=sys.stderr,
                flush=True,
            )


def select_components(
    label: int,
    candidates: Sequence[Component],
    primary: Component,
    expected_z: Dict[int, float],
    axis,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Classify components as keep, adjacent-level relabel, or delete."""
    labels = sorted(expected_z)
    decisions: List[Dict[str, object]] = []
    spacing = max(local_spacing(label, expected_z), 5.0)
    expected_array = np.asarray([expected_z[item] for item in labels])

    for component in candidates:
        z = float(component.center_world[2])
        radial = float(np.linalg.norm(component.center_world[:2] - axis(z)))
        nearest_label = labels[int(np.argmin(np.abs(expected_array - z)))]
        ratio = component.voxel_count / max(primary.voxel_count, 1)
        action = "keep" if component is primary else "delete"
        target_label = label
        reason = "primary" if component is primary else "kept"
        if component is primary:
            pass
        elif component.volume_mm3 < args.min_component_volume_mm3:
            reason = "too_small"
        elif component.below_min_hu_fraction > 0.50:
            reason = "mostly_below_ct_hu_limit"
        elif nearest_label != label:
            if (
                abs(nearest_label - label) <= args.relabel_max_level_difference
                and radial <= anatomical_radius_mm(nearest_label, args)
                and component.bone_fraction >= args.relabel_min_bone_fraction
            ):
                action = "relabel_candidate"
                target_label = nearest_label
                reason = "closer_to_adjacent_vertebra_level"
            else:
                reason = "closer_to_nonadjacent_level_or_outside_target_corridor"
        elif abs(z - expected_z[label]) > 0.75 * spacing:
            reason = "outside_expected_superior_inferior_band"
        elif radial > anatomical_radius_mm(label, args):
            reason = "outside_spinal_axis_corridor"
        else:
            action = "keep"

        decisions.append(
            {
                "component": component,
                "action": action,
                "target_label": int(target_label),
                "reason": reason,
                "nearest_label": int(nearest_label),
                "radial_distance_mm": radial,
                "volume_ratio_to_primary": float(ratio),
            }
        )
    return decisions


def process_arrays(
    case: str,
    ct: np.ndarray,
    prediction: np.ndarray,
    affine: np.ndarray,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run ShapeKit-Geodesic on aligned CT and 0..24 labels without writing files.

    The caller supplies the shared voxel-to-world affine. File-backed callers
    should use ``process_case``, which also verifies the two image affines and
    adds source/output paths to the report.
    """
    if args is None:
        args = build_parser().parse_args(["--ct-root", ".", "--prediction-root", "."])
    ct = np.asanyarray(ct)
    prediction = np.asanyarray(prediction)
    affine = np.asarray(affine)
    if ct.ndim != 3 or prediction.ndim != 3:
        raise ValueError(
            f"{case}: CT and prediction must both be 3D; got "
            f"{ct.shape} and {prediction.shape}"
        )
    if ct.shape != prediction.shape:
        raise ValueError(
            f"{case}: CT shape {ct.shape} != prediction shape {prediction.shape}"
        )
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"{case}: affine must be a finite 4x4 matrix")

    if not (
        np.issubdtype(ct.dtype, np.integer)
        or np.issubdtype(ct.dtype, np.floating)
    ) or not arrays_are_finite(ct):
        raise ValueError(f"{case}: CT must contain only finite real numeric values")

    prediction = validate_prediction_labels(prediction, case)

    spacing = nib.affines.voxel_sizes(affine).astype(np.float64)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(f"{case}: voxel spacing must be finite and positive")
    if not math.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError(f"{case}: affine must define a finite, non-zero voxel volume")
    components_by_label: Dict[int, List[Component]] = {}
    for label in range(1, 25):
        components = extract_components(
            label, prediction, ct, affine, voxel_volume_mm3, args.min_hu
        )
        if components:
            components_by_label[label] = components

    if not components_by_label:
        raise ValueError(f"{case}: prediction contains no vertebra labels")

    primary, expected_z, axis = choose_primary_components(
        components_by_label, args.max_axis_distance_mm
    )
    output = np.zeros_like(prediction, dtype=np.uint8)
    report_labels: Dict[str, object] = {}
    assignments: Dict[int, List[Component]] = {label: [] for label in range(1, 25)}
    input_counts = np.bincount(prediction.ravel(), minlength=25)

    for label in range(1, 25):
        name = VERTEBRA_NAMES[label - 1]
        candidates = components_by_label.get(label, [])
        if not candidates:
            report_labels[name] = {"label": label, "present": False}
            continue

        decisions = select_components(
            label,
            candidates,
            primary[label],
            expected_z,
            axis,
            args,
        )
        component_report: List[Dict[str, object]] = []
        for decision in decisions:
            component = decision.pop("component")
            source_gap = bbox_gap_mm(component.bbox, primary[label].bbox, spacing)
            target_label = int(decision["target_label"])
            action = str(decision["action"])
            reason = str(decision["reason"])
            if (
                action == "keep"
                and component is not primary[label]
                and decision["volume_ratio_to_primary"] < args.large_secondary_ratio
                and source_gap > args.secondary_gap_mm
            ):
                action = "delete"
                reason = "small_component_too_far_from_primary"

            target_gap: Optional[float] = None
            if action == "relabel_candidate":
                target_gap = bbox_gap_mm(
                    component.bbox, primary[target_label].bbox, spacing
                )
                if target_gap <= args.relabel_target_gap_mm:
                    action = "relabel"
                    assignments[target_label].append(component)
                else:
                    action = "delete"
                    reason = "adjacent_level_candidate_too_far_from_target"
            elif action == "keep":
                assignments[label].append(component)

            component_report.append(
                {
                    "component_id": component.component_id,
                    "action": action,
                    "reason": "primary" if component is primary[label] else reason,
                    "source_label": name,
                    "output_label": (
                        VERTEBRA_NAMES[target_label - 1] if action != "delete" else None
                    ),
                    "voxel_count": component.voxel_count,
                    "volume_mm3": round(component.volume_mm3, 3),
                    "center_world_mm": [round(float(value), 3) for value in component.center_world],
                    "bone_fraction_hu_ge_130": round(component.bone_fraction, 5),
                    "radial_distance_mm": round(float(decision["radial_distance_mm"]), 3),
                    "distance_to_source_primary_bbox_mm": round(source_gap, 3),
                    "distance_to_target_primary_bbox_mm": (
                        round(target_gap, 3) if target_gap is not None else None
                    ),
                    "nearest_predicted_level": VERTEBRA_NAMES[int(decision["nearest_label"]) - 1],
                }
            )

        report_labels[name] = {
            "label": label,
            "present": True,
            "anatomical_region": ANATOMICAL_REGION[label],
            "input_voxels": int(input_counts[label]),
            "expected_world_z_mm": round(float(expected_z[label]), 3),
            "components": component_report,
        }

    # Refinement is performed after all component decisions so that an erroneous
    # fragment can be transferred to its anatomically adjacent target label.
    for label in range(1, 25):
        name = VERTEBRA_NAMES[label - 1]
        selected = assignments[label]
        if not selected:
            continue
        refinement = refine_local_mask(
            selected,
            label,
            ct,
            prediction,
            output,
            affine,
            axis,
            expected_z,
            spacing,
            anatomical_radius_mm(label, args),
            args.terminal_guard_factor,
            args.min_hu,
            args.closing_radius_mm,
            args.max_hole_volume_mm3,
            voxel_volume_mm3,
        )
        report_labels[name].update(refinement)

    output_counts = np.bincount(output.ravel(), minlength=25)
    # Stage two: count CT-supported body instances independently of the corrupt
    # L1..T7 IDs, then propagate them along bone connectivity. The historical
    # component/refinement records below describe stage one; final voxel totals
    # and transition counts describe the complete pipeline.
    thoracolumbar_report = {"status": "disabled"}
    if not getattr(args, "disable_thoracolumbar_refinement", False):
        if __package__:
            from .thoracolumbar_refinement import refine_thoracolumbar
        else:
            from thoracolumbar_refinement import refine_thoracolumbar
        output, thoracolumbar_report = refine_thoracolumbar(
            ct, prediction, output, affine
        )
        print(f"  L1-T7 refinement: {thoracolumbar_report['status']}", flush=True)
        output_counts = np.bincount(output.ravel(), minlength=25)
    transitions = label_transition_counts(prediction, output)
    for label in range(1, 25):
        name = VERTEBRA_NAMES[label - 1]
        report_labels[name]["output_voxels"] = int(output_counts[label])
        report_labels[name]["incoming_relabelled_voxels"] = int(
            transitions[1:, label].sum() - transitions[label, label]
        )
        report_labels[name]["outgoing_relabelled_voxels"] = int(
            transitions[label, 1:].sum() - transitions[label, label]
        )

    diagonal_foreground = int(np.trace(transitions[1:, 1:]))
    input_foreground = int(transitions[1:, :].sum())
    output_foreground = int(transitions[:, 1:].sum())
    removed = input_foreground - diagonal_foreground
    removed_to_background = int(transitions[1:, 0].sum())
    added = int(transitions[0, 1:].sum())
    changed_labels = int(transitions[1:, 1:].sum()) - diagonal_foreground
    report: Dict[str, object] = {
        "case": case,
        "shape": list(prediction.shape),
        "spacing_mm": [float(value) for value in spacing],
        "pipeline_version": "2026-09-11-ct-body-geodesic",
        "thoracolumbar_refinement": thoracolumbar_report,
        "component_record_scope": "legacy stage before thoracolumbar refinement; final output counts follow both stages",
        "parameters": {
            "min_hu": args.min_hu,
            "min_component_volume_mm3": args.min_component_volume_mm3,
            "max_axis_distance_mm": args.max_axis_distance_mm,
            "secondary_gap_mm": args.secondary_gap_mm,
            "large_secondary_ratio": args.large_secondary_ratio,
            "relabel_max_level_difference": args.relabel_max_level_difference,
            "relabel_target_gap_mm": args.relabel_target_gap_mm,
            "relabel_min_bone_fraction": args.relabel_min_bone_fraction,
            "lumbar_radius_mm": args.lumbar_radius_mm,
            "thoracic_radius_mm": args.thoracic_radius_mm,
            "cervical_radius_mm": args.cervical_radius_mm,
            "upper_cervical_radius_allowance_mm": args.upper_cervical_radius_allowance_mm,
            "terminal_guard_factor": args.terminal_guard_factor,
            "closing_radius_mm": args.closing_radius_mm,
            "max_hole_volume_mm3": args.max_hole_volume_mm3,
        },
        "summary": {
            "input_foreground_voxels": input_foreground,
            "output_foreground_voxels": output_foreground,
            "removed_or_relabelled_voxels": removed,
            "removed_to_background_voxels": removed_to_background,
            "added_voxels": added,
            "directly_relabelled_voxels": changed_labels,
        },
        "labels": report_labels,
    }
    return output, report


def process_case(
    case: str,
    ct_path: Path,
    prediction_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    reported_output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    ct_image = nib.load(str(ct_path))
    prediction_image = nib.load(str(prediction_path))
    if len(ct_image.shape) != 3 or len(prediction_image.shape) != 3:
        raise ValueError(
            f"{case}: CT and prediction must both be 3D; got "
            f"{ct_image.shape} and {prediction_image.shape}"
        )
    if ct_image.shape != prediction_image.shape:
        raise ValueError(
            f"{case}: CT shape {ct_image.shape} != prediction shape {prediction_image.shape}"
        )
    if not np.all(np.isfinite(ct_image.affine)) or not np.all(
        np.isfinite(prediction_image.affine)
    ):
        raise ValueError(f"{case}: CT and prediction affines must contain finite values")
    if not np.allclose(ct_image.affine, prediction_image.affine, rtol=1e-5, atol=1e-3):
        raise ValueError(f"{case}: CT and prediction affine matrices differ")

    output, array_report = process_arrays(
        case,
        np.asanyarray(ct_image.dataobj),
        np.asanyarray(prediction_image.dataobj),
        prediction_image.affine,
        args,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_nifti(output, prediction_image, output_dir / "combined_labels.nii.gz")
    segmentation_dir = output_dir / "segmentations"
    for label, name in enumerate(VERTEBRA_NAMES, start=1):
        atomic_save_nifti(
            (output == label).astype(np.uint8),
            prediction_image,
            segmentation_dir / f"{name}.nii.gz",
        )

    report: Dict[str, object] = {
        "case": case,
        "ct_path": str(ct_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "output_path": str((reported_output_dir or output_dir).resolve()),
        **array_report,
    }
    atomic_save_json(report, output_dir / "postprocessing_report.json")
    return report


def discover_cases(ct_root: Path, prediction_root: Path, requested: Iterable[str]) -> List[str]:
    requested = list(requested)
    if requested:
        return requested
    return sorted(
        directory.name
        for directory in prediction_root.iterdir()
        if directory.is_dir()
        and (directory / "combined_labels.nii.gz").is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anatomy-aware post-processing for 24-class vertebra predictions"
    )
    parser.add_argument("--ct-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case", action="append", default=[], help="case ID; repeat as needed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="transactionally replace complete existing case directories",
    )
    parser.add_argument("--min-hu", type=float, default=-250.0)
    parser.add_argument("--min-component-volume-mm3", type=float, default=8.0)
    parser.add_argument("--max-axis-distance-mm", type=float, default=65.0)
    parser.add_argument("--secondary-gap-mm", type=float, default=12.0)
    parser.add_argument("--large-secondary-ratio", type=float, default=0.03)
    parser.add_argument("--relabel-max-level-difference", type=int, default=1)
    parser.add_argument("--relabel-target-gap-mm", type=float, default=16.0)
    parser.add_argument("--relabel-min-bone-fraction", type=float, default=0.60)
    parser.add_argument("--lumbar-radius-mm", type=float, default=75.0)
    parser.add_argument("--thoracic-radius-mm", type=float, default=58.0)
    parser.add_argument("--cervical-radius-mm", type=float, default=55.0)
    parser.add_argument("--upper-cervical-radius-allowance-mm", type=float, default=8.0)
    parser.add_argument("--terminal-guard-factor", type=float, default=1.35)
    parser.add_argument("--closing-radius-mm", type=float, default=1.5)
    parser.add_argument("--max-hole-volume-mm3", type=float, default=50.0)
    parser.add_argument(
        "--disable-thoracolumbar-refinement", action="store_true",
        help="run only the previous component/morphology pipeline for comparison",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.ct_root = args.ct_root.expanduser().resolve()
    args.prediction_root = args.prediction_root.expanduser().resolve()
    args.output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else args.prediction_root.with_name(args.prediction_root.name + "Postprocessed")
    )
    if not args.ct_root.is_dir():
        parser.error(f"CT root does not exist: {args.ct_root}")
    if not args.prediction_root.is_dir():
        parser.error(f"prediction root does not exist: {args.prediction_root}")
    if paths_overlap(args.output_root, args.ct_root):
        parser.error("--output-root must not overlap --ct-root")
    if paths_overlap(args.output_root, args.prediction_root):
        parser.error("--output-root must not overlap --prediction-root")
    if args.output_root.exists() and not args.output_root.is_dir():
        parser.error(f"output root is not a directory: {args.output_root}")
    finite_float_attributes = (
        "min_hu", "min_component_volume_mm3", "max_axis_distance_mm",
        "secondary_gap_mm", "large_secondary_ratio", "relabel_target_gap_mm",
        "relabel_min_bone_fraction",
        "lumbar_radius_mm", "thoracic_radius_mm", "cervical_radius_mm",
        "upper_cervical_radius_allowance_mm", "terminal_guard_factor",
        "closing_radius_mm", "max_hole_volume_mm3",
    )
    for attribute in finite_float_attributes:
        if not math.isfinite(getattr(args, attribute)):
            parser.error(f"--{attribute.replace('_', '-')} must be finite")
    for attribute in (
        "min_component_volume_mm3", "max_axis_distance_mm", "secondary_gap_mm",
        "relabel_target_gap_mm", "lumbar_radius_mm", "thoracic_radius_mm",
        "cervical_radius_mm", "upper_cervical_radius_allowance_mm",
        "terminal_guard_factor", "closing_radius_mm", "max_hole_volume_mm3",
    ):
        if getattr(args, attribute) < 0:
            parser.error(f"--{attribute.replace('_', '-')} must be non-negative")
    if args.relabel_max_level_difference < 0:
        parser.error("--relabel-max-level-difference must be non-negative")
    if not 0 <= args.large_secondary_ratio <= 1:
        parser.error("--large-secondary-ratio must be between 0 and 1")
    if not 0 <= args.relabel_min_bone_fraction <= 1:
        parser.error("--relabel-min-bone-fraction must be between 0 and 1")

    cases = discover_cases(args.ct_root, args.prediction_root, args.case)
    if not cases:
        parser.error("no cases with both ct.nii.gz and combined_labels.nii.gz were found")
    try:
        validate_case_ids(cases)
    except ValueError as error:
        parser.error(str(error))
    invalid_destinations = [
        case
        for case in cases
        if (args.output_root / case).is_symlink()
        or (
            (args.output_root / case).exists()
            and not (args.output_root / case).is_dir()
        )
    ]
    if invalid_destinations:
        parser.error(
            "output paths are not regular directories for: "
            + ", ".join(invalid_destinations)
        )
    conflicts = [case for case in cases if (args.output_root / case).exists()]
    if conflicts and not args.overwrite:
        parser.error(
            "output already exists for: " + ", ".join(conflicts)
            + "; pass --overwrite to replace files in those case directories"
        )

    inputs: Dict[str, Tuple[Path, Path]] = {}
    for case in cases:
        ct_path = args.ct_root / case / "ct.nii.gz"
        prediction_path = args.prediction_root / case / "combined_labels.nii.gz"
        if not ct_path.is_file():
            raise FileNotFoundError(f"{case}: missing {ct_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"{case}: missing {prediction_path}")
        inputs[case] = (ct_path, prediction_path)

    print(f"Processing {len(cases)} case(s) -> {args.output_root}", flush=True)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_root.name}.staging-", dir=str(args.output_root.parent)
    ) as staging_directory:
        staging_root = Path(staging_directory)
        for index, case in enumerate(cases, start=1):
            ct_path, prediction_path = inputs[case]
            print(f"[{index}/{len(cases)}] {case}", flush=True)
            report = process_case(
                case,
                ct_path,
                prediction_path,
                staging_root / case,
                args,
                reported_output_dir=args.output_root / case,
            )
            summary = report["summary"]
            print(
                "  staged: removed={removed_to_background_voxels:,}, "
                "relabelled={directly_relabelled_voxels:,}, added={added_voxels:,}, "
                "output={output_foreground_voxels:,}".format(**summary),
                flush=True,
            )
        publish_case_directories(staging_root, args.output_root, cases, args.overwrite)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
