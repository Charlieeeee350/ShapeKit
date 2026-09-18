"""Adapt ShapeKit-Geodesic to ShapeKit's per-organ mask pipeline.

The supplied two-stage algorithm uses vertebra-only labels 1=L5 .. 24=C1.
ShapeKit's combined output retains its configured labels (normally 26..49).
Axis permutations/flips are lossless; differently sampled CTs are rejected.
"""

from pathlib import Path
import zlib

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation, axcodes2ornt, io_orientation, ornt_transform,
)

from . import postprocessing_vertebrae as engine

VERTEBRA_NAMES = engine.VERTEBRA_NAMES


def _load_ct_aligned(ct_path, reference_ras):
    if ct_path is None or not Path(ct_path).is_file():
        raise ValueError(f"CT not found ({ct_path})")
    ct_image = nib.load(str(ct_path))
    if len(ct_image.shape) != 3 or not np.all(np.isfinite(ct_image.affine)):
        raise ValueError("CT must be 3D with a finite affine")
    ct_image = nib.as_closest_canonical(ct_image)
    if ct_image.shape != reference_ras.shape or not np.allclose(
        ct_image.affine, reference_ras.affine, rtol=1e-5, atol=1e-3
    ):
        raise ValueError("CT and masks must share the same physical voxel grid")
    ct = np.asanyarray(ct_image.dataobj)
    if not (
        np.issubdtype(ct.dtype, np.integer)
        or np.issubdtype(ct.dtype, np.floating)
    ) or not engine.arrays_are_finite(ct):
        raise ValueError("CT must contain only finite real numeric HU values")
    return ct


def postprocessing_vertebrae_geodesic(
    patient_id, segmentation_dict, reference_img, ct_path, logger,
    report_path=None,
):
    """Run both stages, restore input orientation, and optionally save QA JSON.

    Missing, unreadable, or incompatible CTs use the legacy mask-only engine.
    Invalid segmentation geometry and algorithm errors are surfaced to callers.
    Non-vertebra masks are left untouched by this adapter.
    """
    def save_report(report):
        report.update(case=patient_id, requested_engine="shapekit_geodesic")
        if report_path is not None:
            engine.atomic_save_json(report, Path(report_path))

    present = [name for name in VERTEBRA_NAMES
               if segmentation_dict.get(name) is not None
               and np.any(segmentation_dict[name])]
    if not present:
        logger.info(f"[ShapeKit-Geodesic] {patient_id}: no vertebra masks present")
        save_report({"status": "skipped", "reason": "no vertebra masks present"})
        return segmentation_dict

    if len(reference_img.shape) != 3 or not np.all(np.isfinite(reference_img.affine)):
        raise ValueError(f"{patient_id}: reference must be 3D with a finite affine")
    shape = reference_img.shape
    prediction = np.zeros(shape, dtype=np.uint8)
    for label, name in enumerate(VERTEBRA_NAMES, start=1):
        mask = segmentation_dict.get(name)
        if mask is None:
            continue
        if mask.shape != shape:
            raise ValueError(f"{patient_id}: {name} grid {mask.shape} != reference {shape}")
        prediction[mask > 0] = label

    original_orientation = io_orientation(reference_img.affine)
    ras_orientation = axcodes2ornt(("R", "A", "S"))
    to_ras = ornt_transform(original_orientation, ras_orientation)
    from_ras = ornt_transform(ras_orientation, original_orientation)
    reference_ras = reference_img.as_reoriented(to_ras)
    try:
        ct = _load_ct_aligned(ct_path, reference_ras)
    except (OSError, EOFError, ValueError, zlib.error,
            nib.filebasedimages.ImageFileError) as exc:
        from .vertebrae_postprocessing import postprocessing_vertebrae as legacy

        logger.warning(
            f"[ShapeKit-Geodesic] {patient_id}: {exc}; "
            "falling back to the legacy shapekit vertebrae module"
        )
        result = legacy(patient_id, segmentation_dict, logger=logger)
        save_report({"status": "fallback", "engine_used": "shapekit", "reason": str(exc)})
        return result

    output, report = engine.process_arrays(
        patient_id, ct, apply_orientation(prediction, to_ras), reference_ras.affine,
    )
    output = apply_orientation(output, from_ras)
    for label, name in enumerate(VERTEBRA_NAMES, start=1):
        mask = (output == label).astype(np.uint8)
        if mask.any() or segmentation_dict.get(name) is not None:
            segmentation_dict[name] = mask
    report.update(
        engine_used="shapekit_geodesic", status="processed",
        ct_path=str(Path(ct_path).resolve()),
        processing_axis_codes=list(nib.aff2axcodes(reference_ras.affine)),
        output_axis_codes=list(nib.aff2axcodes(reference_img.affine)),
    )
    save_report(report)
    logger.info(
        f"[ShapeKit-Geodesic] {patient_id}: "
        f"thoracolumbar={report['thoracolumbar_refinement']['status']}; "
        f"summary={report['summary']}"
    )
    return segmentation_dict
