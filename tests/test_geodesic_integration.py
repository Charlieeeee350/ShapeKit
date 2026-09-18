"""Exercise physical alignment, label mapping, fallback and main.py output."""

import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import nibabel as nib
import numpy as np
import pytest
import yaml
from nibabel.orientations import axcodes2ornt, ornt_transform

from utils import postprocessing_vertebrae as engine
from utils.vertebrae_geodesic import postprocessing_vertebrae_geodesic

REPO = Path(__file__).resolve().parents[1]


def synthetic_case():
    labels = np.zeros((24, 26, 40), dtype=np.uint8)
    for label, start in ((1, 4), (2, 15), (3, 26)):
        labels[8:16, 9:17, start:start + 7] = label
    ct = np.full(labels.shape, -1000, dtype=np.int16)
    ct[labels > 0] = 400
    affine = np.diag([2., 2., 2., 1.])
    return labels, ct, affine


def orient(image, codes):
    return image.as_reoriented(ornt_transform(
        nib.orientations.io_orientation(image.affine), axcodes2ornt(codes),
    ))


@pytest.mark.parametrize("codes", [("R", "A", "S"), ("L", "P", "S"), ("S", "A", "L")])
def test_adapter_matches_core_and_restores_orientation(tmp_path, codes):
    labels, ct, affine = synthetic_case()
    expected, _ = engine.process_arrays("case", ct, labels, affine)
    reference = orient(nib.Nifti1Image(labels, affine), codes)
    label_array = np.asanyarray(reference.dataobj)
    ct_path = tmp_path / "ct.nii.gz"
    # CT and masks may have different storage axes while sharing a physical grid.
    nib.save(orient(nib.Nifti1Image(ct, affine), ("P", "S", "R")), ct_path)
    masks = {name: (label_array == label).astype(np.uint8)
             for label, name in enumerate(engine.VERTEBRA_NAMES, 1)}
    liver = np.ones(reference.shape, dtype=np.uint8)
    masks["liver"] = liver
    report_path = tmp_path / "qa" / "report.json"
    result = postprocessing_vertebrae_geodesic(
        "case", masks, reference, ct_path, logging, report_path,
    )
    expected = np.asanyarray(orient(nib.Nifti1Image(expected, affine), codes).dataobj)
    for label, name in enumerate(engine.VERTEBRA_NAMES, 1):
        np.testing.assert_array_equal(result[name], expected == label)
    assert result["liver"] is liver
    report = json.loads(report_path.read_text())
    assert report["engine_used"] == "shapekit_geodesic"
    assert report["processing_axis_codes"] == ["R", "A", "S"]
    assert report["output_axis_codes"] == list(codes)


@pytest.mark.parametrize("problem", ["missing", "shifted", "wrong_shape", "nan", "truncated", "corrupt"])
def test_unusable_ct_falls_back_and_records_reason(tmp_path, monkeypatch, problem):
    import utils.vertebrae_postprocessing as legacy

    labels, ct, affine = synthetic_case()
    reference = nib.Nifti1Image(labels, affine)
    ct_path = tmp_path / "ct.nii.gz"
    if problem == "shifted":
        shifted = affine.copy()
        shifted[0, 3] = 20
        nib.save(nib.Nifti1Image(ct, shifted), ct_path)
    elif problem == "wrong_shape":
        nib.save(nib.Nifti1Image(ct[:-1], affine), ct_path)
    elif problem == "nan":
        ct = ct.astype(np.float32)
        ct[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(ct, affine), ct_path)
    elif problem in {"truncated", "corrupt"}:
        ct = np.random.default_rng(0).integers(-1000, 1000, labels.shape, dtype=np.int16)
        nib.save(nib.Nifti1Image(ct, affine), ct_path)
        data = ct_path.read_bytes()
        if problem == "truncated":
            data = data[:-100]
        else:
            data = data[:200] + bytes(20) + data[220:]
        ct_path.write_bytes(data)
    masks = {"vertebrae_L5": labels == 1}
    fallback = Mock(return_value=masks)
    monkeypatch.setattr(legacy, "postprocessing_vertebrae", fallback)
    report_path = tmp_path / "report.json"
    result = postprocessing_vertebrae_geodesic(
        "case", masks, reference, ct_path, logging, report_path,
    )
    assert result is masks
    fallback.assert_called_once_with("case", masks, logger=logging)
    report = json.loads(report_path.read_text())
    assert report["status"] == "fallback"
    assert report["engine_used"] == "shapekit"
    assert report["reason"]


def test_empty_masks_skip_and_invalid_mask_grid_raises(tmp_path):
    labels, _, affine = synthetic_case()
    reference = nib.Nifti1Image(labels, affine)
    masks = {"liver": np.ones(labels.shape, dtype=np.uint8)}
    assert postprocessing_vertebrae_geodesic("case", masks, reference, None, logging) is masks
    masks["vertebrae_L5"] = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="grid"):
        postprocessing_vertebrae_geodesic("case", masks, reference, None, logging)


def test_empty_processed_mask_overwrites_copied_input(tmp_path):
    from utils.utils import save_and_combine_segmentations

    labels, _, affine = synthetic_case()
    segmentations = tmp_path / "segmentations"
    segmentations.mkdir()
    stale_path = segmentations / "vertebrae_L4.nii.gz"
    nib.save(nib.Nifti1Image((labels == 2).astype(np.uint8), affine), stale_path)
    masks = {"vertebrae_L4": np.zeros_like(labels), "vertebrae_L5": labels == 1}
    save_and_combine_segmentations(
        masks, {26: "vertebrae_L5", 27: "vertebrae_L4"},
        nib.Nifti1Image(labels, affine), str(tmp_path), True,
    )
    assert not np.any(nib.load(stale_path).get_fdata())
    combined = nib.load(tmp_path / "combined_labels.nii.gz").get_fdata()
    assert 27 not in np.unique(combined)
    np.testing.assert_array_equal(combined == 26, labels == 1)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["main.py", "--log_folder", str(tmp_path / "logs")])
    spec = importlib.util.spec_from_file_location("shapekit_test_main", REPO / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for handler in list(module.post_logger.handlers):
        handler.close()
        module.post_logger.removeHandler(handler)


@pytest.mark.parametrize("selector,expected", [
    ("shapekit_geodesic", "postprocessing_vertebrae_geodesic"),
    ("shapekit_pro", "postprocessing_vertebrae_pro"),
    ("shapekit_iterative", "postprocessing_vertebrae_songlin"),
    ("shapekit_songlin", "postprocessing_vertebrae_songlin"),
    ("shapekit", "postprocessing_vertebrae"),
])
def test_methods_are_independently_selectable(pipeline, monkeypatch, selector, expected):
    labels, _, affine = synthetic_case()
    masks = {"vertebrae_L5": labels == 1}
    monkeypatch.setattr(pipeline, "vertebrae_engine", selector)
    monkeypatch.setattr(pipeline, "reassign_false_positives", lambda masks, *a, **kw: masks)
    runners = {}
    for name in ("postprocessing_vertebrae_geodesic", "postprocessing_vertebrae_pro",
                 "postprocessing_vertebrae_songlin", "postprocessing_vertebrae"):
        runners[name] = Mock(return_value=masks)
        monkeypatch.setattr(pipeline, name, runners[name])
    result = pipeline.process_organs(
        masks, nib.Nifti1Image(labels, affine), labels,
        {"vertebrae"}, "case", logging, ct_path="ct.nii.gz",
    )
    assert result is masks
    for name, runner in runners.items():
        assert runner.call_count == int(name == expected)


@pytest.mark.parametrize("omit_engine", [False, True])
def test_main_cli_uses_geodesic_default_and_preserves_label_scheme(tmp_path, omit_engine):
    labels, ct, affine = synthetic_case()
    ct[labels == 2] = -1000  # removal must survive the input-folder copy.
    expected, _ = engine.process_arrays("case", ct, labels, affine)
    config = yaml.safe_load((REPO / "config.yaml").read_text())
    assert config["vertebrae_engine"] == "shapekit_geodesic"
    if omit_engine:
        config.pop("vertebrae_engine")
    config["target_organs"] = ["vertebrae"]
    config["organ_adjacency_map"] = {}
    # Exercise CT lookup in an external root as well as input case directories.
    ct_root = tmp_path / "ct_source"
    if omit_engine:
        config["ct_root"] = str(ct_root)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    case_dir = tmp_path / "input" / "case"
    seg_dir = case_dir / "segmentations"
    seg_dir.mkdir(parents=True)
    ct_dir = ct_root / "case" if omit_engine else case_dir
    ct_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(ct, affine), ct_dir / "ct.nii.gz")
    liver = np.zeros_like(labels)
    liver[1:3, 1:3, 1:3] = 1
    nib.save(nib.Nifti1Image(liver, affine), seg_dir / "liver.nii.gz")
    for label, name in enumerate(engine.VERTEBRA_NAMES, 1):
        if np.any(labels == label):
            nib.save(nib.Nifti1Image((labels == label).astype(np.uint8), affine), seg_dir / f"{name}.nii.gz")
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"))
    run = subprocess.run([
        sys.executable, str(REPO / "main.py"),
        "--input_folder", str(tmp_path / "input"),
        "--output_folder", str(tmp_path / "output"),
        "--log_folder", str(tmp_path / "logs"), "--cpu_count", "1",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    output = tmp_path / "output" / "case"
    report_path = output / "vertebrae_geodesic_report.json"
    assert report_path.is_file(), run.stdout + run.stderr
    report = json.loads(report_path.read_text())
    assert report["engine_used"] == "shapekit_geodesic"
    combined_img = nib.load(output / "combined_labels.nii.gz")
    combined = combined_img.get_fdata()
    np.testing.assert_array_equal(combined_img.affine, affine)
    np.testing.assert_array_equal(combined == 5, liver > 0)
    for label in range(1, 25):
        np.testing.assert_array_equal(combined == label + 25, expected == label)
    assert not np.any(nib.load(output / "segmentations" / "vertebrae_L4.nii.gz").get_fdata())
