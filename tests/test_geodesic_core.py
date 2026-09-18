"""Regression tests for the supplied ShapeKit-Geodesic algorithm and file API."""

import json

import nibabel as nib
import numpy as np
import pytest

from utils.postprocessing_vertebrae import (
    VERTEBRA_NAMES,
    build_parser,
    process_arrays,
    process_case,
)
from utils.thoracolumbar_refinement import refine_thoracolumbar


@pytest.fixture(scope="module")
def vertebral_chain():
    """Nine separated 9 mm bone bodies, 20 mm apart, with L2/T6 anchors."""
    shape = (19, 23, 99)
    coordinates = np.indices(shape)
    correct = np.zeros(shape, dtype=np.uint8)
    for index, label in enumerate(range(4, 13)):
        center = np.array([9, 11, 8 + 10 * index])
        squared_distance_mm = np.sum(
            ((coordinates - center[:, None, None, None]) * 2.0) ** 2,
            axis=0,
        )
        correct[squared_distance_mm <= 9.0**2] = label
    ct = np.full(shape, -1000.0, dtype=np.float32)
    ct[correct > 0] = 300.0
    swapped = correct.copy()
    swapped[correct == 6] = 8
    swapped[correct == 8] = 6
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return ct, correct, swapped, affine


def test_geodesic_relabels_stable_chain_and_preserves_non_targets(vertebral_chain):
    ct, correct, swapped, affine = vertebral_chain
    baseline = swapped.copy()
    # Existing non-target foreground inside a target body must remain locked,
    # even though the geodesic partition assigns its surroundings to that body.
    locked_voxel = (9, 11, 28)
    baseline[locked_voxel] = 17
    before = baseline.copy()
    expected = correct.copy()
    expected[locked_voxel] = 17

    output, report = refine_thoracolumbar(ct, swapped, baseline, affine)

    assert report["status"] == "refined"
    assert len(report["body_cores"]) == 9
    assert report["core_search"][0]["chain_found"]
    assert report["relabelled_voxels"] == np.count_nonzero(expected != before)
    assert report["existing_non_target_voxels_changed"] == 0
    assert report["unreachable_target_voxels"] == 0
    np.testing.assert_array_equal(output, expected)
    non_targets = (baseline > 0) & ~np.isin(baseline, range(5, 12))
    np.testing.assert_array_equal(output[non_targets], baseline[non_targets])
    np.testing.assert_array_equal(baseline, before)
    assert not np.shares_memory(output, baseline)


def test_geodesic_keeps_consistent_chain(vertebral_chain):
    ct, correct, _, affine = vertebral_chain

    output, report = refine_thoracolumbar(ct, correct, correct, affine)

    assert report["status"] == "unchanged_consistent"
    assert report["body_core_label_disagreement_fraction"] == 0.0
    np.testing.assert_array_equal(output, correct)


@pytest.mark.parametrize("missing_anchor", [4, 12])
def test_geodesic_skips_when_anchor_is_missing(vertebral_chain, missing_anchor):
    ct, _, swapped, affine = vertebral_chain
    baseline = swapped.copy()
    baseline[baseline == missing_anchor] = 0

    output, report = refine_thoracolumbar(ct, swapped, baseline, affine)

    assert report["status"] == "skipped"
    assert "required anchor label is empty" in report["reasons"]
    np.testing.assert_array_equal(output, baseline)


def test_array_and_file_apis_match_with_geodesic_refinement(vertebral_chain, tmp_path):
    ct, correct, swapped, affine = vertebral_chain
    ct_before, prediction_before = ct.copy(), swapped.copy()
    array_output, array_report = process_arrays("synthetic", ct, swapped, affine)
    ct_path = tmp_path / "ct.nii.gz"
    prediction_path = tmp_path / "prediction.nii.gz"
    output_dir = tmp_path / "output"
    reported_dir = tmp_path / "published"
    reference = nib.Nifti1Image(swapped, affine)
    reference.set_qform(affine, code=1)
    reference.set_sform(affine, code=2)
    nib.save(nib.Nifti1Image(ct, affine), ct_path)
    nib.save(reference, prediction_path)
    args = build_parser().parse_args(
        ["--ct-root", str(tmp_path), "--prediction-root", str(tmp_path)]
    )

    file_report = process_case(
        "synthetic", ct_path, prediction_path, output_dir, args,
        reported_output_dir=reported_dir,
    )

    assert array_report["thoracolumbar_refinement"]["status"] == "refined"
    np.testing.assert_array_equal(array_output, correct)
    np.testing.assert_array_equal(ct, ct_before)
    np.testing.assert_array_equal(swapped, prediction_before)
    assert array_output.dtype == np.uint8
    paths = {"ct_path", "prediction_path", "output_path"}
    assert array_report == {key: value for key, value in file_report.items() if key not in paths}
    assert file_report["ct_path"] == str(ct_path.resolve())
    assert file_report["prediction_path"] == str(prediction_path.resolve())
    assert file_report["output_path"] == str(reported_dir.resolve())
    with (output_dir / "postprocessing_report.json").open() as handle:
        assert json.load(handle) == file_report

    combined = nib.load(output_dir / "combined_labels.nii.gz")
    np.testing.assert_array_equal(np.asanyarray(combined.dataobj), array_output)
    np.testing.assert_array_equal(combined.affine, affine)
    assert combined.get_data_dtype() == np.dtype("uint8")
    assert int(combined.header["qform_code"]) == 1
    assert int(combined.header["sform_code"]) == 2
    mask_paths = list((output_dir / "segmentations").glob("*.nii.gz"))
    assert len(mask_paths) == 24
    for label, name in enumerate(VERTEBRA_NAMES, start=1):
        mask = nib.load(output_dir / "segmentations" / f"{name}.nii.gz")
        np.testing.assert_array_equal(np.asanyarray(mask.dataobj), array_output == label)
        np.testing.assert_array_equal(mask.affine, affine)


@pytest.mark.parametrize(
    ("invalid_label", "message"),
    [
        (-1.0, "labels must be integers from 0 to 24"),
        (25.0, "labels must be integers from 0 to 24"),
        (5.5, "non-integer label values"),
        (np.nan, "NaN or infinite"),
        (np.inf, "NaN or infinite"),
    ],
)
def test_array_api_rejects_invalid_labels(vertebral_chain, invalid_label, message):
    ct, _, swapped, affine = vertebral_chain
    invalid = swapped.astype(np.float32)
    invalid[0, 0, 0] = invalid_label

    with pytest.raises(ValueError, match=message):
        process_arrays("invalid", ct, invalid, affine)
