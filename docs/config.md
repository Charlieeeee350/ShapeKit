<h1 align="center">Config Setting</h1>



1. `subfolder_name`: the name of subfolder in the input/output folder containing organs `.nii.gz` files. For example:
    ```
    INPUT / OUTPUT (--input_folder / --output_folder)
    └── case_001
        └── segmentations <- (subfolder_name)
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```

2. `class_map`: the label mapping dict of organ and their labels.
    > [!WARNING]
    > This parameter will be deprecated soon.

    All organs on this list will be read and loaded, but only the ones listed in target_organs will be processed by ShapeKit.

3. `target_organs`: the organs selected for postprocessing. 

    By adding or deleting the organs listed, you can choose which organs you want to process. For example:
    ```
        target_organs:
            - bladder
            - colon
            - duodenum
            - femur
            - intestine
            - kidney
            - liver
            - lung
            - pancreas
    ```


4. `organ_adjacency_map`: a dictionary used in the `reassign_false_positives` function. 
    
   This section identifies organs that sit close together where the AI might mislabel a border. By listing these anatomical neighbors, you help the software distinguish between touching structures—like the liver and pancreas—to ensure your results are accurate.

    Exmaple:
    ```
    organ_adjacency_map:
        lung_left: [postcava]
        lung_right: [postcava]
        liver: [kidney_right, pancreas]
    ```

    This means that during segmentation:
	(1) Parts of the predicted `lung_left` may be false positives that actually belong to `postcava`.
    (2) Similarly, `liver` may mistakenly include areas from `kidney_right` or `pancreas`.

    **Note: This map is one-directional**, i.e., if `lung_left` → `postcava` is defined, it does not imply the reverse (`postcava` → `lung_left`). This directionality reflects common misclassification patterns, not anatomical symmetry.

5. `affine_reference_file_name`: file to load affine reference info.

6. `if_save_combined_label`: boolean parameter that controls whether to save the combined labels as a .nii.gz file after processing. For example:

    ```
    OUTPUT
    └── case_001
        ├── combined_labels.nii.gz <- (if_save_combined_label)
        └── segmentations
                ├── liver.nii.gz
                ...
                └── veins.nii.gz
    ```
7. `vertebrae_engine`: which vertebrae module to run. The default is
   **ShapeKit-Geodesic**; all methods remain independently selectable.

   | Value | Method | CT requirement |
   | --- | --- | --- |
   | `shapekit_geodesic` | **ShapeKit-Geodesic (default)**: CT-supported component and morphology processing, followed by conservative L1–T7 body-core geodesic partitioning using L2/T6 anchors | Required; falls back to legacy `shapekit` if missing, unreadable, or geometrically incompatible |
   | `shapekit_pro` | ShapeKit-Pro: evidence-gated vertebra label repair within the prediction envelope | Required; falls back to legacy `shapekit` if absent |
   | `shapekit_iterative` / `shapekit_songlin` | ShapeKit-Iterative: iterative anatomical consistency refinement | None |
   | `shapekit` | Legacy mask-based module | None |

   ```yaml
   vertebrae_engine: shapekit_geodesic
   ```

   Geodesic's thoracolumbar stage skips relabeling when stable body-core
   evidence is unavailable, the existing identities are already consistent,
   or the affine is oblique and would require resampling. Its first-stage
   output is retained in those cases. Decisions are recorded in
   `<output_case>/vertebrae_geodesic_report.json`.

   The main pipeline preserves its existing 26–49 vertebrae label scheme
   (L5–C1). Its Geodesic adapter handles internal 1–24 labels and lossless
   CT/mask axis reorientation to RAS, then restores the input mask orientation.
   The [standalone Geodesic command](../README.md#ct-guided-geodesic-vertebrae-engine-shapekit-geodesic)
   accepts vertebrae-only 1–24 label volumes instead.

8. `ct_file_name` / `ct_root`: how `shapekit_geodesic` and `shapekit_pro`
   find the CT. They first look for `<input_case>/<ct_file_name>`; when
   `ct_root` is set they also try `<ct_root>/<case_id>/<ct_file_name>`.

   ```yaml
   ct_file_name: ct.nii.gz
   # ct_root: /path/to/original/ct/cases
   ```

   Geodesic requires CT and prediction masks to describe the same voxel grid
   after any lossless axis permutation/flips. It does not resample mismatched
   images; the main pipeline logs a fallback to legacy `shapekit` instead.
