<!-- <h1 align="center">ShapeKit</h1> -->

<div align="center">
  <img src="./docs/Gemini_version.png" alt="ShapeKit" width="100%">
</div>

<!-- <div align="center">

![logo](./docs/ShapeKit.png) -->

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=BodyMaps/ShapeKit&left_color=%2363C7E6&right_color=%23CEE75F)
[![GitHub Repo stars](https://img.shields.io/github/stars/BodyMaps/ShapeKit?style=social)](https://github.com/BodyMaps/ShapeKit/stargazers)
<a href="https://twitter.com/bodymaps317">
        <img src="https://img.shields.io/twitter/follow/BodyMaps?style=social" alt="Follow on Twitter" />
</a><br/>  

</div>

# Introduction
**ShapeKit** is a plug-and-play post-processing toolkit that enables researchers and clinicians to correct anatomical errors in AI-predicted segmentations without retraining models. It integrates seamlessly into existing pipelines and supports robust, anatomy-aware refinement across multiple organs and datasets.

Using a parallelized Python workflow, ShapeKit combines, calibrates, and refines multi-organ segmentations, leading to up to **15% improvement in Dice Similarity Coefficient (DSC)** and producing consistent outputs suitable for downstream analysis.

# Paper

<b>ShapeKit</b> <br/>
[Junqi Liu*](https://kumakuma2002.github.io/), Dongli He*, [Wenxuan Li](https://scholar.google.com/citations?hl=en&user=tpNZM2YAAAAJ), Ningyu Wang, [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Zongwei Zhou](https://www.zongweiz.com/) <br/>
*Johns Hopkins University* <br/>
*Equal contribution. <br/>
MICCAI 2025 Workshop on Shape in Medical Imaging

<a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://www.cs.jhu.edu/~zongwei/publication/liu2025shapekit.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> <a href='http://www.cs.jhu.edu/~zongwei/poster/liu2025miccaiw_shapekit.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a>
  
# Installation

To set up environment, see [INSTALL.md](docs/INSTALL.md) for details.

```bash
git clone https://github.com/Charlieeeee350/ShapeKit.git
cd ShapeKit
pip install -r requirements.txt
```

# Use ShapeKit

<details>
<summary style="margin-left: 25px;">Organize your data</summary>
<div style="margin-left: 25px;">
    
```bash
INPUT or OUTPUT
└── case_001
    ├── ct.nii.gz (input CT for Geodesic or Pro)
    ├── combined_labels.nii.gz (optional)
    └── segmentations
            ├── liver.nii.gz
            ...
            └── veins.nii.gz
```
</div>
</details>

```bash
export INPUT="/path/to/your/input/folder"
export OUTPUT="/path/to/your/output/folder"
export CPU_NUM=16
export LOG="logs/folder_named_after_your_task"

python -W ignore main.py --input_folder $INPUT --output_folder $OUTPUT --cpu_count $CPU_NUM --log_folder $LOG --continue_prediction
```

The processing process will be recorded as `debug.log` and `postprocessing.log`,and are stored under the directory `LOG`.

# Plug-and-Play Configuration
Tell ShapeKit which anatomical structures you are interested in by modifying the `config.yaml` file.

<details>
<summary style="margin-left: 25px;">Check for details 🔍</summary>
<div style="margin-left: 25px;">

### How to choose your interested anatomical structures:

Open the `config.yaml`file and list the anatomical structures you want to process under `target_organs`. It’s as easy as checking boxes on a form.

```
# plug-and-play like Lego! choose organs for processing

target_organs: (example)
  - bladder
  - colon
  - duodenum
  - femur
  - intestine
  - kidney
  - liver
  - lung
  - pancreas
  - vertebrae
```

**<mark>For detailed configuration setting, please check [the config instructions 🌞](docs/config.md)</mark>.**.

Before running any commands, please ensure that `config.yaml` is properly configured. But don't worry! **Most of the configurations do not need to be changed at all.**
</details>

# Vertebrae Methods

Choose one vertebrae method with `vertebrae_engine` in `config.yaml`.
**ShapeKit-Geodesic is the default**, alongside the independently selectable
ShapeKit-Pro and ShapeKit-Iterative methods. Other organ processing is unchanged.

| Method | `vertebrae_engine` | CT required |
| --- | --- | --- |
| **ShapeKit-Geodesic (default)** | `shapekit_geodesic` | Yes; falls back to legacy `shapekit` if CT is unavailable or incompatible |
| ShapeKit-Pro | `shapekit_pro` | Yes; falls back to legacy `shapekit` if CT is absent |
| ShapeKit-Iterative | `shapekit_iterative` or `shapekit_songlin` | No |
| Legacy ShapeKit | `shapekit` | No |

# CT-Guided Geodesic Vertebrae Engine (ShapeKit-Geodesic)

This method integrates the supplied [postprocessing pipeline](utils/postprocessing_vertebrae.py)
and [thoracolumbar refinement](utils/thoracolumbar_refinement.py) as a two-stage
vertebrae engine:

1. CT-supported component filtering, reassignment, and anatomical morphology
   refine the predicted vertebrae masks.
2. A conservative L1–T7 refinement uses reliable L2 and T6 anchors to identify
   ordered vertebral body cores and partition the foreground by geodesic distance.
   It skips relabeling when stable anatomical evidence is unavailable or the
   existing labels already agree with the detected cores.

The default configuration is:

```yaml
vertebrae_engine: shapekit_geodesic
ct_file_name: ct.nii.gz
# ct_root: /path/to/ct/cases
```

The CT is read from `<input_case>/<ct_file_name>`, with
`<ct_root>/<case_id>/<ct_file_name>` as a fallback location when configured.
The main pipeline converts its vertebrae labels (26 = L5 … 49 = C1) to the
method's internal labels (1 = L5 … 24 = C1), losslessly reorients compatible
CT and mask voxel axes to RAS, and restores the output to the input mask
orientation and label scheme. Oblique data can pass through the first stage;
the thoracolumbar stage skips data that would require explicit resampling.
Missing, unreadable, or geometrically incompatible CT causes a logged fallback
to the legacy `shapekit` mask-based engine. No additional dependencies are needed.
Each Geodesic result includes `vertebrae_geodesic_report.json` in the case
output folder, recording component decisions and the thoracolumbar stage status.

The supplied pipeline is also available as a standalone command:

```bash
python -m utils.postprocessing_vertebrae \
  --ct-root /path/to/ct/cases \
  --prediction-root /path/to/vertebrae/predictions \
  --output-root /path/to/postprocessed/cases
```

The standalone command expects `<ct-root>/<case_id>/ct.nii.gz` and
`<prediction-root>/<case_id>/combined_labels.nii.gz`. Its input contains
**vertebrae only**, labeled 1 = L5 … 24 = C1, with 0 for background;
do not pass the main pipeline's multi-organ combined labels directly to it.
For each case it writes 24 binary vertebrae masks, `combined_labels.nii.gz`,
and `postprocessing_report.json`. Use `--help` for its tuning parameters.

# Evidence-Gated Vertebrae Engine (ShapeKit-Pro)

ShapeKit-Pro is an alternative that repairs vertebrae against the case
**CT image** with an
evidence-gated engine that **recolors label errors inside the prediction
envelope instead of deleting bone**:

- fragments are re-attached through CT-certified bone corridors, never
  discarded when they are real bone;
- level-band mass misassignment (e.g. a collapsed L1 split between
  neighbors) is re-arbitrated at image-detected disc planes;
- the posterior arch is rebuilt from pedicle roots, and one-level-down
  spinous chains on fused spines are repaired by a caudal-flow
  re-derivation;
- every risky stage carries its own defect meter and **reverts itself**
  when it cannot prove improvement, so one parameter set is safe across
  clean and pathological cases at scale.

Enable it in `config.yaml`:

```yaml
vertebrae_engine: shapekit_pro   # select Pro instead of the default Geodesic method
ct_file_name: ct.nii.gz          # looked up inside each input case folder
# ct_root: /path/to/ct/cases     # fallback root when CTs live elsewhere
```

No new dependencies (numpy, scipy, nibabel, scikit-image and
connected-components-3d are already required). CPU only; ~2 min for a
2.5 mm case and ~30 min for a 0.7 mm whole-spine case on 2 cores, with a
peak of roughly 9 GB on the latter — budget `--cpu_count` accordingly.
When a case has no reachable CT the engine logs it and falls back to the
legacy `shapekit` vertebrae module.

Measured on the AbdomenAtlasDemo cases (identical parameters, per-stage QA
and verification tooling in the
[ShapeKit-Pro repository](https://github.com/aj-das-research/jhu-bodymaps-warmup)):
both cases reach zero structural audit flags (fragmentation, ordering,
size, emptiness; exactly 24 components), the collapsed L1 is restored from
23.3 to 62.9 cm3 at detected disc planes, and every spinous process is
re-attached to its own vertebra on the fused case.

# Iterative Vertebrae Engine (ShapeKit-Iterative)

Alongside the Geodesic and Pro vertebrae engines, ShapeKit offers a
**VerSe-inspired iterative refinement** module that runs an anatomic
consistency cycle on the predicted vertebrae masks alone (no CT required):

- **residual reassignment** — recovers unassigned spine voxels and
  reassigns them to the nearest vertebra by 3D centroid distance;
- **gap detection & filling** — detects anomalously large Z-axis gaps
  between consecutive vertebrae and assigns residual components to the
  missing level;
- **fishing for boundary vertebrae** — extrapolates beyond the detected
  inferior/superior boundaries to recover L5 or C1 when missing;
- **duplicate removal** — merges overlapping detections via IoU thresholding;
- **anatomical size consistency** — validates vertebrae sizes against
  region-group medians (lumbar > thoracic > cervical) and removes outliers;
- **iterative convergence** — repeats the full clean → reassign → fill →
  reallocate loop until the change rate drops below 1% or max iterations
  (3) are reached.

This module is adapted from Meng et al., "Vertebrae localization,
segmentation and identification using a graph optimization and an
anatomic consistency cycle" (2022,
[https://gitlab.inria.fr/spine/vertebrae_segmentation](https://gitlab.inria.fr/spine/vertebrae_segmentation)).

Select it in `config.yaml`:

```yaml
vertebrae_engine: shapekit_iterative   # shapekit_songlin remains a supported alias
```

No CT image is needed — the module works from prediction masks alone.
Output is compatible with the existing 26-based label scheme
(26 = L5 … 49 = C1). Verified to produce identical results to the
SuPreM standalone postprocessing pipeline on the AbdomenAtlasDemo
benchmark cases.

# Key Functions
In addition to these general utilities, anatomical-structures-specific correction functions are available in [organs_postprocessing.py](organs_postprocessing.py).

Please check the details in [functions guide book 📖.](docs/functions.md)

# Related Articles

```
@article{liu2025shapekit,
  title={ShapeKit},
  author={Liu, Junqi and He, Dongli and Li, Wenxuan and Wang, Ningyu and Yuille, Alan L and Zhou, Zongwei},
  journal={arXiv preprint arXiv:2506.24003},
  year={2025}
}
```

# Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research, the Patrick J. McGovern Foundation Award, and the National Institutes of Health (NIH) under Award Number R01EB037669. We would like to thank the Johns Hopkins Research IT team in [IT@JH](https://researchit.jhu.edu/) for their support and infrastructure resources where some of these analyses were conducted; especially [DISCOVERY HPC](https://researchit.jhu.edu/research-hpc/). Paper content is covered by patents pending.
