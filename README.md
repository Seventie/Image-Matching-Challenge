# Image Matching Challenge 2022

Production-ready computer vision pipeline for the [Kaggle Image Matching Challenge 2022](https://www.kaggle.com/competitions/image-matching-challenge-2022), organized into two notebooks:

- `01_image_matching_eda_cv_fundamentals.ipynb`: data understanding and baseline CV foundations.
- `02_feature_matching_pipeline.ipynb`: end-to-end feature matching, evaluation, and Kaggle submission pipeline.

## What This Repository Delivers

- Scene-level dataset analysis (distribution, covisibility, intrinsics, extrinsics, image properties).
- Classical local features: SIFT, ORB, AKAZE.
- Matching strategies: BF + ratio test, FLANN + ratio test.
- Robust geometry: RANSAC, LMEDS, USAC_MAGSAC++.
- Learned matcher path: LoFTR (Kornia, optional).
- Competition metric evaluation with mAA.
- Test-time inference + submission CSV generation (SIFT ratio path and optional LoFTR path).

## Kaggle-Ready Execution

### Dataset location
These notebooks are configured for Kaggle paths:

- `/kaggle/input/image-matching-challenge-2022`

### Notebook order
Run in this order:

1. `01_image_matching_eda_cv_fundamentals.ipynb`
2. `02_feature_matching_pipeline.ipynb`

### Runtime recommendations

- CPU runtime is enough for classical pipelines.
- Enable GPU to run LoFTR efficiently.
- Keep Internet disabled unless explicitly needed.

### Optional dependency for LoFTR
If LoFTR is enabled, ensure `torch` and `kornia` are available in the Kaggle environment.

## Repository Structure

```text
Google Image Matching/
|-- Readme.md
|-- 01_image_matching_eda_cv_fundamentals.ipynb
|-- 02_feature_matching_pipeline.ipynb
|-- n1/
|-- n2/
|-- old/
```

## Pipeline Summary

### Notebook 1: EDA and CV Fundamentals

- Loads IMC 2022 training metadata.
- Analyzes scene composition and pair covisibility.
- Visualizes camera parameters and image characteristics.
- Demonstrates SIFT/ORB/AKAZE extraction and baseline detector parity checks.
- Includes camera calibration + undistortion demonstration.
- Includes stereo disparity/depth visualization.
- Includes both SIFT-based and ORB-based epipolar geometry visualization.

### Notebook 2: Feature Matching Pipeline

- Builds reusable preprocessing utilities.
- Extracts and compares SIFT, ORB, and optional LoFTR correspondences.
- Estimates fundamental matrix with robust estimators.
- Recovers relative pose and computes mAA-style quality metrics.
- Reports per-scene and aggregate runtime performance indicators.
- Adds test inference helpers from n2 notebooks:
  - `estimate_f_single_pair_ratio(...)`
  - `estimate_f_single_pair_loftr(...)`
  - `generate_submission_csv(...)`
- Supports Kaggle-ready `submission.csv` export from `test.csv`.

## Evaluation Metric (mAA)

The project follows the competition metric:

- Rotation thresholds: 1 to 10 degrees.
- Translation thresholds: 0.2 to 5 meters.
- Accuracy per threshold pair averaged into scene-level mAA.
- Final score aggregated across scenes.

## Reproducibility Notes

- Random seeds are fixed in both notebooks.
- Visual and numerical outputs are computed directly from IMC 2022 data.
- Preprocessing and evaluation functions are centralized for consistent reuse.
- Plotting has been standardized to **Matplotlib only** (Plotly removed) for better Kaggle rendering compatibility.
- The notebooks now avoid hardcoded benchmark score tables and rely on runtime-computed metrics.


