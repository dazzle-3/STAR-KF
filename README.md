# STAR-KF

STAR-KF is a MATLAB workflow for separating deformation and atmospheric turbulence in InSAR time series. It combines two-stage Kalman estimation with spatial atmospheric reconstruction and uncertainty feedback.

## Requirements

- MATLAB R2022b was used for testing.
- Statistics and Machine Learning Toolbox for spatial diagnostics.
- Parallel Computing Toolbox for parallel execution and the entry point's pool management.

## Run with your own data

From the repository root in MATLAB:

```matlab
cfg = star_kf_v4_config();
output_dir = fullfile(pwd, 'results_new');
outputs = run_star_kf_v4('path/to/input', output_dir, cfg);
```

The input directory requires these files:

| File | Variables |
| --- | --- |
| `input_data.mat` | `ph_mat`, points by interferograms in radians. `xy`, planar coordinates in metres, either x/y or ID/x/y. |
| `ph_mm_full.mat` | `ph_mm_full`, points by acquisitions in line-of-sight millimetres. `day`, acquisition times. `ifgday`, one pair of acquisition times per interferogram, ordered consistently with `ph_mat`. |
| `ground_truth.mat` (optional) | Truth fields supported by `utils/load_experiment_data.m`, used for post-run evaluation. |

Use the same point order in both required files. Check that the reference acquisition index (`cfg.i_master`), wavelength, and phase-to-displacement scale match your data and sign convention.

The default joint convergence threshold is `0.01` (1%), with at most five outer iterations. The joint score is the maximum relative Frobenius change in deformation, atmospheric phase, and working variance on fixed common valid support. The code starts counting at iteration three and requires two consecutive scores below the threshold. The exit reason is available in `outputs.result.termination_reason`.

The output directory includes `joint_inversion_results.mat`, a reproducibility manifest, and evaluation summaries. Key results include `deformation_joint_final` in LOS mm and `atmosphere_epoch_final_rad` in radians. See [diagnostic fields](DIAGNOSTIC_FIELDS.md) for more details.

## Verify the installation

The smoke test generates its own small synthetic input and exercises the public entry point.

```matlab
results = runtests(fullfile('tests', 'test_run_entrypoint_small.m'));
assertSuccess(results);
```

To run the full regression suite:

```matlab
addpath('tests');
run_all_tests();
```
