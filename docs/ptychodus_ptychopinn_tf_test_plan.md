# Ptychodus → PtychoPINN (TF) Train/Infer Test Plan

## Goal
Validate a full ptychodus-driven train → load → infer workflow for the
PtychoPINN TF backend using the existing dataset:
`../PtychoPINN/datasets/Run1084_recon3_postPC_shrunk_3.npz`, and save
reconstruction artifacts for visual inspection.

## Scope
- PtychoPINN must be installed and importable.
- TensorFlow backend only (no PyTorch/Lightning path).
- Use ptychodus to load data, export `train_data.npz` / `test_data.npz`,
  train, infer, and save reconstruction outputs.

## Preconditions
1) Environment
   - `ptychopinn` and `ptycho` are installed and importable.
   - TensorFlow dependencies are available (TF path only).

2) Dataset
   - Input NPZ exists at:
     `../PtychoPINN/datasets/Run1084_recon3_postPC_shrunk_3.npz`
   - File contains `diffraction`, `xcoords_start`, `ycoords_start`,
     `probeGuess`, `objectGuess`.

3) Plugins
   - Use `SLAC_NPZ` readers to load both diffraction and product from the NPZ.
     This provides the scan positions, probe, and object guesses needed for export.

## Key Constraints (Why these steps are required)
- The raw NPZ is not in the `diff3d` schema expected by ptycho training.
  The ptychodus export step converts it to the correct training format.
- Geometry metadata must be non-zero; otherwise object-plane pixel sizes
  become 0 and coordinate mapping fails during export/reconstruction.
- Reconstruction is executed in a background task; a foreground task
  update is required to apply the new product to the repository.

## Plan

## Progress Checklist
- [ ] Environment: `ptychopinn` + `ptycho` importable; TF backend active.
- [ ] Data loaded via `SLAC_NPZ` (patterns + product).
- [ ] Metadata set: detector distance, probe energy, exposure time.
- [ ] Exported `train_data.npz` and copied `test_data.npz`.
- [ ] Training completed; `wts.h5.zip` saved.
- [ ] Inference completed; output product updated.
- [ ] Artifacts saved: `recon_product.h5`, `recon_object.npy`.

### Phase 1: Setup and Data Load
1) Start `ModelCore` and open patterns using `SLAC_NPZ`.
2) Open the product (also via `SLAC_NPZ`) from the same dataset.
3) Update metadata on the product to ensure valid geometry:
   - `detector_distance_m` (e.g., 0.75)
   - `probe_energy_eV` (e.g., 8000.0)
   - `exposure_time_s` (e.g., 0.1)

### Phase 2: Export Training Data
1) Select reconstructor `"PtychoPINN/PINN"`.
2) Export training data via ptychodus API to produce `train_data.npz`.
3) Duplicate `train_data.npz` → `test_data.npz` in the same directory.

### Phase 3: Train (TF)
1) Configure PtychoPINN settings:
   - `nepochs = 1` (short run for integration test)
   - `batch_size = 4`
   - `gridsize = 1`
   - `output_dir = <work_dir>/model_out`
2) Call `workflow_api.train_reconstructor(<train_dir>, <model_out>)`.
3) Verify `wts.h5.zip` exists in `<model_out>`.

### Phase 4: Inference + Save Artifacts
1) Run reconstruction with `block=True`.
2) Call `model.run_tasks()` to apply the reconstructed product update.
3) Save outputs for visual inspection:
   - `recon_product.h5` (product with updated object)
   - `recon_object.npy` (object array dump for plotting)

## Acceptance Criteria
- `train_data.npz` and `test_data.npz` exist and contain required keys
  (`xcoords`, `ycoords`, `diff3d`, `probeGuess`, `objectGuess`).
- Training finishes without error and creates `wts.h5.zip`.
- Reconstruction completes and produces a new product with a non-empty
  object array.
- Artifacts for inspection are written (`recon_product.h5`,
  `recon_object.npy`).

## Suggested File Layout
```
<work_dir>/
  train/
    train_data.npz
    test_data.npz
  model_out/
    wts.h5.zip
    logs/
  recon_product.h5
  recon_object.npy
```

## Notes and Follow-Ups
- If training time is too long, reduce `nepochs` or limit groups by
  updating `TrainingConfig` defaults in the wrapper (not currently exposed).
- If geometry still fails, verify metadata values and detector pixel size
  settings in ptychodus defaults.
