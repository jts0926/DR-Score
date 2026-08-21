# DR Score

Methods-aligned code for knee localisation, deep-learning-based radiomics model
training, and knee-level DR Score inference from posteroanterior radiographs.
The repository ends at DR Score generation; downstream clinical comparisons,
association analyses, heatmaps, and other post-hoc analyses are not included.

No study radiographs, participant records, outcomes, private paths, or model
weights are stored in this GitHub repository. The cohort labels `OAI`, `MOST`,
`KICK`, and `MenTOR` are retained only to document the supported data schema.

## Quick start

### 1. Install

Python 3.11 is required. The supplied Conda environment installs the verified
CUDA 11.8 builds of PyTorch 2.7.1 and torchvision 0.22.1:

```powershell
conda env create -f environment.yml
conda activate dr_score
pip install -e .
python -m ipykernel install --user --name dr_score --display-name "Python (DR Score)"
```

For CPU-only use, create a Python 3.11 environment, install matching CPU builds
of PyTorch and torchvision, and then run `pip install -e .`.

### 2. Download the model weights

Download both files from the
[DR Score model-weights folder on MEGA](https://mega.nz/folder/b3JGwLxB#TPElfEDOINvw10WS95Ra2w)
and place them exactly as follows:

```text
checkpoints/
|-- knee_detector.pt
`-- drscore_final.pt
```

Expected files:

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `knee_detector.pt` | 165,729,471 | `8414DD151ACE4BDDA83A3218EAF377C3502284BB1E7A64C8079CCAF4F6283A58` |
| `drscore_final.pt` | 53,622,631 | `3B9B382773B64F1057B00DAF5F1C1D1E08226521A46292FA709AA3370973AFE4` |

On Windows, verify the downloads with:

```powershell
Get-FileHash checkpoints\knee_detector.pt -Algorithm SHA256
Get-FileHash checkpoints\drscore_final.pt -Algorithm SHA256
```

The downloaded weights are ignored by Git and must not be committed to this
repository.

### 3. Run the synthetic example

The repository includes a fully synthetic, AI-generated bilateral knee
radiograph. It contains no source-cohort image or participant data.

```powershell
python scripts\infer_bilateral.py examples\synthetic_bilateral_knee_example.png `
  --output-dir outputs\single_case
```

The verified example produces a higher score for the synthetic severely
osteoarthritic right knee than for the near-normal left knee (approximately
3.356 versus 1.407). Small numerical differences can occur across compatible
hardware and software builds. These values demonstrate the code path only and
are not clinical findings or performance estimates.

The same workflow is available in
`notebooks/single_bilateral_inference.ipynb`.

## Single-image inference

Supply a standing bilateral PA knee radiograph:

```powershell
python scripts\infer_bilateral.py path\to\bilateral_radiograph.png `
  --output-dir outputs\single_case
```

The command writes:

```text
bilateral_detection_and_scores.png
left_knee_crop.png
right_knee_crop.png
left_knee_model_orientation.png
right_knee_model_orientation.png
dr_scores.csv
inference_metadata.json
```

The default assumes standard radiographic display convention: the patient's
right knee appears on the left side of the image. For the opposite convention:

```powershell
python scripts\infer_bilateral.py image.png --patient-left-on-image-left
```

Laterality must be verified before use. If two confident, non-overlapping knee
detections are not found, inference raises an error rather than returning an
incomplete result.

## Implemented pipeline

The implementation follows the manuscript and Supplementary Methods:

1. A COCO-pretrained Faster R-CNN with a ResNet-50 FPN backbone localises both
   knees. The released detector was fine-tuned on 100 participant-disjoint
   bilateral radiographs and evaluated on 61 independent radiographs.
2. The detected horizontal span and box centre define a square ROI. A 10-pixel
   margin is added, coordinates are clipped to the image boundary, and right
   knees are reflected to a common anatomical orientation.
3. Each crop is converted to grayscale, processed with CLAHE (clip limit 2.0;
   8 x 8 tiles), independently z-score normalised, and resized to 630 x 630
   pixels. No cohort-specific intensity normalisation or physical-spacing
   resampling is applied.
4. Each image is divided into a non-overlapping 7 x 7 grid of 90 x 90-pixel
   patches. Patches are resized to 224 x 224 and encoded using an
   ImageNet-pretrained ResNet-18 adapted to single-channel input.
5. The final model uses 512-dimensional embeddings, positional encoding, five
   representational neighbours (`R=5`), spatial radius one (`S=1`), AttMIL
   aggregation, and a neural Cox survival head.
6. Raw risk is mapped to the 0-4 DR Score scale using fixed primary
   out-of-fold bounds: `4 * (raw + 4.236161) / (3.5887303 + 4.236161)`.
   Bounds are never refitted on an inference image or external cohort.

Localisation and prognostic modelling were developed as separate components;
the single-image entry point applies their fixed checkpoints sequentially.

## DR Score model training

Create `data/metadata.csv` using `examples/metadata.example.csv`. Required
columns are:

```text
cohort,participant_id,knee_id,side,image_path,time_months,event
```

Both knees from the same participant must share one pseudonymous
`participant_id`. Listed images must be quality-controlled unilateral knee
crops. Run the prespecified primary seed:

```powershell
python scripts\train_drscore.py --output-dir outputs\primary
```

Repeat the complete procedure with the two sensitivity seeds:

```powershell
python scripts\train_drscore.py --output-dir outputs\all_seeds `
  --include-sensitivity-seeds
```

Training uses fivefold participant-grouped outer cross-validation. Each outer
iteration assigns approximately 70%, 10%, and 20% of participants to training,
validation, and untouched testing. Validation loss selects the checkpoint;
outer-test data are not used for model selection. The primary seed is 1029 and
sensitivity seeds are 2029 and 3029.

The final configuration uses batch size 8, eight-step gradient accumulation,
AdamW with AMSGrad (learning rate 8e-6; weight decay 1e-6), ExponentialLR
(gamma 0.999 per epoch), Cox-loss L2 coefficient 1e-4, float32 precision, and a
maximum of 30 epochs. Early stopping begins after ten completed epochs and has
patience six. No weighted outcome sampling or stochastic image augmentation is
used. See `configs/final_model.yaml` and `notebooks/DR_Score_pipeline.ipynb`.

## Knee-detector training

Create a private metadata table using
`examples/detector_metadata.example.csv`. LabelMe JSON files must contain two
rectangular knee annotations, and the `split` column must reproduce the
prespecified participant-disjoint 100-radiograph training and 61-radiograph
independent test sets:

```powershell
python scripts\train_detector.py path\to\detector_metadata.csv
```

The command exports a clean detector checkpoint and knee-level IoU and recall
results. The independent detector test set is never used for optimisation.

## Privacy checks

Run before every public commit:

```powershell
python scripts\privacy_check.py .
```

The check rejects medical-image containers, historical checkpoint formats,
private paths, email addresses, and embedded credentials. `.gitignore` excludes
local data, downloaded weights, outputs, logs, medical images, and notebook
state while explicitly allowing the synthetic example. Always inspect
`git status` before pushing because automated checks cannot recognise every
possible identifier.

## Repository layout

```text
checkpoints/   Download instructions; model weights are not tracked by Git
configs/       Methods-aligned configuration and private path template
data/          Private-data boundary and local metadata location
drscore/       Preprocessing, detector, model, survival loss, training, inference
examples/      Synthetic radiograph and privacy-safe metadata schemas
notebooks/     Output-free Jupyter workflows
scripts/       Command-line entry points and privacy checks
tests/         Geometry, preprocessing, splitting, and loading tests
```

This software is provided for research reproducibility and is not a medical
device.
