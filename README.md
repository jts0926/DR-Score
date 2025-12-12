# DR Score: Deep-Learning Radiomics for Knee OA Progression

This repository provides the official implementation of the **DR Score**, a continuous imaging biomarker derived from knee radiographs to assess osteoarthritis severity and predict progression risk.  
The model generates a **0–4 DR Score** from a single X-ray using a Spatial–Representational kNN Attention MIL framework.

---

## Features
- Automated DR Score from a single radiograph  
- Attention-based MIL with survival prediction  
- Training pipeline (MOST + OAI)  
- External evaluation (MenTOR, KICK)  
- Lightweight script for daily inference

---

## Important Note on Training

The original training datasets (MOST, OAI, MenTOR, KICK) are **not publicly redistributed** due to data-access restrictions.  
As a result, **model training cannot be reproduced directly from this repository**.

---

## Inference (Recommended Usage)

### Step 1: Install requirements
Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Download trained model
Download the pretrained DR Score model from the following link:

https://mega.nz/file/biRRDRLL#BPT-AHqsNX5utshvNw9yea5Gj5Hh1VWI9NQtEMChh-g

Place the downloaded model file into:
```bash
trained_model/
```

### Step 3: Prepare input image
Place the knee X-ray image(s) to be evaluated into:
```bash
new_image/
```

### Step 4: Run inference notebook
Launch Jupyter Notebook, open and execute:
```bash
single_inference.ipynb
```
The notebook will:
- Load the pretrained model
- Perform inference on the provided image(s)
- Output the raw DR Score and the rescaled DR Score (0–4)

---

## License
This repository is provided for research use only.
For clinical or commercial use, please contact the authors.
