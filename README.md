# Spondylolisthesis Classification Model 
## via Spine Geometric Feature Extraction

## Project Overview

This project leverages machine learning to classify spinal conditions using geometric features extracted from the **BUU-LSPINE dataset**. The goal is to accurately predict the presence of specific spinal displacements based on geometric coordinates and demographic data.

The model specifically classifies patients into three categories:

* **0:** Normal
* **1:** Anterolisthesis
* **2:** Retrolisthesis

## Dataset Details

**Source:** [BUU-LSPINE Dataset](https://services.informatics.buu.ac.th/spine/)

For this project, only the **LA (Lateral) views** were used. The dataset consists of a CSV file containing the labeled coordinates of the spine from **L1 to S1**.

### Original Dataset Distribution

Below is the breakdown of the raw data by condition and gender. *(Note: While the original dataset contains Laterolisthesis cases, this project's model focuses strictly on Normal, Anterolisthesis, and Retrolisthesis cases).*

| Diagnosis / Condition | Female (F) | Male (M) | Grand Total | Included in Model |
| --- | --- | --- | --- | --- |
| **Normal** | 1,856 | 1,123 | 2,979 |  (Class 0) |
| **Anterolisthesis** | 343 | 121 | 464 |  (Class 1) |
| **Retrolisthesis** | 94 | 100 | 194 |  (Class 2) |
| **Left Laterolisthesis** | 44 | 17 | 61 | (not included in LA view) |
| **Right Laterolisthesis** | 43 | 26 | 69 | (not included in LA view) |
| *Total Disorders* | *424* | *197* | *621* | - |

## Methodology

1. **Feature Engineering:** Geometric features (such as step-off distances, slip distances, compression ratios, and tilt angles) were systematically calculated using the L1-S1 coordinate data.
2. **Preprocessing:** Target labels were encoded to `[0, 1, 2]` to comply with XGBoost's multi-class requirements.
3. **Model Selection:** An **XGBoost Classifier** was trained and fine-tuned for this task.

### Model Hyperparameters

* `n_estimators`: 1000
* `learning_rate`: 0.05
* `max_leaves`: 500
* `random_state`: 1
* `eval_metric`: 'mlogloss'

---

## Model Performance

### Cross-Validation Accuracy

The model was evaluated using 5-fold cross-validation, yielding highly consistent results with an **average accuracy of 89.26%**.

* **Fold Accuracies:** `[0.8863, 0.8782, 0.8917, 0.9012, 0.9053]`

### Confusion Matrix Analysis

Based on the validation set, the model demonstrates excellent discriminative ability, particularly for identifying "Normal" cases.

* **Class 0 (Normal):** Flawless classification. 3,037 correct predictions with zero false positives or false negatives.
* **Class 1 (Anterolisthesis):** 455 correct predictions. Misclassified as Retrolisthesis 9 times.
* **Class 2 (Retrolisthesis):** 165 correct predictions. Misclassified as Anterolisthesis 29 times.

*Note: The model shows slight confusion between Class 1 and Class 2, which is expected given both conditions involve vertebral slippage in opposing directions.*

---

## Feature Importance

The model's decisions are heavily driven by localized geometric measurements of the lower lumbar spine (L4, L5) and sacrum (S1). Below are the top 10 most influential features driving the model's predictions.

| Rank | Feature | Importance Score | Description/Notes |
| --- | --- | --- | --- |
| **1** | `L4_L5_step_off` | 0.1387 | The most dominant predictor |
| **2** | `slip_dist_L4_L5` | 0.0665 | Measures forward/backward slip |
| **3** | `slip_dist_L5_S1` | 0.0650 | Critical junction for spondylolisthesis |
| **4** | `age` | 0.0580 | Demographic factor |
| **5** | `perct_spond` | 0.0490 | Percentage of spondylolisthesis |
| **6** | `gender` | 0.0339 | Demographic factor |
| **7** | `p_disl_h_L4b_L5a` | 0.0234 | Posterior dislocation height |
| **8** | `compr_ratio_L3b_L4a` | 0.0227 | Compression ratio |
| **9** | `compr_ratio_L2b_L3a` | 0.0213 | Compression ratio |
| **10** | `v_tilt_L3b` | 0.0209 | Vertebral tilt |

*(Note: 32 additional geometric features contribute remaining fractional importance to the model, totaling 42 features).*

---

## Usage

**1. Clone the repository and install dependencies:**

```bash
pip install pandas numpy xgboost scikit-learn matplotlib

```

**2. Explore the code:**
You can view the full training process, feature extraction, and evaluation in the [Kaggle Notebook](https://www.kaggle.com/code/lavanyanigam/lumbar-spine-detection).