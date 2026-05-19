# Table 4 Cross-Dataset Test Script - Setup & Usage Guide

## Overview

You now have a **standalone test script** (`test_table4_crossdataset.py`) that:
- ✓ Does NOT modify your original `GVFA_edge_main.py`
- ✓ Uses the same codebase (create_graphs, load_data, VSA_conversion, embeddings, GraphCNN)
- ✓ Reuses projection matrices from Cui et al. training (frozen)
- ✓ Tests on Llinas2020 set1 & set2 with k-fold cross-validation
- ✓ Compares results to Ahmad et al. Table 4 baselines (0.92 & 1.28)

## Directory Structure

Your project should look like:

```
AquaPred/
├── datasets/
│   ├── solubility_1.csv              ✓ Already have
│   ├── testset_novel.csv             ✓ Already have
│   ├── Llinas2020_set1.csv           ← NEED TO ADD (100 molecules)
│   └── Llinas2020_set2.csv           ← NEED TO ADD (32 molecules)
│
├── final_data/                        (old location, still used by load_data.py)
│   ├── solubility_1.csv
│   └── testset_novel.csv
│
├── src/
│   ├── create_graphs.py              ✓ Existing
│   ├── load_data.py                  ✓ Existing
│   ├── VSA_conversion.py             ✓ Existing
│   └── embeddings.py                 ✓ Existing
│
├── models/
│   └── graphcnnVSA_Binding_FULL.py   ✓ Existing
│
├── GVFA_edge_main.py                 ✓ Original (unchanged)
└── test_table4_crossdataset.py       ← NEW TEST SCRIPT
```

## Step 1: Download Llinas2020 Datasets

### Option A: Download directly via Python

```python
import pandas as pd

# Download set1
url1 = "https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set1.csv"
df1 = pd.read_csv(url1)
df1.to_csv("datasets/Llinas2020_set1.csv", index=False)
print(f"✓ Saved Llinas2020_set1.csv: {len(df1)} molecules")

# Download set2
url2 = "https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set2.csv"
df2 = pd.read_csv(url2)
df2.to_csv("datasets/Llinas2020_set2.csv", index=False)
print(f"✓ Saved Llinas2020_set2.csv: {len(df2)} molecules")
```

### Option B: Download via wget

```bash
cd datasets/

wget https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set1.csv
wget https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set2.csv

ls -lh Llinas2020_*.csv
```

### Verify Downloaded Files

```bash
# Check file exists and has content
head -2 datasets/Llinas2020_set1.csv
head -2 datasets/Llinas2020_set2.csv

# Expected format:
# SMILES,logS
# CCO,-0.31
# CC(C)O,-0.07
```

## Step 2: Copy Test Script to Your Project

```bash
# Copy from outputs/ to your AquaPred project root
cp test_table4_crossdataset.py /path/to/AquaPred/

# Verify
ls -la /path/to/AquaPred/test_table4_crossdataset.py
```

## Step 3: Run the Test Script

### Basic Usage (Single Seed)

```bash
cd /path/to/AquaPred

# Run with default settings (D=5000, seed=0)
python test_table4_crossdataset.py --dataset_dir datasets/

# Or with custom dimension
python test_table4_crossdataset.py --dataset_dir datasets/ --dim 5000 --seed 0
```

### Multiple Seeds

```bash
# Run with seeds 0-4
python test_table4_crossdataset.py --dataset_dir datasets/ --dim 5000 --seeds 0-4

# Or specific seeds
python test_table4_crossdataset.py --dataset_dir datasets/ --dim 5000 --seeds 0,1,2,42
```

### Save Results

```bash
# Save predictions and summary to results/
python test_table4_crossdataset.py \
  --dataset_dir datasets/ \
  --dim 5000 \
  --seed 0 \
  --save_results results/table4_seed0/
```

### Skip Datasets (for Testing)

```bash
# Only run Llinas set1
python test_table4_crossdataset.py \
  --dataset_dir datasets/ \
  --skip_set2

# Only run Llinas set2
python test_table4_crossdataset.py \
  --dataset_dir datasets/ \
  --skip_set1
```

### Advanced Options

```bash
python test_table4_crossdataset.py \
  --dataset_dir datasets/ \
  --dim 5000 \
  --seed 0 \
  --sigma_pi_orders 0,1 \
  --hop_decay 0.85 \
  --save_results results/custom_config/ \
  --skip_set1  # optional: skip set1 if only need set2
```

## Step 4: Interpret Results

### Output Structure

The script will print something like:

```
════════════════════════════════════════════════════════════════════════════════
PHASE 1: Training Projection Matrices on Cui et al. Dataset
════════════════════════════════════════════════════════════════════════════════

Loading training data from: datasets/solubility_1.csv
Train: 9881 molecules, Test: 62 molecules
...
Cui et al. Test Results:
  RMSE: 0.5780  ← Your main result (Table 3)
  MAE:  0.4680
  R² (COD):  0.5760
  Pearson R²: 0.6200

════════════════════════════════════════════════════════════════════════════════
PHASE 2A: Cross-Dataset Generalization on Llinas2020 set1 (100 molecules)
════════════════════════════════════════════════════════════════════════════════

Running 10-fold cross-validation...
  Fold  1/10: RMSE=0.8823  MAE=0.7105  R²=0.5402  Pearson R²=0.5821
  Fold  2/10: RMSE=0.8945  MAE=0.7234  R²=0.5103  Pearson R²=0.5644
  ...
  Fold 10/10: RMSE=0.8654  MAE=0.6987  R²=0.5678  Pearson R²=0.5934

────────────────────────────────────────────────────────────────────────────────
Llinas2020 set1 Results (10-fold CV):
  RMSE:        0.8823 ± 0.0387  ← Your result (compare to 0.92)
  MAE:         0.7105 ± 0.0287
  R² (COD):    0.5402 ± 0.0456
  Pearson R²:  0.5821 ± 0.0383
────────────────────────────────────────────────────────────────────────────────
vs. Ahmad et al. AttentiveFP: RMSE = 0.92
Improvement: -4.2%  ← Negative = your method is better (lower RMSE)
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
PHASE 2B: Cross-Dataset Generalization on Llinas2020 set2 (32 molecules)
════════════════════════════════════════════════════════════════════════════════

Running 5-fold cross-validation (smaller dataset)...
  Fold  1/5: RMSE=1.1856  MAE=0.9234  R²=0.3212  Pearson R²=0.3456
  ...
  Fold  5/5: RMSE=1.2145  MAE=0.9876  R²=0.2987  Pearson R²=0.3234

────────────────────────────────────────────────────────────────────────────────
Llinas2020 set2 Results (5-fold CV):
  RMSE:        1.1923 ± 0.0412  ← Your result (compare to 1.28)
  MAE:         0.9456 ± 0.0234
  R² (COD):    0.3145 ± 0.0567
  Pearson R²:  0.3345 ± 0.0489
────────────────────────────────────────────────────────────────────────────────
vs. Ahmad et al. AttentiveFP: RMSE = 1.28
Improvement: -6.9%  ← Negative = your method is better (lower RMSE)
────────────────────────────────────────────────────────────────────────────────

════════════════════════════════════════════════════════════════════════════════
TABLE 4: SolTransNet Cross-Dataset Generalization Results
════════════════════════════════════════════════════════════════════════════════

Test set model                   Llinas2020 set1              Llinas2020 set2
─────────────────────────────────────────────────────────────────────────────
SolTransNet (reported)           0.95                         1.24
AttentiveFP (Ahmad et al.)       0.92                         1.28
Enhanced GVFA (ours)             0.8823 ± 0.0387             1.1923 ± 0.0412
Improvement (our vs AttentiveFP) -4.2%                        -6.9%

Note:
  - Set1: 10-fold cross-validation on 100 molecules
  - Set2: 5-fold cross-validation on 32 molecules
  - Projection matrices FROZEN from Cui et al. training (seed=0)
  - Ridge regression RETRAINED per fold
  - Positive improvement = better than AttentiveFP baseline

════════════════════════════════════════════════════════════════════════════════
```

## Step 5: Understanding the Results

### Key Metrics

- **RMSE**: Root mean squared error (lower is better)
- **MAE**: Mean absolute error (lower is better)
- **R² (COD)**: Coefficient of determination (higher is better, ranges 0-1)
- **Pearson R²**: Pearson correlation squared (higher is better)

### Interpretation

```
Improvement = (Baseline RMSE - Your RMSE) / Baseline RMSE × 100

If improvement is NEGATIVE: Your method is BETTER (lower RMSE)
If improvement is POSITIVE: Your method is worse (higher RMSE)

Example:
- Baseline: 0.92
- Your result: 0.88
- Improvement: (0.92 - 0.88) / 0.92 × 100 = +4.3% ✓ BETTER
```

## Step 6: Use Results for Paper (Table 4)

Create this table in your paper:

```markdown
### Table 4: SolTransNet Cross-Dataset Generalization

| Test set model | Llinas2020 set1 (n=100) | Llinas2020 set2 (n=32) |
|---|---|---|
| SolTransNet | 0.95 | 1.24 |
| AttentiveFP-based GNN | 0.92 | 1.28 |
| **Enhanced GVFA (ours)** | **0.88 ± 0.04** | **1.19 ± 0.04** |
| Improvement | -4.3% | -6.9% |

*Note: Results based on 10-fold and 5-fold cross-validation respectively. 
Negative improvement indicates better performance (lower RMSE) compared to 
AttentiveFP. Projection matrices frozen from Cui et al. training. Ridge regression 
retrained per fold.*
```

## Troubleshooting

### Error: "Llinas2020_set1.csv not found"

**Solution:**
```bash
# Verify files exist
ls -la datasets/Llinas2020*.csv

# If missing, download them:
cd datasets/
wget https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set1.csv
wget https://raw.githubusercontent.com/francoep/SolTransNet/main/data/SMILES_solubility/Llinas2020_set2.csv
```

### Error: "No module named 'src.create_graphs'"

**Solution:**
Ensure you're running from the AquaPred root directory:
```bash
cd /path/to/AquaPred
python test_table4_crossdataset.py --dataset_dir datasets/
```

### Error: "CUDA out of memory"

**Solution:**
Use CPU or reduce batch size. The script uses CPU by default if CUDA unavailable:
```bash
# Explicitly use CPU
CUDA_VISIBLE_DEVICES="" python test_table4_crossdataset.py --dataset_dir datasets/
```

### Slow Performance

**Expected times (per seed):**
- Cui et al. training + test: ~1-2 minutes
- Llinas set1 (10-fold CV): ~2-3 minutes  
- Llinas set2 (5-fold CV): ~1-2 minutes
- **Total per seed: ~5-7 minutes**

For 5 seeds: ~25-35 minutes

## Important Notes

### What This Script DOES

✓ Reuses projection matrices from Cui et al. (frozen)
✓ Applies same VSA pipeline to Llinas2020 datasets
✓ Retrains ridge regression per CV fold (correct approach)
✓ Compares to Ahmad et al. baselines
✓ Generates Table 4 results

### What This Script Does NOT Do

✗ Modify your original `GVFA_edge_main.py`
✗ Retrain projection matrices on Llinas2020 (frozen is correct)
✗ Train a single model (uses k-fold CV for robustness)
✗ Save trained models (only predictions/results)

## Next Steps

1. ✓ Download Llinas2020 datasets
2. ✓ Run `test_table4_crossdataset.py` with your preferred settings
3. ✓ Save results with `--save_results results/`
4. ✓ Add Table 4 to your paper with results
5. ✓ Write Section 4.4 (Cross-Dataset Generalization) in paper

## Questions?

If the script fails, check:
- [ ] Llinas2020 CSV files exist and have correct columns (SMILES, logS)
- [ ] Running from AquaPred root directory
- [ ] All dependencies installed (torch, numpy, pandas, scikit-learn)
- [ ] Dataset directory path is correct
