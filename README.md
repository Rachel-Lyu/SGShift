# SGShift

Sparse Generalized Additive Models to Explain Concept Shift

## Repository Layout

```text
.
├── README.md                 ← This file
├── requirements.txt          ← Exact dependency pins
├── codes/
│   ├── SGShift.py            ← Core models & utilities
│   ├── simu_support2.py      ← Simulation driver (SUPPORT2)
│   ├── simu_DiabetesReadmission.py
│   ├── simu_CovidCom.py
│   ├── support2.ipynb        ← Real-data notebook with results printed (SUPPORT2)
│   ├── DiabetesReadmission.ipynb
│   └── CovidCom.ipynb
└── data/
    ├── support2_processed.csv
    ├── source_DiabetesReadmission.csv
    ├── target_DiabetesReadmission.csv
    ├── preprocess_support2.ipynb
    ├── preprocess_DiabetesReadmission.ipynb
    └── All of Us Controlled Tier Dataset v8.ipynb
```

## Installation

SGShift uses **Python ≥ 3.9** —the minimum supported version for scikit-learn ≥ 1.0 and many scientific packages.

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install project dependencies
pip install -r requirements.txt
```

## Data

* Keep `support2_processed.csv`, `source_DiabetesReadmission.csv`, and `target_DiabetesReadmission.csv` in the `data/` folder.
* Or run `data/preprocess_support2.ipynb` and `data/preprocess_DiabetesReadmission.ipynb` to regenerate these files if needed.
* For non-public COVID-19 data, execute `data/All of Us Controlled Tier Dataset v8.ipynb` on All of Us Workbench with Controlled Tier Dataset v8.

## Running Simulations and Real-Data Experiments

### Notebooks for Real-Data Experiments

Open the corresponding `.ipynb` notebooks in **Jupyter Lab** or **VS Code** to reproduce all real data results in the paper.

### Command-line for Simulations

All simulation scripts share a common argument parser:

```bash
python codes/simu_support2.py \
  --task classification \
  --model logistic_regression \
  --solver logistic_regression \
  --n_test 1000 \
  --B 1000 \
  --shift_magnitude_y 2.0 \
  --random_state 42
```

Supported values for `--model` / `--solver` are listed in *SGShift.py* such as decision-tree, SVM, logistic-regression, Gradient Boosting, and their regression counterparts.

## License

SGShift is released under the **MIT License**.