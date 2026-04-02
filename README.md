# Cell2Phys: Neuro-Symbolic Digital Twins

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A scalable neuro-symbolic digital twin framework integrating Large Language Model (LLM) semantic reasoning with physics-informed ODEs for simulating metabolic physiology and drug responses across multiple scales.

---

## 🎯 Overview

**Cell2Phys** bridges the gap between single-cell transcriptomics and whole-organism physiology by:

- **Neuro-Symbolic Integration**: Combining LLM-based parameter inference with mechanistic ODE models
- **Adaptive Semantic Caching (ASC)**: Accelerating simulations through intelligent parameter reuse
- **Multi-Scale Modeling**: Simulating cellular dynamics (beta-cells, hepatocytes) to systemic glucose regulation
- **In Silico Drug Trials**: Predicting therapeutic responses without physical experimentation

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jie-Ni/Cell2Phys.git
   cd Cell2Phys
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required dataset**
   
   The simulation requires a pre-integrated single-cell atlas:
   
   - **File**: `bastidas_ponce_2019.h5ad`
   - **Source**: [Zenodo DOI: 10.5281/zenodo.18331267](https://doi.org/10.5281/zenodo.18331267)
   - **Destination**: Place in `data/` directory
   
   > **Note**: This atlas integrates GSE81608 (Human Islets) and TS_Liver (Tabula Sapiens) as detailed in the manuscript.

---

## 🚀 Usage

Run the core in silico drug trial simulation:

```bash
python main.py
```

### What This Does

1. Initializes a cohort of digital twins from single-cell transcriptomic data
2. Enables Adaptive Semantic Caching (ASC) for parameter inference
3. Simulates a two-arm drug trial (Placebo vs. Metformin) using LSODA
4. Exports metrics (AUC, Time-in-Range) to `results/data/trial_summary_stats.csv`

### Expected Output

```
💊 [Cell2Phys] Starting In Silico Clinical Trial...
   Creating 20 Virtual Patients...
   
🚀 Running Simulation Loop (LSODA + ASC)...
   ...Simulating Arm: Placebo
   ...Simulating Arm: Metformin (Therapeutic)

📊 Trial Summary:
                           Arm  Drug_Conc_uM  ...  Time_in_Range_Pct
                       Placebo           0.0  ...              45.23
  Metformin (Therapeutic)                150.0  ...              78.91

✅ Data exported to: results/data/
```

---

## 🔬 Key Features

### 1. **LLM-Driven Parameterization**
Gene expression profiles → Kinetic parameters via semantic reasoning

### 2. **Adaptive Semantic Caching (ASC)**
- **Challenge**: LLM inference is computationally expensive
- **Solution**: Cache parameter mappings using FAISS vector similarity
- **Impact**: ~3-5x speedup with <5% error

### 3. **Mechanistic ODE Foundation**
- Hill kinetics for insulin secretion
- Glucotoxicity models for beta-cell plasticity
- Receptor-mediated hepatic glucose production

### 4. **Retrieval-Augmented Generation (RAG)**
- Grounds LLM reasoning in scientific literature
- Reduces hallucinations and improves biological plausibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⚠️ Disclaimer

This is a research prototype. The simulations are for academic purposes only and should not be used for clinical decision-making without proper validation.

---

## 📧 Contact

For questions or collaborations, please open an issue on this repository.
