# PhysioNet-Deep-Imputation-Critique
"A channel-normalized critique of deep sequential imputation models (BRITS, SAITS) on the PhysioNet 2012 ICU dataset
# Alignment With Reality Is Not Optional: A Critique of Deep Sequential Imputation[cite: 2]

**Project ID:** ML-T2-017 | **Domain:** Time-Series ML - Healthcare Data Reconstruction[cite: 2]  
**Organization:** LearnDepth Academy (Track 2 - Advanced Machine Learning Internship)[cite: 2]

## 📌 Overview
Published benchmarks report state-of-the-art point-imputation MAE near 0.24 for deep sequential imputers (BRITS, SAITS) on the PhysioNet 2012 ICU dataset[cite: 3]. However, this figure is a micro-average that is heavily biased toward dense, easily predictable vitals (like heart rate), masking the models' failure on sparse, heavy-tailed clinical labs[cite: 3]. 

This project forensically audits this claim by re-evaluating BRITS and SAITS against classical baselines using a strict **channel-normalized, macro-averaged MAE**[cite: 3].

## 🚀 Key Contributions & Findings
* **The "SOTA Illusion":** Proved that under a clinically rigorous macro-averaged metric, BRITS and SAITS actually underperform a trivial constant-mean-imputation baseline[cite: 3].
* **Deterministic Superiority:** Demonstrated that for short-gap missingness, deterministic methods like Linear Interpolation and LOCF lead decisively on both pointwise accuracy and distributional fidelity[cite: 3].
* **Dual Failure Modes:** Diagnosed two coexisting error patterns in deep neural imputers: genuine predicted-variance collapse on sparse channels, and active hallucination on continuous vitals[cite: 3].
* **Pipeline Debugging:** Identified and corrected a silent validation-based early-stopping defect in the PyPOTS v1.5 library[cite: 4, 7].

## 🛠️ Technologies Used
* **Languages & Frameworks:** Python, PyTorch, PyPOTS (for BRITS/SAITS)
* **Data Processing:** Pandas, NumPy, SciPy (KS-Statistic)
* **Machine Learning:** Scikit-learn (MICE, k-NN, Random Forest for downstream AUROC/AUPRC evaluation)

## 📄 Artifacts
* `ML-T2-017_Finall.ipynb`: The complete, data-isolated experimental pipeline featuring synthetic masking, a custom `ClippedAdam` optimizer, and physiologically-bounded output clamping.
* `ML-T2-017_Technical_Paper.pdf`: The full academic report detailing the methodology, error analysis, and downstream clinical utility findings.
