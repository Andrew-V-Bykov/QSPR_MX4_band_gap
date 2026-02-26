# Curated Dataset, Structure–Property Relationships and Machine Learning Prediction of Band Gaps for One-Dimensional Bismuth(III) and Antimony(III) Halides

## Andrei V. Bykov, Andrei V. Shevelkov

#### <i>Lomonosov Moscow State University, Department of Chemistry, Moscow, Russia</i>

## Abstract

Vast structural diversity and remarkable optoelectronic properties of organic-inorganic and all-inorganic metal halides continue to stimulate huge research interest. While the outstanding performance of lead halide perovskites has defined the field of perovskite photovoltaics, concerns regarding lead toxicity and long-term stability have driven the search for lead-free alternatives. Low-dimensional bismuth(III) and antimony(III) halides have emerged as promising candidates; however, their rational design remains hindered by the lack of established structure–property relationships and structure predictive possibilities. In an effort to fill this knowledge gap, we report here an open curated dataset of one-dimensional (1D) bismuth(III) and antimony(III) halides featuring edge-sharing \{MX<sub>4</sub>\}<sup>–</sup> chain-type anions (M – Bi, Sb; X – Cl, Br, I), including experimentally determined optical band gap values. Leveraging crystallographically accessible structural parameters, we constructed two descriptor spaces for compounds with an <i>α</i>-\{MX<sub>4</sub>\}<sup>–</sup> anion-type: a primary space based on M–X bond lengths and X–M–X bond angles (Full Geometric Descriptor Space), and a reduced space comprising average bond lengths and distortion parameters of building units \[MX<sub>6</sub>\] (Reduced Distortion Descriptor Space). Both spaces additionally incorporate descriptors capturing the interchain X···X contacts, anion composition, and crystallographic experiment temperatures. Given the limited data volume, we deliberately employed classical machine learning (ML) algorithms with cross-validation and stability analysis. The resulting models achieve prediction errors of 0.10 and 0.11 eV for the two descriptor spaces, approaching a typical experimental uncertainty of optical band gap determination. The feature importance analysis reveals that geometric descriptors of the inorganic anionic framework dominate over noncovalent interchain interaction descriptors in governing the band gap. To promote reproducibility and further development in the field, we released a complete dataset of group 15 1D-halometallates, trained models, and supporting materials in an open-access repository under the MIT license.

## 📄 Associated Publication

This repository accompanies the manuscript:

> ...

If you use this code or dataset in your research, please cite the above publication.

## 📂 Repository Structure

```Plain text
.
├── data/                  # MX4 group 15 metal halide Dataset, Curated Set (task for ML development)
├── models/                # Trained models (.pickle)
├── notebooks/             # Reproducible Jupyter notebooks and Code for future model usage
├── scripts/               # Helper scripts
├── images/                # General figures
├── tables/                # General tables
├── environment.yml
└── LICENSE
```

## ▶️ Reproducing the Results and Future Investigation

To reproduce the workflow run notebooks **1** and **2**.

- Notebook **1_data_analysis.ipynb** contains dataset analysis data and data preparation workflow.

- Notebook **2_band_gap_prediction.ipynb** machine learning development workflow and result interpritations.

- Notebooke **3_future_models_usage.ipynb** contains example how to use trained models for the band gap prediction and aimed for future investigation.

## 👨‍🎓 Funding

> The research was carried out by support of the Non-commercial Foundation for the Advancement of Science and Education INTELLECT.

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

## 💬 Contact

Andrei V. Bykov – andrei.bykov@chemistry.msu.ru