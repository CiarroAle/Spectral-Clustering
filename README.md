# Spectral Clustering Analysis and Comparison

This project, developed in **MATLAB**, provides a comprehensive analysis of **Spectral Clustering** and compares its performance with traditional clustering algorithms like **K-Means** and **DBSCAN**. The study focuses on how different algorithms handle complex, non-globular data structures.

## Project Overview

Traditional algorithms like K-Means often struggle with datasets where clusters are not linearly separable or have complex geometries. This project explores the effectiveness of the Spectral Clustering approach, leveraging graph theory and Laplacian matrices to identify clusters in 2D and 3D spaces.

### Key Features
* **Similarity Graph Construction:** Implementation of $k$-nearest neighbors ($k$-NN) graphs.
* **Laplacian Analysis:** Comparison between **Unnormalized** and **Normalized** Laplacian matrices.
* **Eigen-computation:** Use of the **Inverse Power Method** for manual eigenvalue and eigenvector calculation.
* **Benchmarking:** Direct performance comparison against K-Means and DBSCAN.

---

## Datasets

The repository includes three main datasets:
1.  **Spiral.mat**: 312 points forming three intertwined spirals (challenging for distance-based methods).
2.  **Circle.mat**: 900 points featuring two concentric rings and a separate globular cluster.
3.  **clustering_data.csv**: A 3D dataset consisting of 396 points distributed across 6 clusters.

---

## 🛠️ Implemented Techniques

### 1. Spectral Clustering
The algorithm transforms the data into a lower-dimensional space using the eigenvectors of the Laplacian matrix $L$:
* **Unnormalized Laplacian:** $L = D - W$
* **Normalized Laplacian:** $L_{sym} = D^{-1/2} L D^{-1/2}$

### 2. K-Means Clustering
Used as a baseline comparison. It performs well on the 3D dataset but fails to capture the intricate shapes of the Spiral and Circle datasets.

### 3. DBSCAN
A density-based algorithm used to identify clusters of arbitrary shape and handle noise (outliers).

---

## Getting Started

1.  **Prerequisites:** Ensure you have **MATLAB** installed.
2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/spectral-clustering-analysis.git](https://github.com/your-username/spectral-clustering-analysis.git)
    ```
3.  **Run the Script:**
    Open MATLAB and execute the main file:
    ```matlab
    HW_SC_Ciarrocchi.m
    ```

---

## Results Summary

Based on the technical report (`Spectral Clustering.pdf`):
* **Spectral Clustering** successfully identifies complex structures when the $k$ parameter for the similarity graph is correctly tuned (optimal results often found with $k=10$ or $k=40$).
* **DBSCAN** is highly effective for 2D shapes but sensitive to the $\epsilon$ (epsilon) and `min_pts` parameters.
* **K-Means** is only suitable for globular clusters and fails on the spiral/circular patterns.

---

## Repository Structure

* `HW_SC_Ciarrocchi.m`: Main MATLAB script containing the full implementation.
* `Spectral Clustering.pdf`: Detailed technical report with mathematical derivations and plots.
* `*.mat` / `*.csv`: Data files used for the experiments.

---

**Author:** Alessandro Ciarrocchi  
**Academic Context:** Politecnico di Torino
