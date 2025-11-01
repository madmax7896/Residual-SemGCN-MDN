# Residual SemGCN-MDN: A Hybrid Graph Neural Network Approach for Time Series Forecasting

This project implements a **Residual Semantic Graph Convolutional Network (SemGCN)** combined with a **Mixture Density Network (MDN)** head to model the **spatiotemporal uncertainty**.

It also integrates a **residual learning framework**, where a simple **Linear Regression baseline** models the deterministic trend, and the SemGCN–MDN models the **residual uncertainty** and nonlinear dependencies between nodes.

---

## 🚀 Key Features

| Component | Description |
|:--|:--|
| **SemGCN Layer** | Graph convolution layer with LayerNorm, dropout, ReLU, and residual skip connections. |
| **Mixture Density Network (MDN)** | Predicts multiple Gaussian modes (π, μ, σ) for modeling stochastic outcomes. |
| **Residual Learning** | Combines a linear regression baseline and a GCN residual learner. |
| **Hybrid Loss** | Uses a combination of MDN negative log-likelihood, Huber loss, and KL regularization. |
| **Cosine Annealing Scheduler** | Smooth cyclical learning rate adjustment for stable convergence. |
| **Early Stopping** | Prevents overfitting based on validation loss. |
| **England COVID-19 Dataset** | Uses graph-structured temporal data from PyTorch Geometric Temporal. |

---

## 🧰 Architecture Overview
![Architecture](https://github.com/madmax7896/Residual-SemGCN-MDN/blob/main/architecture.png?raw=true)

---

## 📦 Dependencies

```bash
pip install torch torch-geometric torch-geometric-temporal scikit-learn numpy
