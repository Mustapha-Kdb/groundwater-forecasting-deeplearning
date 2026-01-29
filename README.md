# 💧 Groundwater Level Prediction via Deep Learning & Transfer Learning

This project implements a high-performance deep learning pipeline to predict **Groundwater Level (GWL)** fluctuations in Australian aquifers with a 6-month lead time. The core strategy uses a "1-to-6" forecasting approach: using environmental data from a specific month to predict the water level half a year later.

## 🎯 Project Overview
Accurate groundwater forecasting is critical for managing water resources in regions prone to drought. This study leverages environmental inputs—**Precipitation, Temperature, Evapotranspiration, and NDVI**—to model piezometric levels across more than 2,500 wells.

## 🚀 Advanced Modeling Strategy
The project focuses on the efficiency of **Transfer Learning** to overcome the limitations of site-specific data:

* **Individual Models:** Training dedicated LSTM or GRU networks for each well independently.
* **Global Model:** Training a large-scale "General Model" on the combined data of all 2,500+ wells to capture universal hydrological patterns across Australia.
* **Fine-Tuning Strategy:** Adapting the pre-trained Global Model to specific local wells. This approach significantly reduces the Mean Absolute Error (MAE) compared to training from scratch.



## 🛠️ Technical Implementation
* **Architectures:** Comparative analysis between **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit) to identify the best cell for long-term dependencies.
* **Forecasting Horizon:** Designed to predict values at $t+6$ using current environmental state $t$.
* **Data Pipeline:** Includes Z-score normalization, handling of missing hydrological values, and sliding window creation for sequential training.

## 📂 Project Structure
* `GWL_1_for_6.ipynb`: The primary production notebook. Includes data ingestion, Deep Learning architectures (PyTorch/TensorFlow), Global training, and Fine-Tuning loops.
* `documentation/Rapport_Mustapha-Genouiz.pdf`: Comprehensive technical report analyzing the methodology and the performance gains of the fine-tuning strategy.
* `models/` : The different models generated throughout the prediction process.

## ⚙️ Installation & Usage

### 1. Requirements
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```
### 2. Execution
Open and run GWL_1_for_6.ipynb. The notebook is self-contained and handles the data preprocessing before launching the training and evaluation phases.

---
*Developed by Mustapha Genouiz - Master in AI at Avignon University.*
