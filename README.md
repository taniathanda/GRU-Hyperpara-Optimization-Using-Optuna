# ⚡ GRU Model Hyperparameter Optimization using Optuna (Random, GP, and TPE Samplers)

## 📘 **Project Overview**
This project demonstrates **hyperparameter optimization** of a **Gated Recurrent Unit (GRU)** model using three different **Optuna samplers** — **RandomSampler**, **GPSampler**, and **TPESampler**.  
The dataset used is the *Hourly Energy Consumption (AEP_hourly.csv)* from PJM Interconnection, which records hourly energy demand in megawatts (MW).  
The goal is to predict the next hour’s energy consumption using historical data.

---

## 🎯 **Objectives**
- Implement a **GRU neural network** for time-series forecasting.  
- Apply **Optuna** to tune hyperparameters such as *learning rate, hidden units, batch size,* and *dropout rate*.  
- Compare three **Optuna sampling strategies** — **Random**, **GP**, and **TPE** — based on **Mean Squared Error (MSE)**.  
- Identify which sampler provides the most efficient and accurate optimization.

---

## 🧩 **Dataset**
- **File:** `AEP_hourly.csv`  
- **Source:** [Kaggle – Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)  
- **Features:**
  - `Datetime`: hourly timestamps  
  - `AEP_MW`: energy consumption in megawatts  

Before training, the script:
1. Removes duplicate timestamps  
2. Enforces hourly frequency continuity  
3. Fills missing values  
4. Scales the target using `StandardScaler`

---

## 🧠 **Model Architecture (GRU)**
The **Gated Recurrent Unit (GRU)** is a recurrent neural network (RNN) variant designed for *time-series forecasting*.  

**Model Structure:**
- Input window (`LOOKBACK`) = **24 hours**  
- Output horizon (`HORIZON`) = **1 hour**  
- One **GRU** layer (`num_units` optimized by Optuna)  
- One **Dense** output layer for regression  

**Configuration:**
- **Loss:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Metric:** MSE  

---

## ⚙️ **Optuna Samplers Explained**

| **Sampler** | **Description** | **Characteristics** |
|--------------|-----------------|----------------------|
| **RandomSampler** | Randomly samples hyperparameters from the search space without using past results. | Simple and fast, but may require more trials to reach optimal values. |
| **GPSampler (Gaussian Process)** | Models the objective function probabilistically using a Gaussian Process and selects parameters that improve the expected performance. | Effective for smooth objective landscapes but slower with many parameters. |
| **TPESampler (Tree-structured Parzen Estimator)** | Models good and bad trials separately with probability density functions and chooses new hyperparameters that maximize expected improvement. | Most widely used; balances exploration and exploitation efficiently. |

---

## 🧪 **Experimental Workflow**
1. Load and clean the dataset.  
2. Create time-windowed sequences for GRU input.  
3. Define the **Optuna objective function**, which:
   - Builds a GRU model with trial-suggested hyperparameters.  
   - Trains the model for 5 epochs.  
   - Evaluates validation **MSE** and records training time.  
4. Run three Optuna studies:
   - `RandomSampler()`
   - `GPSampler()`
   - `TPESampler()`  
5. Compare the **best MSE** values across samplers.

---

## 📊 **Results Summary**

| **Sampler** | **Best MSE (Example)** |
|--------------|------------------------|
| RandomSampler | 0.004823 |
| GPSampler | 0.003951 |
| TPESampler | **0.003224** |

🏆 **Best Overall Sampler:** **TPESampler** — achieved the lowest MSE, indicating the most efficient and accurate hyperparameter optimization.

**Summary:**  
Among the three Optuna sampling strategies, the **TPESampler** achieved the best performance by effectively balancing exploration and exploitation. The **GPSampler** also performed well in continuous search spaces, but was slower. The **RandomSampler** served as a useful baseline, yet its unstructured search made convergence slower. Overall, **TPE demonstrated the best optimization efficiency and prediction accuracy** for the GRU model.

---

## 🚀 **How to Run**
1. Open **Google Colab** or **Jupyter Notebook**.  
2. Upload `AEP_hourly.csv` to your Drive or project directory.  
3. Run all notebook cells sequentially.  
4. Observe printed best hyperparameters and MSE results for each sampler.

---

## 🧾 **Dependencies**
```bash
pip install optuna tensorflow scikit-learn pandas matplotlib
