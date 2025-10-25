# ⚡ Energy Efficiency Prediction

An interactive **Streamlit web app** that predicts **Energy Usage** of buildings using trained **Random Forest** and **XGBoost** machine learning models.

---

## 🚀 Features
- Predict building energy efficiency 
- Choose between **Random Forest** or **XGBoost** model  
- Simple and intuitive **Streamlit interface**  
- Modular structure — easy to update with new models  

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/energy-efficiency-predictor.git
cd energy-efficiency-predictor
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```bash
cd webapp
streamlit run app.py
```
---

## 📊 Models Used
```bash
| Model          | MAE    | RMSE   | R²     |
|----------------|--------|--------|--------|
| Random Forest  | 0.089  | 0.149  | 0.976  |
| XGBoost        | 0.078  | 0.114  | 0.986  |
```
🏆 **XGBoost** performs the best with the lowest error and highest R² score.

---
