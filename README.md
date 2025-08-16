# 🎓 Student Performance Indicator  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)  
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green?logo=flask&logoColor=white)  
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML%20Library-orange?logo=scikit-learn&logoColor=white)  
![Status](https://img.shields.io/badge/Status-Active-success)  
![License](https://img.shields.io/badge/License-MIT-yellow)  
![GitHub Stars](https://img.shields.io/github/stars/shiv-ansh-singh-dev/student_performance_indicator?style=social)  

---

A complete **end-to-end Machine Learning project** that predicts a student’s **Math score** based on demographic, social, and academic features.  

This project demonstrates **industry-level ML pipeline building** including **data ingestion, preprocessing, model training, evaluation, deployment via Flask, and CI/CD readiness**.  

---

## 📌 Problem Statement  

The goal is to **predict Math performance of students** based on attributes like:  

- 👨‍🎓 Gender  
- 🌍 Race/Ethnicity  
- 🎓 Parental education level  
- 🍱 Lunch type  
- 📚 Test preparation course  
- 📖 Reading score  
- ✍️ Writing score  

👉 This helps educators **identify at-risk students** and optimize learning strategies.  

---

## 📊 Exploratory Data Analysis (EDA)  

### 🔍 Key Insights:  
- 🎓 **Parental education level** strongly correlates with student scores.  
- ✅ Students who **completed test preparation** course perform better.  
- 🍱 Students with **standard lunch** outperform those with free/reduced lunch.  
- 🔗 **Reading & Writing scores** are highly correlated with Math scores.  

📈 **Visualizations include:** Distribution plots, Heatmaps, and Feature comparisons.  

---

## 🏗 Pipeline & Industry-Level Workflow  

This project follows a **modular and scalable ML pipeline** (similar to production systems):  

```
Data Ingestion → Data Transformation → Model Training → Model Evaluation → Deployment
```

🔹 **Data Ingestion** → Reads raw data, handles missing values.  
🔹 **Data Transformation** → Uses `ColumnTransformer` with:  
   - `SimpleImputer` → Handle missing values  
   - `OneHotEncoder` → Encode categorical features  
   - `StandardScaler` → Feature scaling  
🔹 **Model Training** → Trained multiple ML algorithms (Linear Regression, Random Forest, Gradient Boosting, XGBoost).  
🔹 **Model Evaluation** → Compared using **R², RMSE, MAE**.  
🔹 **Deployment** → Best model + preprocessor serialized with **dill** and deployed with Flask.  

✅ This pipeline ensures **reusability, modularity, and maintainability** → just like real **industry ML systems**.  

---

## 🚀 Model Training  

📓 **Model Training Notebook Highlights:**  
- Tested multiple regression algorithms.  
- Tuned hyperparameters with `GridSearchCV`.  
- **Best Model Achieved:**  
  - 🎯 **R² Score ≈ 0.85** on test set  

---

## 🌐 Web Application  

The application is built with **Flask + HTML + CSS (dark theme UI)**.  

🔹 **User Input Form** → Collects student details  
🔹 **Backend Pipeline** → Preprocesses & predicts  
🔹 **Output** → Displays predicted Math score  

---

## 🖼 Screenshots  

**1️⃣ Home Page**  
<img width="1917" alt="Home Page" src="https://github.com/user-attachments/assets/49caa886-3274-4864-aeee-6635b00eca4a" />  

**2️⃣ Prediction Result**  
<img width="1891" alt="Result 1" src="https://github.com/user-attachments/assets/227ac22c-6377-4f19-8b00-8d0aff3b2c5a" />  
<img width="1879" alt="Result 2" src="https://github.com/user-attachments/assets/8d9ef0ec-a156-4100-8c23-4b2b1c61d85e" />  

---

## 📂 Project Structure  

<img width="655" alt="Project Structure" src="https://github.com/user-attachments/assets/ff883c4f-282c-4079-93ae-c5524a9e2457" />  

---

## 💻 How to Run Locally  

```bash
# 1️⃣ Clone the Repository
git clone https://github.com/shiv-ansh-singh-dev/student_performance_indicator.git
cd student_performance_indicator

# 2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# 4️⃣ Run Flask App
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

➡ Open in browser: **http://127.0.0.1:5000/**  

---

## ✨ Features  

✔️ End-to-End ML Project (EDA → Model → Deployment)  
✔️ Automated Data Preprocessing Pipeline  
✔️ Multiple ML Models with Evaluation  
✔️ Flask Web Application with Interactive Dark UI  
✔️ Industry-standard **Logging & Exception Handling**  

---

## 🔮 Future Enhancements  

- 🚀 Deploy on **Heroku / Render / AWS**  
- 🔄 Add **CI/CD pipelines (GitHub Actions)**  
- 📊 Extend dataset with **study hours, attendance, socio-economic status**  
- 📈 Build an **interactive dashboard with Streamlit**  

---

## 👨‍💻 Author  

**Shiv Ansh Singh**  
🔗 [GitHub Profile](https://github.com/shiv-ansh-singh-dev)  

---

✨ This project is a **showcase of how ML models are built, evaluated, and deployed** in real-world industry applications.  
