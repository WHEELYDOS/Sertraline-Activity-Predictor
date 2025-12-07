# Sertraline Activity Predictor 🧪💊

A machine learning–powered Flask web application that predicts **sertraline treatment activity** (e.g., response / class outcome) from structured input features.

The project includes:
- A **trained ML pipeline** (multiple models saved as `.pkl` files)
- A **Flask web app** for interactive prediction
- A **Jupyter notebook** for model training, evaluation, and experimentation

---

## 🚀 Project Overview

Sertraline is a commonly prescribed antidepressant. This project explores how machine learning can be used to predict treatment-related activity based on tabular patient/medication data.

The workflow includes:

1. **Data preprocessing & feature engineering**  
2. **Training multiple classification models**  
   - Logistic Regression  
   - Naive Bayes  
   - Random Forest  
   - Gradient Boosting  
3. **Model selection & saving trained models as `.pkl` files**  
4. **Deploying the best model(s) via a Flask API + HTML UI**

> 🔐 Note: The underlying dataset is not included here if it is sensitive or proprietary. The repository focuses on the end-to-end ML + deployment pipeline.

---

## 🧱 Project Structure

```text
Sertraline-Activity-Predictor/
├── medeallian data/              # (Likely) raw / processed data files
├── templates/                    # HTML templates for the Flask app
│   └── ...                       # e.g., index.html, result.html, etc.
├── Gradient_Boosting_Model.pkl   # Trained Gradient Boosting model
├── Logistic_Regression_Model.pkl # Trained Logistic Regression model
├── Naive_Bayes_Model.pkl         # Trained Naive Bayes model
├── Random_Forest_Model.pkl       # Trained Random Forest model
├── feature_names.pkl             # Feature name list used during training
├── flask_app.py                  # Flask web application
├── model.ipynb                   # Model training & experimentation notebook
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
````

---

## 🛠 Tech Stack

**Core:**

* Python
* Scikit-learn
* Flask

**Supporting:**

* Pandas, NumPy (data handling)
* Joblib / Pickle (for saving models)
* HTML (Flask templates)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/WHEELYDOS/Sertraline-Activity-Predictor.git
cd Sertraline-Activity-Predictor
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Flask App

Make sure you are in the project root directory and your virtual environment (if any) is activated.

```bash
python flask_app.py
```

The app will typically start at:

```text
http://127.0.0.1:5000/
```

Open this URL in your browser to:

* Fill in the required input fields (features used by the model)
* Submit the form
* View the predicted sertraline activity / class result

---

## 📊 Model Training & Experimentation

All training logic and experiments are stored in:

* `model.ipynb`

Open this notebook in Jupyter:

```bash
jupyter notebook model.ipynb
```

From there you can:

* Explore the dataset (if available)
* Perform preprocessing & feature engineering
* Train multiple models (Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting)
* Evaluate performance using metrics like accuracy, precision, recall, F1-score, etc.
* Export updated models as `.pkl` files to be used by `flask_app.py`

> ❗ If you retrain models, ensure the **feature order** and **preprocessing pipeline** stay consistent, or update `feature_names.pkl` and the Flask app accordingly.

---

## 📥 Input Features

The exact set of features is stored in:

* `feature_names.pkl`

The Flask app expects input fields corresponding to these features. When extending or modifying the app:

* Keep the form inputs in `templates/` aligned with `feature_names.pkl`
* Ensure the same preprocessing steps used during training are applied before prediction

---

## 🔮 Possible Improvements

Some ideas for extending this project:

* Add exploratory data analysis (EDA) visualizations
* Implement robust preprocessing pipelines (e.g., `sklearn.Pipeline`)
* Add model comparison dashboards on the UI
* Support probability outputs and confidence scores
* Deploy the app on a cloud platform (Render, Railway, Heroku, etc.)

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork the repo
2. Create a new branch (`feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

## 📄 License

This project is provided for educational and experimental purposes.
Add a formal license here if you plan to open-source it (e.g., MIT, Apache 2.0).

---

## 👤 Author

**Harsh Vardhan Sahu (WHEELYDOS)**
Feel free to reach out via GitHub for questions or collaboration.

