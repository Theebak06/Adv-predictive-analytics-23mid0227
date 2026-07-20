House Price Prediction 

A project that predicts house prices using different regression algorithms and compares their performance. The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and evaluation.

Project Overview

The objective of this project is to build accurate house price prediction models using real estate data. Multiple regression techniques are implemented and compared to determine the best-performing model.

Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature scaling using StandardScaler
- Train-Test Split
- Multiple regression model implementation
- Hyperparameter tuning using GridSearchCV
- Model evaluation using standard regression metrics
- Performance comparison of different models

Machine Learning Models Used

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression (Degree 2)
- Random Forest Regressor (Hyperparameter Tuned)

Evaluation Metrics
The models are evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Adjusted R² Score

 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Dataset

The project uses the **Real Estate Valuation Dataset** containing features such as:

- Transaction Date
- House Age
- Distance to Nearest MRT Station
- Number of Convenience Stores
- Latitude
- Longitude
- House Price per Unit Area (Target Variable)

Workflow
1. Load the dataset
2. Perform Exploratory Data Analysis (EDA)
3. Preprocess the data
4. Split into training and testing datasets
5. Train multiple regression models
6. Perform hyperparameter tuning using GridSearchCV
7. Evaluate model performance
8. Compare results and identify the best model

Outputs
The project generates:
- Target variable distribution plot
- Correlation heatmap
- Model performance comparison table
- Best model selection based on evaluation metrics

Project Structure
House-Price-Prediction/
│
├── 23MID0227_LAB01_HOUSEPRICE.html
├── dataset_a_eda_plots.png
├── README.md
└── dataset/
```
## 🚀 How to Run
1. Clone the repository
```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```
2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
```
3. Update the dataset path in the code if required.
4. Run the notebook or Python script.

Results

Among the implemented models, the Tuned Random Forest Regressor achieved the best prediction performance with the lowest prediction error and highest R² score.

Future Improvements
- XGBoost Regressor
- LightGBM
- CatBoost
- Model deployment using Flask or Streamlit
- Feature engineering
- Cross-validation on larger datasets



  *****House Price Prediction using the Ames Housing Dataset*****

A  project that predicts residential house prices using the "Ames Housing Dataset". The project focuses on data preprocessing, exploratory data analysis (EDA), feature engineering, regression model development, hyperparameter tuning, and model evaluation.

Project Overview
This project aims to predict house sale prices using the **Ames Housing Dataset**, which contains detailed information about residential properties. Multiple regression models are implemented and compared to identify the best-performing algorithm for price prediction.
Features
- Data preprocessing and cleaning
- Handling missing values
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Correlation analysis
- Multiple regression models
- Hyperparameter tuning using GridSearchCV
- Model performance comparison
- House price prediction

---
Dataset
This project uses the **Ames Housing Dataset**, a popular dataset for regression problems containing detailed information on residential homes in Ames, Iowa.

Dataset Information

- Number of Records: 1460
-Features: 80 explanatory variables
- Target Variable: SalePrice

Some important features include:

- Overall Quality
- Overall Condition
- Year Built
- Year Remodeled
- Total Basement Area
- First Floor Area
- Garage Area
- Garage Cars
- GrLivArea (Above Ground Living Area)
- Full Bathrooms
- Lot Area
- Neighborhood

---

Machine Learning Models
The following regression algorithms were implemented:
- Linear Regression
- Polynomial Regression
- Random Forest Regressor
- Tuned Random Forest Regressor (GridSearchCV)


Evaluation Metrics

The models are evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

Project Workflow
1. Load the Ames Housing Dataset
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection & Engineering
5. Train-Test Split
6. Train Regression Models
7. Hyperparameter Tuning using GridSearchCV
8. Evaluate Models
9. Compare Results
10. Predict House Prices
Project Structure

House-Price-Prediction/
│── 23MID0227_LAB01_HOUSEPRICE.html
│── README.md
│── dataset/
│    └── AmesHousing.csv
│── images/
│    ├── correlation_heatmap.png
│    └── price_distribution.png
Installation

Clone the repository:

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the notebook or HTML-exported notebook to reproduce the analysis.

---

Results
The models were compared based on regression performance metrics. After hyperparameter tuning, the **Random Forest Regressor** achieved the best predictive performance, demonstrating its ability to capture the complex relationships within the Ames Housing Dataset.
Future Improvements
- Implement XGBoost and LightGBM
- Add CatBoost Regressor
- Deploy the model using Streamlit or Flask
- Perform advanced feature engineering
- Use cross-validation for improved model robustness

