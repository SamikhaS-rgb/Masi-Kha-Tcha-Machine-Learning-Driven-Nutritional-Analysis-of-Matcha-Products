import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import statsmodels.api as sm
from scipy.stats import ttest_ind, sem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('/Users/samikhasrinivasan/Downloads/Matcha_nutritional_profile.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Inspect the data
print(df.head())
print(df.info())

# Handle missing values if any
df.dropna(inplace=True)

# Add Total_Calories using available data (protein only for now)
df['Total_Calories'] = df['Protein'] * 4

# Encode categorical variable
df['Category_Code'] = df['Category'].astype('category').cat.codes

# Visualizations
sns.pairplot(df[['Fiber_g', 'Antoxidant_Level_mg', 'Protein', 'Total_Calories']])
plt.suptitle("Pairplot of Nutritional Components", y=1.02)
plt.show()

px.scatter(df, x='Protein', y='Antoxidant_Level_mg', color='Category', title='Protein vs Antioxidants by Category').show()

# Train-test split
X = df[['Fiber_g', 'Antoxidant_Level_mg', 'Protein', 'Category_Code']]
y = df['Total_Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# SHAP for interpretability
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Statistical test: Compare protein between categories
culinary = df[df['Category'] == 'Culinary']['Protein']
standard = df[df['Category'] == 'Standard']['Protein']
t_stat, p_value = ttest_ind(culinary, standard, equal_var=False)

print(f"T-test for Protein: t-stat = {t_stat:.3f}, p = {p_value:.3f}")
