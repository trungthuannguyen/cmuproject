import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Header with logo
logo_path = "housing_prediction/team3vn_cmu.jpg"
# Center the logo on the page
col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
with col1:
    st.write("")
with col2:
    st.write("")
with col3:
    st.write("")
with col4:
    st.image(logo_path, width=220, caption="")
with col5:
    st.write("")
with col6:
    st.write("")
with col7:
    st.write("")
with col8:
    st.write("")
with col9:
    st.write("")

# Header
st.markdown(
    "<h1 style='text-align: center; font-size: 30px; background-color: red; color: #FFFFFF'; margin: 20px;padding: 20px;>"
    "&nbsp; Welcome to Team3VN-CMU House Price Prediction &nbsp;"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 30px; background-color: #f2f2f2; line-height: 0.3;'>Predict House Prices</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center; font-size: 26px; background-color: #f8f8f8; color: blue; line-height: 0.3;'>"
    "Member: Dieu - Man - Sanh - Thuan - Tinh - Trinh"
    "</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; font-size: 26px; background-color: #f8f8f8; color: gray; line-height: 0.3;'>"
    "CMU 2023"
    "</h3>",
    unsafe_allow_html=True
)

# Function to load the dataset
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to train and evaluate the Linear Regression model
def train_model(df):
    st.write("### Model Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("#### Model Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model, "housing_prediction/LinearRegression.pkl")
    return model

# Function to train and evaluate the Random Forest model
def train_model_random_forest(df):
    st.write("### Model Random Forest Training and Evaluation")

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)

    st.write("#### Model Random Forest Performance")
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))
    save_model(model_rf, "housing_prediction/RandomForest.pkl")
    return model_rf

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to predict house prices using Linear Regression
def predict_price_linear_regression(model, input_data):
    # Ensure input_data has the same number of features as the training dataset
    if input_data.shape[1] != model.coef_.shape[0]:
        raise ValueError("Number of features in input data does not match the model")

    prediction = model.predict(input_data)
    return prediction

# Function to predict house prices using Random Forest
def predict_price_random_forest(model_rf, input_data):
    prediction_rf = model_rf.predict(input_data)
    return prediction_rf

# Function to visualize the predicted prices using a pie chart
def visualize_prediction_pie(prediction_lr, prediction_rf):
    labels = ['Linear Regression', 'Random Forest']
    # Ensure that the predictions are non-negative (you can set negative values to 0)
    prediction_lr = np.maximum(prediction_lr, 0)
    prediction_rf = np.maximum(prediction_rf, 0)
    sizes = [prediction_lr[0], prediction_rf[0]]
    explode = (0.1, 0.0)  # explode the first slice

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Result')
    plt.show()
    st.pyplot(fig)

def main():
    st.write("**Upload the dataset file (CSV format)**")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("### Dataset Summary")
        st.write(df.head())
        st.write("### Dataset Shape")
        st.write(df.shape)
        st.write("### Dataset Description")
        st.write(df.describe())

        model_lr = train_model(df)
        model_rf = train_model_random_forest(df)

        st.write("### House Price Prediction")

        st.write("**Enter the following features to get the predicted price:**")
        input_col1, input_col2 = st.columns(2)

        with input_col1:
            # Add sliders for input features
            st.write("**CRIM**:")
            crim = st.slider('CRIM', df['CRIM'].min(), df['CRIM'].max(), df['CRIM'].mean())
            st.write("**ZN**:")
            zn = st.slider('ZN', df['ZN'].min(), df['ZN'].max(), df['ZN'].mean())
            st.write("**INDUS**:")
            indus = st.slider('INDUS', df['INDUS'].min(), df['INDUS'].max(), df['INDUS'].mean())
            st.write("**CHAS**:")
            chas = st.slider('CHAS', df['CHAS'].min(), df['CHAS'].max(), df['CHAS'].mean())
            st.write("**NOX**:")
            nox = st.slider('NOX', df['NOX'].min(), df['NOX'].max(), df['NOX'].mean())
            st.write("**RM**:")
            rm = st.slider('RM', df['RM'].min(), df['RM'].max(), df['RM'].mean())
            st.write("**AGE**:")
            age = st.slider('AGE', df['AGE'].min(), df['AGE'].max(), df['AGE'].mean())
            st.write("**DIS**:")
            dis = st.slider('DIS', df['DIS'].min(), df['DIS'].max(), df['DIS'].mean())

        with input_col2:
            # Add sliders for the remaining input features
            st.write("**RAD**:")
            rad = st.slider('RAD', df['RAD'].min(), df['RAD'].max(), df['RAD'].mean())
            st.write("**TAX**:")
            tax = st.slider('TAX', df['TAX'].min(), df['TAX'].max(), df['TAX'].mean())
            st.write("**PTRATIO**:")
            ptratio = st.slider('PTRATIO', df['PTRATIO'].min(), df['PTRATIO'].max(), df['PTRATIO'].mean())
            st.write("**B**:")
            b = st.slider('B', df['B'].min(), df['B'].max(), df['B'].mean())
            st.write("**LSTAT**:")
            lstat = st.slider('LSTAT', df['LSTAT'].min(), df['LSTAT'].max(), df['LSTAT'].mean())
            st.write("**MEDV**:")
            medv = st.slider('MEDV', df['MEDV'].min(), df['MEDV'].max(), df['MEDV'].mean())

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'CRIM': [crim],
            'ZN': [zn],
            'INDUS': [indus],
            'CHAS': [chas],
            'NOX': [nox],
            'RM': [rm],
            'AGE': [age],
            'DIS': [dis],
            'RAD': [rad],
            'TAX': [tax],
            'PTRATIO': [ptratio],
            'B': [b],
            'LSTAT': [lstat],
            'MEDV': [medv]
        })

        st.write("### Selected Feature Values")
        st.write(input_data)

        # Make predictions using both Linear Regression and Random Forest models
        prediction_lr = predict_price_linear_regression(model_lr, input_data)
        prediction_rf = predict_price_random_forest(model_rf, input_data)

        st.write("### Predicted House Price")
        st.write("Using Linear Regression:", prediction_lr)
        st.write("Using Random Forest:", prediction_rf)

        # Visualize the predicted prices using a pie chart
        visualize_prediction_pie(prediction_lr, prediction_rf)

if __name__ == "__main__":
    main()
