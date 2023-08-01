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

# Function to get min, max, and mean values from the dataset description
def get_dataset_stats(df):
    stats = {}
    stats['CRIM'] = (df['CRIM'].min(), df['CRIM'].max(), df['CRIM'].mean())
    stats['ZN'] = (df['ZN'].min(), df['ZN'].max(), df['ZN'].mean())
    stats['INDUS'] = (df['INDUS'].min(), df['INDUS'].max(), df['INDUS'].mean())
    stats['CHAS'] = (df['CHAS'].min(), df['CHAS'].max(), df['CHAS'].mean())
    stats['NOX'] = (df['NOX'].min(), df['NOX'].max(), df['NOX'].mean())
    stats['RM'] = (df['RM'].min(), df['RM'].max(), df['RM'].mean())
    stats['AGE'] = (df['AGE'].min(), df['AGE'].max(), df['AGE'].mean())
    stats['DIS'] = (df['DIS'].min(), df['DIS'].max(), df['DIS'].mean())
    stats['RAD'] = (df['RAD'].min(), df['RAD'].max(), df['RAD'].mean())
    stats['TAX'] = (df['TAX'].min(), df['TAX'].max(), df['TAX'].mean())
    stats['PTRATIO'] = (df['PTRATIO'].min(), df['PTRATIO'].max(), df['PTRATIO'].mean())
    stats['B'] = (df['B'].min(), df['B'].max(), df['B'].mean())
    stats['LSTAT'] = (df['LSTAT'].min(), df['LSTAT'].max(), df['LSTAT'].mean())
    stats['MEDV'] = (df['MEDV'].min(), df['MEDV'].max(), df['MEDV'].mean())
    return stats

# Function to get slider values from the dataset description stats
def get_slider_values(stat):
    return stat[0], stat[1], stat[2]

# Function to describe the attribute information
def describe_attributes():
    st.write("## Data Set Characteristics")
    st.write("- The Boston Housing dataset contains information about various features of houses in Boston.")
    st.write("- It includes attributes such as per capita crime rate, proportion of residential land zoned for lots over 25,000 sq.ft., average number of rooms per dwelling, etc.")
    st.write("- The target variable is the median value of owner-occupied homes in thousands of dollars.")
    st.write("- The dataset consists of 506 instances and 13 input features.")
    st.write('===================================================================')
    st.write("## Attribute Information")
    st.write("- CRIM: Per capita crime rate by town")
    st.write("- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.")
    st.write("- INDUS: Proportion of non-retail business acres per town")
    st.write("- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    st.write("- NOX: Nitric oxides concentration (parts per 10 million)")
    st.write("- RM: Average number of rooms per dwelling")
    st.write("- AGE: Proportion of owner-occupied units built prior to 1940")
    st.write("- DIS: Weighted distances to five Boston employment centers")
    st.write("- RAD: Index of accessibility to radial highways")
    st.write("- TAX: Full-value property tax rate per $10,000")
    st.write("- PTRATIO: Pupil-teacher ratio by town")
    st.write("- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town")
    st.write("- LSTAT: Percentage of lower status of the population")
    st.write("- MEDV: Median value of owner-occupied homes in $1000s")
    st.write('===================================================================')

# Function to explore the dataset
def explore_data(df):
    describe_attributes()
    st.write("### Dataset Summary")
    st.write(df.head())
    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Dataset Description")
    st.write(df.describe())

    # Data visualization
    st.write("### Data Visualization")
    st.write("#### Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df['RM'], df['MEDV'])
    ax.set_xlabel('RM: Average number of rooms per dwelling')
    ax.set_ylabel('Median value of owner-occupied homes in $1000s')
    st.pyplot(fig)

    st.write("#### Histogram")
    fig, ax = plt.subplots()
    ax.hist(df['MEDV'])
    ax.set_xlabel('Median value of owner-occupied homes in $1000s')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("#### Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("#### Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, orient='h', palette='Set2')
    st.pyplot(fig)

    st.write("#### Pair Plot")
    fig = sns.pairplot(df, diag_kind='kde')
    st.pyplot(fig)

    st.write("#### Bar Plot")
    fig, ax = plt.subplots()
    df['RAD'].value_counts().sort_index().plot(kind='bar')
    ax.set_xlabel('RAD: Index of accessibility to radial highways')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.write("#### KDE Plot")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df['DIS'], shade=True)
    ax.set_xlabel('DIS: Weighted distances to five Boston employment centers')
    ax.set_ylabel('Density')
    st.pyplot(fig)

# Function to save the trained model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to train and evaluate the model
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

def visualize_slider_values(crim, indus, nox, age, rad, ptratio, lstat, zn, chas, rm, dis, tax, b, medv):
    features = ['CRIM', 'INDUS', 'NOX', 'AGE', 'RAD', 'PTRATIO', 'LSTAT', 'ZN', 'CHAS', 'RM', 'DIS', 'TAX', 'B', 'MEDV']
    values = [crim, indus, nox, age, rad, ptratio, lstat, zn, chas, rm, dis, tax, b, medv]

    fig, ax = plt.subplots()
    ax.bar(features, values, color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'teal', 'lime', 'indigo', 'yellow'])
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature Value')
    ax.set_title('Selected Feature Values')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def main():
    st.write("**Upload the dataset file (CSV format)**")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        explore_data(df)
        model_lr = train_model(df)
        model_rf = train_model_random_forest(df)
        
        st.write("### House Price Prediction")

        st.write("**Enter the following features to get the predicted price:**")
        input_col1, input_col2 = st.columns(2)

        stats = get_dataset_stats(df)

        with input_col1:
            st.write("**CRIM**:")
            min_crim, max_crim, mean_crim = get_slider_values(stats['CRIM'])
            crim = st.slider('crim', min_crim, max_crim, mean_crim)

            st.write("**INDUS**:")
            min_indus, max_indus, mean_indus = get_slider_values(stats['INDUS'])
            indus = st.slider('indus', min_indus, max_indus, mean_indus)

            st.write("**NOX**:")
            min_nox, max_nox, mean_nox = get_slider_values(stats['NOX'])
            nox = st.slider('nox', min_nox, max_nox, mean_nox)

            st.write("**AGE**:")
            min_age, max_age, mean_age = get_slider_values(stats['AGE'])
            age = st.slider('age', min_age, max_age, mean_age)

            st.write("**RAD**:")
            min_rad, max_rad, mean_rad = get_slider_values(stats['RAD'])
            rad = st.slider('rad', min_rad, max_rad, mean_rad)

            st.write("**PTRATIO**:")
            min_ptratio, max_ptratio, mean_ptratio = get_slider_values(stats['PTRATIO'])
            ptratio = st.slider('ptratio', min_ptratio, max_ptratio, mean_ptratio)

            st.write("**LSTAT**:")
            min_lstat, max_lstat, mean_lstat = get_slider_values(stats['LSTAT'])
            lstat = st.slider('lstat', min_lstat, max_lstat, mean_lstat)

        with input_col2:
            st.write("**ZN**:")
            min_zn, max_zn, mean_zn = get_slider_values(stats['ZN'])
            zn = st.slider('zn', min_zn, max_zn, mean_zn)

            st.write("**CHAS**:")
            min_chas, max_chas, mean_chas = get_slider_values(stats['CHAS'])
            chas = st.slider('chas', min_chas, max_chas, mean_chas)

            st.write("**RM**:")
            min_rm, max_rm, mean_rm = get_slider_values(stats['RM'])
            rm = st.slider('rm', min_rm, max_rm, mean_rm)

            st.write("**DIS**:")
            min_dis, max_dis, mean_dis = get_slider_values(stats['DIS'])
            dis = st.slider('dis', min_dis, max_dis, mean_dis)

            st.write("**TAX**:")
            min_tax, max_tax, mean_tax = get_slider_values(stats['TAX'])
            tax = st.slider('tax', min_tax, max_tax, mean_tax)

                        st.write("**B**:")
            min_b, max_b, mean_b = get_slider_values(stats['B'])
            b = st.slider('b', min_b, max_b, mean_b)

            st.write("**MEDV**:")
            min_medv, max_medv, mean_medv = get_slider_values(stats['MEDV'])
            medv = st.slider('medv', min_medv, max_medv, mean_medv)

        visualize_slider_values(crim, indus, nox, age, rad, ptratio, lstat, zn, chas, rm, dis, tax, b, medv)

        submitted = st.button('Predict Price')

        if submitted:
            input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv]])
            prediction_lr = predict_price_linear_regression(model_lr, input_data)
            st.write("### **Predicted House Price using Linear Regression:**", prediction_lr)

            prediction_rf = predict_price_random_forest(model_rf, input_data)
            st.write("### **Predicted House Price using Random Forest:**", prediction_rf)

            visualize_prediction_pie(prediction_lr, prediction_rf)

if __name__ == "__main__":
    main()
