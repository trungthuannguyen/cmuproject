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
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Add MinMaxScaler
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
def load_data(file_path):
    return pd.read_csv(file_path)

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
    st.write("- PTRATION: Pupil-teacher ratio by town")
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
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 6})
    st.pyplot(fig)

    st.write("#### Box Plot")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, orient='h', palette='Set2')
    st.pyplot(fig)

    # Create a copy of the DataFrame before plotting the pairplot
    df_copy = df.copy()

    st.write("#### Pair Plot")
    # Pass the original DataFrame to the pairplot function
    fig = sns.pairplot(df, diag_kind='kde')
    st.pyplot(fig)

    st.write("#### KDE Plot")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df['DIS'], shade=True)
    ax.set_xlabel('DIS: Weighted distances to five Boston employment centers')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    # Bar Plot for the 'RAD' feature
    st.write("#### Bar Plot")
    fig, ax = plt.subplots()
    df['RAD'].value_counts().sort_index().plot(kind='bar')
    ax.set_xlabel('RAD: Index of accessibility to radial highways')
    ax.set_ylabel('Count')
    # Add value above each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points', fontsize=8, fontweight='bold')
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
    if len(input_data) != model.coef_.shape[0]:
        raise ValueError("Number of features in input data does not match the model")

    prediction = model.predict([input_data])
    return prediction

# Function to predict house prices using Random Forest
def predict_price_random_forest(model_rf, input_data):
    prediction_rf = model_rf.predict(input_data)
    return prediction_rf
    
# Function to visualize the predicted prices using a pie chart
def visualize_prediction_pie(prediction_lr, prediction_rf):
    labels = ['Linear Regression', 'Random Forest']
    sizes = [prediction_lr, prediction_rf]
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

# Function to visualize the selected feature values using a bar chart
def visualize_slider_values(df, values):
    # List of features to be visualized
    features = ['LSTAT', 'INDUS', 'NOX', 'RM', 'TAX', 'DIS', 'AGE']

    # Include 'PTRATION' in the features list if it exists in the DataFrame
    if 'PTRATION' in df.columns:
        features.append('PTRATION')

    # Scale the selected features using MinMaxScaler from sklearn.preprocessing
    min_max_scaler = MinMaxScaler()
    x = df.loc[:, features]
    y = df['MEDV']
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=features)

    # Create subplots for each feature and plot regression lines
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        sns.regplot(y=y, x=x[feature], ax=axs[i])  # Scatter plot with regression line for each feature
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    # Create a bold and distinct title for the content
    st.markdown("### Relationship between Selected Features and House Prices")

    # Display the plot using Streamlit's st.pyplot()
    st.pyplot(fig)

# ##########
# Function to process and display the most promising types
def process_prediction_results(df, prediction_lr, prediction_rf):
    # Create a new DataFrame to store the predictions
    prediction_df = df.copy()

    # Repeat the scalar predictions to match the length of the DataFrame
    prediction_lr_array = np.repeat(prediction_lr, len(df))
    prediction_rf_array = np.repeat(prediction_rf, len(df))

    # Add the predictions to the DataFrame
    prediction_df['Predicted Price (Linear Regression)'] = prediction_lr_array
    prediction_df['Predicted Price (Random Forest)'] = prediction_rf_array

    # Sort the DataFrame by the predicted prices in descending order
    prediction_df = prediction_df.sort_values(by='Predicted Price (Linear Regression)', ascending=False)

    # Display the most promising types
    st.write("### Most Promising Types")
    st.write(prediction_df.head())

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

        with input_col1:
            st.write("**CRIM**:")
            crim = st.slider('crim', df['CRIM'].min(), df['CRIM'].max(), df['CRIM'].mean())
            st.write("**INDUS**:")
            indus = st.slider('indus', df['INDUS'].min(), df['INDUS'].max(), df['INDUS'].mean())
            st.write("**NOX**:")
            nox = st.slider('nox', df['NOX'].min(), df['NOX'].max(), df['NOX'].mean())
            st.write("**AGE**:")
            age = st.slider('age', df['AGE'].min(), df['AGE'].max(), df['AGE'].mean())
            st.write("**RAD**:")
            rad = st.slider('rad', float(df['RAD'].min()), float(df['RAD'].max()), float(df['RAD'].mean()))
            st.write("**LSTAT**:")
            lstat = st.slider('lstat', df['LSTAT'].min(), df['LSTAT'].max(), df['LSTAT'].mean())
            st.write("**ZN**:")
            zn = st.slider('zn', df['ZN'].min(), df['ZN'].max(), df['ZN'].mean())

        # Include 'PTRATION' here to match the 14 features used for training the Random Forest model
        with input_col2:
            st.write("**CHAS**:")
            chas = st.slider('chas', df['CHAS'].min(), df['CHAS'].max(), df['CHAS'].mean())
            st.write("**RM**:")
            rm = st.slider('rm', df['RM'].min(), df['RM'].max(), df['RM'].mean())
            st.write("**DIS**:")
            dis = st.slider('dis', df['DIS'].min(), df['DIS'].max(), df['DIS'].mean())
            st.write("**TAX**:")
            tax = st.slider('tax', df['TAX'].min(), df['TAX'].max(), df['TAX'].mean())
            st.write("**PTRATION**:")
            ptration = st.slider('ptration', df['PTRATION'].min(), df['PTRATION'].max(), df['PTRATION'].mean())
            st.write("**B_1000**:")
            b_1000 = st.slider('b_1000', df['B_1000'].min(), df['B_1000'].max(), df['B_1000'].mean())
            st.write("**MEDV**:")
            medv = st.slider('medv', df['MEDV'].min(), df['MEDV'].max(), df['MEDV'].mean())

        values = [crim, indus, nox, age, rad, ptration, lstat, zn, chas, rm, dis, tax, b_1000, medv]
        visualize_slider_values(df, values)

        submitted = st.button('Predict Price')
        
        # Check if the 'Predict Price' button is clicked or not
        if 'submitted' not in locals():
            submitted = False
    
        if submitted:
            # Perform the prediction using the linear regression model
            prediction_lr = predict_price_linear_regression(model_lr, values)
            st.write("### **Predicted House Price using Linear Regression:**", prediction_lr)
    
            # Reshape the values list to a 2D array before passing it to the random forest model
            values_2d = np.array([values])
    
            # Perform the prediction using the random forest model
            prediction_rf = predict_price_random_forest(model_rf, values_2d)
            st.write("### **Predicted House Price using Random Forest:**", prediction_rf)
    
            # Convert scalar predictions to arrays
            prediction_lr = np.array(prediction_lr)
            prediction_rf = np.array(prediction_rf)
    
            # Visualize the predicted prices using a pie chart
            visualize_prediction_pie(prediction_lr, prediction_rf)
    
            # Process and display the most promising types
            process_prediction_results(df, prediction_lr, prediction_rf)
            
if __name__ == "__main__":
    main()
