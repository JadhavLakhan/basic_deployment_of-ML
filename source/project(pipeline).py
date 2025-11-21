import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def data_ingestion(path):
    df = pd.read_csv(path)
    print('Data Loaded Successfully!')
    return df

def perform_eda(df):
    print('--- DATA SUMMARY ---')
    print(df.head())
    print('\nInformation about data:')
    print(df.info())
    print('\nMissing values:\n', df.isnull().sum())
    print('\nStatistics:\n', df.describe())

def feature_engineering(df):
    for i in df.columns:
        df[i] = df[i].astype(float)
    print("Feature Engineering Done!")
    return df

def feature_selection(df):
    X = df.drop(columns='Salary')
    y = df['Salary']
    return X, y

def model_training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    print('Model trained and saved.')

    return model, X_test, y_test

def model_evaluation(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- MODEL EVALUATION ---")
    print("MSE:", mse)
    print("R2 Score:", r2)

if __name__ == '__main__':
    df = data_ingestion(r'D:\end to end python\End to end CICD pipeline\data\data.csv')
    perform_eda(df)
    df = feature_engineering(df)
    X, y = feature_selection(df)
    model, X_test, y_test = model_training(X, y)
    model_evaluation(model, X_test, y_test)
