import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

def preprocess_data(df):
    try:
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])

        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        logging.info("Data preprocessing completed.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        return None, None, None, None
