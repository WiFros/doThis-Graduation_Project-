# Import modules
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load pipeline.pickle
with open('models/pipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)


# We use this function for prediction in app.py
def predict_pipeline(data):
    return predict(loaded_pipe, data)


# Function : preprocessing for input data
def preprocess_input(data):

    # Make DataFrame
    data = pd.DataFrame(data,
                        columns=['Gender', 'Age', 'Location', 'Financial Condition', 'Internet Type', 'Network Type', 'Class Duration'],)

    # Label Encoding
    df_clean = pd.read_csv("data/clean.csv")
    df_clean_copy = df_clean.copy()

    # Add user data into dataset
    X_df = df_clean_copy.drop(
        columns=['Education Level', 'IT Student', 'Device', 'Adaptivity Level'])
    df = X_df.to_numpy()
    df = np.append(df, data, axis=0)

    # Normalization (dataset & user)
    le = LabelEncoder()
    org_df = pd.DataFrame(df, columns=[
                          'Gender', 'Age', 'Location', 'Financial Condition', 'Internet Type', 'Network Type', 'Class Duration'])
    for feature in ['Gender', 'Age', 'Location', 'Financial Condition', 'Internet Type', 'Network Type', 'Class Duration']:
        org_df[feature] = le.fit_transform(org_df[feature].values)
    scaler = StandardScaler()
    df = scaler.fit_transform(org_df)

    # Check user input
    user = df[-1]

    return user


# Function : preprocessing for exist data
def predict(model, data):
    # Make list into ndarray
    data = str(data)
    data = [data.split(',')]

    # Preprocessing
    preprocessed_df = preprocess_input(data)
    predictions = model.predict(preprocessed_df.reshape(1, -1))

    pred_to_label = {0: 'High', 1: 'Low', 2: 'Moderate'}

    # Make a list of result
    result = []
    for pred in predictions:
        result.append({'pred:': int(pred), 'label': pred_to_label[pred]})

    return result


# Test
if __name__ == "__main__":
    # test_1
    data = "Boy,16-20,No,Mid,Wifi,3G,03-06"
    predictions = predict_pipeline(data)
    print(predictions)
    # test_2
    data = "Boy,11-15,No,Mid,Wifi,4G,03-06"
    predictions = predict_pipeline(data)
    print(predictions)
    # test_3
    data = "Boy,11-15,Yes,Rich,Mobile Data,4G,03-06"
    predictions = predict_pipeline(data)
    print(predictions)
