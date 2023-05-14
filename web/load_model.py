import pickle
import pandas as pd
import catboost as cb
import numpy as np





def predict_individual(id):
    with open("./models/model_catboost.pkl", "rb") as file:
        model = pickle.load(file)

    X = pd.read_csv('../Datasets/X.csv')

    year = int(id[0:4])
    nweek = int(id[4:6])
    product_number = int(id.split('-')[1])
    a = X[X.product_number==product_number].iloc[0]
    a.nweek, a.year = nweek, year
    prediction = model.predict(a)
    return prediction

def predict_csv(X_pred):
    with open("./models/model_catboost.pkl", "rb") as file:
        model = pickle.load(file)

    X = pd.read_csv('../Datasets/X.csv')

    X_pred["year"] = X_pred["id"].apply(lambda x: x[:4]).astype(np.int64)
    X_pred["nweek"] = X_pred["id"].apply(lambda x: x[4:6]).astype(np.int64)
    X_pred["product_number"] = X_pred["id"].apply(lambda x: x.split('-')[1]).astype(np.int64)

    df_merge = X.drop(columns = ["nweek", "year"])
    df_merge = df_merge.drop_duplicates()

    X_pred = X_pred.reset_index().merge(df_merge,on=['product_number']).set_index('index')

    X_pred = X_pred.sort_index()
    X_pred = X_pred[X.columns]
    # X_pred = X_pred.drop(columns = ["id"])

    prediction = model.predict(X_pred)

    return prediction

    

