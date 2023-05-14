import pickle
import pandas as pd
import catboost as cb





def predict_individual(id):
    with open("HP\web\models\model_catboost.pkl", "rb") as file:
        model = pickle.load(file)

    X = pd.read_csv('HP/Datasets/X.csv')

    year = int(id[0:4])
    nweek = int(id[4:6])
    product_number = int(id.split('-')[1])
    a = X[X.product_number==product_number].iloc[0]
    a.nweek, a.year = nweek, year
    prediction = model.predict(a)
    return prediction


    

