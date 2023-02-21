"""Main module."""

import warnings
import argparse
import os
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Trains model when data is pushed in processed folder")
    parser.add_argument('--data', type=str, required=True, help="Path to the csv file added in data/processed folder")

    return parser.parse_args()


def preprocess_data(data):
    data = pd.read_csv(data)
    y = data['SalePrice']
    X = data.drop(columns="SalePrice")

    return X, y


def train_model(X, y, test_size):
    # Declare the type of each column in the dataset(example: category, numeric, text)
    column_types = {'TypeOfDewelling': 'category',
                    'BldgType': 'category',
                    'AbvGroundLivingArea': 'numeric',
                    'Neighborhood': 'category',
                    'KitchenQuality': 'category',
                    'NumGarageCars': 'numeric',
                    'YearBuilt': 'numeric',
                    'RemodelYear': 'numeric',
                    'ExternalQuality': 'category',
                    'LotArea': 'numeric',
                    'LotShape': 'category',
                    'Fireplaces': 'numeric',
                    'NumBathroom': 'numeric',
                    'Basement1Type': 'category',
                    'Basement1SurfaceArea': 'numeric',
                    'Basement2Type': 'category',
                    'Basement2SurfaceArea': 'numeric',
                    'TotalBasementArea': 'numeric',
                    'GarageArea': 'numeric',
                    '1stFlrArea': 'numeric',
                    '2ndFlrArea': 'numeric',
                    'Utilities': 'category',
                    'OverallQuality': 'category',
                    'SalePrice': 'numeric'
                    }

    # feature_types is used to declare the features the model is trained on
    feature_types = {i: column_types[i] for i in column_types if i != 'SalePrice'}

    # Pipeline to fill missing values, transform and scale the numeric columns
    numeric_features = [key for key in feature_types.keys() if feature_types[key] == "numeric"]
    numeric_transformer = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                    ('scaler', StandardScaler())])

    # Pipeline to fill missing values and one hot encode the categorical values
    categorical_features = [key for key in feature_types.keys() if feature_types[key] == "category"]
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    # Initiate Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Pipeline for the Random Forest Model
    reg_random_forest = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', RandomForestRegressor())])

    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=30)

    # Fit and score your model
    reg_random_forest.fit(X_train, y_train)
    print("model score: %.3f" % reg_random_forest.score(X_test, y_test))

    # Prepare data to upload on Giskard
    test_data = pd.concat([X_test, y_test], axis=1)

    return reg_random_forest, test_data


if __name__ == "__main__":
    time = datetime.now()
    dt_string = time.strftime("%d%m%Y%H%M%S")
    args = parse_args()
    X, y = preprocess_data(args.data)
    model, test_data = train_model(X, y, 0.2)

    # Dumping
    dname = os.path.dirname(os.path.abspath(__file__))
    dname = dname.split("src", 1)[0]
    trained_model_path = os.path.join(dname, 'models/trained')
    data_path = os.path.join(dname, 'data/achived')
    if not os.path.isdir(trained_model_path):
        os.system('mkdir ' + trained_model_path)
    if not os.path.isdir(data_path):
        os.system('mkdir ' + data_path)

    model_filename = trained_model_path + '/house_pricing_regression_model_' + dt_string + '.pkl'
    test_data_filename = data_path + '/test_data_' + dt_string + '.zip'
    pickle.dump(model, open(model_filename, 'wb'))
    test_data.to_pickle(test_data_filename, compression='zip')
