from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run, Dataset
from azureml.core.workspace import Workspace


run = Run.get_context()
ws = run.experiment.workspace
found = False
key = "Credit-Card-Churners"
description_text = "Credit Card Churners DataSet for Udacity Capstone"

if key in ws.datasets.keys():
        found = True
        dataset = ws.datasets[key]

def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()

    # Drop last two columns (unneeded)
    x_df.drop(x_df.columns[-2:],inplace=True, axis=1)

    # Drop CLIENTNUM columns
    x_df.drop("CLIENTNUM",inplace=True, axis=1)

    # Encode unknown values as np.NaN
    x_df = x_df.replace('Unknown', np.NaN)

    # Fill ordinal missing values with modes (Education_Level and Income_Category columns)
    x_df['Education_Level'] = x_df['Education_Level'].fillna('Graduate')
    x_df['Income_Category'] = x_df['Income_Category'].fillna('Less than $40K')

    # Encode binary columns
    x_df = binary_encode(x_df, 'Attrition_Flag', positive_value='Attrited Customer')
    x_df = binary_encode(x_df, 'Gender', positive_value='M')

    # Encode ordinal columns
    education_ordering = [
        'Uneducated',
        'High School',
        'College',
        'Graduate',
        'Post-Graduate',
        'Doctorate'
    ]
    income_ordering = [
        'Less than $40K',
        '$40K - $60K',
        '$60K - $80K',
        '$80K - $120K',
        '$120K +'
    ]

    x_df = ordinal_encode(x_df, 'Education_Level', ordering=education_ordering)
    x_df = ordinal_encode(x_df, 'Income_Category', ordering=income_ordering)

    # Encode nominal columns
    x_df = onehot_encode(x_df, 'Marital_Status', prefix='Marital_Status')
    x_df = onehot_encode(x_df, 'Card_Category', prefix='Card_Category')

    # Split df into X and y
    X = x_df.drop('Attrition_Flag', axis=1).copy()
    y = x_df['Attrition_Flag'].copy()

    # Scale X with a standard scaler
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="The minimum number of samples required to be at a leaf node.")

    args = parser.parse_args()

    if args.max_depth == 0:
        max_depth = None
    else:
        max_depth = args.max_depth

    run.log("Num Estimators:", np.float(args.n_estimators))
    run.log("Max Depth:", max_depth)
    run.log("Min Samples Split:", np.int(args.min_samples_split))
    run.log("Min Samples Leaf:", np.int(args.min_samples_leaf))

    x, y = clean_data(dataset)

    # TODO: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=0)

    # Train Random Forest Model
    model = RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)

    # calculate accuracy
    Accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(Accuracy))

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.joblib')

if __name__ == '__main__':
    main()
