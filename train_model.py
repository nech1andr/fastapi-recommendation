# train_model.py
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV, cross_validate
import logging

logging.basicConfig(level=logging.INFO)

def load_and_prepare_data(filepath, rating_scale):
    """nacteni a priprava dat pro trenovani"""
    df = pd.read_parquet(filepath)
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[['User-ID', 'Book-Title', 'Book-Rating']], reader)
    return data

def optimize_model(data, param_grid):
    """provedeni optimalizaci hyperparametru pomoci gridsearchcv"""
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
    gs.fit(data)
    best_params = gs.best_params['rmse']
    logging.info(f"Nejlepsi parametry: {best_params}")
    return gs.best_estimator['rmse'], best_params

def evaluate_model(model, data):
    """evaluace modelu pomoci krizove validace"""
    cv_results = cross_validate(model, data, measures=['rmse', 'mae'], cv=5, verbose=True)
    logging.info(f"Křížová validace výsledky: {cv_results}")
    return cv_results

def train_and_save_model(model, trainset, output_path):
    """trenovani modelu a ukladani do souboru"""
    model.fit(trainset)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("SVD model trénován a uložen.")


data = load_and_prepare_data('cleaned_dataset.parquet', rating_scale=(1, 10))
trainset, testset = train_test_split(data, test_size=0.2)

param_grid = {
    'n_factors': [50, 100, 150],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.05, 0.1]
}

optimized_model, best_params = optimize_model(data, param_grid)
evaluate_model(optimized_model, data)

train_and_save_model(optimized_model, trainset, 'svd_model.pkl')