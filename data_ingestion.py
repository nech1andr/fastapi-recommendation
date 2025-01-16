#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import io
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def download_csv(path: str, sep=',', encoding='cp1251') -> pd.DataFrame:
    """stahovani nebo cteni souboru csv z cesty k souboru"""
    try:
        return pd.read_csv(
            path, 
            sep=sep, 
            encoding=encoding, 
            dtype={'Book-Title': str},
            low_memory=False
        )
    except Exception as e:
        logging.error(f"Failed to read CSV from {path}: {e}")
        raise

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """odstraneni whitespaces a prevod sloupcu textu na mala pismena"""
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip()
    return df

def impute_year_publication(df: pd.DataFrame) -> pd.DataFrame:
    """chybejici nebo neplatny rok publikovani byl imputovan medianem nebo pevnou hodnotou"""
    # prevod na ciselne hodnoty, neplatne hodnoty budou nahrazeny hodnotou NaN
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')

    # vypocet median roku nebo pouzit aktualni rok, pokud nejsou k dispozici zadne platne roky
    if df['Year-Of-Publication'].dropna().size > 0:
        median_year = int(df['Year-Of-Publication'].dropna().median())
    else:
        median_year = datetime.now().year

    # nahradit nan medianem roku
    df['Year-Of-Publication'] = df['Year-Of-Publication'].fillna(median_year).astype(int)

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """pridani novych atributu do datove sady"""
    current_year = datetime.now().year

    # novy atribut - rok publikace
    df['Publication-Age'] = current_year - df['Year-Of-Publication']

    # novy atribut - delka nazvu
    df['Title-Length'] = df['Book-Title'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    # prevod vybranych sloupcu na kategorialni datovy typ pro optimalizaci pameti
    categorical_cols = ['Book-Author', 'Publisher']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # prumerne hodnoceni podle autora a vydavatele
    author_avg_rating = df.groupby('Book-Author', observed=False)['Book-Rating'].transform('mean')
    publisher_avg_rating = df.groupby('Publisher', observed=False)['Book-Rating'].transform('mean')
    df['Author-Avg-Rating'] = author_avg_rating
    df['Publisher-Avg-Rating'] = publisher_avg_rating

    return df

def clean_and_merge_data(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """vycisteni a slouceni udaju o hodnoceni a knihach a pote provedeni feature engineeringu"""
    # odstranei nulovych hodnoceni a duplicit
    ratings = ratings[ratings['Book-Rating'] != 0].drop_duplicates()
    books = books.drop_duplicates()

    # volani f-ci clean_text_columns
    books = clean_text_columns(books)
    ratings = clean_text_columns(ratings)

    # sjednoceni datasetu pres ISBN s vyuzitim inner join
    merged = pd.merge(ratings, books, on='ISBN', how='inner')

    # odstraneni radku s chybejicimi zakladnimi hodnotami
    merged.dropna(subset=['Book-Title', 'Book-Author', 'User-ID', 'Book-Rating'], inplace=True)

    # volani f-ce impute_year_publication
    merged = impute_year_publication(merged)

    # timto volanim fce pridame nove atributy
    merged = add_features(merged)

    return merged

# cesta do datasetu
ratings_path = r'./data/Ratings.csv'
users_path = r'./data/Users.csv'
books_path = r'./data/Books.csv'

# nacteni dat
ratings_df = download_csv(ratings_path)
users_df = download_csv(users_path)
books_df = download_csv(books_path)

# cisteni a sjednoceni dat
cleaned_dataset = clean_and_merge_data(ratings_df, books_df)

# ulozeni df do parquet pro dalsi vyuziti
cleaned_dataset.to_parquet('cleaned_dataset.parquet', index=False)
logging.info("Data byla úspěšně vyčištěna a uložena.")

# verifikace zda parquet soubor byl nacten
df = pd.read_parquet('cleaned_dataset.parquet')
logging.info(f"Loaded dataset with shape: {df.shape}")


