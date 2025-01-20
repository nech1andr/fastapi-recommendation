

#!/usr/bin/env python
# # coding: utf-8

import pandas as pd
import numpy as np
import requests
import io
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def download_csv(path: str, sep=',', encoding='cp1251') -> pd.DataFrame:
    """Stažení nebo čtení souboru CSV z cesty k souboru"""
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
    """Odstranění whitespace a převod textových sloupců na malá písmena"""
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip()
    return df

def impute_year_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Chybějící nebo neplatný rok publikace je imputován mediánem nebo pevnou hodnotou"""
    # převod na číselné hodnoty, neplatné hodnoty budou nahrazeny NaN
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')

    # výpočet mediánu roku nebo použití aktuálního roku, pokud nejsou k dispozici žádné platné roky
    if df['Year-Of-Publication'].dropna().size > 0:
        median_year = int(df['Year-Of-Publication'].dropna().median())
    else:
        median_year = datetime.now().year

    # nahradit NaN mediánem roku
    df['Year-Of-Publication'] = df['Year-Of-Publication'].fillna(median_year).astype(int)

    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Přidání nových atributů do datové sady"""
    current_year = datetime.now().year

    # nový atribut - věk publikace
    df['Publication-Age'] = current_year - df['Year-Of-Publication']

    # nový atribut - délka názvu
    df['Title-Length'] = df['Book-Title'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

    # převod vybraných sloupců na kategorický datový typ pro optimalizaci paměti
    categorical_cols = ['Book-Author', 'Publisher']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # průměrné hodnocení podle autora a vydavatele
    author_avg_rating = df.groupby('Book-Author', observed=False)['Book-Rating'].transform('mean')
    publisher_avg_rating = df.groupby('Publisher', observed=False)['Book-Rating'].transform('mean')
    df['Author-Avg-Rating'] = author_avg_rating
    df['Publisher-Avg-Rating'] = publisher_avg_rating

    return df

def clean_and_merge_data(ratings: pd.DataFrame, books: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Vyčištění a sloučení dat o hodnoceních, knihách a uživatelích a provedení feature engineeringu"""
    # odstranění nulových hodnocení a duplicit
    ratings = ratings[ratings['Book-Rating'] != 0].drop_duplicates()
    books = books.drop_duplicates()

    # volání funkce clean_text_columns
    books = clean_text_columns(books)
    ratings = clean_text_columns(ratings)
    users = clean_text_columns(users)

    # sloučení datasetů přes ISBN s využitím inner join
    merged = pd.merge(ratings, books, on='ISBN', how='inner')

    # odstranění řádků s chybějícími základními hodnotami
    merged.dropna(subset=['Book-Title', 'Book-Author', 'User-ID', 'Book-Rating'], inplace=True)

    # imputace roku publikace
    merged = impute_year_publication(merged)

    # přidání nových atributů
    merged = add_features(merged)

    # sloučení s informacemi o uživatelích na základě sloupce 'User-ID'
    merged = pd.merge(merged, users, on='User-ID', how='left')

    # Imputace chybějících hodnot ve sloupci Age pomocí mediánu
    if 'Age' in merged.columns:
        # Převedení sloupce Age na numerický typ, pokud ještě není
        merged['Age'] = pd.to_numeric(merged['Age'], errors='coerce')
        median_age = merged['Age'].median()
        merged['Age'] = merged['Age'].fillna(median_age)

    return merged

# Cesty k datasetům
ratings_path = r'./data/Ratings.csv'
users_path = r'./data/Users.csv'
books_path = r'./data/Books.csv'

# Načtení dat
ratings_df = download_csv(ratings_path)
users_df = download_csv(users_path)
books_df = download_csv(books_path)

# Čištění a sloučení dat včetně informací o uživatelích
cleaned_dataset = clean_and_merge_data(ratings_df, books_df, users_df)

# Uložení DataFrame do Parquet pro další využití
cleaned_dataset.to_parquet('cleaned_dataset.parquet', index=False)
logging.info("Data byla úspěšně vyčištěna a uložena.")

# Verifikace, zda byl Parquet soubor načten
df = pd.read_parquet('cleaned_dataset.parquet')
logging.info(f"Loaded dataset with shape: {df.shape}")
