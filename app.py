from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import pickle
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# namontovani statickzch souboru na cestu /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# zde zacina endpoint pro korenovou URL, ktery vlastne vraci index.html
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# nacteni natrenovaneho SVD modelu
try:
    with open('svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    logging.info("SVD model načten.")
except Exception as e:
    logging.error(f"Chyba při načítání SVD modelu: {e}")
    raise

# prubeh nacteni vycistenych dat pro ziskani seznamu knih
try:
    df = pd.read_parquet('cleaned_dataset.parquet')
    all_books = df['Book-Title'].unique()
except Exception as e:
    logging.error(f"Chyba při načítání dat: {e}")
    raise

# definice modelu vstupniho pozadavku
class RecommendationRequest(BaseModel):
    user_id: str
    favorite_book: str

# definovani endpointu pro doporuceni
@app.post("/recommend")
def recommend_books(request: RecommendationRequest):
    user_id = request.user_id.strip()
    favorite_book = request.favorite_book.lower().strip()

    # overeni zda existuji knihy v datech
    if favorite_book not in all_books:
        raise HTTPException(status_code=404, detail="Kniha nenalezena v datasetu")

    try:
        predictions = []
        # generovani odhadu pro vsechny knihy krome oblibene
        for book in all_books:
            if book == favorite_book:
                continue
            pred = svd_model.predict(user_id, book)
            predictions.append((book, pred.est))

        # serazeni a vyber top 10 doporuceni
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = predictions[:10]
    except Exception as e:
        logging.error(f"Chyba při generování doporučení: {e}")
        raise HTTPException(status_code=500, detail="Chyba při generování doporučení")

    # zaokrouhleni hodnoceni na dve desetinna mista
    recommendations = [
        {"book": book, "estimated_rating": round(rating, 2)} 
        for book, rating in top_recommendations
    ]
    return {
        "user_id": user_id,
        "favorite_book": favorite_book,
        "recommendations": recommendations
    }
