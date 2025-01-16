# pouzivam miniconda image jako zaklad
FROM continuumio/miniconda3

# nastaveni pracovni slozky uvnitr kontejneru
WORKDIR /app

# kopirovani environment.yml do kontejneru
COPY environment.yml .

# aktualizace existujiciho conda prosteedi podle environment.yml
RUN conda env update --file environment.yml --prune

# nastaveni shellu, aby pouzival base conda prostreedi
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# namontovani statickych souboru na cestu /static
COPY static /app/static

# kopirovani zbytku aplikace do pracovniho adresare
COPY . .

# exponovani portu, na kterem bezi aplikace
EXPOSE 8000

# spusteni aplikace
CMD ["conda", "run", "-n", "base", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
