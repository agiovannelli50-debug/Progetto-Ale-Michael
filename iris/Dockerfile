#prendiamo versione slim di python cosi i tempi si riducono
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html iris_model.joblib iris_model_knn.joblib ./

EXPOSE 8000

#uvicorn fa girare fastAPI
#"app:app" indica percorso per trovare fastapi (app.py --> app=...)
#"--host", "0.0.0.0" per dire a docker di accettare connessioni anche dall'esterno
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]