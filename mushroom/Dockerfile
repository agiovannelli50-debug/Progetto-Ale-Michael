# Prendiamo versione slim di python cosi i tempi si riducono
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY app.py index.html logistic_regression_model.pkl xgboost_model.pkl scaler.pkl neural_network_model.keras ./

EXPOSE 8000

#"app:app" indica percorso per trovare fastapi (app.py --> app=...)
#"--host", "0.0.0.0" per dire a docker di accettare connessioni anche dall'esterno
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]