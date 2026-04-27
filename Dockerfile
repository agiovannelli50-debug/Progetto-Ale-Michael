# Base image
FROM python:3.13.9

# Work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
# Ensure the trained models are included in the image
COPY app.py ./
COPY iris_model.joblib iris_model_knn.joblib ./

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]