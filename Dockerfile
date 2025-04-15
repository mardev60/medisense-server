FROM python:3.11-slim

LABEL maintainer="marde"

WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Exposition du port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 