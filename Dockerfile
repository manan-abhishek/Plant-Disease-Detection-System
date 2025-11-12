# ---------- base image ----------
FROM python:3.10-slim

# ---------- workdir ----------
WORKDIR /app

# ---------- copy project ----------
COPY . .

# ---------- install dependencies ----------
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---------- start Flask via gunicorn ----------
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000"]
