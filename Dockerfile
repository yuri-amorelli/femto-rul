FROM python:3.12-slim

WORKDIR /app

# No PyTorch and no scikit-learn: the served model is a JSON bundle
# (artifacts/serving/, written by scripts/export_serving.py; see src/serving.py).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the service needs at runtime: code + serving bundle. No data, no notebooks,
# no training artifacts.
COPY src/ src/
COPY api/ api/
COPY artifacts/serving/ artifacts/serving/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
