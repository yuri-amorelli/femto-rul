FROM python:3.12-slim

WORKDIR /app

# PyTorch CPU-only, PRIMA del resto: niente CUDA nel container (~2.5 GB risparmiati)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Solo cio' che serve a runtime: codice + artefatti. Niente data, niente notebook.
COPY src/ src/
COPY api/ api/
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]