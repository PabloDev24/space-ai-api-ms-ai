FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first — pip's resolver otherwise pulls the CUDA build (~6GB of
# unused nvidia-* wheels) to satisfy docling/easyocr on this GPU-less App Service.
# Split into separate RUN layers (smaller image layers push more reliably than
# one multi-GB layer over constrained networks).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir docling[easyocr]
RUN pip install --no-cache-dir chromadb fastembed
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
