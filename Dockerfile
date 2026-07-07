# CrudeWatch server image — for hosting on a VM you control (behind Tailscale).
# Build:  docker build -t crudewatch .
# Run:    docker run -d --restart unless-stopped -p 127.0.0.1:8501:8501 --name crudewatch crudewatch
FROM python:3.11-slim

WORKDIR /srv/crudewatch

# Install runtime deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source + data.
COPY . .

# Bake the parquet cache so the container starts instantly (needs data/raw_files.xlsx).
RUN python scripts/prebuild_cache.py

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
