FROM python:3.12-slim

# System deps for building wheels (pandas / pyarrow / numpy).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the layer caches across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source — entry script + multi-page views + strategy module + runtime data.
# .dockerignore excludes tests/, notebooks/, dev pipeline scripts, and caches.
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 streamlit \
 && chown -R streamlit:streamlit /app
USER streamlit

EXPOSE 8080

# Healthcheck against Streamlit's built-in /_stcore/health endpoint. Uses Python's
# stdlib so we don't need to install curl/wget.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8080/_stcore/health', timeout=3).status == 200 else 1)" \
  || exit 1

# CORS / XSRF off so websockets work through the cloud proxy (Fly, Render, Cloud Run).
# Headless = no auto-launch of a browser inside the container.
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
