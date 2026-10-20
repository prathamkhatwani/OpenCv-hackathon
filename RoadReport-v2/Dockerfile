FROM node:24.19.0-bookworm-slim AS ui
WORKDIR /ui
COPY roadreport/frontend/package*.json ./
RUN npm ci
COPY roadreport/frontend/ ./
RUN npm run build
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 USE_LOCAL_STORAGE=true ROADREPORT_DB=/data/roadreport.db LOCAL_STORAGE_DIR=/data/storage FRONTEND_DIST=/app/frontend/dist
WORKDIR /app/backend
COPY roadreport/backend/requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY roadreport/backend/ ./
COPY --from=ui /ui/dist /app/frontend/dist
RUN useradd --uid 10001 --create-home roadreport && mkdir /data && chown roadreport:roadreport /data
USER roadreport
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
