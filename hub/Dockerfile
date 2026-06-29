FROM python:3.12-slim

# Create a non-root user; app runs entirely as this user at runtime.
RUN useradd --create-home --shell /bin/bash quipu-hub

WORKDIR /app

# Copy dependency list first so layer is cached unless requirements change.
COPY hub/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the hub package.
COPY hub/ ./hub/

# Create persistent-data directory owned by the non-root user.
RUN mkdir -p /data && chown quipu-hub:quipu-hub /data

# Switch to non-root for the remainder of the image and all runtime.
USER quipu-hub

EXPOSE 8000

# --workers 1 is intentional: WAL mode + single process avoids connection
# contention on SQLite. See db-architect notes in TASK-013.
CMD ["uvicorn", "hub.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
