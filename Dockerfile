FROM python:3.10-slim

# Create user to comply with Hugging Face Space security requirements
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the environment source code
COPY --chown=user . /app

# Install OpenEnv and environment dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Enable OpenEnv web interface
ENV ENABLE_WEB_INTERFACE="true"

# Hugging Face routes port 7860 automatically based on your README.md
EXPOSE 7860

# Start the OpenEnv web server
CMD ["python", "-m", "hospital_triage.server.app", "--port", "7860", "--host", "0.0.0.0"]
