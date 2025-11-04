# Use a Python version compatible with MediaPipe/OpenCV (e.g., 3.11)
FROM python:3.11-slim-bullseye

# 1. INSTALL SYSTEM DEPENDENCIES (FIXES libGL.so.1 ERROR)
# libgl1 is the primary missing package.
# libgomp1 is required by some numpy/ML library optimizations.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        v4l-utils \
        # Clean up to reduce final image size
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# 2. SET UP WORK DIRECTORY
WORKDIR /python

# 3. INSTALL PYTHON DEPENDENCIES
# Use the stable versions from your requirements.txt
COPY requirements.txt .

# Upgrade pip for best dependency resolution
RUN pip install --no-cache-dir --upgrade pip

# Install main requirements (OpenCV, MediaPipe, NumPy, websockets, Qdrant)
# Note: jax/jaxlib installation will be handled after this step if needed.
RUN pip install --no-cache-dir -r requirements.txt

# --- JAX/JAXLIB OPTIONAL FIX (If needed) ---
# Since jaxlib can be tricky, install the CPU version explicitly here.
RUN pip install --no-cache-dir jaxlib==0.4.28
RUN pip install --no-cache-dir jax==0.4.28
# -------------------------------------------


# 4. COPY PROJECT CODE
COPY . .

EXPOSE 7777