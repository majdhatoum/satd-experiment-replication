# Repro image for the SATD experiment
# - Headless plotting
# - Wheels for numpy/scipy/statsmodels/scikit-learn/matplotlib
# - No source compiles in the common case

FROM python:3.11-slim

# System deps mostly for matplotlib font rendering + git (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    fonts-dejavu-core \
    git \
 && rm -rf /var/lib/apt/lists/*

# Prevent interactive prompts, and make matplotlib run headless
ENV MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user (optional but nice)
RUN useradd -ms /bin/bash runner
USER runner

# Workdir where we’ll mount the repo from host
WORKDIR /workspace

# Install Python deps from your pinned lock if present, falling back to base requirements
# Copy just the requirement files to leverage Docker layer caching
COPY --chown=runner:runner requirements.txt requirements.txt
# If you have a freeze file, copy it too; installation tries it first
COPY --chown=runner:runner requirements.lock.txt requirements.lock.txt

# Try strict (locked) install first; if it fails (e.g., platform-specific wheels),
# fall back to the base requirements
RUN --mount=type=cache,target=/home/runner/.cache/pip \
    (pip install -r requirements.lock.txt || pip install -r requirements.txt)

# The code itself is *not* copied into the image; we’ll bind-mount the repo at runtime.
# Default command shows help; override with specific scripts in `docker run`
CMD [ "python", "-c", "print('SATD repro image ready. Bind-mount the repo to /workspace and run analysis scripts.')" ]
