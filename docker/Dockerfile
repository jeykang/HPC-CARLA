FROM carlasim/carla:0.9.10.1

ENV DEBIAN_FRONTEND=noninteractive \
    PROJECT_ROOT=/workspace \
    CARLA_ROOT=/home/carla

# Some Python 3.6 sdists read files using the process locale; ensure UTF-8 to avoid UnicodeDecodeError.
ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

# carlasim/carla image may default to a non-root user; switch to root for apt installs
USER root

# System deps for python packages + debugging + CARLA runtime helpers
# The base image may include NVIDIA/CUDA apt repos with expired/rotated keys; disable them for reproducible builds.
RUN bash -lc 'set -euo pipefail; \
    mkdir -p /root/apt-hold; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do \
      [ -f "$f" ] || continue; \
      if grep -qE "developer\\.download\\.nvidia\\.com/compute/(cuda|machine-learning)" "$f"; then \
        mv -f "$f" "/root/apt-hold/$(basename "$f")"; \
      fi; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      python3-pip python3-setuptools python3-dev \
      git curl ca-certificates \
      locales \
      ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
      libjpeg-turbo8 libtiff5 \
      build-essential gcc g++ \
      lsof psmisc procps jq \
    && rm -rf /var/lib/apt/lists/*'

RUN bash -lc 'set -e; locale-gen en_US.UTF-8 || true; update-locale LANG=C.UTF-8 LC_ALL=C.UTF-8 || true'

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir cython==0.29.33

# Copy only requirement manifests first for better layer caching
WORKDIR /tmp/reqs
COPY leaderboard/requirements.txt /tmp/reqs/leaderboard-requirements.txt
COPY scenario_runner/requirements.txt /tmp/reqs/scenario-runner-requirements.txt

# Base python deps used by evaluator + agents.
# Note: this repo historically used a large dependency set (see carla_official.def).
ARG INSTALL_HEAVY_DEPS=0
RUN python3 -m pip install --no-cache-dir --use-deprecated=legacy-resolver -r /tmp/reqs/scenario-runner-requirements.txt \
 && python3 -m pip install --no-cache-dir --use-deprecated=legacy-resolver -r /tmp/reqs/leaderboard-requirements.txt \
 && if [ "$INSTALL_HEAVY_DEPS" = "1" ]; then \
      # Keep this list intentionally minimal: the base image is Python 3.6,
      # and many modern ML utilities (torchmetrics, fsspec, etc.) now require >=3.8.
      # InterFuser/LAV/TCP primarily need torch(+vision) and a few lightweight extras.
      python3 -m pip install --no-cache-dir --use-deprecated=legacy-resolver --prefer-binary \
        filelock==3.0.12 typing-extensions==4.1.1 packaging==21.3 fsspec==2022.1.0 \
        torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
          -f https://download.pytorch.org/whl/cu113/torch_stable.html \
        easydict==1.13 timm==0.6.12 pytorch-lightning==1.2.5 torchmetrics==0.8.2 \
    ; fi

# Runtime wiring
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}"
# egg path uses a glob; resolve at runtime in entrypoint scripts.
ENV PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist:${PYTHONPATH}"
ENV PYTHONPATH="/workspace:/workspace/leaderboard:/workspace/scenario_runner:${PYTHONPATH}"

WORKDIR /workspace

# Default command is overridden by docker-compose to run docker/node_entrypoint.sh
CMD ["bash", "-lc", "python3 -V && echo 'Image ready' && sleep infinity"]
