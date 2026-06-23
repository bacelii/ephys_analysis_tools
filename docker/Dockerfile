FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/ephys_analysis_tools

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential bash \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install notebook jupyterlab

EXPOSE 8888

# CMD ["bash", "-lc", "pip install -e /workspace/ephys_analysis_tools && exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token='' --ServerApp.password='' --ServerApp.allow_origin='*' --ServerApp.root_dir=/workspace/ephys_analysis_tools --allow-root"]
