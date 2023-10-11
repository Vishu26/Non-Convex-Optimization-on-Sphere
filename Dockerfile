FROM daskdev/dask:latest

SHELL ["/bin/bash", "--login", "-c"]

COPY requirements.txt /tmp/requirements.txt

RUN apt-get update -y && apt-get install -y libstdc++6 gcc && \
conda create -n opt python=3.10 && \
source /opt/conda/bin/activate opt && \
pip install -r /tmp/requirements.txt