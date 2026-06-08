# Use the official Ubuntu base image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install "pybind11[global]"

RUN git clone https://github.com/openfheorg/openfhe-development.git \
    && cd openfhe-development \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. \
    && make -j$(nproc) \
    && make install

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN git clone https://github.com/openfheorg/openfhe-python.git \
    && cd openfhe-python \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

WORKDIR /openfhe-python

WORKDIR /workspace

RUN cp /openfhe-python/build/openfhe*.so /workspace/openfhe.so

RUN set -eux; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    PKG_SRC="/openfhe-python"; \
    mkdir -p "${PY_SITE}/openfhe"; \
    cp -a "${PKG_SRC}/." "${PY_SITE}/openfhe/"

COPY *.txt /workspace/
# Install requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir pytest httpx numpy

COPY *.py /workspace/

RUN mkdir -p /data/keys /data/fhe-encrypted
    
EXPOSE 8000

CMD ["uvicorn", "fhe_app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
