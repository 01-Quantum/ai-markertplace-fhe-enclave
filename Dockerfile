# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV FHE_KEYS_DIR=/data/keys
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Install necessary dependencies for OpenFHE
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

# Install PyBind11
RUN pip3 install "pybind11[global]"

# Clone and build OpenFHE-development
RUN git clone https://github.com/openfheorg/openfhe-development.git \
    && cd openfhe-development \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. \
    && make -j$(nproc) \
    && make install

# Clone and build OpenFHE-Python
RUN git clone https://github.com/openfheorg/openfhe-python.git \
    && cd openfhe-python \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

RUN set -eux; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    PKG_SRC="/openfhe-python"; \
    mkdir -p "${PY_SITE}/openfhe"; \
    cp -a "${PKG_SRC}/." "${PY_SITE}/openfhe/"; \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY fhe_key_gen.py key_storage.py supabase_db.py auth.py fhe_app.py /workspace/

RUN mkdir -p /data/keys

EXPOSE 8000

CMD ["uvicorn", "fhe_app:app", "--host", "0.0.0.0", "--port", "8000"]
