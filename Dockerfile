ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"
ARG MMCV="2.0.1"

# NOTE (known limitation): this base image ships an older Python than the
# project's `requires-python`, and mmsegmentation/mmdetection are cloned from
# their `main` branches. If `pip install .` fails with a Requires-Python error
# or the clones break, bump the PYTORCH/CUDA/MMCV build args. The image could
# not be rebuilt as part of the generalization work (no Docker daemon
# available), so treat it as unverified. See doc/usage.md.
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev \
    libgdal-dev gdal-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# Install MMCV
ARG PYTORCH
ARG CUDA
ARG MMCV
RUN ["/bin/bash", "-c", "pip install openmim"]
RUN ["/bin/bash", "-c", "mim install mmengine"]
RUN ["/bin/bash", "-c", "mim install mmcv==${MMCV}"]

# Install MMSegmentation
RUN git clone -b main https://github.com/open-mmlab/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

# Install MMDetection
RUN git clone -b main https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .

WORKDIR /app

# The project has to be present for `pip install .`: the previous version
# copied only pyproject.toml, so the image contained no code at all.
COPY pyproject.toml README.md ./
COPY src ./src
# Default configuration (docker-compose bind-mounts the live one over it).
COPY config.toml ./

RUN pip install .

# Data, model and results are provided as bind mounts by docker-compose:
#   ./data   -> /app/data    (inputs, read-only)
#   ./model  -> /app/model   (model config + checkpoint, read-only)
#   ./results-> /app/results (outputs)
