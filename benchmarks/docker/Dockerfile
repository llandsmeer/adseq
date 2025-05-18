FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
# docker build --tag mlseq ./docker
# docker run -h=$(hostname)_docker --gpus all -v $(git rev-parse --show-toplevel):/mnt -it mlseq bash

RUN apt update && apt upgrade -y
RUN apt install -y \
    python3 \
    python3-pip

RUN pip install \
    'jax[cuda12]' \
    tqdm \
    pytest \
    optax \
    numpy \
    openvino \
    onnxruntime \
    matplotlib

WORKDIR /mnt



# -v 
