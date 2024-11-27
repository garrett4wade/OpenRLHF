FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt-get -y install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
RUN pip uninstall xgboost transformer_engine flash_attn -y

COPY ./OpenRLHF/requirements.txt /
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements.txt && rm /requirements.txt
COPY ./vllm/requirements-cuda.txt /
COPY ./vllm/requirements-common.txt /
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements-cuda.txt && rm /requirements-cuda.txt

COPY ./vllm /vllm
RUN VLLM_USE_PRECOMPILED=1 pip3 install -e /vllm --no-build-isolation

COPY ./wheels/flash_attn-2.6.1-cp310-cp310-linux_x86_64.whl /
RUN pip install /flash_attn-2.6.1-cp310-cp310-linux_x86_64.whl && rm /flash_attn-2.6.1-cp310-cp310-linux_x86_64.whl

COPY ./docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

COPY ./OpenRLHF /openrlhf
RUN pip3 install -e /openrlhf --no-build-isolation
