FROM 10.119.12.14:5000/llm/llm-gpu

ENV MAX_JOBS=32
COPY ./requirements.txt /requirements.txt
# RUN pip3 install flash-attn==2.4.2 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
RUN pip3 install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation && rm /requirements.txt
COPY ./requirements-vllm.txt /requirements-vllm.txt
RUN pip3 install -r /requirements-vllm.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && rm /requirements-vllm.txt
RUN pip3 install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

# somehow we should reinstall the following packages after installing vllm
RUN pip3 uninstall -y transformer_engine
COPY ./flash-attention/dist/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl
RUN pip3 uninstall -y flash_attn && pip3 install /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl \
    && rm /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl