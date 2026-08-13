FROM nvcr.io/nvidia/vllm:26.07-py3

# Cosmos3-Nano's reasoner architecture is provided by NVIDIA's vLLM plugin.
# Pin the source commit so rebuilding this deployment is reproducible.
RUN git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cosmos-framework.git /tmp/cosmos-framework \
    && cd /tmp/cosmos-framework \
    && git checkout 103c5d1687d290b050e4890f48ff7a38b12742ef -- packages/vllm-cosmos3 packages/transformers-cosmos3 \
    && pip install --no-cache-dir --no-deps ./packages/transformers-cosmos3 ./packages/vllm-cosmos3 \
    && rm -rf /tmp/cosmos-framework

ENTRYPOINT ["vllm", "serve"]
