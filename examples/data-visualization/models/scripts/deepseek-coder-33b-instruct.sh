CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model path/to/deepseek-coder-33b-instruct \
    --port=8003 \
    --gpu-memory-utilization=1 \
    --max-model-len=16380 \
    --enforce-eager \