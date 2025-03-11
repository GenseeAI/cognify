CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model path/to/CodeLlama-34b-Instruct-hf \
    --port=8006 \
    --gpu-memory-utilization=1 \
    --enforce-eager \
    --tensor-parallel-size 2