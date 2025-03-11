CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
    --model path/to/WizardCoder-33B-V1.1 \
    --port=8004 \
    --gpu-memory-utilization=1 \
    --max-model-len=16380 \
    --enforce-eager \
    --tensor-parallel-size=2 \