CUDA_VISIBLE_DEVICES=1,0,5 python -m vllm.entrypoints.openai.api_server \
    --model path/to/WizardCoder-Python-34B-V1.0 \
    --port=8005 \
    --gpu-memory-utilization=1 \
    --max-model-len=16380 \
    --enforce-eager \