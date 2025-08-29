import os
import sys
import torch

def main():
    # Add project root to path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(root_dir)

    from models import choose_model_class

    model_name = "Qwen/Qwen3-MoE-A2.7B-Instruct"
    LLM = choose_model_class(model_name)

    prompt = "You are a helpful AI assistant. Briefly introduce ShadowKV in 2 sentences."

    # Run with full attention as baseline
    llm = LLM(
        model_name=model_name,
        device='cuda:0',
        batch_size=1,
        max_length=2048,
        attn_mode='full',
        sparse_budget=1024,
        minference=True,
    )
    input_ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    outputs = llm.generate(input_ids, gen_len=64, temperature=0.7)
    print("\n=== Full Attention Output ===\n", outputs[0])

    # Clean up before next run
    del llm.kv_cache
    del llm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run with ShadowKV
    llm = LLM(
        model_name=model_name,
        device='cuda:0',
        batch_size=1,
        max_length=2048,
        attn_mode='shadowkv_cpu',
        sparse_budget=1024,
        minference=True,
    )
    input_ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    outputs = llm.generate(input_ids, gen_len=64, temperature=0.7)
    print("\n=== ShadowKV Output ===\n", outputs[0])


if __name__ == "__main__":
    main()