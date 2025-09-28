import torch

from src.utils.inference import GRPOInference


def main() -> int:
    # Define model and adapter
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    adapter_path = "outputs/qwen2.5-3b-grpo/checkpoint-2000"
    device_map = "cuda"
    dtype = torch.bfloat16
    max_new_tokens = 128
    temperature = 1.0
    top_p = 0.9
    do_sample = True

    # Define context and question
    context = "Paris is the capital of France."
    question = "What is the capital of France?"

    # Define inference
    infer = GRPOInference(
        model_id=model_id,
        adapter_path=adapter_path,
        device_map=device_map,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )

    # Generate adversarial sentence
    result = infer.generate_one(context=context, question=question)
    print(result["answer"])


if __name__ == "__main__":
    main()
