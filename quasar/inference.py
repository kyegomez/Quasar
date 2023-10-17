import time

import torch
from transformers import (
    LlamaTokenizer,
    MistralForCausalLM,
)

tokenizer = LlamaTokenizer.from_pretrained(
    "./collectivecognition-run6", trust_remote_code=True
)
model = MistralForCausalLM.from_pretrained(
    "./collectivecognition-run6",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_flash_attention_2=True,
)
benchmarks = [
    "Hello, tell me about the history of the United States",
    "Roleplay as a scientist, who just discovered artificial general intelligence. What do you think about this discovery? What possibilities are there now?",
]

index = 0
for obj in benchmarks:
    index += 1
    if index < 1:
        continue
    else:
        start_time = time.time()  # Start timing
        prompt = f"USER:\n{obj}\n\nASSISTANT:\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_ids = model.generate(
            input_ids, max_new_tokens=2048, temperature=None
        )  # , do_sample=True, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(
            generated_ids[0][input_ids.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_space=True,
        )
        print(f"Response  {index}: {response}")

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate time taken for the iteration
        print(f"Time taken for Response {index}: {elapsed_time:.4f} seconds")
        print(f"tokens total: {len(tokenizer.encode(response))}")
