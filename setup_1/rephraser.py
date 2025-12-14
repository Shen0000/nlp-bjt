import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Function for rewriting ---
def rewrite(text, tokenizer, model):
    messages = [
        {"role": "user",
         "content": f"Rewrite the following paragraph in 20 words: {text}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **model_inputs,
        max_new_tokens=120,
    )

    generated_ids = output[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def main(source, i, rephraser, model_used):

    # --- Load model and tokenizer ---
    model_name = rephraser
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # --- Load JSON properly ---
    if (i == 1):
        data_path = f"/home/common/nlp-bjt/news_data/data/sample/{source}_sampled.jsonl"
    else:
        data_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/data/rephrased_1.jsonl"
    # --- Load JSONL ---
    data = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # --- Output file ---
    out_path = f"/home/common/nlp-bjt/setup_1/{source}/{model_used}/data/rephrased_{i}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as out_f:
        for line in data:
            human_rewrite = rewrite(line["human"], tokenizer, model)
            llm_rewrite   = rewrite(line["llm"], tokenizer, model)
            out_obj = {
                "human": human_rewrite,
                "llm": llm_rewrite
            }
            out_f.write(json.dumps(out_obj) + "\n")
    print("Done! Saved to:", out_path)

# --- Execution ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python rephraser.py <data_source> <iteration> <rephraser> <rephraser_name>")
        sys.exit(1)
    source_name = sys.argv[1]
    iteration = int(sys.argv[2])
    rephraser = sys.argv[3]
    model = sys.argv[4]
    main(source_name, iteration, rephraser, model)
