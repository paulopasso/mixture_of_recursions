import os

# Ensure HF can go online for this one-time bootstrap
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

TOKENIZER_REPO = "HuggingFaceTB/SmolLM-135M"
MODEL_REPO = "HuggingFaceTB/SmolLM-360M"

def main():
	print("Downloading tokenizer to local cache:", TOKENIZER_REPO, flush=True)
	_ = AutoTokenizer.from_pretrained(TOKENIZER_REPO, local_files_only=False)

	print("Downloading model to local cache:", MODEL_REPO, flush=True)
	_ = AutoModelForCausalLM.from_pretrained(MODEL_REPO, local_files_only=False)

	print("Done. You can now run in offline mode.", flush=True)

if __name__ == "__main__":
	main()
