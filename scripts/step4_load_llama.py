"""
STEP 4: Load LLaMA-3-8B (4-bit quantized, frozen)

HONESTY NOTE - READ THIS FIRST:
This script CANNOT be tested in the current sandbox. Reasons:
  1. No GPU available here (torch.cuda.is_available() == False)
  2. LLaMA-3-8B weights are several GB and require a Hugging Face account +
     license acceptance (Meta's gated repo) - this sandbox's network is
     restricted to PyPI/npm/GitHub-type domains, not huggingface.co downloads
  3. 4-bit quantization (bitsandbytes) requires CUDA

This script is written to be CORRECT based on standard practice, but it has
NOT been run successfully end-to-end by me. You must run this on Kaggle
(free T4 GPU) and report back what happens - especially any package version
conflicts, which are extremely common with bitsandbytes + transformers +
accelerate version mismatches.

BEFORE RUNNING ON KAGGLE, YOU MUST:
  1. Create a Hugging Face account (if you don't have one)
  2. Go to: huggingface.co/meta-llama/Meta-Llama-3-8B
  3. Accept Meta's license agreement (takes a few hours to a day for approval,
     sometimes instant - don't assume instant, check today)
  4. Generate a HF access token: huggingface.co/settings/tokens
  5. In Kaggle, add the token as a Kaggle Secret (do NOT hardcode it in the
     notebook) - Kaggle has a "Secrets" feature in notebook settings for this

This is another external approval dependency like ScanNet - check it today,
not when you're ready to run this script.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"


def load_frozen_llama(hf_token=None):
    """
    Loads LLaMA-3-8B in 4-bit quantization, fully frozen (no gradients).

    Args:
        hf_token: Hugging Face access token (required for gated model).
                   On Kaggle, retrieve via:
                   from kaggle_secrets import UserSecretsClient
                   hf_token = UserSecretsClient().get_secret("HF_TOKEN")

    Returns:
        model, tokenizer
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )

    # Freeze everything - LLaMA does not get fine-tuned in this architecture.
    # Only the projection layer (Step 3) is trainable.
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    print(f"Loaded {MODEL_NAME}")
    print(f"Hidden size: {model.config.hidden_size}  <-- VERIFY this matches "
          f"Step 3's OUTPUT_DIM (currently set to 4096)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"(should be 0 - model is fully frozen)")

    return model, tokenizer


def embed_text_tokens(model, tokenizer, text, device="cuda"):
    """
    Converts text into LLaMA's input embedding space, so we can later
    concatenate [projected_point_tokens, text_embeddings] before feeding
    into the model.

    Returns:
        embeddings: (1, seq_len, hidden_size) tensor
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeddings = embedding_layer(input_ids)
    return embeddings


if __name__ == "__main__":
    print("=" * 70)
    print("THIS SCRIPT REQUIRES A GPU AND HUGGING FACE GATED MODEL ACCESS.")
    print("It will NOT run in a CPU-only sandbox. Run this on Kaggle.")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("\nNo GPU detected. Stopping here - this confirms the sandbox")
        print("limitation described above. Do not attempt to run further")
        print("in this environment.")
    else:
        # This branch only executes on a real GPU machine (Kaggle/Colab)
        # On Kaggle, get the token like this instead of pasting it directly:
        # from kaggle_secrets import UserSecretsClient
        # hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        hf_token = None  # <-- SET THIS on Kaggle before running
        model, tokenizer = load_frozen_llama(hf_token=hf_token)

        # Quick sanity test
        sample_embeds = embed_text_tokens(model, tokenizer, "What color is the chair?")
        print(f"\nSample text embedding shape: {sample_embeds.shape}")
