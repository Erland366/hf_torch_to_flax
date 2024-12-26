import gc
import os
import argparse
import torch
import jax
import jax.numpy as jnp
from transformers import AutoConfig, AutoModelForCausalLM, FlaxAutoModelForCausalLM, AutoTokenizer
from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
import numpy as np
import platform
import time
import subprocess
from dotenv import load_dotenv
from huggingface_hub import create_repo, HfApi, ModelCard

load_dotenv()

# Disable JAX's memory preallocation to potentially prevent OOM errors
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def clean_gpu():
    for _ in range(10):
        torch.cuda.empty_cache()
        gc.collect()
        jax.clear_backends()

def compare_weights(flax_params, pt_state_dict, rtol=1e-5, atol=1e-3, verbose=False):
    """
    Compares Flax and PyTorch model weights and generates a comparison table.

    Args:
        flax_params: The Flax model's parameters (nested dictionary).
        pt_state_dict: The PyTorch model's state_dict.
        rtol: Relative tolerance for jnp.allclose().
        atol: Absolute tolerance for jnp.allclose().
        verbose: If True, print detailed comparison results.

    Returns:
        all_close: Boolean indicating if all weights are close.
        comparison_table: A list of dictionaries, where each dictionary represents a row in the table.
    """

    all_close = True
    mismatched_keys = []
    comparison_table = []

    for pt_key, pt_tensor in pt_state_dict.items():
        if "embed_tokens" in pt_key:
            flax_key_tuple = pt_key.replace("weight", "embedding")
        elif "input_layernorm" in pt_key or "post_attention_layernorm" in pt_key or "norm" in pt_key:
            flax_key_tuple = pt_key
        else:
            flax_key_tuple = pt_key.replace("weight", "kernel")

        flax_key_tuple = flax_key_tuple.split(".")

        flax_param_group = flax_params
        for key_part in flax_key_tuple:
            if key_part in flax_param_group:
                flax_param_group = flax_param_group[str(key_part)]
            else:
                if verbose:
                    print(f"Key mismatch: {pt_key} not found in Flax params")
                all_close = False
                mismatched_keys.append(pt_key)
                flax_param_group = None
                break
        
        if flax_param_group is None:
            continue

        flax_tensor = jnp.asarray(flax_param_group)

        if flax_key_tuple[-1] == "kernel" and len(pt_tensor.shape) == 2:
            pt_tensor = pt_tensor.T

        pt_tensor_np = pt_tensor.detach().cpu().float().numpy()
        flax_tensor_np = np.array(flax_tensor)

        is_close = jnp.allclose(flax_tensor_np, pt_tensor_np, rtol=rtol, atol=atol)
        if not is_close:
            all_close = False
            mismatched_keys.append(pt_key)
            diff = np.abs(flax_tensor_np - pt_tensor_np)
            max_diff = diff.max()
            mean_diff = diff.mean()
            std_diff = diff.std()
        else:
            max_diff = 0.0
            mean_diff = 0.0
            std_diff = 0.0

        if verbose:
            print(f"Comparing: {pt_key} (PyTorch) vs. {flax_key_tuple} (Flax)")
            print(f"  Shapes: {pt_tensor_np.shape} (PyTorch) vs. {flax_tensor_np.shape} (Flax)")
            print(f"  Allclose: {is_close}")

            if not is_close:
                print(f"  Max Diff: {max_diff:.3e}")
                print(f"  Mean Diff: {mean_diff:.3e}")
                print(f"  Std Diff: {std_diff:.3e}")

        comparison_table.append({
            "Layer": pt_key,
            "PyTorch Shape": str(pt_tensor_np.shape),
            "Flax Shape": str(flax_tensor_np.shape),
            "Allclose": str(is_close),
            "Max Diff": f"{max_diff:.3e}" if not is_close else "0",
            "Mean Diff": f"{mean_diff:.3e}" if not is_close else "0",
            "Std Diff": f"{std_diff:.3e}" if not is_close else "0",
        })

    print("\nSummary:")
    if all_close:
        print("  All weights are approximately close!")
    else:
        print("  Weight mismatches found!")
        print("  Mismatched keys:", mismatched_keys)

    return all_close, comparison_table

def convert_and_upload(model_name, hf_user, max_pos_embed_divisor_start=1, safe_serialization=False):
    """Converts a PyTorch model to Flax, handles OOM, generates README, uploads to HF Hub."""
    start_time = time.time()
    dtype = "float32"

    config = AutoConfig.from_pretrained(model_name)
    original_max_pos_embed = config.max_position_embeddings
    max_pos_embed_divisor = max_pos_embed_divisor_start

    while max_pos_embed_divisor <= original_max_pos_embed:
        try:
            config.max_position_embeddings = original_max_pos_embed // max_pos_embed_divisor
            print(f"Trying max_position_embeddings: {config.max_position_embeddings}")

            flax_model = FlaxAutoModelForCausalLM.from_config(config, dtype=getattr(jnp, dtype))
            pt_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True
            )

            flax_state_dict = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), flax_model)

            jax.clear_backends()

            flax_model.params = flax_state_dict

            all_close, comparison_table = compare_weights(flax_model.params, pt_model.state_dict(), verbose=False)

            del pt_model
            if all_close:
                print(f"Successfully converted with max_position_embeddings: {config.max_position_embeddings}")
                break
            else:
                print("Weight check failed. Retrying with a different max_position_embeddings setting.")
                max_pos_embed_divisor *= 2
                del flax_state_dict
                del flax_model
                clean_gpu()

        except (RuntimeError, ValueError) as e:
            print(f"Error during conversion: {e}")
            if "out of memory" in str(e).lower() or "Failed to allocate" in str(e):
                print("Out of memory error. Trying a smaller max_position_embeddings.")
                max_pos_embed_divisor *= 2
                try:
                    del flax_state_dict
                    del flax_model
                    clean_gpu()
                except Exception as _:
                    pass
            else:
                raise

    if max_pos_embed_divisor > original_max_pos_embed:
        raise RuntimeError("Failed to convert the model even with the smallest max_position_embeddings setting.")

    config.max_position_embeddings = original_max_pos_embed // max_pos_embed_divisor

    hardware_info = {
        "CPU": platform.processor(),
        "RAM": f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.0 ** 3):.2f} GB",
        "OS": platform.platform(),
        "JAX version": jax.__version__,
        "Flax version": "Not available", 
        "Transformers version": "Not available",
    }

    try:
        import flax
        hardware_info["Flax version"] = flax.__version__
    except ImportError:
        print("Flax library not found. Flax version will not be included in hardware info.")
    try:
        import transformers
        hardware_info["Transformers version"] = transformers.__version__
    except ImportError:
        print("Transformers library not found. Transformers version will not be included in hardware info.")

    try:
        gpu_info = torch.cuda.get_device_name(0)
        hardware_info["GPU"] = gpu_info
    except:
        hardware_info["GPU"] = "None"

    end_time = time.time()
    conversion_time = end_time - start_time

    repo_name = f"{hf_user}/{model_name.split('/')[-1]}-JAX"
    try:
        repo_url = create_repo(repo_name, private=False, exist_ok=True)
        print(f"Repository '{repo_name}' created or already exists.")
    except Exception as e:
        print(f"An error occurred while creating the repository: {e}")
    
    api = HfApi()

    license_text = get_license(model_name)

    with open("testing_chamber/model_card.txt", "r") as f:
        readme_template = f.read()

    readme_content = readme_template.format(
        model_name=model_name,
        repo_name=repo_name,
        original_model_org=model_name.split("/")[0],
        model_size=config.num_hidden_layers,  # You can make this more descriptive
        original_max_pos_embed=original_max_pos_embed,
        new_max_pos_embed=config.max_position_embeddings,
        weight_comparison_table=generate_markdown_table(comparison_table),
        hardware_info=generate_hardware_info_string(hardware_info),
        conversion_time=f"{conversion_time:.2f} seconds",
        license=license_text
    )

    # Save README.md
    readme_file = "model_card.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    # Push model to Hub
    flax_model.push_to_hub(repo_name, config=config, safe_serialization=safe_serialization, commit_message="Add model")
    print(f"Model uploaded to: {repo_name}")

    # Push tokenizer to Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.push_to_hub(repo_name, commit_message="Add tokenizer")
    print(f"Tokenizer uploaded to: {repo_name}")

    # Upload README.md to the repository
    try:
        api.upload_file(
            path_or_fileobj=readme_file,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",  # Specify that it's a model repository
            commit_message="Update README.md with weight comparison and hardware info"
        )
        print(f"README.md updated and pushed to: {repo_name}")

        # delete file README.md
        os.remove("model_card.md")
    except Exception as e:
        print(f"An error occurred while uploading README.md: {e}")


def generate_markdown_table(comparison_table):
    """Generates a Markdown table from the comparison data."""
    table = ["| Layer | PyTorch Shape | Flax Shape | Allclose | Max Diff | Mean Diff | Std Diff |",
             "| :---- | :------------ | :--------- | :------- | :------- | :-------- | :------- |"]
    for row in comparison_table:
        table.append(f"| {row['Layer']} | {row['PyTorch Shape']} | {row['Flax Shape']} | {row['Allclose']} | {row['Max Diff']} | {row['Mean Diff']} | {row['Std Diff']} |")
    return "\n".join(table)

def generate_hardware_info_string(hardware_info):
    """Generates a formatted string of hardware information."""
    info_lines = ["*   **{}:** {}".format(key, value) for key, value in hardware_info.items()]
    return "\n".join(info_lines)

def get_license(model_id):
    """Retrieves the license of the original model from Hugging Face Hub."""
    api = HfApi()
    try:
        model_info = api.model_info(model_id)
        if model_info.cardData and model_info.cardData["license"]:
            return model_info.cardData["license"]
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error retrieving model info for {model_id}: {e}")
        return "Error"

def generate_markdown_table(comparison_table):
    """Generates a Markdown table from the comparison data."""
    table = ["| Layer | PyTorch Shape | Flax Shape | Allclose | Max Diff | Mean Diff | Std Diff |",
             "| :---- | :------------ | :--------- | :------- | :------- | :-------- | :------- |"]
    for row in comparison_table:
        table.append(f"| {row['Layer']} | {row['PyTorch Shape']} | {row['Flax Shape']} | {row['Allclose']} | {row['Max Diff']} | {row['Mean Diff']} | {row['Std Diff']} |")
    return "\n".join(table)

def generate_hardware_info_string(hardware_info):
    """Generates a formatted string of hardware information."""
    info_lines = ["*   **{}:** {}".format(key, value) for key, value in hardware_info.items()]
    return "\n".join(info_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch Hugging Face model to Flax and upload it to the Hub.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the Hugging Face model to convert (e.g., 'meta-llama/Llama-3.2-3B').")
    parser.add_argument("--hf_user", type=str, required=True, help="Your Hugging Face username.")
    parser.add_argument("--token", type=str, default=None, help="Your Hugging Face API token.")
    parser.add_argument("--max_pos_embed_divisor_start", type=int, default=1, help="Initial divisor for max_position_embeddings.")
    parser.add_argument("--safe_serialization", type=bool, default=False, help="Whether to use safe serialization when pushing to the hub.")
    args = parser.parse_args()

    if os.getenv("HF_TOKEN", None) is None and args.token is None:
        raise ValueError("Please provide your Hugging Face API token with the --token flag or set it as an environment variable `HF_TOKEN`.")

    # Make sure already login
    subprocess.run(f'huggingface-cli login --token={os.environ["HF_TOKEN"]}', 
               shell=True)

    convert_and_upload(args.model_name, args.hf_user, args.max_pos_embed_divisor_start, args.safe_serialization)