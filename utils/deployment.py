import os
import torch
from safetensors.torch import save_file
from huggingface_hub import upload_folder, create_repo

def convert_ego_to_gpt2_safetensors(model_state_dict, output_dir):
    """
    Converts EgoModel state_dict (with Linear layers) to GPT-2 (Conv1D) safetensors format.
    Handles renaming and weight transposition.
    
    Args:
        model_state_dict (dict): The PyTorch state dict of the trained EgoModel.
        output_dir (str): Directory to save the safetensors file.
    """
    print(f"Starting conversion to Standard GPT-2 format...")
    os.makedirs(output_dir, exist_ok=True)
    
    new_state_dict = {}
    
    for key, val in model_state_dict.items():
        if key.startswith("module."): key = key[7:]
        new_key = key
        
        # 1. Embeddings
        if "token_embedding.weight" in key:
            new_key = "transformer.wte.weight"
            
        # 2. Transformer Blocks
        elif key.startswith("transformer_blocks."):
            inner_key = key.replace("transformer_blocks.", "")
            inner_key = inner_key.replace("self_attention.", "attn.")
            inner_key = inner_key.replace("pre_attention_norm.", "ln_1.")
            inner_key = inner_key.replace("feed_forward.", "mlp.")
            inner_key = inner_key.replace("pre_ffn_norm.", "ln_2.")
            
            # Map Attention internal layers
            inner_key = inner_key.replace("c_q.", "c_attn_q.")
            inner_key = inner_key.replace("c_k.", "c_attn_k.")
            inner_key = inner_key.replace("c_v.", "c_attn_v.")
            inner_key = inner_key.replace("c_proj.", "c_proj.")
            
            # Map MLP
            inner_key = inner_key.replace("expansion_layer.", "c_fc.")
            inner_key = inner_key.replace("output_projection.", "c_proj.")
            
            new_key = f"transformer.h.{inner_key}"
        
        # 3. Final Norm
        elif "final_norm" in key:
            new_key = key.replace("final_norm", "transformer.ln_f")
            
        # 4. LM Head
        elif "lm_head.weight" in key:
            new_key = "lm_head.weight"
            val = val.clone()
            
        # 5. Skip RoPE buffers (not needed in state dict for most loaders)
        if any(x in key for x in ["cos", "sin", "mask"]):
            continue
            
        new_state_dict[new_key] = val
        
    # Save
    output_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving converted model to {output_path}...")
    save_file(new_state_dict, output_path, metadata={"format": "pt"})
    print("✅ Conversion Successful!")

def upload_to_huggingface(local_dir, repo_id, token=None):
    """
    Uploads the model folder to Hugging Face Hub.
    """
    print(f"Uploading {local_dir} to {repo_id}...")
    
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model", token=token)
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print("✅ Upload Complete!")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
