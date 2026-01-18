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
        # Handle 'module.' prefix if trained with DDP
        if key.startswith("module."):
            key = key[7:]
            
        new_key = key
        
        # 1. Embeddings
        if "token_embedding.weight" in key:
            new_key = "transformer.wte.weight"
        elif "position_embedding.weight" in key:
            new_key = "transformer.wpe.weight"
            
        # 2. Transformer Blocks (transformer_blocks -> h)
        elif key.startswith("transformer_blocks."):
            # transformer_blocks.0.self_attention.qkv_projection.weight -> transformer.h.0.attn.c_attn.weight
            
            # Remove the descriptive prefix
            inner_key = key.replace("transformer_blocks.", "")
            
            # Map sub-components
            inner_key = inner_key.replace("self_attention.", "attn.")
            inner_key = inner_key.replace("pre_attention_norm.", "ln_1.")
            inner_key = inner_key.replace("feed_forward.", "mlp.")
            inner_key = inner_key.replace("pre_ffn_norm.", "ln_2.")
            
            # Map Attention internal layers
            inner_key = inner_key.replace("qkv_projection.", "c_attn.")
            inner_key = inner_key.replace("output_projection.", "c_proj.")
            
            # Map MLP internal layers (Expansion: fc1 -> c_fc, Output: output_projection -> c_proj)
            inner_key = inner_key.replace("expansion_layer.", "c_fc.")
            # Note: MLP output projection is also named 'output_projection', same as attention
            # Since we replaced "self_attention." with "attn." earlier, 
            # attention keys are "attn.output_projection..." -> "attn.c_proj..."
            # MLP keys are "mlp.output_projection..." -> "mlp.c_proj..."
            # So a single replace works for both if the context prefix is already handled or if the name is unique.
            # However, to be safe and explicit:
            inner_key = inner_key.replace("output_projection.", "c_proj.")
            
            new_key = f"transformer.h.{inner_key}"
            
            # CRITICAL: Transpose Linear weights for Conv1D compatibility
            if "weight" in key and ("c_attn" in key or "c_proj" in key or "fc1" in key or "fc2" in key or "qkv_projection" in key or "output_projection" in key or "expansion_layer" in key):
                 val = val.t().contiguous()
        
        # 3. Final Layer Norm
        elif "final_layer_norm" in key:
            new_key = key.replace("final_layer_norm", "transformer.ln_f")
            
        # 4. LM Head (Shared Weights)
        elif "lm_head.weight" in key:
            new_key = "lm_head.weight"
            # Clone to break shared memory (required for safetensors)
            val = val.clone()
            
        # 5. Skip buffers like attn.mask
        if "attn.mask" in key:
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
