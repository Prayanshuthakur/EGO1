import torch
from transformers import PreTrainedModel, PretrainedConfig
import sys
import os

# Ensure the parent directory is in the path to import Model.ego_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class EgoConfig(PretrainedConfig):
    model_type = "ego"

    def __init__(
        self,
        vocab_size=50257,
        emb_dim=768,
        n_layers=12,
        n_heads=12,
        context_length=1024,
        drop_rate=0.1,
        qkv_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_length = context_length
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias


class EgoForCausalLM(PreTrainedModel):
    config_class = EgoConfig

    def __init__(self, config):
        super().__init__(config)
        # Import inside to avoid circular imports if any, but ensure path is correct
        try:
            from Model.ego_model import EgoModel
        except ImportError:
            # Fallback if specific structure isn't exactly as expected or running from different cwd
            try:
                from ego_model import EgoModel
            except ImportError:
                 # Last resort: try to import from current directory if Model package not found
                import sys
                sys.path.append("Model")
                from ego_model import EgoModel

        self.model = EgoModel(vars(config))

    def forward(self, input_ids):
        return self.model(input_ids)

    def load_pt(self, path):
        print(f"Loading weights from {path}...")
        # Load state dict
        if torch.cuda.is_available():
            state_dict = torch.load(path) # Auto detect device
        else:
             state_dict = torch.load(path, map_location="cpu")
             
        # Check if it's a full checkpoint or just weights
        if "model_state_dict" in state_dict:
            print("Detected full checkpoint, extracting model_state_dict...")
            state_dict = state_dict["model_state_dict"]
            
        # Clean up state dict keys if needed (e.g. remove 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # Load into model
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys: {missing[:5]}...")


