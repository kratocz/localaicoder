"""HuggingFace LLM provider."""

from typing import Optional
from .base import LLMProvider
from ..utils.device_detection import detect_optimal_device, TORCH_AVAILABLE, HUGGINGFACE_AVAILABLE
from ..utils.model_management import check_model_exists_locally, prompt_model_download

try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    HuggingFacePipeline = None


class HuggingFaceProvider(LLMProvider):
    """HuggingFace local LLM provider."""
    
    def __init__(self, model_id: str, device: Optional[str] = None, cli = None):
        self.model_id = model_id
        self.device = device or detect_optimal_device()
        self.cli = cli
    
    def get_llm(self) -> HuggingFacePipeline:
        if not self.is_available():
            raise RuntimeError("HuggingFace provider not available. Install: pip install langchain-huggingface torch transformers")
        
        # Check if model exists locally
        if not check_model_exists_locally(self.model_id):
            if self.cli:
                self.cli.print_info(f"🔍 Checking for model: {self.model_id}")
                self.cli.print_warning("Model not found in local cache.")
                
                # Ask user for download confirmation
                if not prompt_model_download(self.model_id, self.cli):
                    raise RuntimeError(f"Model download cancelled by user. Cannot proceed with {self.model_id}")
            else:
                # Fallback to basic print if no CLI available
                print(f"\n🔍 Checking for model: {self.model_id}")
                print("Model not found in local cache.")
                raise RuntimeError(f"Model {self.model_id} not found locally and no CLI available for confirmation")
        else:
            if self.cli:
                self.cli.print_success(f"Model {self.model_id} found in local cache")
            else:
                print(f"✅ Model {self.model_id} found in local cache")
        
        # Configure device for PyTorch
        device_map = None
        if self.device == 'mps':
            device_map = {"": "mps"}
        elif self.device == 'cuda':
            device_map = "auto"
        else:
            device_map = {"": "cpu"}
        
        try:
            if self.cli:
                self.cli.print_info(f"🚀 Loading {self.model_id} on {self.device}...")
            else:
                print(f"🚀 Loading {self.model_id} on {self.device}...")
                
            return HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                device_map=device_map,
                model_kwargs={
                    "torch_dtype": "auto" if TORCH_AVAILABLE else None,
                    "low_cpu_mem_usage": True,
                },
                pipeline_kwargs={
                    "max_new_tokens": 512,
                    "do_sample": True,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            )
        except Exception as e:
            if self.cli:
                self.cli.print_error(f"Failed to load model {self.model_id}: {e}")
            else:
                print(f"❌ Failed to load model {self.model_id}: {e}")
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def is_available(self) -> bool:
        return HUGGINGFACE_AVAILABLE and TORCH_AVAILABLE