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
            
            # Prepare model_kwargs with MPS/Metal compatibility
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,  # Required for some models like phi-2
            }
            
            # Handle torch_dtype for different devices
            if TORCH_AVAILABLE:
                import torch
                if self.device == 'mps':
                    # For Apple Silicon/MPS, use float32 for better numerical stability
                    model_kwargs["torch_dtype"] = torch.float32
                    # Additional MPS-specific settings for stability
                    model_kwargs["attn_implementation"] = "eager"  # Use eager attention for stability
                elif self.device == 'cuda':
                    model_kwargs["torch_dtype"] = torch.float16
                else:  # CPU
                    model_kwargs["torch_dtype"] = torch.float32
                
            return HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                device_map=device_map,
                model_kwargs=model_kwargs,
                pipeline_kwargs={
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "return_full_text": False,
                    "pad_token_id": 50256,  # Explicit pad token
                    "eos_token_id": 50256   # Explicit end token
                }
            )
        except Exception as e:
            error_msg = str(e)
            
            # Special handling for quantization errors on Apple Silicon
            if self.device == 'mps' and any(keyword in error_msg.lower() for keyword in ['mxfp4', 'quantized', 'gpu']):
                if self.cli:
                    self.cli.print_error(f"Model {self.model_id} is not compatible with Apple Silicon (MPS)")
                    self.cli.print_info("💡 For Apple Silicon, try using Ollama instead:")
                    self.cli.print_info("   Set LLM_PROVIDER='ollama' in your .env file")
                    self.cli.print_info("   Then run: ollama pull llama3.2:3b")
                else:
                    print(f"❌ Model {self.model_id} is not compatible with Apple Silicon (MPS)")
                    print("💡 For Apple Silicon, try using Ollama instead:")
                    print("   Set LLM_PROVIDER='ollama' in your .env file")
                    print("   Then run: ollama pull llama3.2:3b")
                raise RuntimeError(f"Model {self.model_id} requires GPU/CUDA but you're using Apple Silicon. Please use Ollama provider instead.")
            else:
                if self.cli:
                    self.cli.print_error(f"Failed to load model {self.model_id}: {e}")
                else:
                    print(f"❌ Failed to load model {self.model_id}: {e}")
                raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def is_available(self) -> bool:
        return HUGGINGFACE_AVAILABLE and TORCH_AVAILABLE