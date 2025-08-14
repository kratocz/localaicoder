"""Device detection utilities for optimal ML inference."""

import platform
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import HfApi
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def detect_optimal_device() -> str:
    """Auto-detect the best available device for ML inference.
    
    Returns:
        Device string: 'mps' (Metal), 'cuda', or 'cpu'
    """
    if TORCH_AVAILABLE:
        # Check for Apple Silicon Metal support
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        # Check for CUDA support
        elif torch.cuda.is_available():
            return 'cuda'
    
    # Fallback to CPU
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'detected_device': detect_optimal_device(),
        'torch_available': TORCH_AVAILABLE,
        'huggingface_available': HUGGINGFACE_AVAILABLE
    }
    
    if TORCH_AVAILABLE:
        info.update({
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        })
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
    
    return info