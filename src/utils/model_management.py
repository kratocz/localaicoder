"""Model management utilities for HuggingFace models."""

from typing import Optional, Dict, Any
from rich.panel import Panel
from rich.console import Console

console = Console()

try:
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


def check_model_exists_locally(model_id: str) -> bool:
    """Check if HuggingFace model is already downloaded locally."""
    if not HUGGINGFACE_AVAILABLE:
        return False
        
    try:
        # Try to load tokenizer from cache - this will fail if model isn't downloaded
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model information from HuggingFace Hub."""
    if not HUGGINGFACE_AVAILABLE:
        return None
        
    try:
        api = HfApi()
        model_info = api.model_info(model_id)
        
        # Extract relevant information
        size_gb = 0
        try:
            if hasattr(model_info, 'safetensors') and model_info.safetensors:
                if isinstance(model_info.safetensors, dict) and 'parameters' in model_info.safetensors:
                    for file_info in model_info.safetensors['parameters'].values():
                        if isinstance(file_info, dict) and 'size' in file_info:
                            size_gb += file_info['size']
                    size_gb = size_gb / (1024**3)
        except Exception:
            # If we can't get size info, that's OK
            pass
        
        return {
            'id': model_id,
            'downloads': getattr(model_info, 'downloads', 0),
            'size_gb': round(size_gb, 2) if size_gb > 0 else None,
            'tags': getattr(model_info, 'tags', []),
            'pipeline_tag': getattr(model_info, 'pipeline_tag', None),
            'library_name': getattr(model_info, 'library_name', None)
        }
    except Exception as e:
        print(f"Warning: Could not fetch model info for {model_id}: {e}")
        return None


def prompt_model_download(model_id: str, cli) -> bool:
    """Ask user for confirmation before downloading a model."""
    # Create model info display
    info_lines = [f"Model: {model_id}"]
    
    # Get model info
    model_info = get_model_info(model_id)
    if model_info:
        info_lines.append(f"Downloads: {model_info.get('downloads', 'Unknown'):,}")
        if model_info.get('size_gb'):
            info_lines.append(f"Estimated size: ~{model_info['size_gb']} GB")
        if model_info.get('tags'):
            info_lines.append(f"Tags: {', '.join(model_info['tags'][:5])}")  # Show first 5 tags
    
    info_lines.extend([
        "",
        "This model needs to be downloaded from HuggingFace Hub.",
        "The download may take several minutes depending on model size and connection speed."
    ])
    
    # Display model download panel
    panel = Panel(
        "\n".join(info_lines),
        title="📦 Model Download Required",
        border_style="yellow"
    )
    console.print(panel)
    
    # Ask for confirmation
    if cli.confirm_action("Do you want to download this model?"):
        cli.print_success("Download confirmed. Starting model download...")
        return True
    else:
        cli.print_error("Download cancelled by user.")
        return False