import os
import platform
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Optional

from rich.console import Console
from rich.prompt import Confirm
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

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

# Type alias - more specific structure
class PathsDict(TypedDict):
    directories: List[str]
    files: List[str]

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console and CLI
console = Console()
global_cli = None  # Will be initialized in main()

class AdvancedCLI:
    """Advanced CLI with Rich formatting and Prompt Toolkit features."""
    
    def __init__(self):
        self.history = InMemoryHistory()
        
        # Create command completer with slash commands and common phrases
        commands = [
            '/help', '/exit', '/quit', '/clear', '/model', '/config', '/device',
            'What files are in this project?',
            'Read the README.md file', 
            'Analyze main.py for bugs',
            'Create a new Python file',
            'List project files',
            'Show me the code structure',
            'Fix any errors',
            'Run tests'
        ]
        self.completer = WordCompleter(commands, ignore_case=True)
        
    def get_user_input(self, prompt_text: str = "🤖 > ") -> str:
        """Get user input with history, auto-suggest, and completion."""
        try:
            user_input = prompt(
                prompt_text,
                history=self.history,
                # Disable auto-suggest to prevent visual conflicts with completion
                # auto_suggest=AutoSuggestFromHistory(),
                completer=self.completer,
                complete_while_typing=False,  # Only complete on Tab press
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C and Ctrl+D gracefully
            raise
        except OSError:
            # Fallback to basic input for non-interactive terminals
            console.print(prompt_text, end="")
            try:
                return input().strip()
            except (KeyboardInterrupt, EOFError):
                raise
    
    def confirm_action(self, message: str, default: bool = False) -> bool:
        """Ask for user confirmation with Rich styling."""
        try:
            return Confirm.ask(f"[bold yellow]{message}[/bold yellow]", default=default)
        except OSError:
            # Fallback for non-interactive terminals
            console.print(f"[bold yellow]{message}[/bold yellow] (y/n): ", end="")
            try:
                response = input().lower().strip()
                return response in ['y', 'yes'] if response else default
            except (KeyboardInterrupt, EOFError):
                raise
    
    def show_diff_confirmation(self, file_path: str, diff_content: str) -> bool:
        """Show diff and ask for confirmation with Rich formatting."""
        # Create a panel for the diff
        panel = Panel(
            diff_content,
            title=f"[bold]Proposed Changes to: {file_path}[/bold]",
            border_style="yellow",
            expand=False
        )
        console.print(panel)
        
        return self.confirm_action(f"Apply these changes to {file_path}?")
    
    def show_welcome_message(self):
        """Display welcome message with Rich formatting."""
        welcome_text = Text()
        welcome_text.append("🤖 Local AI Coder - Multi-Agent System\n", style="bold blue")
        welcome_text.append("by robot_dreams course\n", style="dim")
        
        panel = Panel(
            welcome_text,
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
    
    def show_help(self):
        """Display help information with Rich formatting."""
        # Create help table
        table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        
        # Add slash commands
        table.add_row("/help, /?", "Show this help message")
        table.add_row("/exit, /quit", "Exit the application")
        table.add_row("/clear", "Clear the conversation memory")
        table.add_row("/model", "Show current model information")
        table.add_row("/config", "Show current configuration")
        table.add_row("/device", "Show device and hardware information")
        table.add_row("Ctrl+D", "Exit the application (EOF)")
        
        console.print(table)
        
        # Add usage examples
        examples_panel = Panel(
            "[bold]Usage Examples:[/bold]\n"
            "• What files are in this project?\n"
            "• Read the README.md file\n"
            "• Create a new Python file with hello world\n"
            "• Analyze main.py and suggest improvements",
            title="Examples",
            border_style="green"
        )
        console.print(examples_panel)
    
    def show_config_info(self, config_data: Dict[str, str]):
        """Display configuration with Rich formatting."""
        table = Table(title="Configuration", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        
        for key, value in config_data.items():
            # Mask API keys
            if 'key' in key.lower() and value:
                value = '***set***'
            table.add_row(key, str(value))
        
        console.print(table)
    
    def show_device_info(self, device_data: Dict[str, Any]):
        """Display device information with Rich formatting."""
        table = Table(title="Device Information", show_header=True, header_style="bold green")
        table.add_column("Component", style="yellow", no_wrap=True)
        table.add_column("Status", style="white")
        
        for key, value in device_data.items():
            # Convert boolean values to checkmarks
            if isinstance(value, bool):
                value = "✅" if value else "❌"
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
    
    def print_info(self, message: str, style: str = "blue"):
        """Print info message with Rich styling."""
        console.print(f"[{style}]{message}[/{style}]")
    
    def print_error(self, message: str):
        """Print error message with Rich styling."""
        console.print(f"[bold red]❌ {message}[/bold red]")
    
    def print_success(self, message: str):
        """Print success message with Rich styling."""
        console.print(f"[bold green]✅ {message}[/bold green]")
    
    def print_warning(self, message: str):
        """Print warning message with Rich styling."""
        console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")

# Device Detection Functions
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

# Model Management Functions
def check_model_exists_locally(model_id: str) -> bool:
    """Check if HuggingFace model is already downloaded locally."""
    try:
        from transformers import AutoTokenizer
        
        # Try to load tokenizer from cache - this will fail if model isn't downloaded
        AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        return True
    except Exception:
        return False

def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model information from HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
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

def prompt_model_download(model_id: str, cli: AdvancedCLI) -> bool:
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

# LLM Provider Abstraction
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """Get configured LLM instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
    
    def get_llm(self) -> ChatOllama:
        return ChatOllama(
            model=self.model, 
            temperature=0, 
            base_url=self.base_url
        )
    
    def is_available(self) -> bool:
        return True  # Always available if ollama package is installed

class HuggingFaceProvider(LLMProvider):
    """HuggingFace local LLM provider."""
    
    def __init__(self, model_id: str, device: Optional[str] = None, cli: Optional[AdvancedCLI] = None):
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

# Function Implementations using LangChain tool decorator
@tool
def list_paths_recursive(directory: str = "", exclude_dirs: List[str] = None, include_hidden: bool = False) -> PathsDict:
    """Get a list of all files and directories recursively in a directory.
    
    Args:
        directory: Directory to scan (defaults to current working directory)
        exclude_dirs: List of directory names to exclude from scanning
        include_hidden: Whether to include hidden directories/files starting with '.' (default: False)
    
    Returns:
        Dictionary with 'directories' and 'files' keys containing sorted lists of paths
    """
    # Handle empty directory string
    if not directory or directory.strip() == "":
        directory = os.getcwd()
    
    # Check if directory exists
    if not os.path.exists(directory):
        return PathsDict(
            directories=[],
            files=[]
        )
    
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'node_modules']
    
    all_dirs = []
    all_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            # Filter directories: exclude specified dirs and optionally hidden dirs
            dirs[:] = [d for d in dirs if (
                d not in exclude_dirs and 
                (include_hidden or not d.startswith('.'))
            )]
            
            # Add directories (only non-excluded ones)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Make path relative to the starting directory
                relative_path = os.path.relpath(dir_path, directory)
                all_dirs.append(relative_path)

            # Add files (filter hidden files if needed)
            for file_name in files:
                # Skip hidden files unless include_hidden is True
                if not include_hidden and file_name.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file_name)
                # Make path relative to the starting directory
                relative_path = os.path.relpath(file_path, directory)
                all_files.append(relative_path)
                
    except (OSError, PermissionError) as e:
        # Return empty result if we can't read the directory
        return PathsDict(
            directories=[],
            files=[]
        )
    
    return PathsDict(
        directories=sorted(all_dirs),
        files=sorted(all_files)
    )

@tool
def read_file(file_path: str, max_lines: int = 100) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (default: 100)
    
    Returns:
        File contents as a string, or error message if file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:max_lines]
            content = ''.join(lines)
            if len(lines) == max_lines:
                content += f"\n... (file truncated, showing first {max_lines} lines)"
            return content
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

@tool  
def write_file(file_path: str, content: str) -> str:
    """Write content to a file with diff preview and user confirmation.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Success or error message
    """
    import difflib
    
    try:
        # Check if file exists and read current content
        existing_content = ""
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            except Exception as e:
                return f"Error reading existing file {file_path}: {str(e)}"
            
            # If content is the same, no need to write
            if existing_content == content:
                return f"No changes needed - file {file_path} already has the same content"
            
            # Generate and display diff
            diff = difflib.unified_diff(
                existing_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"{file_path} (current)",
                tofile=f"{file_path} (proposed)",
                lineterm=""
            )
            
            diff_lines = list(diff)
            if not diff_lines:
                return f"No changes detected in {file_path}"
            
            # Format diff content for Rich display
            diff_content = ""
            for line in diff_lines:
                if line.startswith('---') or line.startswith('+++'):
                    diff_content += f"[bold]{line.rstrip()}[/bold]\n"
                elif line.startswith('@@'):
                    diff_content += f"[cyan]{line.rstrip()}[/cyan]\n"
                elif line.startswith('+'):
                    diff_content += f"[green]{line.rstrip()}[/green]\n"
                elif line.startswith('-'):
                    diff_content += f"[red]{line.rstrip()}[/red]\n"
                else:
                    diff_content += line.rstrip() + "\n"
            
            # Use advanced CLI if available, otherwise fallback
            if global_cli:
                if not global_cli.show_diff_confirmation(file_path, diff_content.rstrip()):
                    return f"Changes to {file_path} cancelled by user"
            else:
                # Fallback to basic confirmation
                print(f"\n{'='*60}")
                print(f"PROPOSED CHANGES TO: {file_path}")
                print(f"{'='*60}")
                print(diff_content)
                print(f"{'='*60}")
                
                while True:
                    user_input = input(f"Apply these changes to {file_path}? (y/n): ").lower().strip()
                    if user_input in ['y', 'yes']:
                        break
                    elif user_input in ['n', 'no']:
                        return f"Changes to {file_path} cancelled by user"
                    else:
                        print("Please enter 'y' for yes or 'n' for no")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        action = "updated" if file_exists else "created"
        return f"Successfully {action} {file_path} ({len(content)} characters)"
        
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"

@tool
def run_command(command: str, working_dir: str = ".") -> str:
    """Execute a shell command and return the output.
    
    Args:
        command: Shell command to execute
        working_dir: Working directory for the command (default: current directory)
    
    Returns:
        Command output or error message
    """
    import subprocess
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=working_dir,
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        
        return output
        
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command '{command}': {str(e)}"

# Define tools by category
file_tools = [list_paths_recursive, read_file, write_file]
command_tools = [run_command]
all_tools = file_tools + command_tools

# Agent roles and their specialized tools
class AgentRole:
    COORDINATOR = "coordinator"
    TASK_PLANNER = "task_planner"
    CODE_ANALYZER = "code_analyzer"
    FILE_MANAGER = "file_manager"
    COMMAND_EXECUTOR = "command_executor"


class MultiAgentCoder:
    """Multi-agent AI Coder system using LangGraph with multiple LLM providers."""

    def __init__(self, 
                 llm_provider: str = os.getenv("LLM_PROVIDER", "huggingface"),
                 model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
                 base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                 device: Optional[str] = os.getenv("HF_DEVICE"),
                 cli: Optional[AdvancedCLI] = None):
        
        # Initialize LLM provider based on configuration
        self.llm_provider_name = llm_provider.lower()
        self.device_info = get_device_info()
        self.cli = cli
        
        # Validate provider name
        valid_providers = ["huggingface", "ollama"]
        if self.llm_provider_name not in valid_providers:
            raise ValueError(f"Unknown LLM_PROVIDER: '{llm_provider}'. Valid options are: {', '.join(valid_providers)}")
        
        if self.llm_provider_name == "huggingface":
            # Use HuggingFace provider
            hf_model_id = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b")
            self.provider = HuggingFaceProvider(model_id=hf_model_id, device=device, cli=cli)
            if not self.provider.is_available():
                raise RuntimeError("HuggingFace provider not available. Install required dependencies: pip install langchain-huggingface torch transformers")
        elif self.llm_provider_name == "ollama":
            # Use Ollama provider
            self.provider = OllamaProvider(model=model, base_url=base_url)
        
        # Get LLM instance
        self.llm = self.provider.get_llm()
        
        # Print configuration info
        if self.cli:
            self.cli.print_info("🤖 Multi-Agent Coder initialized:")
            self.cli.print_info(f"   Provider: {self.llm_provider_name}")
            if self.llm_provider_name == "ollama":
                self.cli.print_info(f"   Model: {model}")
                self.cli.print_info(f"   Ollama Server: {base_url}")
            else:
                self.cli.print_info(f"   Model: {getattr(self.provider, 'model_id', 'N/A')}")
                self.cli.print_info(f"   Device: {self.device_info['detected_device']} (auto-detected)")
        else:
            print(f"🤖 Multi-Agent Coder initialized:")
            print(f"   Provider: {self.llm_provider_name}")
            if self.llm_provider_name == "ollama":
                print(f"   Model: {model}")
                print(f"   Ollama Server: {base_url}")
            else:
                print(f"   Model: {getattr(self.provider, 'model_id', 'N/A')}")
                print(f"   Device: {self.device_info['detected_device']} (auto-detected)")
        
        # Create specialized agent LLMs with different prompts
        self.agents = self._create_agents()
        
        # Create tool nodes
        self.file_tool_node = ToolNode(file_tools)
        self.command_tool_node = ToolNode(command_tools)
        
        # Build the multi-agent graph
        self.graph = self._build_multi_agent_graph()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create specialized agents with different system prompts."""
        
        coordinator_prompt = """
You are the Coordinator agent. Your role is to:
1. Understand user requests and break them into tasks
2. Decide which specialized agents should handle each task
3. Coordinate between agents and synthesize final responses
4. Always respond with your reasoning and next steps
"""
        
        task_planner_prompt = """
You are the Task Planner agent. Your role is to:
1. Analyze complex programming tasks and break them into steps
2. Identify dependencies between tasks
3. Create actionable plans for other agents to execute
4. Prioritize tasks based on importance and complexity
"""
        
        code_analyzer_prompt = """
You are the Code Analyzer agent. Your role is to:
1. Read and analyze source code files
2. Identify bugs, code smells, and improvement opportunities
3. Suggest refactoring and optimization strategies
4. Explain code functionality and architecture
Use file tools to examine code when needed.
"""
        
        file_manager_prompt = """
You are the File Manager agent. Your role is to:
1. Navigate project directory structures
2. Read, create, and modify files safely
3. Organize project files and folders
4. Handle file operations and maintain project structure
Use file management tools exclusively.
"""
        
        command_executor_prompt = """
You are the Command Executor agent. Your role is to:
1. Execute shell commands safely and securely
2. Run tests, builds, and development tools
3. Handle git operations and version control
4. Monitor command outputs and report results
Use command execution tools only.
"""
        
        # Check if LLM supports bind_tools (for Ollama)
        if hasattr(self.llm, 'bind_tools'):
            return {
                AgentRole.COORDINATOR: self.llm.bind_tools([]),
                AgentRole.TASK_PLANNER: self.llm.bind_tools([]),
                AgentRole.CODE_ANALYZER: self.llm.bind_tools(file_tools),
                AgentRole.FILE_MANAGER: self.llm.bind_tools(file_tools),
                AgentRole.COMMAND_EXECUTOR: self.llm.bind_tools(command_tools),
            }
        else:
            # For HuggingFace models, we'll handle tools separately
            return {
                AgentRole.COORDINATOR: self.llm,
                AgentRole.TASK_PLANNER: self.llm,
                AgentRole.CODE_ANALYZER: self.llm,
                AgentRole.FILE_MANAGER: self.llm,
                AgentRole.COMMAND_EXECUTOR: self.llm,
            }
        
    def _build_multi_agent_graph(self):
        """Build the multi-agent LangGraph workflow."""
        
        def coordinator_node(state: MessagesState) -> Dict[str, Any]:
            """Coordinator agent decides which agent should handle the task."""
            messages = state['messages']
            
            # Add coordinator system message
            coordinator_messages = [
                SystemMessage(content="""
You are the Coordinator agent. Analyze the user's request and provide clear routing instructions.

For the user's request, respond with:
1. Brief analysis of what they want
2. Clear routing instruction using one of these phrases:
   - "Route to file_manager" - for file/directory operations (list, read, write files)
   - "Route to code_analyzer" - for code analysis, bug finding, reviews
   - "Route to command_executor" - for running commands, tests, builds
   - "Route to task_planner" - for complex multi-step planning

Example: "User wants to list project files. Route to file_manager to handle directory scanning."
""")
            ] + messages
            
            response = self.agents[AgentRole.COORDINATOR].invoke(coordinator_messages)
            return {"messages": [response]}
            
        def task_planner_node(state: MessagesState) -> Dict[str, Any]:
            """Task planner breaks down complex tasks."""
            messages = state['messages']
            
            planner_messages = [
                SystemMessage(content="""
You are the Task Planner. Break down the user's request into specific, actionable steps.
Create a detailed plan and then route to the appropriate specialized agent for execution.
""")
            ] + messages
            
            response = self.agents[AgentRole.TASK_PLANNER].invoke(planner_messages)
            return {"messages": [response]}
            
        def code_analyzer_node(state: MessagesState) -> Dict[str, Any]:
            """Code analyzer examines and analyzes code."""
            messages = state['messages']
            
            analyzer_messages = [
                SystemMessage(content="""
You are the Code Analyzer. Use file tools to examine code and provide detailed analysis.
Look for bugs, improvements, and explain functionality.
""")
            ] + messages
            
            response = self.agents[AgentRole.CODE_ANALYZER].invoke(analyzer_messages)
            return {"messages": [response]}
            
        def file_manager_node(state: MessagesState) -> Dict[str, Any]:
            """File manager handles file operations."""
            messages = state['messages']
            
            manager_messages = [
                SystemMessage(content="""
You are the File Manager agent. Your job is to handle file and directory operations using the available tools.

Available tools:
- list_paths_recursive: List files and directories in a project
- read_file: Read the contents of a specific file  
- write_file: Write content to a file (with diff preview)

For the user's request, analyze what they need and use the appropriate tool. 
If they want to list files/directories, use list_paths_recursive.
If they want to see file contents, use read_file.
If they want to modify files, use write_file.

Always use tools to complete the task - don't just describe what you would do.
""")
            ] + messages
            
            response = self.agents[AgentRole.FILE_MANAGER].invoke(manager_messages)
            return {"messages": [response]}
            
        def command_executor_node(state: MessagesState) -> Dict[str, Any]:
            """Command executor runs shell commands."""
            messages = state['messages']
            
            executor_messages = [
                SystemMessage(content="""
You are the Command Executor. Use command tools to execute shell commands safely.
Run tests, builds, git operations, and other development commands.
""")
            ] + messages
            
            response = self.agents[AgentRole.COMMAND_EXECUTOR].invoke(executor_messages)
            return {"messages": [response]}
            
        def route_decision(state: MessagesState) -> str:
            """Route to appropriate agent based on coordinator's decision."""
            messages = state['messages']
            last_message = messages[-1]
            
            # Enhanced routing logic
            content = last_message.content.lower()
            
            # Check for explicit routing instructions
            if 'file_manager' in content:
                return 'file_manager'
            elif 'code_analyzer' in content:
                return 'code_analyzer'
            elif 'command_executor' in content:
                return 'command_executor'
            elif 'task_planner' in content:
                return 'task_planner'
            elif 'end' in content or 'complete' in content or 'done' in content:
                return '__end__'
            
            # Infer from task content
            elif any(word in content for word in ['list', 'file', 'directory', 'folder', 'read', 'write']):
                return 'file_manager'
            elif any(word in content for word in ['analyze', 'check', 'review', 'bug', 'code']):
                return 'code_analyzer'
            elif any(word in content for word in ['run', 'execute', 'command', 'test', 'build']):
                return 'command_executor'
            elif any(word in content for word in ['plan', 'task', 'steps', 'strategy']):
                return 'task_planner'
            else:
                # Default to file_manager for file-related tasks
                return 'file_manager'
                
        def tool_router(state: MessagesState) -> str:
            """Route to appropriate tool node."""
            messages = state['messages']
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Check which tools are being called
                tool_names = []
                for tc in last_message.tool_calls:
                    if hasattr(tc, 'function'):
                        tool_names.append(tc.function.name)
                    elif isinstance(tc, dict) and 'function' in tc:
                        tool_names.append(tc['function']['name'])
                    
                if any(tool_name in ['run_command'] for tool_name in tool_names):
                    return 'command_tools'
                else:
                    return 'file_tools'
            return '__end__'
        
        # Create the graph
        workflow = StateGraph(MessagesState)
        
        # Add agent nodes
        workflow.add_node("coordinator", coordinator_node)
        workflow.add_node("task_planner", task_planner_node)
        workflow.add_node("code_analyzer", code_analyzer_node)
        workflow.add_node("file_manager", file_manager_node)
        workflow.add_node("command_executor", command_executor_node)
        
        # Add tool nodes
        workflow.add_node("file_tools", self.file_tool_node)
        workflow.add_node("command_tools", self.command_tool_node)
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        # Add conditional edges from coordinator
        workflow.add_conditional_edges(
            "coordinator",
            route_decision,
            {
                "task_planner": "task_planner",
                "code_analyzer": "code_analyzer", 
                "file_manager": "file_manager",
                "command_executor": "command_executor",
                "__end__": "__end__"
            }
        )
        
        # Add conditional edges from each agent to tools or end
        for agent in ["task_planner", "code_analyzer", "file_manager", "command_executor"]:
            workflow.add_conditional_edges(
                agent,
                tool_router,
                {
                    "file_tools": "file_tools",
                    "command_tools": "command_tools", 
                    "__end__": "__end__"
                }
            )
        
        # Add edges from tools back to coordinator
        workflow.add_edge("file_tools", "coordinator")
        workflow.add_edge("command_tools", "coordinator")
        
        # Compile the graph
        return workflow.compile(checkpointer=MemorySaver())
    
    def run(self, message: str) -> str:
        """Run the agent with a user message."""
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": "default"}}
        
        print(f"\n=== Processing: {message} ===")
        
        all_steps = []
        for step_output in self.graph.stream(initial_state, config):
            print(f"Step: {step_output}")
            all_steps.append(step_output)
        
        # Extract final response - look for the best non-empty response
        best_response = None
        best_node = None
        
        # Check steps in reverse order to find the most recent meaningful response
        for step in reversed(all_steps):
            for node_name, node_output in step.items():
                if 'messages' in node_output and node_output['messages']:
                    final_message = node_output['messages'][-1]
                    if hasattr(final_message, 'content') and final_message.content.strip():
                        content = final_message.content.strip()
                        # Prefer substantial responses over empty ones
                        if len(content) > 10:  # Prefer responses with meaningful content
                            print(f"\nFinal answer from {node_name}: {content}")
                            return content
                        elif not best_response:  # Keep as backup if no better response found
                            best_response = content
                            best_node = node_name
        
        # Fall back to best response found if any
        if best_response:
            print(f"\nFinal answer from {best_node}: {best_response}")
            return best_response
            
        return "Error: No response generated."


def main():
    """Interactive main function with command loop."""
    global global_cli
    
    # Initialize advanced CLI
    cli = AdvancedCLI()
    global_cli = cli  # Set global reference for tools
    
    # Show welcome message
    cli.show_welcome_message()
    
    # Create the Multi-Agent AI Coder system
    agent = MultiAgentCoder(cli=cli)
    
    # Show example usage
    cli.print_info("💡 Example usage:")
    example_prompt = "What directories and files are in the directory? Describe from their names the project."
    cli.print_info(f'   "{example_prompt}"')
    cli.print_info("📝 Type '/help' for commands, '/exit' or Ctrl+D to quit")
    
    # Interactive command loop
    while True:
        try:
            # Get user input with advanced CLI
            user_input = cli.get_user_input()
            
            # Handle empty input
            if not user_input:
                continue
                
            # Handle slash commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                
                if command in ['help', '?']:
                    cli.show_help()
                    continue
                    
                elif command in ['exit', 'quit']:
                    cli.print_success("👋 Goodbye! Thanks for using Local AI Coder!")
                    break
                    
                elif command == 'clear':
                    cli.print_success("🧹 Conversation memory cleared!")
                    # Recreate agent to clear memory
                    agent = MultiAgentCoder(cli=cli)
                    continue
                    
                elif command == 'model':
                    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    config_data = {
                        "Model": model,
                        "Server": base_url
                    }
                    cli.show_config_info(config_data)
                    continue
                    
                elif command == 'config':
                    config_data = {
                        "LLM Provider": os.getenv('LLM_PROVIDER', 'huggingface'),
                        "Ollama Model": os.getenv('OLLAMA_MODEL', 'gpt-oss:20b'),
                        "Ollama Server": os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                        "HF Model": os.getenv('HF_MODEL_ID', 'openai/gpt-oss-20b'),
                        "HF Device": os.getenv('HF_DEVICE', 'auto-detect'),
                        "API Key": os.getenv('OLLAMA_API_KEY', 'not set')
                    }
                    cli.show_config_info(config_data)
                    continue
                
                elif command == 'device':
                    device_info = get_device_info()
                    cli.show_device_info(device_info)
                    continue
                    
                else:
                    cli.print_error(f"Unknown command: /{command}")
                    cli.print_info("Type '/help' for available commands")
                    continue
            
            # Process normal user request
            console.print()  # Add spacing
            result = agent.run(user_input)
            console.print(f"\n" + "="*60)
            
        except KeyboardInterrupt:
            cli.print_warning("\n👋 Interrupted! Use '/exit' to quit properly.")
            continue
            
        except EOFError:
            # Handle Ctrl+D gracefully
            cli.print_success("\n👋 Goodbye! (Ctrl+D detected)")
            break
            
        except Exception as e:
            cli.print_error(f"Error: {e}")
            cli.print_info("Please try again or use '/help' for assistance")
            continue

if __name__ == "__main__":
    #print(list_paths_recursive())
    main()
