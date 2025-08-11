import os
import platform
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict, Optional

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.base import BaseLanguageModel
from langgraph.graph import StateGraph, MessagesState, START, END
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

def prompt_model_download(model_id: str) -> bool:
    """Ask user for confirmation before downloading a model."""
    print(f"\n📦 Model Download Required")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    
    # Get model info
    model_info = get_model_info(model_id)
    if model_info:
        print(f"Downloads: {model_info.get('downloads', 'Unknown'):,}")
        if model_info.get('size_gb'):
            print(f"Estimated size: ~{model_info['size_gb']} GB")
        if model_info.get('tags'):
            print(f"Tags: {', '.join(model_info['tags'][:5])}")  # Show first 5 tags
    
    print(f"{'='*60}")
    print("This model needs to be downloaded from HuggingFace Hub.")
    print("The download may take several minutes depending on model size and connection speed.")
    print()
    
    while True:
        user_input = input("Do you want to download this model? (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            print("✅ Download confirmed. Starting model download...")
            return True
        elif user_input in ['n', 'no']:
            print("❌ Download cancelled by user.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no")

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
    
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or detect_optimal_device()
    
    def get_llm(self) -> HuggingFacePipeline:
        if not self.is_available():
            raise RuntimeError("HuggingFace provider not available. Install: pip install langchain-huggingface torch transformers")
        
        # Check if model exists locally
        if not check_model_exists_locally(self.model_id):
            print(f"\n🔍 Checking for model: {self.model_id}")
            print("Model not found in local cache.")
            
            # Ask user for download confirmation
            if not prompt_model_download(self.model_id):
                raise RuntimeError(f"Model download cancelled by user. Cannot proceed with {self.model_id}")
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
            print(f"❌ Failed to load model {self.model_id}: {e}")
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def is_available(self) -> bool:
        return HUGGINGFACE_AVAILABLE and TORCH_AVAILABLE

# Function Implementations using LangChain tool decorator
@tool
def list_paths_recursive(directory: str = os.getcwd(), exclude_dirs: List[str] = None, include_hidden: bool = False) -> PathsDict:
    """Get a list of all files and directories recursively in a directory.
    
    Args:
        directory: Directory to scan (defaults to current working directory)
        exclude_dirs: List of directory names to exclude from scanning
        include_hidden: Whether to include hidden directories/files starting with '.' (default: False)
    
    Returns:
        Dictionary with 'directories' and 'files' keys containing sorted lists of paths
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'node_modules']
    
    all_dirs = []
    all_files = []
    
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
            print(f"\n{'='*60}")
            print(f"PROPOSED CHANGES TO: {file_path}")
            print(f"{'='*60}")
            
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
            
            # Print diff with colors
            for line in diff_lines:
                if line.startswith('---') or line.startswith('+++'):
                    print(f"\033[1m{line.rstrip()}\033[0m")  # Bold
                elif line.startswith('@@'):
                    print(f"\033[36m{line.rstrip()}\033[0m")  # Cyan
                elif line.startswith('+'):
                    print(f"\033[32m{line.rstrip()}\033[0m")  # Green
                elif line.startswith('-'):
                    print(f"\033[31m{line.rstrip()}\033[0m")  # Red
                else:
                    print(line.rstrip())
            
            print(f"{'='*60}")
            
            # Ask for user confirmation
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
                 model: str = os.getenv("MODEL", "gpt-oss:20b"),
                 base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                 device: Optional[str] = os.getenv("HF_DEVICE")):
        
        # Initialize LLM provider based on configuration
        self.llm_provider_name = llm_provider.lower()
        self.device_info = get_device_info()
        
        if self.llm_provider_name == "huggingface":
            # Use HuggingFace provider
            hf_model_id = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b")
            self.provider = HuggingFaceProvider(model_id=hf_model_id, device=device)
            if not self.provider.is_available():
                print("⚠️  HuggingFace provider not available, falling back to Ollama")
                self.llm_provider_name = "ollama"
                self.provider = OllamaProvider(model=model, base_url=base_url)
        else:
            # Use Ollama provider (default)
            self.provider = OllamaProvider(model=model, base_url=base_url)
        
        # Get LLM instance
        self.llm = self.provider.get_llm()
        
        # Print configuration info
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


def print_help():
    """Display help information about available commands."""
    print("""
╭─────────────────────────────────────────────────────────────╮
│                    Local AI Coder - Help                    │
╰─────────────────────────────────────────────────────────────╯

Available Commands:
  /help, /?     - Show this help message
  /exit, /quit  - Exit the application
  /clear        - Clear the conversation memory
  /model        - Show current model information
  /config       - Show current configuration
  /device       - Show device and hardware information
  
Usage:
  - Type any question or request for the AI agents
  - Use file operations: "read file.py", "list project files"
  - Ask for code analysis: "analyze main.py for bugs"
  - Request help: "how to implement authentication"
  
Examples:
  → "What files are in this project?"
  → "Read the README.md file"
  → "Create a new Python file with hello world"
  → "Analyze main.py and suggest improvements"
  
The multi-agent system will automatically route your request
to the appropriate specialist agent (File Manager, Code Analyzer, etc.)
""")

def main():
    """Interactive main function with command loop."""
    print("""
╭─────────────────────────────────────────────────────────────╮
│          🤖 Local AI Coder - Multi-Agent System             │
│                   by robot_dreams course                    │
╰─────────────────────────────────────────────────────────────╯
""")
    
    # Create the Multi-Agent AI Coder system
    agent = MultiAgentCoder()
    
    # Show example usage
    print("\n💡 Example usage:")
    example_prompt = "What directories and files are in the directory? Describe from their names the project."
    print(f'   "{example_prompt}"')
    print("\n📝 Type '/help' for more commands, '/exit' to quit\n")
    
    # Interactive command loop
    while True:
        try:
            # Get user input
            user_input = input("🤖 > ").strip()
            
            # Handle empty input
            if not user_input:
                continue
                
            # Handle slash commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                
                if command in ['help', '?']:
                    print_help()
                    continue
                    
                elif command in ['exit', 'quit']:
                    print("\n👋 Goodbye! Thanks for using Local AI Coder!")
                    break
                    
                elif command == 'clear':
                    print("\n🧹 Conversation memory cleared!")
                    # Recreate agent to clear memory
                    agent = MultiAgentCoder()
                    continue
                    
                elif command == 'model':
                    model = os.getenv("MODEL", "gpt-oss:20b")
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    print(f"\n📊 Current configuration:")
                    print(f"   Model: {model}")
                    print(f"   Server: {base_url}")
                    continue
                    
                elif command == 'config':
                    print(f"\n⚙️  Configuration:")
                    print(f"   LLM Provider: {os.getenv('LLM_PROVIDER', 'huggingface')}")
                    print(f"   Model: {os.getenv('MODEL', 'gpt-oss:20b')}")
                    print(f"   Ollama Server: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
                    print(f"   HF Model: {os.getenv('HF_MODEL_ID', 'openai/gpt-oss-20b')}")
                    print(f"   HF Device: {os.getenv('HF_DEVICE', 'auto-detect')}")
                    api_key = os.getenv('OLLAMA_API_KEY')
                    print(f"   API Key: {'***set***' if api_key else 'not set'}")
                    continue
                
                elif command == 'device':
                    device_info = get_device_info()
                    print(f"\n🖥️  Device Information:")
                    print(f"   Platform: {device_info['platform']} ({device_info['machine']})")
                    print(f"   Detected Device: {device_info['detected_device']}")
                    print(f"   PyTorch Available: {'✅' if device_info['torch_available'] else '❌'}")
                    print(f"   HuggingFace Available: {'✅' if device_info['huggingface_available'] else '❌'}")
                    
                    if device_info['torch_available']:
                        print(f"   PyTorch Version: {device_info.get('torch_version', 'N/A')}")
                        print(f"   CUDA Available: {'✅' if device_info.get('cuda_available', False) else '❌'}")
                        print(f"   Metal (MPS) Available: {'✅' if device_info.get('mps_available', False) else '❌'}")
                        
                        if device_info.get('cuda_available', False):
                            print(f"   CUDA Devices: {device_info.get('cuda_device_count', 0)}")
                            if device_info.get('cuda_device_name'):
                                print(f"   CUDA Device Name: {device_info['cuda_device_name']}")
                    continue
                    
                else:
                    print(f"❌ Unknown command: /{command}")
                    print("   Type '/help' for available commands")
                    continue
            
            # Process normal user request
            print()  # Add spacing
            result = agent.run(user_input)
            print(f"\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted! Use '/exit' to quit properly.")
            continue
            
        except EOFError:
            print("\n👋 Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again or use '/help' for assistance")
            continue

if __name__ == "__main__":
    #print(list_paths_recursive())
    main()
