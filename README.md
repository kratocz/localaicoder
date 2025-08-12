# Local AI Coder

> **Author**: [Petr Kratochvíl](https://krato.cz) | **Developed with**: [Claude Code](https://claude.ai/code)

This is an AI assistant for programmers and coders. It's open-source and uses a locally running LLM of your choice. All your programming code, questions, etc. stay on your computer.

It's homework for the course [AI Agents](https://robotdreams.cz/course/567-ai-agents) by the company [robot_dreams](https://robotdreams.cz/). This course has been led by [Lukáš Kellerstein](https://www.linkedin.com/in/lukas-kellerstein/).

License: [MIT](LICENSE)

## Current Features

**🤖 Multi-Agent Architecture** with 5 specialized agents:

* **Coordinator Agent**: Analyzes requests and routes to appropriate specialists
* **Task Planner Agent**: Breaks down complex programming tasks into actionable steps  
* **Code Analyzer Agent**: Examines code for bugs, improvements, and explains functionality
* **File Manager Agent**: Handles all file operations (read, write, organize)
* **Command Executor Agent**: Safely runs shell commands, tests, and git operations

**🔧 Technical Features**:
* **LangGraph-based workflow** with intelligent agent routing
* **Multiple LLM providers** - Ollama (remote) or HuggingFace (local)
* **Automatic device detection** - CPU/CUDA/Metal (Apple Silicon) optimization
* **Smart model management** - checks local cache and prompts before downloads
* **Environment-based configuration** via `.env` file
* **Specialized tool sets** per agent for focused functionality
* **ReAct pattern**: Multi-agent Reason → Act → Observe loops
* **Persistent conversation memory** via MemorySaver
* **Dynamic agent coordination** - agents collaborate to solve complex tasks
* **Safe file operations** - `write_file` shows diff preview and requires user confirmation
* **Color-coded diff display** - easily see additions (green) and deletions (red)
* **Smart file filtering** - automatically ignores hidden files/directories (starting with `.`)

## Usage

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure LLM Provider** (optional):
   The system uses **HuggingFace** by default (fully local). Edit `.env` file to customize:
   
   **Option A: Ollama (requires external server) - DEFAULT**
   ```bash
   LLM_PROVIDER="ollama"
   MODEL="gpt-oss:20b"
   OLLAMA_BASE_URL="http://localhost:11434"
   ```
   Start Ollama server: `ollama serve`

   **Option B: HuggingFace (fully local, no external dependencies)**
   ```bash
   LLM_PROVIDER="huggingface" 
   HF_MODEL_ID="openai/gpt-oss-20b"
   # HF_DEVICE="mps"  # Auto-detects optimal device (CPU/CUDA/Metal)
   ```

   **Device Auto-Detection:**
   - 🍎 **Apple Silicon**: Automatically uses Metal Performance Shaders (MPS)
   - 🟢 **NVIDIA GPU**: Automatically uses CUDA acceleration  
   - 💻 **CPU Only**: Falls back to CPU inference

3. **Run the agent**:
   ```bash
   ./run.sh
   ```

   The application starts in **interactive mode** where you can:
   - Ask questions and give tasks to the AI agents
   - Use slash commands for special functions
   - Get continuous assistance in a conversational manner

## Interactive Mode

### Available Slash Commands:
- `/help` or `/?` - Show help and available commands
- `/exit` or `/quit` - Exit the application
- `/clear` - Clear conversation memory (restart agents)
- `/model` - Show current model information
- `/config` - Show current configuration
- `/device` - Show device and hardware information

### Usage Examples:
```bash
🤖 > What files are in this project?
🤖 > Read the README.md file
🤖 > Create a new Python file with a hello world function
🤖 > Analyze main.py for potential improvements
🤖 > /help
🤖 > /exit
```

### Features:
- **Conversation continuity** - agents remember context across requests
- **Error handling** - graceful handling of interrupts and errors
- **Memory management** - use `/clear` to reset conversation history
- **Configuration display** - check current model and server settings

## Multi-Agent Architecture

```
User Request → Coordinator → Specialized Agent → Tools → Back to Coordinator → Result
```

**Flow Details**:

1. **Coordinator** receives user request and analyzes task type
2. **Routes** to appropriate specialized agent:
   - Complex planning → Task Planner
   - Code analysis → Code Analyzer  
   - File operations → File Manager
   - Commands/builds → Command Executor
3. **Specialized agent** executes using domain-specific tools
4. **Results flow back** through Coordinator for synthesis
5. **Memory persisted** across the entire conversation

**LangGraph Nodes**:
- 5 Agent nodes (coordinator + 4 specialists)
- 2 Tool nodes (file tools + command tools)
- Conditional routing based on task analysis
- Persistent state management

### Complex Task Example

For complex tasks, the **Coordinator is called multiple times** to orchestrate the workflow. Here's how a request like *"Analyze this project, find all Python files, check them for errors, and create a summary report"* would be processed:

**Execution Flow:**
1. **1st Coordinator call**: "Need to find Python files" → routes to `File Manager`
2. **File Manager** → calls `list_paths_recursive` → returns to **2nd Coordinator call**
3. **2nd Coordinator call**: "Now analyze the found files" → routes to `Code Analyzer`
4. **Code Analyzer** → calls `read_file` on first Python file → returns to **3rd Coordinator call**  
5. **3rd Coordinator call**: "Continue analyzing next file" → routes to `Code Analyzer`
6. **Code Analyzer** → calls `read_file` on second Python file → returns to **4th Coordinator call**
7. **4th Coordinator call**: "Create summary report" → routes to `File Manager`
8. **File Manager** → calls `write_file` to create report → returns to **5th Coordinator call**
9. **5th Coordinator call**: "Task completed" → `__END__`

This **cyclic workflow** enables the system to:
- **Adapt dynamically** based on intermediate results
- **Coordinate multiple agents** for complex multi-step tasks  
- **Maintain context** throughout the entire process
- **Handle errors gracefully** by adjusting the strategy mid-execution

## LLM Provider Options

### Ollama Provider (External Server)
- **Pros**: Access to latest models, external GPU servers, model switching
- **Cons**: Requires Ollama server installation and setup
- **Best for**: Users with powerful hardware or access to remote Ollama servers

### HuggingFace Provider (Fully Local)
- **Pros**: No external dependencies, automatic device optimization, offline capable
- **Cons**: Limited to HuggingFace Hub models, initial download required
- **Best for**: Users wanting complete local deployment without external services
- **Smart Downloads**: Checks for locally cached models and prompts before downloading new ones

### Device Optimization
- **Auto-detection**: Automatically selects optimal device (CPU/CUDA/Metal)
- **Apple Silicon**: Leverages Metal Performance Shaders for M1/M2/M3 Macs
- **NVIDIA GPUs**: Uses CUDA acceleration for faster inference
- **CPU Fallback**: Works on any system with reasonable performance

### Recommended Models

**HuggingFace (Small & Fast):**
- `microsoft/DialoGPT-small` - Lightweight conversational model
- `distilgpt2` - Compact general-purpose model
- `gpt2` - Classic GPT-2 for basic tasks

**HuggingFace (Better Quality):**
- `openai/gpt-oss-20b` - Large model with excellent quality (default)
- `microsoft/DialoGPT-medium` - Balanced performance/quality
- `microsoft/DialoGPT-large` - Higher quality responses

**Ollama (External):**
- `llama3.1:8b` - Meta's latest 8B parameter model
- `mistral:7b` - Mistral 7B model
- `codellama:13b` - Code-specialized model

## File Management Features

The **File Manager Agent** provides intelligent file operations:

* **`list_paths_recursive`**: Scans project directories with smart filtering
  - Automatically ignores hidden files/directories (starting with `.`)
  - Excludes common build directories (`__pycache__`, `node_modules`)
  - Optional `include_hidden` parameter for full visibility

* **`read_file`**: Safe file reading with content limits
  - Configurable line limits to prevent overwhelming output
  - Error handling for unreadable files

* **`write_file`**: Protected file writing with diff preview
  - Shows color-coded differences before applying changes
  - Requires user confirmation for all modifications
  - Creates directories automatically as needed

## Credits

**Author**: [Petr Kratochvíl](https://krato.cz)  
**Development**: This project was developed with the assistance of [Claude Code](https://claude.ai/code) - Anthropic's AI-powered development environment.

The multi-agent architecture, LangGraph implementation, interactive CLI, and advanced features were collaboratively designed and implemented through AI-assisted development.

## Plans

Potential future enhancements:
- Git integration tools
- Code analysis and refactoring tools  
- Web search capabilities
- Database query tools

