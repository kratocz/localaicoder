# Local AI Coder

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
* **Configurable Ollama connection** - local or remote server support
* **Environment-based configuration** via `.env` file (model, server URL, API key)
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

2. **Start Ollama server** (if not running):
   ```bash
   ollama serve
   ```

3. **Configure settings** (optional):
   Edit the `.env` file to customize:
   ```bash
   # Model selection
   MODEL="your-preferred-model"  # Default: gpt-oss:20b
   
   # Ollama server address  
   OLLAMA_BASE_URL="http://localhost:11434"  # Default: localhost
   
   # For remote Ollama servers:
   # OLLAMA_BASE_URL="http://your-server:11434"
   # OLLAMA_API_KEY="your_api_key"  # If authentication required
   ```

4. **Run the agent**:
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

## Plans

Potential future enhancements:
- Git integration tools
- Code analysis and refactoring tools  
- Interactive chat mode
- Web search capabilities
- Database query tools

