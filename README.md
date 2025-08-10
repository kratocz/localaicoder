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
* **Connection to local Ollama server** (configurable model via MODEL env var)
* **Specialized tool sets** per agent for focused functionality
* **ReAct pattern**: Multi-agent Reason → Act → Observe loops
* **Persistent conversation memory** via MemorySaver
* **Dynamic agent coordination** - agents collaborate to solve complex tasks

## Usage

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Start Ollama server** (if not running):
   ```bash
   ollama serve
   ```

3. **Configure model** (optional):
   ```bash
   export MODEL="your-preferred-model"  # Default: gpt-oss:20b
   ```

4. **Run the agent**:
   ```bash
   ./run.sh
   ```

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

## Plans

Potential future enhancements:
- Git integration tools
- Code analysis and refactoring tools  
- Interactive chat mode
- Web search capabilities
- Database query tools

