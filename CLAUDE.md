# CLAUDE.md - Educational Project Cut-off Notice

## ⚠️ IMPORTANT - Claude Code Development Ended

**This file documents Claude Code assistance that ended on August 15, 2025.**

This project was developed as educational homework for the AI Agents course by robot_dreams. All development using Claude Code assistance ended on **August 15, 2025** to avoid potential conflicts with Anthropic's Claude Code product.

**Any future development of this project must proceed without Claude Code assistance.**

This project is intended EXCLUSIVELY for educational and learning purposes, not for commercial use or as a competitor to professional AI development tools.

---

## Original Development Context (Until August 15, 2025)

This file provided guidance to Claude Code (claude.ai/code) when working with code in this repository during the educational development phase.

## Development Commands

### Running the Application
```bash
# Run the interactive multi-agent system
./run.sh

# Alternative: Direct Python execution via uv
uv run main.py
```

### Package Management
```bash
# Install/sync dependencies
uv sync

# Check current Python version
cat .python-version
```

### Configuration
```bash
# Copy and edit environment configuration
cp .env.example .env
# Then edit .env to configure LLM provider settings
```

### Release Management
```bash
# Update CHANGELOG.md with new version details
# Edit main.py if version number needs updating
# Commit changes and create git tag
git add CHANGELOG.md [other-files]
git commit -m "feat: Description of changes vX.X.X"
git tag vX.X.X
git push && git push --tags
```

**Release Process:**
1. **Determine version number** using Semantic Versioning (semver.org):
   - **MAJOR** (X.0.0): Breaking changes that break backward compatibility
   - **MINOR** (x.X.0): New features that maintain backward compatibility  
   - **PATCH** (x.x.X): Bug fixes and small improvements without new features
2. Update CHANGELOG.md with new version section:
   - Use current Czech Republic date (format: YYYY-MM-DD, e.g., "2025-08-15")
   - Include sections: Changed, Added, Fixed, Technical Details as appropriate
3. Update version number in pyproject.toml to match new version
4. Create commit with conventional commit message format
5. Create git tag matching version number (e.g., v2.3.0)
6. Push both commits and tags to remote

**Semantic Versioning Examples:**
- v2.3.0 → v3.0.0: Major changes (removed LLM provider fallback = breaking change)
- v2.3.0 → v2.4.0: Minor changes (added new agent type or tool)
- v2.3.0 → v2.3.1: Patch changes (fixed bug in file reading, improved error messages)

## Architecture Overview

This is a **multi-agent AI coding assistant** built with LangGraph that orchestrates 5 specialized agents to handle different programming tasks. The system supports both local (HuggingFace) and remote (Ollama) LLM providers.

### Core Components

**MultiAgentCoder** (main.py:442) - Central orchestrator class that:
- Manages LLM provider selection and validation (strict validation, no fallbacks)
- Creates and coordinates 5 specialized agents using LangGraph StateGraph
- Handles conversation memory and persistent state
- Routes requests through conditional agent workflows

**LLM Provider Abstraction** (main.py:154-238):
- `LLMProvider` abstract base class
- `OllamaProvider` for external Ollama servers 
- `HuggingFaceProvider` for fully local inference with device auto-detection
- Provider validation throws `ValueError` for unknown providers (no silent fallbacks)

### Multi-Agent Architecture

**Agent Roles** (main.py:434-440):
1. **COORDINATOR** - Analyzes requests and routes to appropriate specialists
2. **TASK_PLANNER** - Breaks complex tasks into actionable steps
3. **CODE_ANALYZER** - Examines code files for bugs and improvements  
4. **FILE_MANAGER** - Handles file operations (read/write/list)
5. **COMMAND_EXECUTOR** - Safely executes shell commands

**Tool Distribution**:
- **file_tools**: `list_paths_recursive`, `read_file`, `write_file` (used by CODE_ANALYZER, FILE_MANAGER)
- **command_tools**: `run_command` (used by COMMAND_EXECUTOR)
- **Coordinator/Task Planner**: No direct tools (pure routing/planning)

### LangGraph Workflow

**Graph Structure** (main.py:555-749):
```
User Request → Coordinator → [Agent Selection] → Tool Execution → Back to Coordinator → Result
```

**Key Routing Logic**:
- Coordinator analyzes requests and provides explicit routing instructions
- Conditional edges route between agents based on task analysis
- Tool router determines file_tools vs command_tools usage
- Cyclic workflow allows multiple coordinator calls for complex tasks

**Agent Node Functions**:
- Each agent has a dedicated node function with specialized system prompts
- Tool calling capability varies by LLM provider (bind_tools for Ollama)
- Persistent conversation state via MemorySaver

### Configuration System

**Environment Variables** (.env.example):
- `LLM_PROVIDER`: "huggingface" (local) or "ollama" (remote) - **required and validated**
- `OLLAMA_MODEL`, `OLLAMA_BASE_URL`: Ollama configuration  
- `HF_MODEL_ID`, `HF_DEVICE`: HuggingFace configuration with device auto-detection

**Device Detection** (main.py:38-76):
- Auto-detects optimal device: Metal (Apple Silicon) > CUDA > CPU
- Device info accessible via `/device` command
- Supports CPU, CUDA, and Metal Performance Shaders (MPS)

### File Operations

**Safe File Writing** (main.py:310-391):
- `write_file` shows diff preview with color coding before changes
- Requires explicit user confirmation for all file modifications
- Creates directories automatically as needed

**Smart Directory Scanning** (main.py:241-286):
- `list_paths_recursive` excludes hidden files and common build directories by default
- Configurable exclusions and hidden file inclusion

### Interactive CLI Features

**Slash Commands** (main.py:855-913):
- `/help` - Show available commands and usage examples
- `/exit`, `/quit` - Clean application exit
- `/clear` - Reset conversation memory (recreates agents)
- `/model`, `/config` - Show current configuration 
- `/device` - Display hardware and device information

**Error Handling**:
- Graceful keyboard interrupt handling
- Configuration validation with clear error messages
- LLM provider availability checking with helpful guidance

## Version Information

Current version: **2.3.0** (see CHANGELOG.md for complete history)

**Recent Changes (v2.3.0)**:
- Strict LLM provider validation - no automatic fallbacks
- Enhanced error handling for configuration issues
- Clear error messages for unknown LLM_PROVIDER values

## Development Notes

- Single-file architecture (main.py ~936 lines) with clear separation of concerns
- Uses uv for Python package management with pyproject.toml configuration  
- MIT licensed with comprehensive README and changelog
- Dependencies: LangGraph, LangChain, Ollama, HuggingFace libraries, PyTorch
- Interactive mode is the primary interface - not designed for programmatic API usage