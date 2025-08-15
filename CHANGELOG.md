# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## ⚠️ IMPORTANT NOTICE - Educational Project Cut-off

**Claude Code Development End Date: August 15, 2025**

This project was developed as educational homework for the AI Agents course by robot_dreams. All development using Claude Code assistance ended on August 15, 2025. Any future development must proceed without Claude Code to avoid potential conflicts with Anthropic's Claude Code product.

This project is intended EXCLUSIVELY for educational and learning purposes, not for commercial use or as a competitor to professional AI development tools.

---

## [2.7.0] - 2025-08-15

### Fixed
- **Apple Silicon compatibility** - Fixed .env loading order to properly read LLM_PROVIDER setting
- **HuggingFace provider stability** - Enhanced numerical stability for models on Apple Silicon/MPS
- **Model compatibility** - Improved torch_dtype handling and added explicit token settings
- **Initialization order** - Fixed tool initialization before agent creation

### Changed
- **Welcome banner** - Updated attribution to "Petr Kratochvil (krato.cz)"
- **Model recommendations** - Added stable model alternatives for Apple Silicon (gpt2, distilgpt2)
- **Error handling** - Better error messages for quantized models incompatible with MPS
- **Model loading** - Added trust_remote_code and attn_implementation settings for stability

### Technical Details
- **MPS optimizations** - Use torch.float32 for better numerical stability on Apple Silicon
- **Pipeline settings** - Optimized temperature, top_p, top_k for better generation quality
- **Token handling** - Explicit pad_token_id and eos_token_id to prevent generation issues
- **Attention mechanism** - Forced eager attention implementation for MPS compatibility

### Migration Path
- **No breaking changes** - All existing functionality preserved
- **Model updates** - Recommended models: gpt2 (stable), distilgpt2 (smaller), avoid microsoft/phi-2 on MPS
- **Environment variables** - Same .env configuration, better loading order

## [2.6.0] - 2025-08-15

### Changed
- **Major code refactoring** - Restructured monolithic main.py (1108 lines) into modular architecture
- **Package structure** - Organized code into logical packages: agents/, cli/, providers/, tools/, utils/
- **Maintainability improvements** - Separated concerns for better code organization and testability
- **Import optimization** - Clean imports with proper `__init__.py` files for each package

### Technical Details
- **src/agents/** - Multi-agent system components (MultiAgentCoder, AgentRole)
- **src/cli/** - Advanced CLI interface with Rich formatting
- **src/providers/** - LLM provider abstraction (Ollama, HuggingFace)
- **src/tools/** - Agent tools for file operations and command execution
- **src/utils/** - Device detection and model management utilities
- **main.py** - Streamlined entry point (48 lines vs 1108 lines)

### Migration Path
- **No breaking changes** - All functionality preserved, same environment variables
- **Same commands** - All CLI commands and features work identically
- **Dependencies unchanged** - Same requirements, no additional installations needed

## [2.5.0] - 2025-08-15

### Added
- **Advanced CLI interface** - Rich formatting and Prompt Toolkit integration for professional user experience
- **Command history and auto-completion** - Arrow keys for history navigation and Tab completion for commands
- **Rich output formatting** - Beautiful tables, panels, and color-coded displays for better readability
- **Enhanced file operations** - Rich diff preview with syntax highlighting for file changes
- **Keyboard shortcuts** - Ctrl+D (EOF) support for elegant application exit
- **Interactive command help** - Comprehensive help system with usage examples and command descriptions

### Changed
- **User input system** - Replaced basic input() with advanced Prompt Toolkit for better UX
- **Configuration display** - Professional tables for /config and /device commands
- **Error messages** - Rich-formatted error and success messages with consistent styling
- **Welcome interface** - Elegant welcome panel with improved visual hierarchy

### Fixed
- **Tool parameter handling** - Fixed list_paths_recursive to properly handle empty directory strings
- **Input validation** - Robust error handling for non-interactive terminals with graceful fallbacks
- **Visual conflicts** - Disabled auto-suggest to prevent display issues with command typing

### Technical Details
- Added Rich and Prompt Toolkit dependencies for advanced CLI features
- Implemented AdvancedCLI class with history, completion, and formatting capabilities
- Enhanced MultiAgentCoder to use CLI for all user interactions
- Improved list_paths_recursive tool with better error handling and directory validation
- Added fallback mechanisms for non-interactive terminal environments

## [2.4.0] - 2025-08-15

### Added
- **CLAUDE.md documentation** - Comprehensive guidance for Claude Code when working with this repository
- **Development commands** - Running, building, configuration instructions for AI development assistance
- **Architecture documentation** - Multi-agent system overview, LangGraph workflow, and provider abstraction details
- **Release management process** - Semantic versioning guidelines and step-by-step release procedures

### Technical Details
- Created CLAUDE.md with development commands, architecture overview, and release procedures
- Documented multi-agent coordination patterns and LangGraph workflow structure
- Added semantic versioning guidelines (MAJOR.MINOR.PATCH) with examples
- Included Czech Republic date format requirements for changelog entries

## [2.3.0] - 2025-08-15

### Changed
- **Strict LLM provider validation** - Removed automatic fallback between providers
- **Enhanced error handling** - Clear error messages for unknown LLM_PROVIDER values
- **Provider isolation** - Each provider now fails gracefully without falling back to others

### Fixed
- **Configuration validation** - Application now exits with clear error when invalid LLM_PROVIDER is set
- **Fallback removal** - Eliminated silent fallback from HuggingFace to Ollama provider

### Technical Details
- Added provider validation in MultiAgentCoder.__init__()
- ValueError thrown for unknown LLM_PROVIDER with list of valid options
- RuntimeError thrown when HuggingFace dependencies are missing (instead of fallback)
- Improved error messages guide users to correct configuration

## [2.2.0] - 2025-08-12

### Changed
- **Environment variable naming** - Changed `MODEL` to `OLLAMA_MODEL` for better clarity and consistency
- **Default LLM provider** - Reverted default from "huggingface" back to "ollama" in .env.example
- **Configuration clarity** - Improved naming convention to distinguish Ollama-specific settings

### Technical Details
- Updated all references from `MODEL` to `OLLAMA_MODEL` in main.py
- Modified `/config` command output to show "Ollama Model" instead of generic "Model"
- Enhanced configuration display for better provider-specific identification

## [2.1.0] - 2025-08-12

### Added
- **HuggingFace LLM provider** - Fully local LLM support without external dependencies
- **Automatic device detection** - CPU/CUDA/Metal (Apple Silicon) optimization
- **Smart model management** - Model cache detection and download confirmation prompts
- **Provider abstraction layer** - Clean OOP design for multiple LLM providers
- **Enhanced configuration** - New .env options for LLM provider selection
- **`/device` command** - Display detailed hardware and device information
- **Accelerate dependency** - Support for advanced model loading optimizations

### Changed
- **Default LLM provider** changed from "ollama" to "huggingface" for better out-of-box experience
- **Enhanced .env.example** with detailed comments explaining each configuration option
- **Improved error handling** - Graceful fallback from HuggingFace to Ollama on availability issues
- **Updated documentation** - Comprehensive guides for both LLM providers

### Technical Details
- HuggingFacePipeline integration with automatic device mapping
- Model download prompts with size estimates and user consent
- Device auto-detection prioritizing Metal > CUDA > CPU
- Provider-specific tool binding with compatibility layer
- Enhanced dependency management (torch, transformers, huggingface-hub, accelerate)

### Fixed
- **Tool binding compatibility** - Different approaches for Ollama vs HuggingFace models
- **Model info extraction** - Robust parsing of HuggingFace model metadata
- **Import error handling** - Graceful degradation when optional dependencies missing

## [2.0.0] - 2025-08-11

### Added
- **Multi-agent architecture** with 5 specialized agents using LangGraph framework
- **Coordinator Agent** for request analysis and intelligent routing
- **Task Planner Agent** for complex task decomposition  
- **Code Analyzer Agent** with file analysis tools
- **File Manager Agent** with smart file operations
- **Command Executor Agent** for safe shell command execution
- **Interactive CLI mode** with continuous conversation loop
- **Slash commands** (`/help`, `/exit`, `/clear`, `/model`, `/config`)
- **Environment-based configuration** via `.env` file
- **Configurable Ollama server URL** and model selection
- **Diff preview with user confirmation** for file write operations
- **Color-coded diff display** (green additions, red deletions)
- **Smart file filtering** - automatically ignores hidden directories/files
- **Persistent conversation memory** using MemorySaver
- **Error handling and recovery** mechanisms
- **Comprehensive help system** and usage examples

### Changed
- **Complete rewrite** from single-agent to multi-agent architecture (160 → 662 lines)
- **Replaced OllamaReactAgent** with MultiAgentCoder class
- **Enhanced file operations** with safety checks and user interaction
- **Improved project documentation** with detailed architecture description
- **Updated dependencies** to include LangGraph, LangChain, and related packages

### Technical Details
- **LangGraph StateGraph** with MessagesState for workflow management
- **Conditional routing** between specialized agents based on task analysis
- **ReAct pattern implementation** (Reason → Act → Observe loops)
- **Tool segregation** - file tools and command tools for different agent types
- **Cyclic workflow** - agents can collaborate through coordinator orchestration

## [1.0.0] - 2025-07-10

### Added
- **Proof of Concept (POC)** implementation
- Basic **OllamaReactAgent** with simple tool integration
- **Core file operations** (list_paths_recursive, read_file, write_file, run_command)
- **Single-execution mode** - one request, one response
- **Basic Ollama integration** with ChatOllama
- **Project structure** and initial documentation
- **MIT license** and basic README

### Technical Details
- Simple React agent pattern with basic tool calling
- Hardcoded model configuration
- No conversation memory or state persistence
- Direct tool execution without agent specialization
- Single-threaded, synchronous processing

---

## Version Summary

- **v1.0.0**: Basic POC with single-agent React pattern
- **v2.0.0**: Production-ready multi-agent system with LangGraph architecture  
- **v2.1.0**: Multi-provider LLM support with local HuggingFace integration
- **v2.2.0**: Configuration improvements and environment variable consistency
- **v2.3.0**: Strict provider validation and removal of automatic fallbacks
- **v2.4.0**: Added comprehensive CLAUDE.md documentation for AI development assistance
- **v2.5.0**: Advanced CLI interface with Rich formatting and enhanced user experience

## Migration Path

- v1.0.0 → v2.0.0: Complete architectural redesign to multi-agent system
- v2.0.0 → v2.1.0: Added local LLM support and enhanced provider abstraction
- v2.1.0 → v2.2.0: Environment variable renaming (MODEL → OLLAMA_MODEL)
- v2.2.0 → v2.3.0: Enhanced provider validation - ensure correct LLM_PROVIDER is set
- v2.3.0 → v2.4.0: Added CLAUDE.md documentation for improved AI development assistance
- v2.4.0 → v2.5.0: Enhanced CLI with Rich formatting and advanced user interface features
