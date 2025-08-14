# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2025-08-14

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

**Migration Path**: 
- v1.0.0 → v2.0.0: Complete architectural redesign to multi-agent system
- v2.0.0 → v2.1.0: Added local LLM support and enhanced provider abstraction
- v2.1.0 → v2.2.0: Environment variable renaming (MODEL → OLLAMA_MODEL)
- v2.2.0 → v2.3.0: Enhanced provider validation - ensure correct LLM_PROVIDER is set