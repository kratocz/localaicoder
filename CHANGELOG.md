# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

**Migration Path**: v1.0.0 → v2.0.0 represents a complete architectural redesign from proof-of-concept to a sophisticated multi-agent system suitable for complex programming assistance tasks.