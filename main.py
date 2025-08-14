#!/usr/bin/env python3
"""
AI Agents Multi-Agent Coder

A sophisticated multi-agent AI coding assistant using LangGraph with support for
both Ollama and HuggingFace providers. Features advanced CLI interface with
command history, auto-completion, and rich formatting.
"""

import sys
from dotenv import load_dotenv
from src.cli.advanced_cli import AdvancedCLI
from src.agents.multi_agent_coder import MultiAgentCoder
from src.tools.file_tools import set_global_cli

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point for the AI Agents application."""
    try:
        # Initialize CLI
        cli = AdvancedCLI()
        
        # Set global CLI reference for tools
        set_global_cli(cli)
        
        # Display welcome message
        cli.display_welcome()
        
        # Initialize the multi-agent system
        coder = MultiAgentCoder(cli=cli)
        
        # Start interactive mode
        cli.start_interactive_mode(coder)
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()