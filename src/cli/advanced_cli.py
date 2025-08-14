"""Advanced CLI with Rich formatting and Prompt Toolkit features."""

from typing import Dict, Any
from rich.console import Console
from rich.prompt import Confirm
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter

console = Console()


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
    
    def display_welcome(self):
        """Display welcome message."""
        self.show_welcome_message()
    
    def start_interactive_mode(self, agent):
        """Start the interactive command loop."""
        import os
        from ..utils.device_detection import get_device_info
        
        # Show example usage
        console.print("\n💡 Example usage:")
        example_prompt = "What directories and files are in the directory? Describe from their names the project."
        console.print(f'   "{example_prompt}"')
        console.print("\n📝 Type '/help' for more commands, '/exit' to quit\n")
        
        # Interactive command loop
        while True:
            try:
                # Get user input
                user_input = self.get_user_input("🤖 > ").strip()
                
                # Handle empty input
                if not user_input:
                    continue
                    
                # Handle slash commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command in ['help', '?']:
                        self.show_help()
                        continue
                        
                    elif command in ['exit', 'quit']:
                        console.print("\n👋 Goodbye! Thanks for using Multi-Agent AI Coder!")
                        break
                        
                    elif command == 'clear':
                        console.print("\n🧹 Conversation memory cleared!")
                        # Recreate agent to clear memory
                        from ..agents.multi_agent_coder import MultiAgentCoder
                        agent = MultiAgentCoder(cli=self)
                        continue
                        
                    elif command == 'model':
                        model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
                        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                        config_data = {
                            "Model": model,
                            "Server": base_url
                        }
                        self.show_config_info(config_data)
                        continue
                        
                    elif command == 'config':
                        config_data = {
                            "LLM Provider": os.getenv('LLM_PROVIDER', 'huggingface'),
                            "Ollama Model": os.getenv('OLLAMA_MODEL', 'gpt-oss:20b'),
                            "Ollama Server": os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                            "HF Model": os.getenv('HF_MODEL_ID', 'openai/gpt-oss-20b'),
                            "HF Device": os.getenv('HF_DEVICE', 'auto-detect'),
                            "API Key": '***set***' if os.getenv('OLLAMA_API_KEY') else 'not set'
                        }
                        self.show_config_info(config_data)
                        continue
                    
                    elif command == 'device':
                        device_info = get_device_info()
                        self.show_device_info(device_info)
                        continue
                        
                    else:
                        self.print_error(f"Unknown command: /{command}")
                        console.print("   Type '/help' for available commands")
                        continue
                
                # Process normal user request
                console.print()  # Add spacing
                result = agent.run(user_input)
                console.print(f"\n" + "="*60)
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Interrupted! Use '/exit' to quit properly.")
                continue
                
            except EOFError:
                console.print("\n👋 Goodbye!")
                break
                
            except Exception as e:
                self.print_error(f"{e}")
                console.print("   Please try again or use '/help' for assistance")
                continue