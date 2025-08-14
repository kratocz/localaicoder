"""Command execution tools for the multi-agent system."""

import subprocess
from langchain_core.tools import tool


@tool
def run_command(command: str, working_dir: str = ".") -> str:
    """Execute a shell command and return the output.
    
    Args:
        command: Shell command to execute
        working_dir: Working directory for the command (default: current directory)
    
    Returns:
        Command output or error message
    """
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