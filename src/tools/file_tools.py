"""File operation tools for the multi-agent system."""

import os
import difflib
from typing import List, TypedDict
from langchain_core.tools import tool
from rich.console import Console

console = Console()

# Global reference to CLI - will be set from main
global_cli = None

def set_global_cli(cli):
    """Set global CLI reference for tools."""
    global global_cli
    global_cli = cli


class PathsDict(TypedDict):
    """Type alias for paths dictionary."""
    directories: List[str]
    files: List[str]


@tool
def list_paths_recursive(directory: str = "", exclude_dirs: List[str] = None, include_hidden: bool = False) -> PathsDict:
    """Get a list of all files and directories recursively in a directory.
    
    Args:
        directory: Directory to scan (defaults to current working directory)
        exclude_dirs: List of directory names to exclude from scanning
        include_hidden: Whether to include hidden directories/files starting with '.' (default: False)
    
    Returns:
        Dictionary with 'directories' and 'files' keys containing sorted lists of paths
    """
    # Handle empty directory string
    if not directory or directory.strip() == "":
        directory = os.getcwd()
    
    # Check if directory exists
    if not os.path.exists(directory):
        return PathsDict(
            directories=[],
            files=[]
        )
    
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'node_modules']
    
    all_dirs = []
    all_files = []
    
    try:
        for root, dirs, files in os.walk(directory):
            # Filter directories: exclude specified dirs and optionally hidden dirs
            dirs[:] = [d for d in dirs if (
                d not in exclude_dirs and 
                (include_hidden or not d.startswith('.'))
            )]
            
            # Add directories (only non-excluded ones)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Make path relative to the starting directory
                relative_path = os.path.relpath(dir_path, directory)
                all_dirs.append(relative_path)

            # Add files (filter hidden files if needed)
            for file_name in files:
                # Skip hidden files unless include_hidden is True
                if not include_hidden and file_name.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file_name)
                # Make path relative to the starting directory
                relative_path = os.path.relpath(file_path, directory)
                all_files.append(relative_path)
                
    except (OSError, PermissionError):
        # Return empty result if we can't read the directory
        return PathsDict(
            directories=[],
            files=[]
        )
    
    return PathsDict(
        directories=sorted(all_dirs),
        files=sorted(all_files)
    )


@tool
def read_file(file_path: str, max_lines: int = 100) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (default: 100)
    
    Returns:
        File contents as a string, or error message if file cannot be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:max_lines]
            content = ''.join(lines)
            if len(lines) == max_lines:
                content += f"\n... (file truncated, showing first {max_lines} lines)"
            return content
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


@tool  
def write_file(file_path: str, content: str) -> str:
    """Write content to a file with diff preview and user confirmation.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Success or error message
    """
    try:
        # Check if file exists and read current content
        existing_content = ""
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            except Exception as e:
                return f"Error reading existing file {file_path}: {str(e)}"
            
            # If content is the same, no need to write
            if existing_content == content:
                return f"No changes needed - file {file_path} already has the same content"
            
            # Generate and display diff
            diff = difflib.unified_diff(
                existing_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"{file_path} (current)",
                tofile=f"{file_path} (proposed)",
                lineterm=""
            )
            
            diff_lines = list(diff)
            if not diff_lines:
                return f"No changes detected in {file_path}"
            
            # Format diff content for Rich display
            diff_content = ""
            for line in diff_lines:
                if line.startswith('---') or line.startswith('+++'):
                    diff_content += f"[bold]{line.rstrip()}[/bold]\n"
                elif line.startswith('@@'):
                    diff_content += f"[cyan]{line.rstrip()}[/cyan]\n"
                elif line.startswith('+'):
                    diff_content += f"[green]{line.rstrip()}[/green]\n"
                elif line.startswith('-'):
                    diff_content += f"[red]{line.rstrip()}[/red]\n"
                else:
                    diff_content += line.rstrip() + "\n"
            
            # Use advanced CLI if available, otherwise fallback
            if global_cli:
                if not global_cli.show_diff_confirmation(file_path, diff_content.rstrip()):
                    return f"Changes to {file_path} cancelled by user"
            else:
                # Fallback to basic confirmation
                print(f"\n{'='*60}")
                print(f"PROPOSED CHANGES TO: {file_path}")
                print(f"{'='*60}")
                print(diff_content)
                print(f"{'='*60}")
                
                while True:
                    user_input = input(f"Apply these changes to {file_path}? (y/n): ").lower().strip()
                    if user_input in ['y', 'yes']:
                        break
                    elif user_input in ['n', 'no']:
                        return f"Changes to {file_path} cancelled by user"
                    else:
                        print("Please enter 'y' for yes or 'n' for no")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Write the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        action = "updated" if file_exists else "created"
        return f"Successfully {action} {file_path} ({len(content)} characters)"
        
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"