import json
import os
from dotenv import load_dotenv
from ollama import chat, ChatResponse
from typing import List, Dict, Any, TypedDict

# Type alias - more specific structure
class PathsDict(TypedDict):
    directories: List[str]
    files: List[str]

# Load environment variables from .env file
load_dotenv()

# Function Implementations
def list_paths_recursive(directory: str = os.getcwd(), exclude_dirs: List[str] = None) -> PathsDict:
    """Get a list of all files and directories recursively in a directory."""
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.idea', '.git', 'node_modules']
    
    all_dirs = []
    all_files = []
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from dirs list to prevent os.walk from entering them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        # Add directories (only non-excluded ones)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Make path relative to the starting directory
            relative_path = os.path.relpath(dir_path, directory)
            all_dirs.append(relative_path)

        # Add files
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Make path relative to the starting directory
            relative_path = os.path.relpath(file_path, directory)
            all_files.append(relative_path)
    
    return PathsDict(
        directories=sorted(all_dirs),
        files=sorted(all_files)
    )

# Define tools for Ollama (mixed format as shown in original)
tools = [
    list_paths_recursive,  # Direct function reference
    {
        "type": "function",
        "function": {
            "name": list_paths_recursive.__name__,
            "description": "Use this function to get list of all directories and files recursively in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    # "ticker": {
                    #     "type": "string",
                    #     "description": "The ticker symbol for the stock, e.g. GOOG",
                    # }
                },
                "required": [],
                "optional": ["directory", "exclude_dirs"],
            },
        },
    },
]

# Available functions for calling
available_functions = {
    list_paths_recursive.__name__: list_paths_recursive,
}


class OllamaReactAgent:
    """A "ReAct" (Reason and Act) agent using Ollama."""

    def __init__(self, model: str = os.getenv("MODEL", "gpt-oss:20b")):
        self.model = model
        self.max_iterations = 10

    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the "ReAct" loop until we get a final answer.
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Call the LLM
            response: ChatResponse = chat(
                self.model,
                messages=messages,
                tools=tools,
            )

            print(f"LLM Response: {response.message}")

            # Check if there are tool calls
            if response.message.tool_calls:
                # Add the assistant's message to history
                messages.append(response.message)

                # Process ALL tool calls
                for tool_call in response.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments

                    print(f"Executing tool: {function_name}({function_args})")

                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)

                    print(f"Tool result: {function_response}")

                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })

                # Continue the loop to get the next response
                continue

            else:
                # No tool calls - we have our final answer
                final_content = response.message.content

                # Add the final assistant message to history
                messages.append(response.message)

                print(f"\nFinal answer: {final_content}")
                return final_content

        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    # Create a ReAct agent
    agent = OllamaReactAgent()

    # Example 1: Simple query (single tool call)
    prompt = "What directories and files are in the directory? Describe from their names them the project."
    print(f"=== Example 1: {prompt} ===")
    messages1 = [
        {"role": "system", "content": "You are a useful coding assistant. Use the provided tools whenever you need."},
        {"role": "user", "content": prompt},
    ]
    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1}")

if __name__ == "__main__":
    #print(list_paths_recursive())
    main()
