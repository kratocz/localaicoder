import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Type alias - more specific structure
class PathsDict(TypedDict):
    directories: List[str]
    files: List[str]

# Load environment variables from .env file
load_dotenv()

# Function Implementations using LangChain tool decorator
@tool
def list_paths_recursive(directory: str = os.getcwd(), exclude_dirs: List[str] = None) -> PathsDict:
    """Get a list of all files and directories recursively in a directory.
    
    Args:
        directory: Directory to scan (defaults to current working directory)
        exclude_dirs: List of directory names to exclude from scanning
    
    Returns:
        Dictionary with 'directories' and 'files' keys containing sorted lists of paths
    """
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
    """Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
    
    Returns:
        Success or error message
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file {file_path}: {str(e)}"

@tool
def run_command(command: str, working_dir: str = ".") -> str:
    """Execute a shell command and return the output.
    
    Args:
        command: Shell command to execute
        working_dir: Working directory for the command (default: current directory)
    
    Returns:
        Command output or error message
    """
    import subprocess
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

# Define tools by category
file_tools = [list_paths_recursive, read_file, write_file]
command_tools = [run_command]
all_tools = file_tools + command_tools

# Agent roles and their specialized tools
class AgentRole:
    COORDINATOR = "coordinator"
    TASK_PLANNER = "task_planner"
    CODE_ANALYZER = "code_analyzer"
    FILE_MANAGER = "file_manager"
    COMMAND_EXECUTOR = "command_executor"


class MultiAgentCoder:
    """Multi-agent AI Coder system using LangGraph and Ollama."""

    def __init__(self, model: str = os.getenv("MODEL", "gpt-oss:20b")):
        # Initialize the LLM
        self.llm = ChatOllama(model=model, temperature=0)
        
        # Create specialized agent LLMs with different prompts
        self.agents = self._create_agents()
        
        # Create tool nodes
        self.file_tool_node = ToolNode(file_tools)
        self.command_tool_node = ToolNode(command_tools)
        
        # Build the multi-agent graph
        self.graph = self._build_multi_agent_graph()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create specialized agents with different system prompts."""
        
        coordinator_prompt = """
You are the Coordinator agent. Your role is to:
1. Understand user requests and break them into tasks
2. Decide which specialized agents should handle each task
3. Coordinate between agents and synthesize final responses
4. Always respond with your reasoning and next steps
"""
        
        task_planner_prompt = """
You are the Task Planner agent. Your role is to:
1. Analyze complex programming tasks and break them into steps
2. Identify dependencies between tasks
3. Create actionable plans for other agents to execute
4. Prioritize tasks based on importance and complexity
"""
        
        code_analyzer_prompt = """
You are the Code Analyzer agent. Your role is to:
1. Read and analyze source code files
2. Identify bugs, code smells, and improvement opportunities
3. Suggest refactoring and optimization strategies
4. Explain code functionality and architecture
Use file tools to examine code when needed.
"""
        
        file_manager_prompt = """
You are the File Manager agent. Your role is to:
1. Navigate project directory structures
2. Read, create, and modify files safely
3. Organize project files and folders
4. Handle file operations and maintain project structure
Use file management tools exclusively.
"""
        
        command_executor_prompt = """
You are the Command Executor agent. Your role is to:
1. Execute shell commands safely and securely
2. Run tests, builds, and development tools
3. Handle git operations and version control
4. Monitor command outputs and report results
Use command execution tools only.
"""
        
        return {
            AgentRole.COORDINATOR: self.llm.bind_tools([]),
            AgentRole.TASK_PLANNER: self.llm.bind_tools([]),
            AgentRole.CODE_ANALYZER: self.llm.bind_tools(file_tools),
            AgentRole.FILE_MANAGER: self.llm.bind_tools(file_tools),
            AgentRole.COMMAND_EXECUTOR: self.llm.bind_tools(command_tools),
        }
        
    def _build_multi_agent_graph(self):
        """Build the multi-agent LangGraph workflow."""
        
        def coordinator_node(state: MessagesState) -> Dict[str, Any]:
            """Coordinator agent decides which agent should handle the task."""
            messages = state['messages']
            
            # Add coordinator system message
            coordinator_messages = [
                SystemMessage(content="""
You are the Coordinator agent. Analyze the user's request and decide:
1. What type of task this is (code analysis, file management, command execution, etc.)
2. Which specialized agent should handle it
3. Provide reasoning for your decision

Respond with your analysis and route to the appropriate agent:
- For code analysis: route to 'code_analyzer'
- For file operations: route to 'file_manager'  
- For running commands: route to 'command_executor'
- For planning complex tasks: route to 'task_planner'
- If task is complete: route to 'end'
""")
            ] + messages
            
            response = self.agents[AgentRole.COORDINATOR].invoke(coordinator_messages)
            return {"messages": [response]}
            
        def task_planner_node(state: MessagesState) -> Dict[str, Any]:
            """Task planner breaks down complex tasks."""
            messages = state['messages']
            
            planner_messages = [
                SystemMessage(content="""
You are the Task Planner. Break down the user's request into specific, actionable steps.
Create a detailed plan and then route to the appropriate specialized agent for execution.
""")
            ] + messages
            
            response = self.agents[AgentRole.TASK_PLANNER].invoke(planner_messages)
            return {"messages": [response]}
            
        def code_analyzer_node(state: MessagesState) -> Dict[str, Any]:
            """Code analyzer examines and analyzes code."""
            messages = state['messages']
            
            analyzer_messages = [
                SystemMessage(content="""
You are the Code Analyzer. Use file tools to examine code and provide detailed analysis.
Look for bugs, improvements, and explain functionality.
""")
            ] + messages
            
            response = self.agents[AgentRole.CODE_ANALYZER].invoke(analyzer_messages)
            return {"messages": [response]}
            
        def file_manager_node(state: MessagesState) -> Dict[str, Any]:
            """File manager handles file operations."""
            messages = state['messages']
            
            manager_messages = [
                SystemMessage(content="""
You are the File Manager. Use file tools to handle all file operations.
Read, write, and organize files as requested.
""")
            ] + messages
            
            response = self.agents[AgentRole.FILE_MANAGER].invoke(manager_messages)
            return {"messages": [response]}
            
        def command_executor_node(state: MessagesState) -> Dict[str, Any]:
            """Command executor runs shell commands."""
            messages = state['messages']
            
            executor_messages = [
                SystemMessage(content="""
You are the Command Executor. Use command tools to execute shell commands safely.
Run tests, builds, git operations, and other development commands.
""")
            ] + messages
            
            response = self.agents[AgentRole.COMMAND_EXECUTOR].invoke(executor_messages)
            return {"messages": [response]}
            
        def route_decision(state: MessagesState) -> str:
            """Route to appropriate agent based on coordinator's decision."""
            messages = state['messages']
            last_message = messages[-1]
            
            # Simple routing logic based on message content
            content = last_message.content.lower()
            
            if 'code_analyzer' in content or 'analyze' in content:
                return 'code_analyzer'
            elif 'file_manager' in content or 'file' in content:
                return 'file_manager'
            elif 'command_executor' in content or 'command' in content or 'run' in content:
                return 'command_executor'
            elif 'task_planner' in content or 'plan' in content:
                return 'task_planner'
            elif 'end' in content or 'complete' in content:
                return 'end'
            else:
                # Default to file_manager for most programming tasks
                return 'file_manager'
                
        def tool_router(state: MessagesState) -> str:
            """Route to appropriate tool node."""
            messages = state['messages']
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                # Check which tools are being called
                tool_names = []
                for tc in last_message.tool_calls:
                    if hasattr(tc, 'function'):
                        tool_names.append(tc.function.name)
                    elif isinstance(tc, dict) and 'function' in tc:
                        tool_names.append(tc['function']['name'])
                    
                if any(tool_name in ['run_command'] for tool_name in tool_names):
                    return 'command_tools'
                else:
                    return 'file_tools'
            return 'end'
        
        # Create the graph
        workflow = StateGraph(MessagesState)
        
        # Add agent nodes
        workflow.add_node("coordinator", coordinator_node)
        workflow.add_node("task_planner", task_planner_node)
        workflow.add_node("code_analyzer", code_analyzer_node)
        workflow.add_node("file_manager", file_manager_node)
        workflow.add_node("command_executor", command_executor_node)
        
        # Add tool nodes
        workflow.add_node("file_tools", self.file_tool_node)
        workflow.add_node("command_tools", self.command_tool_node)
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        # Add conditional edges from coordinator
        workflow.add_conditional_edges(
            "coordinator",
            route_decision,
            {
                "task_planner": "task_planner",
                "code_analyzer": "code_analyzer", 
                "file_manager": "file_manager",
                "command_executor": "command_executor",
                "end": "__end__"
            }
        )
        
        # Add conditional edges from each agent to tools or end
        for agent in ["task_planner", "code_analyzer", "file_manager", "command_executor"]:
            workflow.add_conditional_edges(
                agent,
                tool_router,
                {
                    "file_tools": "file_tools",
                    "command_tools": "command_tools", 
                    "end": "__end__"
                }
            )
        
        # Add edges from tools back to coordinator
        workflow.add_edge("file_tools", "coordinator")
        workflow.add_edge("command_tools", "coordinator")
        
        # Compile the graph
        return workflow.compile(checkpointer=MemorySaver())
    
    def run(self, message: str) -> str:
        """Run the agent with a user message."""
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)]
        }
        
        # Run the graph
        config = {"configurable": {"thread_id": "default"}}
        
        print(f"\n=== Processing: {message} ===")
        
        final_state = None
        for step_output in self.graph.stream(initial_state, config):
            print(f"Step: {step_output}")
            final_state = step_output
        
        # Extract final response
        if final_state and 'agent' in final_state:
            final_message = final_state['agent']['messages'][-1]
            print(f"\nFinal answer: {final_message.content}")
            return final_message.content
        
        return "Error: No response generated."


def main():
    # Create the Multi-Agent AI Coder system
    agent = MultiAgentCoder()

    # Example: Simple query
    prompt = "What directories and files are in the directory? Describe from their names the project."
    result = agent.run(prompt)
    print(f"\nResult: {result}")

if __name__ == "__main__":
    #print(list_paths_recursive())
    main()
