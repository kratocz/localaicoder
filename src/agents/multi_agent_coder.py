"""Multi-agent AI Coder system using LangGraph with multiple LLM providers."""

import os
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .agent_roles import AgentRole
from ..providers.ollama import OllamaProvider
from ..providers.huggingface import HuggingFaceProvider
from ..tools.file_tools import list_paths_recursive, read_file, write_file
from ..tools.command_tools import run_command
from ..utils.device_detection import get_device_info


class MultiAgentCoder:
    """Multi-agent AI Coder system using LangGraph with multiple LLM providers."""

    def __init__(self, 
                 llm_provider: str = os.getenv("LLM_PROVIDER", "huggingface"),
                 model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
                 base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                 device: Optional[str] = os.getenv("HF_DEVICE"),
                 cli = None):
        
        # Initialize LLM provider based on configuration
        self.llm_provider_name = llm_provider.lower()
        self.device_info = get_device_info()
        self.cli = cli
        
        # Validate provider name
        valid_providers = ["huggingface", "ollama"]
        if self.llm_provider_name not in valid_providers:
            raise ValueError(f"Unknown LLM_PROVIDER: '{llm_provider}'. Valid options are: {', '.join(valid_providers)}")
        
        if self.llm_provider_name == "huggingface":
            # Use HuggingFace provider
            hf_model_id = os.getenv("HF_MODEL_ID", "openai/gpt-oss-20b")
            self.provider = HuggingFaceProvider(model_id=hf_model_id, device=device, cli=cli)
            if not self.provider.is_available():
                raise RuntimeError("HuggingFace provider not available. Install required dependencies: pip install langchain-huggingface torch transformers")
        elif self.llm_provider_name == "ollama":
            # Use Ollama provider
            self.provider = OllamaProvider(model=model, base_url=base_url)
        
        # Get LLM instance
        self.llm = self.provider.get_llm()
        
        # Create tool nodes FIRST
        self.file_tools = [list_paths_recursive, read_file, write_file]
        self.command_tools = [run_command]
        self.file_tool_node = ToolNode(self.file_tools)
        self.command_tool_node = ToolNode(self.command_tools)
        
        # Print configuration info
        if self.cli:
            self.cli.print_info("🤖 Multi-Agent Coder initialized:")
            self.cli.print_info(f"   Provider: {self.llm_provider_name}")
            if self.llm_provider_name == "ollama":
                self.cli.print_info(f"   Model: {model}")
                self.cli.print_info(f"   Ollama Server: {base_url}")
            else:
                self.cli.print_info(f"   Model: {getattr(self.provider, 'model_id', 'N/A')}")
                self.cli.print_info(f"   Device: {self.device_info['detected_device']} (auto-detected)")
        else:
            print(f"🤖 Multi-Agent Coder initialized:")
            print(f"   Provider: {self.llm_provider_name}")
            if self.llm_provider_name == "ollama":
                print(f"   Model: {model}")
                print(f"   Ollama Server: {base_url}")
            else:
                print(f"   Model: {getattr(self.provider, 'model_id', 'N/A')}")
                print(f"   Device: {self.device_info['detected_device']} (auto-detected)")
        
        # Create specialized agent LLMs with different prompts
        self.agents = self._create_agents()
        
        # Build the multi-agent graph
        self.graph = self._build_multi_agent_graph()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create specialized agents with different system prompts."""
        
        # Check if LLM supports bind_tools (for Ollama)
        if hasattr(self.llm, 'bind_tools'):
            return {
                AgentRole.COORDINATOR: self.llm.bind_tools([]),
                AgentRole.TASK_PLANNER: self.llm.bind_tools([]),
                AgentRole.CODE_ANALYZER: self.llm.bind_tools(self.file_tools),
                AgentRole.FILE_MANAGER: self.llm.bind_tools(self.file_tools),
                AgentRole.COMMAND_EXECUTOR: self.llm.bind_tools(self.command_tools),
            }
        else:
            # For HuggingFace models, we'll handle tools separately
            return {
                AgentRole.COORDINATOR: self.llm,
                AgentRole.TASK_PLANNER: self.llm,
                AgentRole.CODE_ANALYZER: self.llm,
                AgentRole.FILE_MANAGER: self.llm,
                AgentRole.COMMAND_EXECUTOR: self.llm,
            }
        
    def _build_multi_agent_graph(self):
        """Build the multi-agent LangGraph workflow."""
        
        def coordinator_node(state: MessagesState) -> Dict[str, Any]:
            """Coordinator agent decides which agent should handle the task."""
            messages = state['messages']
            
            # Add coordinator system message
            coordinator_messages = [
                SystemMessage(content="""
You are the Coordinator agent. Analyze the user's request and provide clear routing instructions.

For the user's request, respond with:
1. Brief analysis of what they want
2. Clear routing instruction using one of these phrases:
   - "Route to file_manager" - for file/directory operations (list, read, write files)
   - "Route to code_analyzer" - for code analysis, bug finding, reviews
   - "Route to command_executor" - for running commands, tests, builds
   - "Route to task_planner" - for complex multi-step planning

Example: "User wants to list project files. Route to file_manager to handle directory scanning."
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
You are the File Manager agent. Your job is to handle file and directory operations using the available tools.

Available tools:
- list_paths_recursive: List files and directories in a project
- read_file: Read the contents of a specific file  
- write_file: Write content to a file (with diff preview)

For the user's request, analyze what they need and use the appropriate tool. 
If they want to list files/directories, use list_paths_recursive.
If they want to see file contents, use read_file.
If they want to modify files, use write_file.

Always use tools to complete the task - don't just describe what you would do.
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
            
            # Enhanced routing logic
            content = last_message.content.lower()
            
            # Check for explicit routing instructions
            if 'file_manager' in content:
                return 'file_manager'
            elif 'code_analyzer' in content:
                return 'code_analyzer'
            elif 'command_executor' in content:
                return 'command_executor'
            elif 'task_planner' in content:
                return 'task_planner'
            elif 'end' in content or 'complete' in content or 'done' in content:
                return '__end__'
            
            # Infer from task content
            elif any(word in content for word in ['list', 'file', 'directory', 'folder', 'read', 'write']):
                return 'file_manager'
            elif any(word in content for word in ['analyze', 'check', 'review', 'bug', 'code']):
                return 'code_analyzer'
            elif any(word in content for word in ['run', 'execute', 'command', 'test', 'build']):
                return 'command_executor'
            elif any(word in content for word in ['plan', 'task', 'steps', 'strategy']):
                return 'task_planner'
            else:
                # Default to file_manager for file-related tasks
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
            return '__end__'
        
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
                "__end__": "__end__"
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
                    "__end__": "__end__"
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
        
        all_steps = []
        for step_output in self.graph.stream(initial_state, config):
            print(f"Step: {step_output}")
            all_steps.append(step_output)
        
        # Extract final response - look for the best non-empty response
        best_response = None
        best_node = None
        
        # Check steps in reverse order to find the most recent meaningful response
        for step in reversed(all_steps):
            for node_name, node_output in step.items():
                if 'messages' in node_output and node_output['messages']:
                    final_message = node_output['messages'][-1]
                    if hasattr(final_message, 'content') and final_message.content.strip():
                        content = final_message.content.strip()
                        # Prefer substantial responses over empty ones
                        if len(content) > 10:  # Prefer responses with meaningful content
                            print(f"\nFinal answer from {node_name}: {content}")
                            return content
                        elif not best_response:  # Keep as backup if no better response found
                            best_response = content
                            best_node = node_name
        
        # Fall back to best response found if any
        if best_response:
            print(f"\nFinal answer from {best_node}: {best_response}")
            return best_response
            
        return "Error: No response generated."