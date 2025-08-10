#!/usr/bin/env python3
"""Simplified multi-agent Local AI Coder with working LangGraph implementation"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Type alias
class PathsDict(TypedDict):
    directories: List[str]
    files: List[str]

# Load environment variables
load_dotenv()

# Tools
@tool
def list_paths_recursive(directory: str = ".", exclude_dirs: List[str] = None) -> PathsDict:
    """Get a list of all files and directories recursively in a directory."""
    if exclude_dirs is None:
        exclude_dirs = ['.venv', '__pycache__', '.idea', '.git', 'node_modules']
    
    all_dirs = []
    all_files = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            relative_path = os.path.relpath(dir_path, directory)
            all_dirs.append(relative_path)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, directory)
            all_files.append(relative_path)
    
    return PathsDict(
        directories=sorted(all_dirs),
        files=sorted(all_files)
    )

@tool
def read_file(file_path: str, max_lines: int = 50) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:max_lines]
            content = ''.join(lines)
            if len(lines) == max_lines:
                content += f"\\n... (truncated, showing first {max_lines} lines)"
            return content
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

# All tools
tools = [list_paths_recursive, read_file]

class SimpleMultiAgent:
    """Simplified multi-agent system"""
    
    def __init__(self, model: str = os.getenv("MODEL", "gpt-oss:20b")):
        self.llm = ChatOllama(model=model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tool_node = ToolNode(tools)
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build simplified multi-agent graph"""
        
        def coordinator(state: MessagesState) -> Dict[str, Any]:
            """Coordinator agent"""
            messages = state['messages']
            
            # Add system message for coordinator
            coordinator_messages = [
                SystemMessage(content="""You are a Coordinator for a programming assistant. 
Analyze the user's request and either:
1. Handle simple questions directly
2. Use tools if you need to examine files or directories
3. Route complex tasks to file_manager

Be concise and helpful.""")
            ] + messages
            
            response = self.llm_with_tools.invoke(coordinator_messages)
            return {"messages": [response]}
        
        def file_manager(state: MessagesState) -> Dict[str, Any]:
            """File manager agent"""
            messages = state['messages']
            
            file_messages = [
                SystemMessage(content="""You are a File Manager agent. 
Use the available file tools to help with file and directory operations.
Be thorough in examining project structure.""")
            ] + messages
            
            response = self.llm_with_tools.invoke(file_messages)
            return {"messages": [response]}
        
        def should_continue(state: MessagesState) -> str:
            """Decide next step"""
            messages = state['messages']
            last_message = messages[-1]
            
            # Check for tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Simple routing based on content
            content = getattr(last_message, 'content', '').lower()
            if 'file' in content and 'manager' in content:
                return "file_manager"
            elif any(word in content for word in ['done', 'complete', 'finished']):
                return "__end__"
            else:
                return "__end__"
        
        # Create graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("coordinator", coordinator)
        workflow.add_node("file_manager", file_manager)
        workflow.add_node("tools", self.tool_node)
        
        # Set entry point
        workflow.set_entry_point("coordinator")
        
        # Add edges
        workflow.add_conditional_edges(
            "coordinator",
            should_continue,
            {
                "tools": "tools",
                "file_manager": "file_manager", 
                "__end__": "__end__"
            }
        )
        
        workflow.add_conditional_edges(
            "file_manager", 
            should_continue,
            {
                "tools": "tools",
                "__end__": "__end__"
            }
        )
        
        workflow.add_edge("tools", "coordinator")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def run(self, message: str) -> str:
        """Run the agent"""
        initial_state = {"messages": [HumanMessage(content=message)]}
        config = {"configurable": {"thread_id": "default"}}
        
        print(f"\\n=== Processing: {message} ===")
        
        final_state = None
        step_count = 0
        max_steps = 10
        
        for step_output in self.graph.stream(initial_state, config):
            step_count += 1
            if step_count > max_steps:
                print("Max steps reached, stopping...")
                break
                
            print(f"Step {step_count}: {list(step_output.keys())}")
            final_state = step_output
            
        # Extract final response
        if final_state:
            for node_name, node_output in final_state.items():
                if 'messages' in node_output:
                    final_message = node_output['messages'][-1]
                    if hasattr(final_message, 'content'):
                        print(f"\\nFinal answer: {final_message.content}")
                        return final_message.content
        
        return "No response generated."

def main():
    """Test the simplified multi-agent system"""
    print("Creating Simple Multi-Agent system...")
    agent = SimpleMultiAgent()
    
    prompt = "What directories and files are in this directory? Describe the project."
    result = agent.run(prompt)
    print(f"\\nResult: {result}")

if __name__ == "__main__":
    main()