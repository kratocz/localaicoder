#!/usr/bin/env python3
"""Simple test of the multi-agent system"""

import os
from main import MultiAgentCoder

def test_simple():
    try:
        print("Creating MultiAgentCoder...")
        agent = MultiAgentCoder()
        print("Agent created successfully!")
        
        print("Testing with simple prompt...")
        prompt = "List the files in current directory"
        result = agent.run(prompt)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()