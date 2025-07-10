# Local AI Coder

This is an AI assistant for programmers and coders. It's open-source and uses a locally running LLM of your choice. All your programming code, questions, etc. stay on your computer.

It's homework for the course [AI Agents](https://robotdreams.cz/course/567-ai-agents) by the company [robot_dreams](https://robotdreams.cz/). This course has been led by [Lukáš Kellerstein](https://www.linkedin.com/in/lukas-kellerstein/).

License: [MIT](LICENSE)

## Current Features

Currently works:

* Connection to the local Ollama server.
* Tool: `list_paths_recursive`
* A predefined question forcing tool usage.
* The chain: prompt → LLM → tool → LLM → … → tool → LLM → answer
