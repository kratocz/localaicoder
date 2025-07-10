# Local AI Coder

This is an AI assistant pro programmers and coders. It's open-source, and it's using locally running LLM of your choice. All your programming code, questions, etc. stays on your computer.

It's homework on the course [AI Agents](https://robotdreams.cz/course/567-ai-agents) by the company [robot_dreams](https://robotdreams.cz/). This course has been led by [Lukáš Kellerstein](https://www.linkedin.com/in/lukas-kellerstein/).

Licence: [MIT](LICENSE)

## Current Features

Now works:

* Connection to the local ollama server.
* Tool: `list_paths_recursive`
* Predefined question forcing tool usage.
* The chain: prompt → LLM thing → tool → LLM thing → … → tool → LLM thing → answer
