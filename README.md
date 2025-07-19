---
license: cc
title: Clinician AI Assistant
sdk: gradio
emoji: 🏃
colorFrom: green
colorTo: blue
pinned: true
---

# Clinician Assistant LG

A conversational assistant designed to help clinicians in Kenya access patient data and clinical guidelines efficiently. This project uses LangChain, LangGraph, and OpenAI models to provide context-aware, tool-augmented chat for clinical decision support.

---

## Features

- **RAG (Retrieval-Augmented Generation):** Retrieve relevant HIV clinical guidelines using natural language queries.
- **SQL Chain:** Query and summarize patient data from a structured database.
- **IDSR Check:** Analyze clinical cases for possible notifiable diseases and suggest clarifying questions.
- **PHI Filtering:** Detect and filter personally identifiable information from free text.
- **Session Management:** Supports persistent chat sessions with unique thread IDs.
- **Gradio UI:** Simple web interface for clinicians to interact with the assistant.

---

## Project Structure

```
.
├── app.py   # Gradio web app entry point
├── chat.py
├── chatlib/    # Core logic, tools, and utilities
│   ├── assistant_node.py
│   ├── guidlines_rag_agent_li.py
│   ├── idsr_check.py
│   ├── logger.py
│   ├── patient_all_data.py
│   ├── patient_sql_agent.py
│   ├── phi_filter.py
│   └── state_types.py
├── config.env.example  # env example
├── Dockerfile  # Docker image
├── main.py
├── Makefile    # Common dev commands (lint, test, format, install, run, clean)
├── pyproject.toml   # uv Python dependencies
├── README.md      # Project documentation
├── requirements.txt      # Python dependencies
└── uv.lock             
```
---

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd clinician-assistant-lg
   ```

2. **Install dependencies:**
   ```sh
   make install
   ```

3. **Set up environment variables:**
   - Copy `config.env.example` to `config.env` and fill in your OpenAI and LangSmith API keys.

4. **Run the app:**
   ```sh
   python app.py
   ```
   Or use the Makefile:
   ```sh
   make run
   ```

---

## Usage

- Open the Gradio web interface (URL will be shown in the terminal).
- Enter a clinical question and (optionally) a patient identifier.
- The assistant will use the appropriate tools to answer, referencing guidelines and patient data as needed.

---

## Development

- **Lint code:** `make lint`
- **Format code:** `make format`
- **Run tests:** `make test`

---

## Requirements

See `pyproject.toml` for the full list. Key dependencies:
   - black>=25.1.0
   - dateparser>=1.2.2
   - faiss-cpu>=1.11.0
   - gradio>=5.36.2
   - langchain-community>=0.3.27
   - langchain-openai>=0.3.27
   - langgraph>=0.5.2
   - llama-index>=0.12.48
   - pandas>=2.3.1
   - pylint>=3.3.7
   - python-dotenv>=1.1.1

---

## Notes

- Patient data and guideline retrieval are tailored for Kenyan clinical workflows.
- The assistant is not a substitute for clinical judgment.
- All PHI is filtered to protect patient privacy.

---

## License

MIT License

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Gradio](https://github.com/gradio-app/gradio)
