# Groq Fullstack LangGraph Research Agent

This project demonstrates a fullstack research application using a React frontend and a LangGraph-powered backend agent.

The backend agent performs iterative, tool-augmented web research: it dynamically generates search queries, gathers information from the web using free, keyless search tools, reflects on the gathered information to detect knowledge gaps, and refines its search until it can produce a grounded answer with explicit citations.

The project is designed as a universal research agent, not tailored to any specific topic or query, and serves as a reference architecture for research-augmented conversational AI using LangGraph and open LLMs.

## Features

- Fullstack application with a React frontend and LangGraph backend
- Multi-step LangGraph research agent (query → search → reflection → refinement → answer)
- Dynamic query generation using Groq LLM (llama-3.3-70b-versatile)
- Web research via free, no-API-key metasearch
- Reflective reasoning to identify knowledge gaps
- Final answers grounded in retrieved sources with explicit citations
- CLI interface for fast local testing
- Hot-reloading during development

## Project Structure

frontend/
  React application (Vite)

backend/
  src/
    agent/
      graph.py
      state.py
      prompts.py
      utils.py
      tools_and_schemas.py
      configuration.py
  examples/
    cli_research.py
  app.py
  .env
  pyproject.toml

The backend uses a src-layout. Make sure backend/src is on PYTHONPATH or use an editable install.

## Getting Started

Prerequisites:
- Node.js 18+
- Python 3.11+
- Groq API key

Environment setup (from backend directory):

cp .env.example .env

Edit .env:

GROQ_API_KEY=your_actual_groq_api_key

Web search does not require any API keys.

## Install Dependencies

Backend:

cd backend
pip install -e .

Frontend:

cd frontend
npm install

## Run Development Servers

make dev

Frontend: http://localhost:5173/app
Backend: http://localhost:2024

Alternatively:

cd backend
langgraph dev

cd frontend
npm run dev

## How the Backend Agent Works

The LangGraph agent defined in backend/src/agent/graph.py follows a universal research loop:

1. Generate search queries from the user question
2. Perform web research using free, keyless search tools
3. Produce grounded summaries with explicit source references
4. Reflect on results and detect knowledge gaps
5. Iteratively refine search queries if needed
6. Synthesize a final answer strictly grounded in collected sources

The agent is topic-agnostic and works for arbitrary research questions.

## CLI Usage

cd backend
python examples/cli_research.py "What are the latest trends in renewable energy?"

The CLI runs the full research loop and prints the final grounded answer.

## Deployment Notes

In production, the backend can serve the optimized frontend build.

LangGraph supports production deployments backed by PostgreSQL for persistence and Redis for streaming and background execution.

Production infrastructure is optional. The project works fully in local and CLI mode without Redis or Postgres.

LangGraph deployment documentation:
https://langchain-ai.github.io/langgraph/concepts/deployment_options/

## Technologies Used

Frontend:
React
Vite
Tailwind CSS
Shadcn UI

Backend:
Python 3.11+
FastAPI
LangGraph

LLM:
Groq llama-3.3-70b-versatile

Web Search:
Free no-API-key metasearch

## License

This project is licensed under the Apache License 2.0.
See the LICENSE file for details.
