# GraphRAG: Multi-Source Knowledge Graph Analysis

A powerful Graph-based Retrieval-Augmented Generation (RAG) system that transforms unstructured data from various sources into a structured knowledge graph for intelligent querying. This project bridges the gap between raw text and semantic relationships, extracting entities and their connections to power context-aware natural language search.

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)

## Overview

Traditional RAG systems often struggle with complex relationships between entities (e.g., linking specific fixes to bugs or features across disparate documents). **GraphRAG** addresses this by building a unified knowledge graph of developers, components, features, and concepts, allowing for more nuanced and context-aware information retrieval.

While the current implementation focuses on **GitHub data (READMEs and Issues)**, the architecture is designed to scale across multiple data sources including documentation, Slack, Jira, and internal wikis.

## Architecture

The pipeline consists of four modular phases:

1.  **Ingestion (`ingest.py`)**: Data connectors for external sources. (Current implementation uses `PyGitHub` to scrape READMEs and closed issues from repositories). Data is chunked and saved to `output.json`.
2.  **Extraction (`extract.py`)**: Processes text chunks through a local LLM (via LM Studio) to identify nodes (Entities) and edges (Relationships). Results are saved to `entities.json`.
3.  **Loading (`load_graph.py`)**: Connects to a Neo4j database and uses `MERGE` operations to populate the knowledge graph with nodes and relationships.
4.  **Querying (`query.py`)**: A natural language interface that:
    *   Translates user questions into Cypher queries.
    *   Executes the queries against the Neo4j database.
    *   Summarizes the results into a human-readable answer.

## Tech Stack

| Technology | Badge | Purpose |
| :--- | :--- | :--- |
| **Python** | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core Language |
| **Neo4j** | ![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white) | Graph Database |
| **LangChain** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white) | Orchestration |
| **OpenAI API** | ![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white) | Local LLM Interface |
| **uv** | ![uv](https://img.shields.io/badge/uv-F01F7A?logo=uv&logoColor=white) | Package Management |

## Prerequisites

- **Python**: 3.13 or higher.
- **Neo4j**: A running instance (local or AuraDB).
- **LM Studio**: Running an OpenAI-compatible server at `http://localhost:1234`.
- **API Keys**: Required for the data sources being ingested (e.g., GitHub Personal Access Token).

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/KshitijBharambe/GraphRAG.git
    cd GraphRAG
    ```

2.  **Install dependencies**:
    Using `uv`:
    ```bash
    uv sync
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    # Current Data Source Credentials
    GitHub_Token=your_github_token
    
    # Graph Database Credentials
    NEO4J_URI=neo4j+s://your-db-uri
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password
    ```

## Usage

Run the pipeline in order:

1.  **Ingest data**:
    ```bash
    uv run ingest.py
    ```
2.  **Extract entities**: (Ensure LM Studio is running, or configure your API of choice by changing the base URL, currently supports openai API)
    ```bash
    uv run extract.py
    ```
3.  **Load the graph**:
    ```bash
    uv run load_graph.py
    ```
4.  **Ask questions**:
    ```bash
    uv run query.py
    ```


## Example Queries

- "What are the core features of the shadcn-ui project?"
- "Which developers have fixed bugs related to performance?"
- "How does this feature depend on the underlying framework?"

---
*Note: This project is a demonstration of GraphRAG concepts and is currently configured for specific GitHub repositories.*

