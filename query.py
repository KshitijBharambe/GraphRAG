"""
Phase 5: Ask questions in English -> LLM generates Cypher -> query Neo4j -> LLM summarizes.

Requires:
  - Neo4j credentials in .env
  - LM Studio running at localhost:1234
"""

import os

import dotenv
from neo4j import GraphDatabase
from openai import OpenAI

dotenv.load_dotenv()

# --- Neo4j ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# --- LLM ---
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="")

# --- Graph schema description for the LLM ---
GRAPH_SCHEMA = """
The Neo4j database has:
- Nodes labeled :Entity with properties: name (string), label (one of: Developer, Repository, Feature, Bug, Concept), repo (string)
- Relationships of type :RELATES_TO with property: type (one of: WROTE, CONTAINS, FIXES, DEPENDS_ON, EXPLAINS)
"""

CYPHER_PROMPT = """You are a Neo4j Cypher expert. Given a user question and a graph schema, generate a Cypher query to answer it.

Schema:
{schema}

Rules:
- Return ONLY the Cypher query, no explanation
- Use case-insensitive matching with toLower() for name lookups
- Always RETURN meaningful fields
- If you can't answer, return: MATCH (n) RETURN 'No relevant query' AS result LIMIT 1

Question: {question}
"""

ANSWER_PROMPT = """Based on the following database results, answer the user's question in plain English.
If the results are empty, say you couldn't find relevant information.

Question: {question}

Database results:
{results}
"""


def generate_cypher(question: str) -> str:
    """Ask the LLM to translate English into Cypher."""
    response = client.chat.completions.create(
        model="lm-studio",
        messages=[
            {
                "role": "system",
                "content": "You generate Neo4j Cypher queries. Return ONLY the query.",
            },
            {
                "role": "user",
                "content": CYPHER_PROMPT.format(schema=GRAPH_SCHEMA, question=question),
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )
    cypher = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if cypher.startswith("```"):
        cypher = cypher.split("\n", 1)[1]
        cypher = cypher.rsplit("```", 1)[0].strip()

    return cypher


def run_cypher(cypher: str) -> list[dict]:
    """Execute a Cypher query and return results."""
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def summarize(question: str, results: list[dict]) -> str:
    """Ask the LLM to summarize raw DB results into a human answer."""
    response = client.chat.completions.create(
        model="lm-studio",
        messages=[
            {
                "role": "system",
                "content": "You summarize database results into clear, concise answers.",
            },
            {
                "role": "user",
                "content": ANSWER_PROMPT.format(question=question, results=results),
            },
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def main():
    print("GraphRAG Query Engine")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Ask: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        # Step 1: English -> Cypher
        cypher = generate_cypher(question)
        print(f"\nGenerated Cypher:\n  {cypher}\n")

        # Step 2: Run against Neo4j
        try:
            results = run_cypher(cypher)
            print(f"Raw results: {results}\n")
        except Exception as e:
            print(f"Cypher query failed: {e}\n")
            continue

        # Step 3: Summarize
        answer = summarize(question, results)
        print(f"Answer: {answer}\n")

    driver.close()


if __name__ == "__main__":
    main()
