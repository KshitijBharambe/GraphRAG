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
IMPORTANT: Every single node in the database has the label :Entity. There are NO other node labels.
Node properties:
  - name (string): the entity's name
  - label (string): one of Developer, Repository, Feature, Bug, Concept
  - repo (string): the GitHub repo it came from

Relationships: (a:Entity)-[:RELATES_TO {type: "..."}]->(b:Entity)
  - The relationship is always :RELATES_TO
  - The relationship type is stored in the property `type`, which is one of: WROTE, CONTAINS, FIXES, DEPENDS_ON, EXPLAINS
"""

CYPHER_PROMPT = """You are a Neo4j Cypher expert. Generate a Cypher query to answer the user's question.

{schema}

CRITICAL RULES — violating these will break the query:
1. EVERY node uses the label :Entity. NEVER use :Feature, :Developer, :Bug, :Concept, :Repository.
2. Filter by type using the `label` property: WHERE n.label = "Feature"
3. Filter by name using CONTAINS for flexibility: WHERE toLower(n.name) CONTAINS "keyword"
4. Only use a relationship pattern when the question asks about connections between entities.
5. For simple listing/counting questions, use a single node pattern: MATCH (n:Entity)
6. Correct clause order: MATCH ... WHERE ... RETURN ... LIMIT ...
7. Return ONLY the raw Cypher. No explanation, no markdown fences.

Examples:
Q: "List all Feature nodes"
A: MATCH (n:Entity) WHERE n.label = "Feature" RETURN n.name, n.repo LIMIT 50

Q: "How many entities are there by type?"
A: MATCH (n:Entity) RETURN n.label AS type, count(n) AS count ORDER BY count DESC

Q: "Show all nodes from the shadcn-ui repo"
A: MATCH (n:Entity) WHERE n.repo = "shadcn-ui/ui" RETURN n.name, n.label LIMIT 50

Q: "What bugs exist in the next.js repo?"
A: MATCH (n:Entity) WHERE n.repo = "vercel/next.js" AND n.label = "Bug" RETURN n.name LIMIT 10

Q: "What features are connected to tailwind?"
A: MATCH (a:Entity)-[:RELATES_TO]->(b:Entity) WHERE toLower(a.name) CONTAINS "tailwind" AND b.label = "Feature" RETURN b.name, b.label LIMIT 10

Q: "List all developers"
A: MATCH (n:Entity) WHERE n.label = "Developer" RETURN n.name, n.repo LIMIT 20

Question: {question}
Cypher: """

ANSWER_PROMPT = """Based on the following database results, answer the user's question in plain English.
If the results are empty, say you couldn't find relevant information.

Question: {question}

Database results:
{results}
"""


_ENTITY_LABELS = {"Feature", "Developer", "Bug", "Concept", "Repository"}


def _fix_cypher(cypher: str) -> str:
    """Rewrite rogue node labels (e.g. MATCH (n:Feature)) into :Entity + WHERE filter."""
    import re

    label_pattern = "|".join(_ENTITY_LABELS)

    def _replace(m: re.Match) -> str:
        var = m.group(1) or "n"
        label = m.group(2)
        return f'({var}:Entity) WHERE {var}.label = "{label}"'

    return re.sub(rf"\((\w*):({label_pattern})\)", _replace, cypher)


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

    cypher = _fix_cypher(cypher)
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


def check_db() -> int:
    """Return the number of Entity nodes in the database."""
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN count(n) AS total")
        return result.single()["total"]


def main():
    print("GraphRAG Query Engine")

    count = check_db()
    if count == 0:
        print("\nWARNING: The database is empty!")
        print("Run the pipeline first:")
        print("  1. python ingest.py      -> output.json")
        print("  2. python extract.py     -> entities.json")
        print("  3. python load_graph.py  -> Neo4j populated")
        print()
    else:
        print(f"Connected. {count} entities in graph.\n")

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
