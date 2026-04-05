"""
Phase 4: Load extracted entities into Neo4j using MERGE (not CREATE).

Requires Neo4j AuraDB credentials in .env:
  NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=your_password
"""

import json
import os

import dotenv
from neo4j import GraphDatabase

dotenv.load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def load_nodes(tx, nodes):
    """MERGE each node so duplicates across chunks collapse into one."""
    for node in nodes:
        # Dynamic labels aren't supported in parameterized Cypher,
        # so we use a generic Entity label + a 'type' property.
        tx.run(
            """
            MERGE (n:Entity {name: $name})
            SET n.label = $label, n.repo = $repo
            """,
            name=node["name"],
            label=node["label"],
            repo=node.get("repo", "unknown"),
        )


def load_edges(tx, edges):
    """MERGE each relationship between existing nodes."""
    for edge in edges:
        # Same approach: generic REL type with a 'type' property
        tx.run(
            """
            MATCH (a:Entity {name: $source})
            MATCH (b:Entity {name: $target})
            MERGE (a)-[r:RELATES_TO {type: $relation}]->(b)
            """,
            source=edge["source"],
            target=edge["target"],
            relation=edge["relation"],
        )


def clear_graph(tx):
    """Delete all nodes and relationships."""
    tx.run("MATCH (n) DETACH DELETE n")


def main():
    import sys
    clear = "--clear" in sys.argv

    with open("entities.json", "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    with driver.session() as session:
        if clear:
            print("Clearing existing graph...")
            session.execute_write(clear_graph)
            print("  Graph cleared.")

        print(f"Loading {len(nodes)} nodes and {len(edges)} edges into Neo4j...")
        session.execute_write(load_nodes, nodes)
        print("  Nodes loaded.")
        session.execute_write(load_edges, edges)
        print("  Edges loaded.")

    driver.close()
    print("Done! Open Neo4j Browser to explore your graph.")


if __name__ == "__main__":
    main()
