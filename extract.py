"""
Phase 3: Send each chunk to a local LLM and extract graph entities (nodes + edges).
Saves results to entities.json.

Requires LM Studio running at http://localhost:1234/v1
Uses LM Studio's built-in parallel inference — no need to load multiple models.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel

# --- Pydantic schemas to force structured output ---


class Node(BaseModel):
    name: str
    label: str  # Developer, Repository, Feature, Bug, Concept


class Edge(BaseModel):
    source: str
    target: str
    relation: str  # WROTE, CONTAINS, FIXES, DEPENDS_ON, EXPLAINS


class ExtractionResult(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


# --- LLM setup ---
BASE_URL = "http://127.0.0.1:1234/v1"

# Single shared client — LM Studio handles concurrent requests internally
client = OpenAI(base_url=BASE_URL, api_key="lm-studio")

SYSTEM_PROMPT = """You are a strict information extraction system. You extract high-level entities and relationships from software project text and return ONLY valid JSON.

You must return a JSON object with two arrays: "nodes" and "edges".

Each node has:
- "name": a short, human-readable name (2-4 words max, Title Case)
- "label": one of [Developer, Repository, Feature, Bug, Concept]

Label definitions:
- Developer: a person or GitHub username (e.g. "shadcn", "Tim Neutkens")
- Repository: a GitHub project or library (e.g. "Next.js", "Tailwind CSS", "shadcn/ui")
- Feature: a user-facing capability or component (e.g. "Dark Mode", "Calendar Component", "CLI Tool")
- Bug: a reported problem or error (e.g. "Hydration Error", "Build Failure")
- Concept: a technical idea or pattern (e.g. "CSS Variables", "Server Components", "Design Tokens")

Each edge has:
- "source": the name of the source node (must exactly match a node name above)
- "target": the name of the target node (must exactly match a node name above)
- "relation": one of [WROTE, CONTAINS, FIXES, DEPENDS_ON, EXPLAINS]

STRICT RULES — if you break these, the pipeline will fail:
- NEVER extract: function names, variable names, file paths, HTML tags, CSS class names, code snippets, or raw identifiers like `verbose()`, `<select>`, `SHADCN_VERBOSE`, or `apps/v4/calendar.tsx`
- ONLY extract real-world concepts a human would naturally talk about
- Names must be plain English, not code. Bad: "forceMount". Good: "Force Mount Prop"
- If you find nothing meaningful, return {"nodes": [], "edges": []}
- Return ONLY the JSON object. No explanation, no markdown."""

USER_TEMPLATE = """Extract entities and relationships from this text:

\"\"\"
{text}
\"\"\"
"""


def get_loaded_model() -> str:
    """Return the ID of the first model currently loaded in LM Studio."""
    models = client.models.list()
    if models.data:
        return models.data[0].id
    return ""  # LM Studio uses whatever is loaded


def extract_from_chunk(text: str, model_id: str) -> dict | None:
    """Send a chunk to the LLM and parse the structured response."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        parsed = json.loads(raw)
        result = ExtractionResult(**parsed)
        return result.model_dump()

    except (json.JSONDecodeError, Exception) as e:
        print(f"  Extraction failed: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=2,
        help="Number of concurrent requests to send to LM Studio (default: 4). "
        "Match this to the 'Parallel Requests' setting in LM Studio.",
    )
    args = parser.parse_args()

    with open("output.json", "r") as f:
        chunks = json.load(f)

    model_id = get_loaded_model()
    print(f"Model: {model_id or '(default)'}")
    print(f"Parallel requests: {args.parallel}")
    print(f"Chunks to process: {len(chunks)}\n")

    all_nodes = []
    all_edges = []
    lock = threading.Lock()
    completed = 0

    def process(idx: int):
        chunk = chunks[idx]
        result = extract_from_chunk(chunk["text"], model_id)
        return idx, chunk, result

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(process, i): i for i in range(len(chunks))}
        for future in as_completed(futures):
            try:
                idx, chunk, result = future.result()
            except Exception as e:
                print(f"  Chunk failed unexpectedly: {e}")
                result, chunk = None, {}

            with lock:
                completed += 1
                print(f"  [{completed}/{len(chunks)}] chunk {futures[future]}")
                if result:
                    for node in result["nodes"]:
                        node["repo"] = chunk.get("repo", "unknown")
                    all_nodes.extend(result["nodes"])
                    all_edges.extend(result["edges"])

    # Deduplicate nodes by (name, label)
    seen = set()
    unique_nodes = []
    for node in all_nodes:
        key = (node["name"].lower(), node["label"])
        if key not in seen:
            seen.add(key)
            unique_nodes.append(node)

    entities = {"nodes": unique_nodes, "edges": all_edges}

    with open("entities.json", "w") as f:
        json.dump(entities, f, indent=2)

    print(f"\nExtracted {len(unique_nodes)} unique nodes, {len(all_edges)} edges")
    print("Saved to entities.json")


if __name__ == "__main__":
    main()
