"""
Phase 1: Scrape GitHub repos (README + closed issues) and chunk into output.json
"""

import json
import os

import dotenv
from github import Auth, Github
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

token = os.getenv("GitHub_Token")
if not token:
    raise RuntimeError("GitHub_Token not found in .env")

# --- Config ---
REPOS = [
    "shadcn-ui/ui",
    "tailwindlabs/tailwindcss",
    "vercel/next.js",
]
MAX_ISSUES = 15  # per repo, keep it manageable

# --- Setup ---
g = Github(auth=Auth.Token(token))
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

all_chunks = []

for repo_name in REPOS:
    print(f"Scraping {repo_name}...")
    repo = g.get_repo(repo_name)

    # README
    try:
        readme_text = repo.get_readme().decoded_content.decode("utf-8")
        for i, chunk in enumerate(splitter.split_text(readme_text)):
            all_chunks.append({
                "source": "readme",
                "repo": repo_name,
                "chunk_id": i,
                "text": chunk,
            })
    except Exception as e:
        print(f"  Could not get README for {repo_name}: {e}")

    # Closed issues
    issues = repo.get_issues(state="closed", sort="updated", direction="desc")
    count = 0
    for issue in issues:
        if count >= MAX_ISSUES:
            break
        if not issue.body:
            continue
        for i, chunk in enumerate(splitter.split_text(issue.body)):
            all_chunks.append({
                "source": "issue",
                "repo": repo_name,
                "title": issue.title,
                "chunk_id": i,
                "text": chunk,
            })
        count += 1

print(f"Total chunks: {len(all_chunks)}")

with open("output.json", "w") as f:
    json.dump(all_chunks, f, indent=2)

print("Saved to output.json")
