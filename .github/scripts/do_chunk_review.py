import os
import re

import openai
import google.generativeai as genai
import qdrant_client as qdrant
from ask_review_example import apply_review_for_one_file


def parse_diff(diff_content):
    files = {}
    current_file = None
    current_changes = []
    line_number = 0

    for line in diff_content.split('\n'):
        if line.startswith('diff --git'):
            if current_file:
                files[current_file] = current_changes
            current_file = re.search(r'b/(.+)$', line).group(1)
            current_changes = []
            line_number = 0
        elif line.startswith('@@'):
            match = re.search(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
            if match:
                line_number = int(match.group(2)) - 1
        elif line.startswith('+') or line.startswith('-') or line.startswith(' '):
            line_number += 1
            current_changes.append((line_number, line))

    if current_file:
        files[current_file] = current_changes

    return files

def group_changes(changes):
    grouped = []
    current_group = []
    last_line_number = None

    for line_number, line in changes:
        if last_line_number is None or line_number == last_line_number + 1:
            current_group.append((line_number, line))
        else:
            if current_group:
                grouped.append(current_group)
            current_group = [(line_number, line)]
        last_line_number = line_number

    if current_group:
        grouped.append(current_group)

    return grouped


def initialize_clients():
    '''
    Initialize API clients for OpenAI, Gemini, and Qdrant.
    Ensure the following environment variables are set:
      - OPENAI_API_KEY
      - GOOGLE_API_KEY
      - QDRANT_HOST
      - QDRANT_PORT
    '''
    openai_client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gemini_client = genai.GenerativeModel("gemini-1.5-flash")
    
    qdrant_client = qdrant.QdrantClient(
        host=os.environ["QDRANT_HOST"],
        port=os.environ["QDRANT_PORT"]
    )
    
    return openai_client, gemini_client, qdrant_client


def format_code_chunk(file, grouped_changes):
    """Format the code changes into a readable string."""
    chunk = f"File name: {file}\nChanges:\n"
    
    for group in grouped_changes:
        start_line, end_line = group[0][0], group[-1][0]
        chunk += f"  Lines {start_line}-{end_line}:\n"
        chunk += "".join(f"    {line}\n" for _, line in group)
    
    return chunk


def get_relevant_guidelines(client, openai_client, code_chunk):
    """Retrieve relevant coding guidelines using vector search."""
    results = client.search(
        collection_name="coding_guideline",
        query_vector=openai_client.embeddings.create(
            input=[code_chunk],
            model="text-embedding-3-small",
        ).data[0].embedding,
    )
    
    return [r.payload for r in results if r.score > 0.3]


def main():

    # Initialize clients
    try:
        openai_client, gemini_client, qdrant_client = initialize_clients()
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        return
    except Exception as e:
        print(f"Unknown error initializing clients: {e}")
        return

    with open('files.diff/sample-001.diff', 'r') as f:
        diff_content = f.read()

    parsed_diff = parse_diff(diff_content)

    for file, changes in parsed_diff.items():

        grouped_changes = group_changes(changes)
        code_chunk = format_code_chunk(file, grouped_changes)
        guideline = get_relevant_guidelines(qdrant_client, openai_client, code_chunk)

        # If no guideline is found, skip the review
        if not guideline:
            print(f"No guideline found for {file}")
            continue

        apply_review_for_one_file(gemini_client, code_chunk, guideline, file)

if __name__ == "__main__":
    main()
