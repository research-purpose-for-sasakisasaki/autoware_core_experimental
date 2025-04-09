import os
import re
import time
from collections import defaultdict

import openai
import tiktoken
import google.generativeai as genai
import qdrant_client as qdrant
from ask_review_example import apply_review_for_one_file


def parse_git_diff_to_dict(diff_text) -> dict[str, str]:
    lines = diff_text.splitlines()
    file_diffs = defaultdict(list)
    current_file = None
    in_hunk = False
    old_line_num = None
    new_line_num = None

    for line in lines:
        # Detect file diff header
        if line.startswith("diff --git"):
            match = re.match(r'^diff --git a/(.*?) b/(.*)', line)
            if match:
                current_file = match.group(2)  # use destination file path
                file_diffs[current_file].append(line)
                in_hunk = False
            continue

        if current_file is None:
            continue  # Skip lines before any file diff appears

        # Skip headers (but store them for context)
        if line.startswith(("index ", "--- ", "+++ ")):
            file_diffs[current_file].append(line)
            continue

        # Detect hunk header
        hunk_match = re.match(r'^@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
        if hunk_match:
            old_line_num = int(hunk_match.group(1))
            new_line_num = int(hunk_match.group(2))
            file_diffs[current_file].append(f"{line}  (old_line={old_line_num}, new_line={new_line_num})")
            in_hunk = True
            continue

        # Ignore anything outside hunks
        if not in_hunk:
            file_diffs[current_file].append(line)
            continue

        # Process diff lines inside hunk
        if line.startswith('+') and not line.startswith('+++'):
            file_diffs[current_file].append(f"{new_line_num:4d} + {line[1:]}")
            new_line_num += 1
        elif line.startswith('-') and not line.startswith('---'):
            file_diffs[current_file].append(f"{old_line_num:4d} - {line[1:]}")
            old_line_num += 1
        else:
            # Context line
            file_diffs[current_file].append(f"{new_line_num:4d}   {line[1:]}")
            new_line_num += 1
            old_line_num += 1

    # Convert lists to joined strings
    return {fname: '\n'.join(contents) for fname, contents in file_diffs.items()}


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # fallback if model isn't explicitly supported
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def initialize_clients() -> tuple[openai.Client, genai.GenerativeModel, qdrant.QdrantClient]:
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


RELEVANT_SCORE_THRESHOLD = 0.7
def get_relevant_guidelines(client, openai_client, code_chunk):
    """Retrieve relevant coding guidelines using vector search."""
    results = client.search(
        collection_name="coding_guideline",
        query_vector=openai_client.embeddings.create(
            input=[code_chunk],
            model="text-embedding-3-small",
        ).data[0].embedding,
    )
    
    return [r.payload for r in results if r.score > RELEVANT_SCORE_THRESHOLD]


MAX_TOKENS = 8000    # < 8192: slightly less than 8192 to ensure the token count is less than 8192
SUPPORTED_FILE_EXTENSIONS = [".cpp", ".c", ".hpp", ".h", ".py", ".md", ".txt", ".rst"]
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_file", type=str, required=True)
    args = parser.parse_args()

    # Initialize clients
    try:
        openai_client, gemini_client, qdrant_client = initialize_clients()
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        return
    except Exception as e:
        print(f"Unknown error initializing clients: {e}")
        return

    with open(args.diff_file, 'r') as f:
        diff_content = f.read()

    parsed_diff = parse_git_diff_to_dict(diff_content)

    review_targets = []

    for file, changes in parsed_diff.items():

        # Skip if the file extension is not supported
        if not any(file.endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS):
            print(f"Skipping {file} because it is not supported")
            continue

        # Count tokens
        count = count_tokens(changes)

        # Consider a case when the number of tokens can be dividable by MAX_TOKENS with remainder or not
        n_chunks = count // MAX_TOKENS + 1 if count % MAX_TOKENS > 0 else count // MAX_TOKENS

        if n_chunks == 0:
            print(f"No chunks found for {file}")
            continue

        # Although the number of tokens and length of the code chunk are different,
        # let's roughtly divide the code chunk into n_chunks.
        chunk_size = len(changes) // n_chunks

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(changes))
            code_chunk = changes[start:end]

            # Cut off last line if it is incomplete like as below
            '''
            225 -     const auto backward_distance = 40.0;
            226 -     const auto forward_dis
            '''

            # Check if the last character is not a newline
            if code_chunk[-1] != '\n':
                # Find the position of the last newline
                last_newline_pos = code_chunk.rfind('\n')
                # Skip if no newline is found
                if last_newline_pos == -1:
                    continue

                # Cut off the last line
                code_chunk = code_chunk[:last_newline_pos]

            guideline = get_relevant_guidelines(qdrant_client, openai_client, code_chunk)

            # If no guideline is found, skip the review
            if not guideline:
                print(f"No guideline found for {file}")
                continue

            review_targets.append((file, code_chunk, guideline))

    for file, code_chunk, guideline in review_targets:
        apply_review_for_one_file(gemini_client, code_chunk, guideline, file)
        time.sleep(1)    # To avoid rate limit

if __name__ == "__main__":
    main()
