import os
import json
import requests
import google.generativeai as genai

# Do not read via config, but read from environment variables
try:
    COMMIT_SHA = os.getenv('COMMIT_SHA')
    REPO_OWNER = os.getenv('REPO_OWNER')
    REPO_NAME = os.getenv('REPO_NAME')
    PR_INDEX = os.getenv('PR_INDEX')
except KeyError:
    print("Please set the following environment variables: COMMIT_SHA, REPO_OWNER, REPO_NAME, PR_INDEX")
    exit(1)

# GitHub API configuration
try:
    github_token = os.getenv('GITHUB_TOKEN')
except KeyError:
    print("Please set the following environment variables: GITHUB_TOKEN")
    exit(1)

# API endpoint
url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{PR_INDEX}/reviews"

# Request headers
headers = {
    "Authorization": f"token {github_token}",
    "Accept": "application/vnd.github.v3+json"
}


# Function to ask for a review
def apply_review_for_one_file(gemini_client: genai.GenerativeModel, code_for_review: str, guideline: str, file_name: str):
    # Request payload
    payload = {
        "commit_id": COMMIT_SHA,
        "body": 'PLEASE FILL THIS',
        "event": "COMMENT",
        "comments": [
            {
                "path": file_name,
                "position": "PLEASE PROVIDE THE LINE NUMBER",
                "body": "PLEASE PROVIDE THE SUGGESTION"
            }
        ]
    }

    # Ask for a review
    question = f'''

Based on the following code guideline, provide a insightful and respectful review with the following format.
But ensure to detect the potential issue in the code.

Your review will be applied via GitHub API. Therefore,
  - please provide the body,
  - file name,
  - line number (position): please start counting from 1 just after the "Changes:",
  - and the suggestion.
in the following JSON format. Note that the property name must be enclosed by double quotes to be valid JSON.

Format:
```
{payload}
```

The following is the code for review.

Code:
```
{code_for_review}
```

Guideline:
```
{guideline}
```

'''

    try:    
        # Ask here
        answer = gemini_client.generate_content(question)
    
        # Remove unnecessary parts
        answer_text = answer.text.replace("```json\n", "").replace("\n```", "")

        # Make the POST request
        response = requests.post(url, headers=headers, json=json.loads(answer_text))

        # Check if the request was successful
        if response.status_code == 200:
            print("Review comment created successfully!")
            return answer_text 
        else:
            raise Exception(f"Error creating review comment. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    pass
