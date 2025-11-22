import os
import csv
from datetime import datetime
from huggingface_hub import login
from transformers import pipeline

# Hugging Face API token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN secret is missing!")

# Authenticate Hugging Face
login(HF_TOKEN)

# Model for text generation
MODEL_ID = "tiiuae/falcon-7b-instruct"
generator = pipeline("text-generation", model=MODEL_ID, device=-1, use_auth_token=HF_TOKEN)

# File paths
KEYWORDS_FILE = "keywords.csv"
POSTS_DIR = "_posts"
os.makedirs(POSTS_DIR, exist_ok=True)

# Read first keyword
with open(KEYWORDS_FILE, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    keywords = list(reader)

if not keywords:
    print("No keywords left to process.")
    exit(0)

keyword = keywords.pop(0)[0]
print("Generating post for keyword:", keyword)

# Generate AI content
prompt = f"Write a detailed, unique product review article for: {keyword}. Include headings, pros/cons, and short SEO-friendly paragraphs."
result = generator(prompt, max_new_tokens=400, do_sample=True)[0]["generated_text"]

print("Generated content preview:", result[:100], "...")

# Save markdown post
today = datetime.now().strftime("%Y-%m-%d")
filename = f"{POSTS_DIR}/{today}-{keyword.replace(' ', '-')}.md"
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"---\ntitle: \"{keyword}\"\ndate: {today}\n---\n\n")
    f.write(result)

print(f"Saved post to {filename}")

# Update keywords.csv
with open(KEYWORDS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(keywords)
