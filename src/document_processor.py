import os
import glob
import json
import frontmatter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a content analysis assistant for a newsletter/podcast processing pipeline. You will be given existing metadata (title, subtitle, date, tags, source_filename) plus the raw body text of the content. Your job is to generate ONLY two new fields: "summary" and "keywords".

Rules for "summary":
- 2-4 sentences capturing the core argument or main takeaways
- Written in clear, complete prose (no cut-off sentences, no stray quote marks)
- Do not include the title/subtitle verbatim — add value beyond them

Rules for "keywords":
- 5-8 items, each a real topic, concept, named entity, or theme actually discussed in the content
- Never include: URLs, file paths, footnote/reference numbers, generic phrases like "part 1", 'step 1', firstly, secondly, or fragments of the title/subtitle
- Prefer specific nouns/phrases a reader would use to search for this content

Return ONLY valid JSON in this exact structure, with no other text:
{
  "summary": "...",
  "keywords": ["...", "...", "..."]
}
"""

def load_document(filepath, default_tag=""):
    post = frontmatter.load(filepath)
    tags = post.get("tags", [])
    if not tags and default_tag:
        tags = [default_tag]
    metadata = {
        "title": post.get("title", ""),
        "subtitle": post.get("subtitle", ""),
        "date": post.get("date", ""),
        "tags": tags,
        "source_filename": filepath,
    }
    return metadata, post.content

def process_document(metadata, body_text, model_name="gpt-4o-mini"):
    user_prompt = f"""Existing metadata:
Title: {metadata['title']}
Subtitle: {metadata['subtitle']}
Tags: {metadata['tags']}

Body content:
{body_text}"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )

    llm_output = json.loads(response.choices[0].message.content)

    return {
        **metadata,
        "keywords": llm_output["keywords"],
        "summary": llm_output["summary"],
        "model": model_name
    }

def process_knowledge_base():
    results = []
    folder_tags = {
        "knowledge_base/primary/newsletters/": "newsletter",
        "knowledge_base/primary/podcasts/": "podcast",
    }
    # for folder, default_tag in folder_tags.items():
    for folder, default_tag in [("knowledge_base/primary/test/", "newsletter")]:
        for filepath in glob.glob(os.path.join(folder, "*.md")):
            metadata, body_text = load_document(filepath, default_tag=default_tag)
            result = process_document(metadata, body_text)
            results.append(result)
            print(f"Processed: {filepath}")
    return results

if __name__ == "__main__":
    all_results = process_knowledge_base()
    with open("knowledge_base/processed/processed_documents_zahra.json", "w") as f:
        json.dump(all_results, f, indent=2)