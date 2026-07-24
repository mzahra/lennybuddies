# from pathlib import Path
# from document_summariser import summarize_markdown_file

# file_path = Path("/Users/muditairan/Desktop/Week3-Project/lennybuddies/knowledge_base/primary/newsletters/beyond-vibe-checks-a-pms-complete-guide-to-evals.md")

# # result = summarize_markdown_file(file_path, model_kind="bart")
# # print(result["note"])
# print("--------------separator-----------")
# result = summarize_markdown_file(file_path, model_kind="t5")
# print(result["note"])


# from pprint import pprint
# from pathlib import Path
# from document_processor import load_knowledge_chunks

# target = "newsletters/beyond-vibe-checks-a-pms-complete-guide-to-evals.md"

# chunks = [
#     chunk for chunk in load_knowledge_chunks()
#     if chunk["path"] == target
# ]

# pprint(chunks)


from pathlib import Path
import json
from document_summariser import summarize_markdown_file

file_path = Path("/Users/muditairan/Desktop/Week3-Project/lennybuddies/knowledge_base/primary/newsletters/essential-reading-for-product-builders-part-1.md")

result = summarize_markdown_file(file_path, model_kind="bart")
print(json.dumps(result, indent=2, ensure_ascii=False))