import json
import pandas as pd

INPUT_PATH = "data/raw/problems_data.jsonl"
OUTPUT_PATH = "data/problems.csv"

records = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        records.append({
            "title": obj.get("title", ""),
            "description": obj.get("description", ""),
            "input_description": obj.get("input_description", ""),
            "output_description": obj.get("output_description", ""),
            "problem_class": obj.get("problem_class", "").lower(),
            "problem_score": obj.get("problem_score", 0.0)
        })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} problems to {OUTPUT_PATH}")
