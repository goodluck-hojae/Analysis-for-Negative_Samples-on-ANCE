import json

input_file = "cleaned_warmup.out"
output_file = "parsed_results.json"

parsed_data = []

with open(input_file, 'r') as file:
    current_entry = None
    for line in file:
        line = line.strip()
        if line.startswith('{"learning_rate"') and line.endswith('}'):
            try:
                current_entry = json.loads(line)
                parsed_data.append(current_entry)
            except json.JSONDecodeError:
                continue

        if "Reranking/Full ranking mrr:" in line:
            try:
                parts = line.split("Reranking/Full ranking mrr:")[1].strip()
                print(parts)
                reranking_mrr, full_ranking_mrr = map(float, parts.split("/"))
                if current_entry:
                    print(current_entry)
                    current_entry['reranking_mrr'] = reranking_mrr
                    current_entry['full_ranking_mrr'] = full_ranking_mrr
                    print(current_entry)
            except (IndexError, ValueError):
                continue

with open(output_file, 'w') as outfile:
    json.dump(parsed_data, outfile, indent=4)

print(f"Parsed {len(parsed_data)} entries. Results saved to {output_file}.")
