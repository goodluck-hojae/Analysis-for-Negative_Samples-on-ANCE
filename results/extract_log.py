import json
import os

input_files = [
    "subset_10_train-a40-26398175-ann_data_4.out",
    "subset_10_train-a40-26398212-ann_data_4_bottom_neg.out",
    "subset_10_train-a40-26449628-ann_data_4_random.out",
    "subset_10_train-a40-26456768-ann_data_4_bottom_neg_only.out"
]

output_dir = "extracted_data"
os.makedirs(output_dir, exist_ok=True)

for input_file in input_files:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")
    extracted_data = []

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('{"learning_rate"') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    extracted_data.append(data)
                except json.JSONDecodeError:
                    continue

    with open(output_file, 'w') as outfile:
        json.dump(extracted_data, outfile, indent=4)

    print(f"Extracted {len(extracted_data)} entries from {input_file}. Saved to {output_file}.")
