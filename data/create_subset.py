import pandas as pd

# Load the data
file_path = "/datasets/ai/msmarco/passage/collection.tsv"  # Replace with the actual path if different
# file_path = "/datasets/ai/msmarco/passage/queries.train.tsv"  
data = pd.read_csv(file_path, sep="\t", header=None)

# Calculate 10% of the data
shuffled_data = data.sample(frac=0.3).reset_index(drop=True)


# Save the reduced data to a new file
output_path = "/home/hojaeson_umass_edu/hojae_workspace/project/ance/data/30/30_passage/collection.tsv" #queries. # Specify output path if needed
shuffled_data.to_csv(output_path, sep="\t", index=False, header=False)

print(f"Reduced file saved to {output_path}")