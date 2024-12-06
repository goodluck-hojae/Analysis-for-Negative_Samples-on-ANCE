
input_file = "warmup.out"  
output_file = "cleaned_warmup.out"  

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if "Iteration:" in line or  "Eval:" in line or not line.strip():
            continue
        outfile.write(line)
