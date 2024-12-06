import json
import matplotlib.pyplot as plt

data_file = "parsed_warmup.json"

with open(data_file, "r") as file:
    data = json.load(file)

steps = [entry["step"] for entry in data]
loss = [entry["loss"] for entry in data]
reranking_mrr = [entry["reranking_mrr"] for entry in data if "reranking_mrr" in entry]
reranking_steps = [entry["step"] for entry in data if "reranking_mrr" in entry]

period = 5
sampled_steps = steps[::period]
sampled_loss = loss[::period]

adjusted_loss = [loss[0]] + [min(l, 0.3) for l in loss[1:]]

plt.figure(figsize=(10, 6))
plt.plot(sampled_steps, sampled_loss, label="Warmup Loss Every 1000 Steps", marker='o', linestyle='-')
plt.axhline(y=0.3, color="red", linestyle="dotted")
plt.title("Warmup - Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.ylim(0, 0.35)
plt.xlim(0, max(steps))
plt.savefig("warm_up_loss.jpg", format='jpg', dpi=300)
plt.savefig("warm_up_loss.pdf", format='pdf', dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(reranking_steps, reranking_mrr, label="Warmup - Reranking MRR", color="red", marker='o', linestyle='-')
plt.title("Warmup - Reranking MRR")
plt.xlabel("Step")
plt.ylabel("MRR")
plt.legend()
plt.grid(True)
plt.savefig("warm_up_mrr.jpg", format='jpg', dpi=300)
plt.savefig("warm_up_mrr.pdf", format='pdf', dpi=300)
