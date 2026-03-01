import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def run_visual_demo(
	model,
    tokenizer,
    aspect_keys, 
    sentence,
	aux_embs,
    device,
    max_length,
    save_heatmap=None,
):

    num_aspects = aux_embs.size(0)

    enc = tokenizer(sentence, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    aux_embs_batch = aux_embs.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model((input_ids, attention_mask, aux_embs_batch))
        _, _, attn_outputs_all, aspect_embs_proj, _ = outputs

    attn = torch.stack([x.squeeze(0) for x in attn_outputs_all], dim=0)
    proj = torch.stack([x.squeeze(0) for x in aspect_embs_proj], dim=0)

    attn_norm = torch.nn.functional.normalize(attn, dim=-1)
    proj_norm = torch.nn.functional.normalize(proj, dim=-1)

    sim = torch.matmul(attn_norm, proj_norm.T).cpu().numpy()
    self_sims = np.diag(sim)

    other_means = np.array([
        (sim[i].sum() - sim[i, i]) / (num_aspects - 1)
        for i in range(num_aspects)
    ])

    margins = self_sims - other_means
    ranked_idx = np.argsort(-margins)

    print("\n==============================")
    print("Aspect Pull Strength Ranking")
    print("==============================")

    for rank, idx in enumerate(ranked_idx):
        print(
            f"{rank+1:2d}. {aspect_keys[idx]} | "
            f"self={self_sims[idx]:.4f} | "
            f"others_mean={other_means[idx]:.4f} | "
            f"margin={margins[idx]:.4f}"
        )

    x = np.arange(num_aspects)

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, self_sims, width=0.4, label="Self-sim")
    plt.bar(x + 0.2, other_means, width=0.4, label="Others mean")

    plt.xticks(x, aspect_keys, rotation=90)
    plt.legend()
    plt.ylabel("Cosine similarity")
    plt.title("Self Similarity vs Mean Others")
    plt.tight_layout()

    if save_heatmap:
        comp_path = os.path.splitext(save_heatmap)[0] + "_comparison.png"
        plt.savefig(comp_path)
        print(f"Saved comparison chart to {comp_path}")
    else:
        plt.show()

def load_aspects_from_file(path):
	with open(path, 'r', encoding='utf-8') as f:
		lines = [l.strip() for l in f if l.strip()]
	return lines
