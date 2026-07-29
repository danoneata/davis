import os

import fire
import torch

from siglip import Dataset, SigLIP, evaluate, load_config


def main(config_name: str, split: str = "dev5"):
    """Evaluate the best checkpoint for `config_name` on `split` (default: dev5).

    Loads `configs/<config_name>.yaml` to find `root` and `checkpoint_dir`, then
    scores `checkpoints/<config_name>/best.pt` on `<root>/<split>_grouped_embeddings.pkl`
    and prints R@1, R@5, R@10.
    """
    cfg = load_config(config_name)
    root = cfg["root"]
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")

    embeddings_path = os.path.join(root, f"{split}_grouped_embeddings.pkl")
    checkpoint_path = os.path.join(checkpoint_dir, config_name, "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SigLIP(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = Dataset(embeddings_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5000, shuffle=False, num_workers=0
    )
    r1, r5, r10 = evaluate(model, dataloader)

    print(f"Config `{config_name}` on split `{split}` (n={len(dataset)}):")
    print(f"  R@1:  {r1:.4f}")
    print(f"  R@5:  {r5:.4f}")
    print(f"  R@10: {r10:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
