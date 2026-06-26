"""Self-supervised autoencoder pretraining for the tactile CNN encoder."""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk

from opentouch.tactile_autoencoder import TactileAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to preprocessed HF dataset.")
    p.add_argument("--output", required=True, help="Path to save pretrained encoder weights.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class TactileOnlyDataset(torch.utils.data.Dataset):
    """Wraps the HF dataset and returns only tactile pressure frames."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        pressure = torch.tensor(
            item["right_pressure_image"], dtype=torch.float32
        ) / 255.0
        # pressure shape: [16, 16]
        return pressure.unsqueeze(0)  # [1, 16, 16]


def main():
    args = parse_args()
    device = torch.device(args.device)

    log.info("Loading dataset...")
    hf_dataset = load_from_disk(args.data)
    dataset = TactileOnlyDataset(hf_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    log.info(f"Dataset size: {len(dataset)} frames")

    model = TactileAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            # batch shape: [B, 1, 16, 16]
            # wrap in fake T=1 sequence for _normalize_input
            x = batch.unsqueeze(1)  # [B, 1, 1, 16, 16]
            reconstruction, original = model(x)
            loss = criterion(reconstruction, original)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        log.info(f"Epoch {epoch+1}/{args.epochs} | loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.encoder.state_dict(), args.output)
    log.info(f"Saved encoder weights to {args.output}")


if __name__ == "__main__":
    main()