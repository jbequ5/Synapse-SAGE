# synapse/meta_rl.py
# SAGE v0.9.15 – Meta-RL Trainer
# Handles supervised fine-tuning of new MoDE specialists and bank entries using the 5-objective vector.

import logging
import torch
import torch.nn as nn
from typing import List

from synapse.fragment import SolveFragment

logger = logging.getLogger(__name__)

class MetaRLTrainer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def supervised_fine_tune(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
                             epochs: int = 8, lr: float = 3e-4):
        """Real supervised fine-tuning on the 5-objective vector."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if epoch % 2 == 0:
                logger.info(f"Meta-RL fine-tune epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

        logger.info("✅ Meta-RL supervised fine-tuning completed")
        return model
