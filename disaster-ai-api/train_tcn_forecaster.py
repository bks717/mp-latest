"""
Training Script for Novel Temporal Convolutional Network with Multi-Head Self-Attention (TCN-Attention).
Multi-Horizon Flood Forecasting across 72-Hour Sequences (T+0, T+6, T+12, T+24, T+48, T+72).
Saves trained weights to tcn_flood_forecaster.pth.
"""

import os
import math
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tcn_attention_forecaster import TCNAttentionForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainTCNAttention")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Regime 72-Hour Sequence Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────
class MultiHorizonFloodDataset(Dataset):
    """
    Synthesizes empirical 72-hour hydrometeorological event sequences across 6 storm regimes:
      1. Monsoon Cloudburst & Riverine Flood (continuous intense precipitation + saturated soil)
      2. Severe Convective Flash Flood (extreme CAPE, sudden high-intensity pulse)
      3. Slow Riverine Surge (high upstream discharge, steady moderate rain)
      4. Frontal Rain System (moderate rain with moderate retention)
      5. Normal Baseflow Conditions (dry or light diurnal showers)
      6. Saturated Post-Storm Runoff (subsiding rain with peak river flow lag)
    """
    def __init__(self, num_samples: int = 6000, seed: int = 42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.seq_len = 72
        self.num_features = 10
        self.horizons = [0, 6, 12, 24, 48, 72]

        self.inputs = []
        self.horizon_probs = []
        self.severity_classes = []
        self.peak_times = []

        for _ in range(num_samples):
            regime = np.random.choice([
                "monsoon_severe", "convective_flash", "riverine_surge",
                "frontal_moderate", "normal_baseflow", "lagged_runoff"
            ], p=[0.20, 0.20, 0.20, 0.15, 0.15, 0.10])

            seq = np.zeros((self.seq_len, self.num_features), dtype=np.float32)
            # Base features: [p, temp, rh, press, wind, cape, sm_surf, sm_root, q, r]

            if regime == "monsoon_severe":
                # High continuous rain, peak around hour 18-30, soil saturated
                peak_hr = np.random.uniform(18, 36)
                p_intensity = np.random.uniform(40, 110) # mm/hr peak
                cape_val = np.random.uniform(800, 2200)
                sm_surf = np.random.uniform(0.85, 0.98)
                sm_root = np.random.uniform(0.80, 0.95)
                base_q = np.random.uniform(2.5, 4.5)

                for t in range(self.seq_len):
                    # Bell shape rainfall around peak_hr
                    rain = p_intensity * np.exp(-0.5 * ((t - peak_hr) / 8.0) ** 2) + np.random.exponential(5.0)
                    seq[t, 0] = min(3.0, rain / 50.0)
                    seq[t, 1] = (np.random.uniform(24, 30) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.85, 0.98)
                    seq[t, 3] = (1013.25 - np.random.uniform(996, 1004)) / 25.0
                    seq[t, 4] = np.random.uniform(20, 55) / 60.0
                    seq[t, 5] = cape_val / 2500.0
                    seq[t, 6] = sm_surf
                    seq[t, 7] = sm_root
                    seq[t, 8] = (base_q + (t / 72.0) * 1.5) / 8.0
                    seq[t, 9] = min(5.0, 1.8 + (t / 72.0) * 2.2) / 2.0

                probs = [0.45, 0.70, 0.92, 0.98, 0.85, 0.65]
                sevs = [2, 3, 4, 4, 3, 2]
                actual_peak = peak_hr + 4.0

            elif regime == "convective_flash":
                # Sudden extreme burst at hour 10-18, high CAPE, rapid subsidence
                peak_hr = np.random.uniform(10, 20)
                p_intensity = np.random.uniform(60, 140)
                cape_val = np.random.uniform(1800, 3200)
                sm_surf = np.random.uniform(0.60, 0.85)
                sm_root = np.random.uniform(0.55, 0.80)

                for t in range(self.seq_len):
                    rain = p_intensity * np.exp(-0.5 * ((t - peak_hr) / 3.0) ** 2)
                    seq[t, 0] = min(3.0, rain / 50.0)
                    seq[t, 1] = (np.random.uniform(28, 35) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.75, 0.95)
                    seq[t, 3] = (1013.25 - np.random.uniform(998, 1008)) / 25.0
                    seq[t, 4] = np.random.uniform(30, 70) / 60.0
                    seq[t, 5] = cape_val / 2500.0
                    seq[t, 6] = min(1.0, sm_surf + (0.3 if t > peak_hr else 0.0))
                    seq[t, 7] = sm_root
                    seq[t, 8] = np.random.uniform(1.2, 2.5) / 8.0
                    seq[t, 9] = (1.5 if t > peak_hr else 0.9) / 2.0

                probs = [0.20, 0.85, 0.95, 0.55, 0.25, 0.10]
                sevs = [1, 4, 4, 2, 1, 0]
                actual_peak = peak_hr + 1.5

            elif regime == "riverine_surge":
                # High upstream discharge wave arriving around hour 24-48, moderate rain
                peak_hr = np.random.uniform(28, 50)
                sm_surf = np.random.uniform(0.75, 0.90)
                sm_root = np.random.uniform(0.78, 0.92)

                for t in range(self.seq_len):
                    rain = np.random.uniform(5, 25)
                    discharge_mult = 1.0 + 3.0 * np.exp(-0.5 * ((t - peak_hr) / 14.0) ** 2)
                    seq[t, 0] = rain / 50.0
                    seq[t, 1] = (np.random.uniform(18, 26) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.80, 0.95)
                    seq[t, 3] = (1013.25 - 1008.0) / 25.0
                    seq[t, 4] = np.random.uniform(15, 35) / 60.0
                    seq[t, 5] = np.random.uniform(300, 900) / 2500.0
                    seq[t, 6] = sm_surf
                    seq[t, 7] = sm_root
                    seq[t, 8] = (3.5 * discharge_mult) / 8.0
                    seq[t, 9] = min(5.0, 1.2 * discharge_mult) / 2.0

                probs = [0.35, 0.50, 0.72, 0.91, 0.88, 0.55]
                sevs = [1, 2, 3, 4, 4, 2]
                actual_peak = peak_hr

            elif regime == "frontal_moderate":
                # Intermittent moderate rain, soil dry to moist, moderate risk
                for t in range(self.seq_len):
                    rain = np.random.exponential(8.0)
                    seq[t, 0] = min(1.5, rain / 50.0)
                    seq[t, 1] = (np.random.uniform(15, 22) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.65, 0.85)
                    seq[t, 3] = (1013.25 - 1012.0) / 25.0
                    seq[t, 4] = np.random.uniform(10, 30) / 60.0
                    seq[t, 5] = np.random.uniform(100, 500) / 2500.0
                    seq[t, 6] = np.random.uniform(0.40, 0.65)
                    seq[t, 7] = np.random.uniform(0.40, 0.60)
                    seq[t, 8] = np.random.uniform(1.0, 2.0) / 8.0
                    seq[t, 9] = np.random.uniform(0.8, 1.3) / 2.0

                probs = [0.10, 0.22, 0.35, 0.45, 0.30, 0.15]
                sevs = [0, 1, 1, 2, 1, 0]
                actual_peak = 24.0

            elif regime == "normal_baseflow":
                # Clear skies / low humidity / low streamflow
                for t in range(self.seq_len):
                    seq[t, 0] = np.random.uniform(0, 1.5) / 50.0
                    seq[t, 1] = (np.random.uniform(18, 28) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.40, 0.65)
                    seq[t, 3] = (1013.25 - 1016.0) / 25.0
                    seq[t, 4] = np.random.uniform(5, 20) / 60.0
                    seq[t, 5] = np.random.uniform(0, 150) / 2500.0
                    seq[t, 6] = np.random.uniform(0.20, 0.40)
                    seq[t, 7] = np.random.uniform(0.25, 0.42)
                    seq[t, 8] = np.random.uniform(0.5, 1.2) / 8.0
                    seq[t, 9] = np.random.uniform(0.6, 1.0) / 2.0

                probs = [0.03, 0.04, 0.04, 0.05, 0.04, 0.03]
                sevs = [0, 0, 0, 0, 0, 0]
                actual_peak = 0.0

            else: # lagged_runoff
                # Past rain ended at hour 0, river peak arrives at hour 12-24
                peak_hr = np.random.uniform(12, 24)
                for t in range(self.seq_len):
                    rain = np.random.uniform(0, 4.0)
                    discharge_mult = 1.0 + 2.5 * np.exp(-0.5 * ((t - peak_hr) / 10.0) ** 2)
                    seq[t, 0] = rain / 50.0
                    seq[t, 1] = (np.random.uniform(20, 26) - 10) / 25.0
                    seq[t, 2] = np.random.uniform(0.70, 0.88)
                    seq[t, 3] = (1013.25 - 1010.0) / 25.0
                    seq[t, 4] = np.random.uniform(10, 25) / 60.0
                    seq[t, 5] = np.random.uniform(200, 600) / 2500.0
                    seq[t, 6] = max(0.4, 0.85 - (t / 72.0) * 0.3)
                    seq[t, 7] = max(0.4, 0.88 - (t / 72.0) * 0.25)
                    seq[t, 8] = (2.8 * discharge_mult) / 8.0
                    seq[t, 9] = min(5.0, 1.4 * discharge_mult) / 2.0

                probs = [0.55, 0.82, 0.90, 0.70, 0.35, 0.15]
                sevs = [2, 3, 4, 3, 1, 0]
                actual_peak = peak_hr

            self.inputs.append(seq)
            self.horizon_probs.append(probs)
            self.severity_classes.append(sevs)
            self.peak_times.append([actual_peak])

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.horizon_probs = np.array(self.horizon_probs, dtype=np.float32)
        self.severity_classes = np.array(self.severity_classes, dtype=np.int64)
        self.peak_times = np.array(self.peak_times, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float32),
            "probs": torch.tensor(self.horizon_probs[idx], dtype=torch.float32),
            "severity": torch.tensor(self.severity_classes[idx], dtype=torch.long),
            "peak_time": torch.tensor(self.peak_times[idx], dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_tcn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Training TCNAttentionForecaster on device: {device}")

    # Dataset split: 80% train, 20% val
    full_dataset = MultiHorizonFloodDataset(num_samples=7000, seed=42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    model = TCNAttentionForecaster(
        in_features=10,
        tcn_channels=64,
        num_heads=4,
        horizons=6,
        dropout=0.15
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_l1 = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_weights_path = os.path.join(os.path.dirname(__file__), "tcn_flood_forecaster.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input"].to(device)
            target_probs = batch["probs"].to(device)
            target_sev = batch["severity"].to(device) # [B, 6]
            target_peak = batch["peak_time"].to(device) # [B, 1]

            optimizer.zero_grad()
            out = model(x)

            loss_prob = criterion_bce(out["horizon_probs"], target_probs)
            # severity logits: [B, 6, 5] -> flatten to [B*6, 5]
            sev_logits_flat = out["severity_logits"].view(-1, 5)
            sev_targets_flat = target_sev.view(-1)
            loss_sev = criterion_ce(sev_logits_flat, sev_targets_flat)
            loss_peak = criterion_l1(out["peak_time_hours"], target_peak)

            total_loss = loss_prob + 0.4 * loss_sev + 0.05 * loss_peak
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * len(x)

        train_loss /= train_size
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_peak = 0.0
        val_sev_correct = 0
        total_eval_points = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                target_probs = batch["probs"].to(device)
                target_sev = batch["severity"].to(device)
                target_peak = batch["peak_time"].to(device)

                out = model(x)
                loss_prob = criterion_bce(out["horizon_probs"], target_probs)
                sev_logits_flat = out["severity_logits"].view(-1, 5)
                sev_targets_flat = target_sev.view(-1)
                loss_sev = criterion_ce(sev_logits_flat, sev_targets_flat)
                loss_peak = criterion_l1(out["peak_time_hours"], target_peak)

                total_loss = loss_prob + 0.4 * loss_sev + 0.05 * loss_peak
                val_loss += total_loss.item() * len(x)

                val_mae_peak += torch.abs(out["peak_time_hours"] - target_peak).sum().item()

                preds_sev = torch.argmax(sev_logits_flat, dim=-1)
                val_sev_correct += (preds_sev == sev_targets_flat).sum().item()
                total_eval_points += len(sev_targets_flat)

        val_loss /= val_size
        val_mae_peak /= val_size
        val_sev_acc = val_sev_correct / max(1, total_eval_points)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                f"Epoch {epoch:02d}/{epochs} — Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Severity Acc: {val_sev_acc:.4f} | Peak Timing MAE: {val_mae_peak:.2f}h"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

    logger.info(f"✅ Saved trained TCNAttentionForecaster to {best_weights_path}")

    # Final summary metrics
    metrics = {
        "model_name": "TCNAttentionForecaster (Dilated Causal 1D Conv + Multi-Head Self-Attention)",
        "receptive_field_hours": 125,
        "horizons_forecasted": [0, 6, 12, 24, 48, 72],
        "training_samples": train_size,
        "validation_samples": val_size,
        "best_val_loss": round(best_val_loss, 4),
        "validation_severity_accuracy": round(val_sev_acc, 4),
        "peak_timing_mae_hours": round(val_mae_peak, 2),
        "weights_path": best_weights_path
    }

    metrics_path = os.path.join(os.path.dirname(__file__), "tcn_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved TCN metrics to {metrics_path}")
    logger.info("=================================================================")
    logger.info("🎉 TCN-Attention Temporal Forecaster Training Complete!")
    logger.info("=================================================================")


if __name__ == "__main__":
    train_tcn()
