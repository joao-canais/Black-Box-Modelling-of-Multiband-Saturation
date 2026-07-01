import os
import json
import time
import random
import numpy as np
from pathlib import Path
import torch.optim as optim
from collections import defaultdict

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

import auraloss.time as auraloss_time
import auraloss.freq as auraloss_freq

print("All libraries imported!")

def set_deterministic_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic_seeds(42)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = str(device)
print(f"Using device = {device}")
print(f"PyTorch version: {torch.__version__}")


# PATHS
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path.home() / "projeto"))
DATASET_DIR = Path(os.environ.get("DATASET_DIR", PROJECT_DIR / "data" / "Dataset"))

CLEAN_DIR   = DATASET_DIR / "clean"
SAT_DIR     = DATASET_DIR / "saturated"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", PROJECT_DIR / "results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# HYPERPARAMETERS
SAMPLE_RATE   = 44100
CHUNK_SAMPLES = 8192

TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.15
TEST_RATIO    = 1 - TRAIN_RATIO - VAL_RATIO

HIDDEN_SIZE   = 96
SKIP_CON      = 1
NUM_LAYERS    = 2

EPOCHS        = 1000
BATCH_SIZE    = 16
LEARNING_RATE = 5e-4
SEGMENT_LEN   = 22050
INIT_LEN      = 0
UP_FR         = 2048
VAL_CHUNK     = 100000
VALIDATION_F  = 2
PATIENCE      = 25
WEIGHT_DECAY  = 1e-5
THRESHOLD     = 1e-4

PRE_FILT      = [-0.85, 1]

LAMBDA_ESR    = 0.45
LAMBDA_MRSTFT = 0.45
LAMBDA_DC     = 0.10

NUM_WORKERS   = 4

# DATASET SPLIT FUNCTIONS
def total_minutes(pairs):
    total_seconds = 0.0
    for clean_path, _ in pairs:
        y_c, sr_clean = torchaudio.load(str(clean_path))
        total_seconds += y_c.shape[-1] / sr_clean
    hours = int(total_seconds // 3600)
    mins  = int((total_seconds % 3600) // 60)
    secs  = total_seconds % 60
    if hours > 0:
        return f"{hours}h {mins}m {round(secs)}s", round(total_seconds)
    elif mins > 0:
        return f"{mins}m {round(secs)}s", round(total_seconds)
    return f"{secs:.1f}s", round(total_seconds)


def set_split_ratio(train, validation, test, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.seed(42)
    train_shuffled      = random.sample(train, len(train))
    validation_shuffled = random.sample(validation, len(validation))
    test_shuffled       = random.sample(test, len(test))
    val_target_len      = round(len(train_shuffled) * (val_ratio / train_ratio))
    test_target_len     = round(len(train_shuffled) * (test_ratio / train_ratio))
    return train_shuffled, validation_shuffled[:val_target_len], test_shuffled[:test_target_len]


def get_style_stats(pairs_list):
    style_pairs = defaultdict(list)
    for c_path, s_path in pairs_list:
        stem  = Path(c_path).stem
        parts = stem.split("_")
        style = parts[4] if len(parts) > 4 and parts[4] in ["FS", "MU", "PK", "ST"] else "UNKNOWN"
        style_pairs[style].append((c_path, s_path))
    stats = {}
    for style, s_pairs in style_pairs.items():
        dur_str, sec = total_minutes(s_pairs)
        stats[style] = {"count": len(s_pairs), "duration": dur_str, "duration_sec": sec}
    return stats


def split_by_instrument(pair_files, train_guitar="1", val_guitar="2", test_guitar="3", set_plit=False):
    buckets = defaultdict(list)
    for clean_path, sat_path in pair_files:
        parts     = Path(clean_path).stem.split("_")
        guitar_id = parts[1]
        buckets[guitar_id].append((clean_path, sat_path))

    train_pairs, val_pairs, test_pairs = [], [], []
    for guitar_id, pairs in buckets.items():
        if guitar_id == train_guitar:
            train_pairs.extend(pairs)
        elif guitar_id == val_guitar:
            val_pairs.extend(pairs)
        else:
            test_pairs.extend(pairs)

    if set_plit:
        train_pairs, val_pairs, test_pairs = set_split_ratio(
            train_pairs, val_pairs, test_pairs
        )

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError("One of the sets is empty. Check the guitar IDs.")

    train_duration, train_sec = total_minutes(train_pairs)
    val_duration,   val_sec   = total_minutes(val_pairs)
    test_duration,  test_sec  = total_minutes(test_pairs)
    total_sec = train_sec + val_sec + test_sec
    print(f"Dataset duration: {total_sec//60}min {total_sec%60:02d}s\n")
    print(f"Train (guitar {train_guitar}): {len(train_pairs)} pairs ({train_duration})")
    print(f"Val   (guitar {val_guitar}):   {len(val_pairs)}   pairs ({val_duration})")
    print(f"Test  (guitar {test_guitar}):  {len(test_pairs)}  pairs ({test_duration})\n")

    return {"train": train_pairs, "val": val_pairs, "test": test_pairs,
            "stats": {"train": get_style_stats(train_pairs),
                      "val":   get_style_stats(val_pairs),
                      "test":  get_style_stats(test_pairs)}}


# DATASET
class SegmentDataset(Dataset):
    def __init__(self, pairs, segment_len=SEGMENT_LEN):
        self.segment_len = segment_len
        self.segments = []
        print(f"  A carregar {len(pairs)} ficheiros para RAM...", flush=True)
        for i, (clean_path, sat_path) in enumerate(pairs):
            x, _ = torchaudio.load(str(clean_path))
            y, _ = torchaudio.load(str(sat_path))
            x = x.squeeze(0)
            y = y.squeeze(0)
            num_frames = x.shape[0]
            for start in range(0, num_frames - segment_len + 1, segment_len):
                self.segments.append((
                    x[start:start + segment_len],
                    y[start:start + segment_len]
                ))
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(pairs)} ficheiros carregados...", flush=True)
        print(f"  Pronto! {len(self.segments)} segmentos em RAM.", flush=True)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


# LSTM MODEL
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, output_size=1,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, skip=SKIP_CON):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.skip        = skip
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_size * 2, output_size)
        self.hidden = None

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(lstm_out)
        if self.skip:
            output = output + x
        return output

    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = tuple(h.detach() for h in self.hidden)

    def reset_hidden(self):
        self.hidden = None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# LOSS FUNCTIONS
class PreEmphasis(nn.Module):
    def __init__(self, coeffs=(-0.85, 1.0)):
        super().__init__()
        self.filter = nn.Conv1d(1, 1, kernel_size=2, bias=False, padding=0)
        self.filter.weight = nn.Parameter(
            torch.tensor([[list(coeffs)]], dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.pad(x, (1, 0))
        x = self.filter(x)
        return x.permute(0, 2, 1)


class CombinedLoss(nn.Module):
    def __init__(self, pre_filt_coeffs=(-0.85, 1.0)):
        super().__init__()
        self.pre_emph = PreEmphasis(coeffs=pre_filt_coeffs)
        self.esr      = auraloss_time.ESRLoss()
        self.dc       = auraloss_time.DCLoss()
        self.mrstft   = auraloss_freq.MultiResolutionSTFTLoss()

    def forward(self, output, target):
        out_pe    = self.pre_emph(output)
        tgt_pe    = self.pre_emph(target)
        output_al = output.permute(0, 2, 1)
        out_pe_al = out_pe.permute(0, 2, 1)
        target_al = target.permute(0, 2, 1)
        tgt_pe_al = tgt_pe.permute(0, 2, 1)
        combined  = (LAMBDA_ESR    * self.esr(out_pe_al, tgt_pe_al)
                   + LAMBDA_DC     * self.dc(output_al, target_al)
                   + LAMBDA_MRSTFT * self.mrstft(output_al, target_al))
        return (combined,
                self.esr(out_pe_al, tgt_pe_al).item(),
                self.dc(output_al, target_al).item(),
                self.mrstft(output_al, target_al).item())


# TRAINING
log_lines = []
def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:    return f"{h}h {m}m {s:.1f}s"
    elif m > 0:  return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def train_epoch(model, loader, optimizer, criterion, device, init_len):
    model.train()
    epoch_loss, valid_batches = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.unsqueeze(-1).to(device)
        y_batch = y_batch.unsqueeze(-1).to(device)
        model.reset_hidden()
        if init_len > 0:
            with torch.no_grad():
                _ = model(x_batch[:, :init_len, :])
        segment = x_batch[:, init_len:, :]
        target  = y_batch[:, init_len:, :]
        if segment.shape[1] < 2048:
            continue
        if torch.mean(target ** 2).item() < THRESHOLD:
            continue
        optimizer.zero_grad()
        output = model(segment)
        loss, *_ = criterion(output, target)
        if torch.isnan(loss) or loss.item() > 20.0:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        epoch_loss   += loss.item()
        valid_batches += 1
    return epoch_loss / max(1, valid_batches)


def validate(model, loader, criterion, device, init_len=INIT_LEN):
    model.eval()
    val_loss = esr_sum = dc_sum = mrstft_sum = 0.0
    with torch.no_grad():
        for x_val, y_val in loader:
            x_val = x_val.unsqueeze(-1).to(device)
            y_val = y_val.unsqueeze(-1).to(device)
            model.reset_hidden()
            if init_len > 0:
                _ = model(x_val[:, :init_len, :])
            out_val = model(x_val[:, init_len:, :])
            combined, esr_val, dc_val, mrstft_val = criterion(out_val, y_val[:, init_len:, :])
            val_loss  += combined.item()
            esr_sum   += esr_val
            dc_sum    += dc_val
            mrstft_sum+= mrstft_val
    n = len(loader)
    return val_loss/n, esr_sum/n, dc_sum/n, mrstft_sum/n


if __name__ == "__main__":

    # Find clean/saturated pairs
    clean_files     = sorted(Path(CLEAN_DIR).rglob("*.wav"))
    saturated_files = sorted(Path(SAT_DIR).rglob("*.wav"))
    sat_index       = {f.stem.removesuffix("_saturated"): f for f in saturated_files}
    pair_files      = [(f, sat_index[f.stem]) for f in clean_files if f.stem in sat_index]
    print(f"Pares válidos: {len(pair_files)}\n")

    # Dataset Split
    dataset_split = split_by_instrument(pair_files, train_guitar="1",
                                        val_guitar="2", test_guitar="3", set_plit=True)

    # Datasets + DataLoaders
    train_dataset = SegmentDataset(dataset_split["train"])
    val_dataset   = SegmentDataset(dataset_split["val"])
    test_dataset  = SegmentDataset(dataset_split["test"])

    train_generator = torch.Generator()
    train_generator.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True, generator=train_generator)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    print(f"Segmentos — train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}\n")

    # Model + Loss + Optimizer
    model    = LSTMModel().to(device)
    loss_fn  = CombinedLoss(pre_filt_coeffs=tuple(PRE_FILT)).to(device)
    optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler= optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(model)
    print(f"Parâmetros treináveis: {model.count_parameters():,}\n")

    # Training loop
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses   = []
    val_losses     = []
    esr_losses     = []
    dc_losses      = []
    mrstft_losses  = []
    epoch_times    = []

    log(f"Hyperparameters: EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | LR={LEARNING_RATE} "
        f"| HIDDEN={HIDDEN_SIZE} | LAYERS={NUM_LAYERS} | SKIP={SKIP_CON} | NUM_WORKERS={NUM_WORKERS}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, INIT_LEN)
        train_losses.append(avg_train_loss)

        if epoch % VALIDATION_F == 0:
            avg_val, avg_esr, avg_dc, avg_mrstft = validate(model, val_loader, loss_fn, device)
            val_losses.append(avg_val)
            esr_losses.append(avg_esr)
            dc_losses.append(avg_dc)
            mrstft_losses.append(avg_mrstft)
            scheduler.step(avg_val)

            log(f"Epoch {epoch:3d}/{EPOCHS} | Train: {avg_train_loss:.5f} | "
                f"Val: {avg_val:.5f} | ESR: {avg_esr:.4f} | DC: {avg_dc:.4f} | "
                f"MRSTFT: {avg_mrstft:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {format_time(time.time()-t0)}")

            if avg_val < best_val_loss:
                best_val_loss  = avg_val
                patience_count = 0
                torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model.pt")
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    log(f"\nEarly stopping na epoch {epoch}.")
                    break

            # Saves history and log every VALIDATION_F epochs
            history = {
                "train_losses":  train_losses,
                "val_losses":    val_losses,
                "esr_losses":    esr_losses,
                "dc_losses":     dc_losses,
                "mrstft_losses": mrstft_losses,
                "val_epochs":    [i * VALIDATION_F for i in range(1, len(val_losses) + 1)]
            }
            with open(f"{RESULTS_DIR}/training_history.json", "w") as f:
                json.dump(history, f, indent=2)
            with open(f"{RESULTS_DIR}/training_log.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
        else:
            log(f"Epoch {epoch:3d}/{EPOCHS} | Train: {avg_train_loss:.5f} | "
                f"Time: {format_time(time.time()-t0)}")

        epoch_times.append(time.time() - t0)

    log(f"\nTraining Total Time: {format_time(sum(epoch_times))}")
    log(f"Best val loss: {best_val_loss:.5f} — model saved in {RESULTS_DIR}/best_model.pt")

    with open(f"{RESULTS_DIR}/training_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Training log saved to {RESULTS_DIR}/training_log.txt")