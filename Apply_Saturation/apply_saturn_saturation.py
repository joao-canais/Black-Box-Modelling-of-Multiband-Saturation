"""
Saturn 2 Saturation Application Script
Used to generate audio datasets (Clean vs Saturated).
"""

import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pedalboard import load_plugin
from pedalboard.io import AudioFile
import scipy.signal


# CONFIGURATION
CONFIG_PATH = Path(__file__).parent / "config_local.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

INPUT_DIR  = Path(config["paths"]["input_dir"])
OUTPUT_DIR = Path(config["paths"]["output_dir"])
VST3_PATH = config["paths"]["vst3_path"]

# Load Preset configuration
PRESET_PATH = Path(__file__).parent / config["processing"]["active_preset"]
with open(PRESET_PATH, "r", encoding="utf-8") as f:
    preset_config = yaml.safe_load(f)

# Dataset filtering config
dataset_filters = config.get("dataset_filters", {})

KEEP_FILTER = dataset_filters.get("keep_filter", [])

# Processing settings
PROCESS_BUFFER_SIZE = config["processing"].get("buffer_size", 512)
AUDIO_DIFF_THRESHOLD = config["processing"].get("audio_diff_threshold", 1e-4)

PRESET_NAME = preset_config.get("name", "Unknown Preset")
PRESET_PARAMETERS = preset_config.get("plugin_state", {})


# AUXILIARY FUNCTIONS
def configure_saturn_2_preset(plugin, preset_parameters):
    """Applies a desired preset to the FabFilter Saturn 2 DSP based on the provided parameter dictionary.
    Configures crossovers, drive per band, EQ, and Envelope Follower modulations."""    
    
    # Force 4-Band architecture if defined in preset
    if "num_active_bands" in preset_parameters and "num_active_bands" in plugin.parameters:
        plugin.num_active_bands = float(preset_parameters["num_active_bands"])

    # Inject variables directly into the plugin engine
    for param_name, value in preset_parameters.items():
        if param_name == "num_active_bands":
            continue
        if param_name in plugin.parameters:
            try:
                setattr(plugin, param_name, value)
            except Exception as e:
                print(f"[Warning] Failed to set {param_name}={value}. Reason: {e}")
                
                
def get_bit_depth_from_dtype(file_dtype: str) -> int:
    """Returns a safe bit-depth integer based on the AudioFile dtype."""
    dtype = (file_dtype or "").lower()
    return next((b for b in [8, 16, 24, 32] if str(b) in dtype), 16)


def filter_valid_files(input_dir: Path) -> list:
    """Filters dataset files based on naming convention mappings."""
    valid_files = []
    use_keep_filter = bool(KEEP_FILTER)



    for p in input_dir.rglob("*.wav"):
        filename = p.stem.upper()
        
        # If no filters are active, include every .wav even without expected tokens.
        if not use_keep_filter:
            valid_files.append(p)
            continue
        
        parts = filename.split("_")

        pluck_style = parts[4] if len(parts) > 4 else None
        
        # A file is valid only if it passes keep filters and is not listed in any exclude filter.
        if pluck_style in KEEP_FILTER:
            valid_files.append(p)
            
    return valid_files


def estimate_latency_samples(clean_mono: np.ndarray, sat_mono: np.ndarray, sr: int, window_seconds: float = 1.0) -> int:
    """Estimate latency using cross-correlation on the highest-energy window."""
    if clean_mono.size == 0 or sat_mono.size == 0:
        return 0

    window_len = int(sr * window_seconds)
    window_len = max(256, min(window_len, clean_mono.size, sat_mono.size))
    if window_len <= 0:
        return 0

    # Find the highest-energy window in the clean signal to avoid silence bias
    energy = np.convolve(clean_mono[:].astype(np.float64) ** 2, np.ones(window_len, dtype=np.float64), mode="valid")
    start_idx = int(np.argmax(energy)) if energy.size > 0 else 0

    clean_seg = clean_mono[start_idx:start_idx + window_len]
    sat_seg = sat_mono[start_idx:start_idx + window_len]

    if clean_seg.size == 0 or sat_seg.size == 0:
        return 0

    correlation = scipy.signal.correlate(sat_seg, clean_seg, mode="full")
    lags = scipy.signal.correlation_lags(len(sat_seg), len(clean_seg), mode="full")
    return int(lags[int(np.argmax(correlation))])


def main():
    print(f"\n--- Saturn 2 Processing Pipeline ---")
    
    # Setup directories
    sat_dir = OUTPUT_DIR / "saturated"
    sat_dir.mkdir(parents=True, exist_ok=True)

    # Validate file inputs
    target_files = filter_valid_files(INPUT_DIR)
    print(f"Discovered {len(target_files)} valid audio pairs to process in {INPUT_DIR}.\n")
    if not target_files:
        return

    # Load Plugin
    print(f"Loading VST3 and injecting hardcoded {PRESET_NAME} preset...\n")
    plugin = load_plugin(VST3_PATH)
    
    # Tracker variables
    metrics = {
        "metadata_mismatch": [],
        "low_saturation_delta": [],
        "success": 0
    }

    # Process all files
    for src in tqdm(target_files, desc="Applying VST Saturation"):
        
        rel_dir = src.parent.relative_to(INPUT_DIR)
        
        current_sat_dir = sat_dir / rel_dir
        sat_dest = current_sat_dir / f"{src.stem}_saturated.wav"

        clean_dir = OUTPUT_DIR / "clean"
        current_clean_dir = clean_dir / rel_dir
        clean_dest = current_clean_dir / src.name

        # If both output files already exist, skip processing and count as success for metrics
        if sat_dest.exists() and clean_dest.exists():
            metrics["success"] += 1 
            continue

        # 1. Read Audio
        with AudioFile(str(src)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
            channels = f.num_channels
            bit_depth = get_bit_depth_from_dtype(f.file_dtype)

        # 2. Reset & Configure Plugin constraints iteratively
        # Must run per iteration to enforce fresh DSP states safely
        configure_saturn_2_preset(plugin, PRESET_PARAMETERS)

        # 3. Process signal natively
        saturated_audio = plugin.process(audio, sample_rate=sr, buffer_size=PROCESS_BUFFER_SIZE, reset=False)

        # 4. Latency Compensation (Dynamic) (turned off)        
        # clean_mono = audio[0] if channels > 1 else audio.flatten()
        # sat_mono = saturated_audio[0] if channels > 1 else saturated_audio.flatten()
        # latency_samples = estimate_latency_samples(clean_mono, sat_mono, sr, window_seconds=1.0)

        # if latency_samples > 0 and saturated_audio.shape[1] > latency_samples:
        #     # Saturn output is delayed: trim start of saturated, end of clean
        #     saturated_audio = saturated_audio[:, latency_samples:]
        #     audio = audio[:, :-latency_samples]
        # elif latency_samples < 0 and audio.shape[1] > abs(latency_samples):
        #     # Saturn output is early: trim start of clean, end of saturated
        #     latency_samples = abs(latency_samples)
        #     audio = audio[:, latency_samples:]
        #     saturated_audio = saturated_audio[:, :-latency_samples]

        # 5. Export Files
        rel_dir = src.parent.relative_to(INPUT_DIR)
        current_sat_dir = sat_dir / rel_dir
        current_sat_dir.mkdir(parents=True, exist_ok=True)
        sat_dest = current_sat_dir / f"{src.stem}_saturated{src.suffix}"


        # Write aligned clean file (trimmed to match saturated length)
        clean_dir = OUTPUT_DIR / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)
        current_clean_dir = clean_dir / rel_dir
        current_clean_dir.mkdir(parents=True, exist_ok=True)
        clean_dest = current_clean_dir / src.name

        with AudioFile(str(sat_dest), "w", sr, saturated_audio.shape[0], bit_depth) as out_f:
            out_f.write(saturated_audio)
        with AudioFile(str(clean_dest), "w", sr, audio.shape[0], bit_depth) as out_f:
            out_f.write(audio)

        # 6. Quality Assurance & Auditing
        rel_delta = float(abs(saturated_audio - audio).mean() / (abs(audio).mean() + 1e-12))
        if rel_delta < AUDIO_DIFF_THRESHOLD:
            metrics["low_saturation_delta"].append(src.name)

        with AudioFile(str(sat_dest)) as out_check:
            if sr != out_check.samplerate or channels != out_check.num_channels:
                metrics["metadata_mismatch"].append(src.name)
            else:
                metrics["success"] += 1


    # OUTPUT
    print("\n--- Process Report ---")
    print(f"Successfully processed and validated: {metrics['success']} / {len(target_files)} files\n")
    
    for filter in dataset_filters:
        print(f"Filter Applied: {filter}")
    
    if metrics["low_saturation_delta"]:
        print(f"\n[!] WARNING: {len(metrics['low_saturation_delta'])} files had extremely low audio delta (Check VST Mix/Drive)")
        for name in metrics["low_saturation_delta"][:5]:
            print(f"  - {name}")

    if metrics["metadata_mismatch"]:
        print(f"\n[!] WARNING: {len(metrics['metadata_mismatch'])} files failed Sample Rate or Channel metadata checks")


if __name__ == "__main__":
    main()
