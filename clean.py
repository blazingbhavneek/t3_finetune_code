import json
import logging
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from dataset import SpeechFineTuningDataset
from tts import ChatterboxTTS

logger = logging.getLogger(__name__)

CHATTERBOX_PROJECT = "./chatterbox-project"


def clean_dataset(
    data_args,
    t3_config,
    model_dir: str,
    m_paths: dict = None,
    device: str = "cpu",
    backup: bool = True,
    remove_audio_files: bool = False,
):
    """
    Clean dataset by validating each item using SpeechFineTuningDataset and removing problematic entries.

    Args:
        data_args: DataArguments instance with dataset configuration
        t3_config: T3Config instance
        model_dir: Path to model directory
        m_paths: Optional dict with model component paths
        device: Device to load models on
        backup: Whether to create backup files before cleaning
        remove_audio_files: Whether to also delete audio files for removed entries

    Returns:
        tuple: (num_original, num_cleaned, num_removed, removed_audio_files)
    """

    dataset_path = Path(data_args.dataset_dir)
    tsv_path = dataset_path / "train.tsv"
    wrd_path = dataset_path / "train.wrd"
    output_dir = Path("/working")
    output_tsv = output_dir / "train.tsv"
    output_wrd = output_dir / "train.wrd"

    if not (tsv_path.exists() and wrd_path.exists()):
        raise ValueError(f"Expected train.tsv and train.wrd in {dataset_path}")

    # Create backups if requested
    if backup:
        backup_dir = output_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(tsv_path, backup_dir / "train.tsv.backup")
        shutil.copy2(wrd_path, backup_dir / "train.wrd.backup")
        logger.info(f"Created backups in {backup_dir}")

    # Read original files
    with open(tsv_path, "r") as f:
        tsv_lines = f.readlines()

    with open(wrd_path, "r") as f:
        wrd_lines = [line.strip() for line in f.readlines()]

    if not tsv_lines:
        raise ValueError("train.tsv is empty")

    # Extract header and data
    absolute_path = "/input/punjabi-speech-recognition/punjabi/Audio files"
    data_lines = tsv_lines[1:]

    if len(data_lines) != len(wrd_lines):
        raise ValueError(
            f"Mismatch between TSV data lines ({len(data_lines)}) and WRD lines ({len(wrd_lines)})"
        )

    logger.info(f"Original dataset has {len(data_lines)} items")

    # Track all file paths and their validity
    all_file_info = []

    for i, (tsv_line, wrd_line) in enumerate(zip(data_lines, wrd_lines)):
        parts = tsv_line.split("\t")
        if len(parts) >= 1:
            rel_path = parts[0].strip()
            full_path = os.path.join(absolute_path, rel_path)
            all_file_info.append(
                {
                    "audio_path": full_path,
                    "text": wrd_line,
                    "original_index": i,
                    "file_exists": os.path.isfile(full_path),
                    "tsv_line": tsv_line,
                }
            )

    # Filter out files that don't exist
    existing_files = [info for info in all_file_info if info["file_exists"]]
    missing_files = [info for info in all_file_info if not info["file_exists"]]

    if missing_files:
        logger.warning(f"Found {len(missing_files)} missing audio files")
        for missing in missing_files[:5]:
            logger.warning(f"Missing: {missing['audio_path']}")
        if len(missing_files) > 5:
            logger.warning(f"... and {len(missing_files) - 5} more missing files")

    # Prepare data for dataset loader
    dataset_items = [
        {"audio": info["audio_path"], "text": info["text"]} for info in existing_files
    ]

    if not dataset_items:
        raise ValueError("No valid audio files found")

    logger.info(f"After file existence check: {len(existing_files)} items remain")

    # Create temporary dataset loader for validation
    try:
        temp_dataset = SpeechFineTuningDataset(
            data_args=data_args,
            t3_config=t3_config,
            hf_dataset=dataset_items,
            is_hf_format=False,
            model_dir=model_dir,
            m_paths=m_paths,
            device=device,
        )
        logger.info("Successfully created temporary dataset for validation")
    except Exception as e:
        logger.error(f"Failed to create dataset loader: {e}")
        raise

    # Validate each item
    valid_items = []
    invalid_items = []

    logger.info("Starting dataset validation...")

    for idx, file_info in enumerate(tqdm(existing_files, desc="Validating items")):
        try:
            item = temp_dataset.__getitem__(idx)

            # Check if item is None or has missing/invalid attributes
            is_valid = True
            validation_errors = []

            if item is None:
                is_valid = False
                validation_errors.append("Dataset __getitem__ returned None")
            else:
                # Validate required keys
                required_keys = [
                    "text_tokens",
                    "text_token_lens",
                    "speech_tokens",
                    "speech_token_lens",
                    "t3_cond_speaker_emb",
                    "t3_cond_prompt_speech_tokens",
                    "t3_cond_emotion_adv",
                ]

                for key in required_keys:
                    if key not in item:
                        is_valid = False
                        validation_errors.append(f"Missing key '{key}'")
                        continue

                    if item[key] is None:
                        is_valid = False
                        validation_errors.append(f"Key '{key}' is None")
                        continue

                    # Check for empty tensors
                    if hasattr(item[key], "numel") and item[key].numel() == 0:
                        is_valid = False
                        validation_errors.append(f"Key '{key}' has empty tensor")
                        continue

                    # Additional validation for specific keys
                    if key in ["text_tokens", "speech_tokens"] and hasattr(
                        item[key], "shape"
                    ):
                        if len(item[key].shape) == 0 or item[key].shape[0] == 0:
                            is_valid = False
                            validation_errors.append(
                                f"Key '{key}' has invalid shape: {item[key].shape}"
                            )

                # Validate tensor lengths are reasonable
                if is_valid:
                    if "text_token_lens" in item and hasattr(
                        item["text_token_lens"], "item"
                    ):
                        text_len = item["text_token_lens"].item()
                        if text_len <= 0 or text_len > data_args.max_text_len:
                            is_valid = False
                            validation_errors.append(
                                f"Invalid text token length: {text_len}"
                            )

                    if "speech_token_lens" in item and hasattr(
                        item["speech_token_lens"], "item"
                    ):
                        speech_len = item["speech_token_lens"].item()
                        if speech_len <= 0 or speech_len > data_args.max_speech_len:
                            is_valid = False
                            validation_errors.append(
                                f"Invalid speech token length: {speech_len}"
                            )

            if is_valid:
                valid_items.append(file_info)
            else:
                invalid_items.append(file_info)
                logger.warning(
                    f"Item {file_info['original_index']} invalid: {'; '.join(validation_errors)}"
                )

        except Exception as e:
            logger.error(
                f"Error validating item {file_info['original_index']}: {str(e)}"
            )
            invalid_items.append(file_info)

    # Combine missing files with invalid items for removal
    items_to_remove = missing_files + invalid_items

    logger.info(f"Validation complete:")
    logger.info(f"  Valid items: {len(valid_items)}")
    logger.info(f"  Invalid items: {len(invalid_items)}")
    logger.info(f"  Missing files: {len(missing_files)}")
    logger.info(f"  Total to remove: {len(items_to_remove)}")

    # Remove audio files if requested
    removed_audio_files = []
    if remove_audio_files and items_to_remove:
        logger.info(
            f"Removing {len([item for item in items_to_remove if item['file_exists']])} audio files..."
        )
        for item_info in items_to_remove:
            if item_info["file_exists"]:
                try:
                    os.remove(item_info["audio_path"])
                    removed_audio_files.append(item_info["audio_path"])
                    logger.debug(f"Removed audio file: {item_info['audio_path']}")
                except Exception as e:
                    logger.error(
                        f"Failed to remove audio file {item_info['audio_path']}: {e}"
                    )

    if len(valid_items) == len(data_lines):
        logger.info("No items were removed - dataset is already clean!")
        return len(data_lines), len(valid_items), 0, []

    logger.info(
        f"Writing cleaned dataset with {len(valid_items)} items to {output_dir}..."
    )

    # Write cleaned TSV
    with open(output_tsv, "w") as f:
        f.write(absolute_path + "\n")
        for item_info in valid_items:
            f.write(item_info["tsv_line"])

    # Write cleaned WRD
    with open(output_wrd, "w") as f:
        for item_info in valid_items:
            f.write(item_info["text"] + "\n")

    num_original = len(data_lines)
    num_cleaned = len(valid_items)
    num_removed = len(items_to_remove)

    logger.info(f"Dataset cleaning complete!")
    logger.info(f"Original items: {num_original}")
    logger.info(f"Cleaned items: {num_cleaned}")
    logger.info(f"Removed items: {num_removed}")
    logger.info(f"Removed audio files: {len(removed_audio_files)}")
    logger.info(f"Removal rate: {num_removed/num_original*100:.2f}%")

    return num_original, num_cleaned, num_removed, removed_audio_files


def run_cleaner(model_args, data_args, device_map):
    with open(model_args.model_config, "r") as file:
        m_paths = json.load(file)
    with tqdm(desc="Loading ChatterboxTTS components", total=4) as pbar:
        pbar.set_description("Loading voice encoder...")
        voice_encoder_path = m_paths["voice_encoder_path"]
        pbar.update(1)

        pbar.set_description("Loading T3 model...")
        t3_path = m_paths["t3_path"]
        pbar.update(1)

        pbar.set_description("Loading S3Gen model...")
        s3gen_path = m_paths["s3gen_path"]
        pbar.update(1)

        pbar.set_description("Initializing complete model...")
        chatterbox_model = ChatterboxTTS.from_specified(
            voice_encoder_path=voice_encoder_path,
            t3_path=t3_path,
            s3gen_path=s3gen_path,
            tokenizer_path=m_paths["tokenizer_path"],
            conds_path=m_paths["conds_path"],
            device_map=device_map,
        )
        pbar.update(1)
        pbar.set_description("Model loading completed")

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp
    num_original, num_cleaned, num_removed, removed_audio = clean_dataset(
        data_args=data_args,
        t3_config=chatterbox_t3_config_instance,
        model_dir=str(CHATTERBOX_PROJECT),
        m_paths=m_paths,
        device="cpu",
        backup=True,
        remove_audio_files=False,
    )
    print("Cleaned: ", num_original, num_cleaned, num_removed, removed_audio)
