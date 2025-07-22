from dataset import T3ForFineTuning
from dataset import SpeechFineTuningDataset, DataArguments, SpeechDataCollator, DetailedLoggingCallback
import datasets
from datasets import concatenate_datasets
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import time
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from tqdm.auto import tqdm
import multiprocessing
import psutil
import shutil
import gc


from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import load_dataset, DatasetDict, VerificationMode, Audio, logging as ds_logging, DownloadConfig

from tts import ChatterboxTTS, Conditionals, punc_norm, REPO_ID
from models.t3.t3 import T3, T3Cond
from models.t3.modules.t3_config import T3Config
from models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE
from models.s3gen import S3GEN_SR
from utils.training_args import CustomTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a json file specifying local paths to models to load."}
        
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."}
    )
    
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_voice_encoder: bool = field(default=True, metadata={"help": "Freeze the Voice Encoder."})
    freeze_s3gen: bool = field(default=True, metadata={"help": "Freeze the S3Gen model (speech token to waveform)."})
    freeze_text_embeddings: Optional[int] = field(default=None, metadata={"help": "Number of original text embedding tokens to freeze (e.g., 704 for original vocab size)."})




config = {
    "voice_encoder_path": "/kaggle/input/chatterbox-weights/ve.safetensors",
    "t3_path": "/kaggle/input/chatterbox-weights/t3_cfg_new.safetensors",
    "s3gen_path": "/kaggle/input/chatterbox-weights/s3gen.safetensors",
    "tokenizer_path": "/kaggle/input/chatterbox-weights/merged_tokenizer.json",
    "conds_path": "/kaggle/input/chatterbox-weights/conds.pt"
}

with open("model_path.json", "w") as f:
    json.dump(config, f, indent=4)


import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(handler)


CHATTERBOX_PROJECT = "./chatterbox-project"


def run_training(model_args, data_args, training_args, is_local=False):

    use_torch_profiler = getattr(training_args, 'use_torch_profiler', False)
    profiler_output_dir = os.path.join(CHATTERBOX_PROJECT, "profiler_output")
    
    if use_torch_profiler:
        os.makedirs(profiler_output_dir, exist_ok=True)
        logger.info(f"PyTorch profiler enabled, output dir: {profiler_output_dir}")

        from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=tensorboard_trace_handler(profiler_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()


    output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(CHATTERBOX_PROJECT, output_dir)
    os.makedirs(training_args.output_dir, exist_ok=True)

    if training_args.resume_from_checkpoint is None:
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt:
            training_args.resume_from_checkpoint = last_ckpt
            logger.info(f"Found existing checkpoint, resuming from: {last_ckpt}")

    global trainer_instance


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    logger.info("Loading ChatterboxTTS model...")

    original_model_dir_for_copy: Optional[Path] = None
    repo_home_weights = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights")
    logger.info(f"Loading model from model config file: {model_args.model_config}")
    with open(model_args.model_config, "r") as file:
        m_paths = json.load(file)
    
    with tqdm(desc="Loading ChatterboxTTS components", total=4) as pbar:
        pbar.set_description("Loading voice encoder...")
        voice_encoder_path =  m_paths["voice_encoder_path"]
        pbar.update(1)
        
        pbar.set_description("Loading T3 model...")
        t3_path =  m_paths["t3_path"]
        pbar.update(1)
        
        pbar.set_description("Loading S3Gen model...")
        s3gen_path =  m_paths["s3gen_path"]
        pbar.update(1)
        
        pbar.set_description("Initializing complete model...")
        
        chatterbox_model = ChatterboxTTS.from_specified(
            voice_encoder_path=voice_encoder_path,
            t3_path=t3_path,
            s3gen_path=s3gen_path,
            tokenizer_path=m_paths["tokenizer_path"],
            conds_path=m_paths["conds_path"],
            device_map=device_map
        )
        pbar.update(1)
        pbar.set_description("Model loading completed")
    
    original_model_dir_for_copy = repo_home_weights

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters(): param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters(): param.requires_grad = False
        logger.info("S3Gen model frozen.")
    for param in t3_model.parameters(): param.requires_grad = True
    
    # # Freeze original text embeddings if specified
    # if model_args.freeze_text_embeddings is not None:
    #     freeze_vocab_size = model_args.freeze_text_embeddings
    #     current_vocab_size = chatterbox_t3_config_instance.text_tokens_dict_size
    #     if current_vocab_size > freeze_vocab_size:
    #         # We'll mask gradients in a training hook instead of setting requires_grad
    #         def mask_old_token_gradients(module, grad_input, grad_output):
    #             if hasattr(module, 'weight') and module.weight.grad is not None:
    #                 module.weight.grad[:freeze_vocab_size] = 0
            
    #         t3_model.text_emb.register_backward_hook(mask_old_token_gradients)
    #         t3_model.text_head.register_backward_hook(mask_old_token_gradients)
    #         logger.info(f"Added gradient masking for original text embeddings (first {freeze_vocab_size} tokens)")
    #     else:
    #         logger.warning(f"Cannot freeze {freeze_vocab_size} tokens - current vocab size is only {current_vocab_size}")
    
    logger.info("T3 model set to trainable.")
    logger.info("Loading and processing dataset...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS

    train_hf_dataset: Union[datasets.Dataset, List[Dict[str,str]]]
    eval_hf_dataset: Optional[Union[datasets.Dataset, List[Dict[str,str]]]] = None 
    streaming = None
    if data_args.dataset_name:
        logger.info(f"Loading dataset '{data_args.dataset_name}' from Hugging Face Hub.")

        if data_args.lang_splits and data_args.lang_paths:
            # Multi-language support: {"de": "Emilia-YODAS/DE/*.tar", "fr": "Emilia-YODAS/FR/*.tar"}
            if len(data_args.lang_splits) != len(data_args.lang_paths):
                raise ValueError("lang_splits and lang_paths must have the same length")
            data_files = {split: path for split, path in zip(data_args.lang_splits, data_args.lang_paths)}
            logger.info(f"Loading multi-language datasets: {list(data_files.keys())}")
        elif data_args.lang_split:
            # Single language support (backward compatibility)
            # {"ja":"Emilia-YODAS/JA/*.tar"}
            if not data_args.lang_path:
                raise ValueError("lang_path must be provided if lang_split is provided")
            data_files = {data_args.lang_split: data_args.lang_path}
        else:
            data_files = None
        

        ds_logging.set_verbosity_info()           # show INFO from datasets
        ds_logging.enable_progress_bar()

        download_config = DownloadConfig()

        logger.info("Loading dataset...")
        import time
        start_time = time.time()
        
        
        # Add progress bar for dataset loading
        with tqdm(desc="Loading dataset", total=1) as pbar:
            if is_local:
                pbar.set_description("Loading local dataset...")
                raw_datasets_loaded = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    data_files=data_files,
                    cache_dir=CHATTERBOX_PROJECT,
                    num_proc=32,
                    download_config=download_config,
                    verification_mode=verification_mode,
                )
            else:
                pbar.set_description("Loading streaming dataset...")
                streaming = True
                raw_datasets_loaded = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    data_files=data_files,
                    # num_proc=32,
                    verification_mode=verification_mode,
                    download_config=download_config,
                    streaming=streaming
                )  
            pbar.update(1)
            pbar.set_description("Dataset loading completed")
            
        logger.info("Dataset loaded.")
        end_time = time.time()
        logger.info(f"Time taken to load dataset: {end_time - start_time} seconds")
        
        if data_args.train_split_name not in raw_datasets_loaded:
            # If train split not found but we have language splits, combine them
            if data_args.lang_splits:
                available_lang_splits = [split for split in data_args.lang_splits if split in raw_datasets_loaded]
                if available_lang_splits:
                    logger.info(f"Train split '{data_args.train_split_name}' not found. Combining language splits: {available_lang_splits}")
                    # Combine all available language splits
                    if streaming:
                        # For streaming datasets, concatenation works differently
                        datasets_to_combine = [raw_datasets_loaded[split] for split in available_lang_splits]
                        train_hf_dataset = concatenate_datasets(datasets_to_combine)
                    else:
                        datasets_to_combine = [raw_datasets_loaded[split] for split in available_lang_splits]
                        train_hf_dataset = concatenate_datasets(datasets_to_combine)
                else:
                    raise ValueError(f"Train split '{data_args.train_split_name}' not found and no language splits available. Available: {list(raw_datasets_loaded.keys())}")
            else:
                raise ValueError(f"Train split '{data_args.train_split_name}' not found. Available: {list(raw_datasets_loaded.keys())}")
        else:
            train_hf_dataset = raw_datasets_loaded[data_args.train_split_name]
        
        if training_args.do_eval:
            with tqdm(desc="Setting up evaluation dataset", total=1) as pbar:
                if data_args.eval_split_name and data_args.eval_split_name in raw_datasets_loaded:
                    eval_hf_dataset = raw_datasets_loaded[data_args.eval_split_name]
                elif "validation" in raw_datasets_loaded: 
                    eval_hf_dataset = raw_datasets_loaded["validation"]
                elif "test" in raw_datasets_loaded: 
                    eval_hf_dataset = raw_datasets_loaded["test"]
                elif data_args.eval_split_size > 0 and hasattr(train_hf_dataset, "__len__") and len(train_hf_dataset) > 1 : # Ensure dataset is splittable
                    pbar.set_description("Splitting train dataset for evaluation...")
                    logger.info(f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}")
                    split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
                    train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
                    logger.info(f"Evaluation set size: {len(eval_hf_dataset)}")
                elif streaming and data_args.lang_splits:
                    # For streaming datasets, use a different language split for eval if available
                    available_eval_splits = [split for split in data_args.lang_splits if split in raw_datasets_loaded and split != data_args.lang_splits[0]]
                    if available_eval_splits:
                        logger.info(f"Using language split '{available_eval_splits[0]}' for evaluation in streaming mode")
                        eval_hf_dataset = raw_datasets_loaded[available_eval_splits[0]]
                    else:
                        logger.warning("Streaming mode: no separate language split available for evaluation. Disabling eval.")
                else: 
                    logger.warning("Evaluation requested but no eval split found/configured or train dataset too small to split. Skipping eval dataset.")
                pbar.update(1)
                pbar.set_description("Evaluation dataset setup completed")
                
        is_hf_format_train, is_hf_format_eval = True, True
    
#################
    else:
        all_files = []
        dataset_path = Path(data_args.dataset_dir)
        tsv_path = dataset_path / "train.tsv"
        wrd_path = dataset_path / "train.wrd"
        if tsv_path.exists() and wrd_path.exists():
            logger.info(f"Loading dataset from TSV and WRD in {dataset_path}")
            with open(tsv_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                raise ValueError("train.tsv is empty")
            absolute_path = "/kaggle/input/punjabi-speech-recognition/punjabi/Audio files"
            if not os.path.isdir(absolute_path):
                raise ValueError(f"Absolute path in train.tsv header does not exist or is not a directory: {absolute_path}")
            data_lines = lines[1:]
            full_paths = []
            for line in data_lines:
                parts = line.split('\t')
                if len(parts) >= 1:
                    rel_path = parts[0].strip()
                    full_path = os.path.join(absolute_path, rel_path)
                    if os.path.isfile(full_path):
                        full_paths.append(full_path)
                    else:
                        logger.warning(f"Audio file not found: {full_path}")
            with open(wrd_path, 'r') as f:
                texts = [line.strip() for line in f.readlines()]
            if len(full_paths) == len(texts):
                all_files = [{"audio": fp, "text": txt} for fp, txt in zip(full_paths, texts)][:500]
                logger.info(f"Limited dataset to 500 audio-text pairs")
                #############################
            else:
                raise ValueError(f"Mismatch between number of valid audio files ({len(full_paths)}) and transcriptions ({len(texts)}) in train.tsv and train.wrd")
        else:
            raise ValueError(f"Expected train.tsv and train.wrd in {dataset_path}")

        if not all_files:
            raise ValueError("No valid audio-text pairs found in dataset_dir")
        logger.info(f"Found {len(all_files)} audio-text pairs")
        np.random.shuffle(all_files)
        train_hf_dataset = all_files
        is_hf_format_train = False

        if data_args.eval_split_size > 0 and training_args.do_eval and len(all_files) > 1:
            with tqdm(desc="Splitting dataset for evaluation", total=1) as pbar:
                split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
                if split_idx == 0:
                    split_idx = 1
                if split_idx == len(all_files):
                    split_idx = len(all_files) - 1
                train_hf_dataset, eval_hf_dataset = all_files[:split_idx], all_files[split_idx:]
                pbar.update(1)
                logger.info(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval")
            is_hf_format_eval = False
#################
        if not all_files: 
            raise ValueError("No data files found from local paths. Check dataset_dir, dataset_dirs, or metadata_file.")
            
        logger.info(f"Found {len(all_files)} audio-text pairs")
        np.random.shuffle(all_files)
        train_hf_dataset = all_files # type: ignore
        
        if data_args.eval_split_size > 0 and training_args.do_eval and len(all_files) > 1:
            with tqdm(desc="Splitting dataset for evaluation", total=1) as pbar:
                split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
                if split_idx == 0 : split_idx = 1 # Ensure at least one for train if eval gets most
                if split_idx == len(all_files): split_idx = len(all_files) -1 # Ensure at least one for eval
                train_hf_dataset, eval_hf_dataset = all_files[:split_idx], all_files[split_idx:] # type: ignore
                pbar.update(1)
                logger.info(f"Split dataset: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval")
                
        is_hf_format_train, is_hf_format_eval = False, False

    
    with tqdm(desc="Initializing training dataset", total=1) as pbar:
        train_dataset = SpeechFineTuningDataset(
            data_args,
            chatterbox_t3_config_instance,
            train_hf_dataset,
            is_hf_format_train,
            model_dir=str(CHATTERBOX_PROJECT),
            m_paths=m_paths,
            device="cpu"
        )
        pbar.update(1)
        pbar.set_description("Training dataset initialized")

    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        with tqdm(desc="Initializing evaluation dataset", total=1) as pbar:
            eval_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_t3_config_instance,
                eval_hf_dataset,
                is_hf_format_eval,
                model_dir=str(original_model_dir_for_copy),
                m_paths=m_paths,
                device="cpu"
            )
            pbar.update(1)
            pbar.set_description("Evaluation dataset initialized")

    # If evaluation was requested but no eval_dataset was built (e.g. streaming), disable eval
    if training_args.do_eval and eval_dataset is None:
        logger.warning("Evaluation requested but no eval_dataset found; disabling evaluation.")
        training_args.do_eval = False
        training_args.eval_strategy = "no"
        if hasattr(training_args, "eval_on_start"):
            training_args.eval_on_start = False
    
    # Configure training arguments for streaming datasets
    if streaming and training_args.max_steps == -1:
        # For streaming datasets, we must set max_steps since __len__ is not available
        # Estimate reasonable max_steps based on epochs and batch size
        estimated_steps_per_epoch = 1000  # Conservative estimate for streaming datasets
        estimated_max_steps = int(training_args.num_train_epochs * estimated_steps_per_epoch)
        training_args.max_steps = estimated_max_steps
        logger.info(f"Streaming mode: Setting max_steps to {estimated_max_steps} (estimated {estimated_steps_per_epoch} steps per epoch)")
        
        # Adjust eval and save steps proportionally
        if training_args.eval_steps and training_args.eval_steps > estimated_max_steps:
            training_args.eval_steps = estimated_max_steps // 10
            logger.info(f"Adjusted eval_steps to {training_args.eval_steps} for streaming mode")
        if training_args.save_steps and training_args.save_steps > estimated_max_steps:
            training_args.save_steps = estimated_max_steps // 10
            logger.info(f"Adjusted save_steps to {training_args.save_steps} for streaming mode")

    with tqdm(desc="Setting up data collator and model", total=2) as pbar:
        data_collator = SpeechDataCollator(chatterbox_t3_config_instance, 
                                           chatterbox_t3_config_instance.stop_text_token,
                                           chatterbox_t3_config_instance.stop_speech_token)
        pbar.update(1)
        pbar.set_description("Data collator created")

        hf_trainable_model = T3ForFineTuning(t3_model, chatterbox_t3_config_instance)
        pbar.update(1)
        pbar.set_description("Model wrapper created")

    
    callbacks = []
    
    # Add detailed logging callback
    callbacks.append(DetailedLoggingCallback())
    
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if use_torch_profiler:
        # Add profiler stepping callback
        class ProfilerCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                prof.step()
        callbacks.append(ProfilerCallback())

    TrainerClass = Trainer

    # Optimize system settings for webdataset remote training
    if data_args.use_webdataset:
        logger.info("Applying webdataset-specific optimizations for remote training...")
        
        # Enable PyTorch optimizations for streaming workloads
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark for consistent input sizes")
        
        # Configure memory management for streaming
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to leave room for data buffers
            memory_fraction = 0.8 if data_args.use_webdataset else 0.9
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"Set CUDA memory fraction to {memory_fraction} for webdataset streaming")
        
        # Log webdataset configuration
        logger.info(f"WebDataset configuration:")
        logger.info(f"  - Shuffle buffer: {data_args.webdataset_shuffle_buffer}")
        logger.info(f"  - DataLoader workers: {training_args.dataloader_num_workers}")
        logger.info(f"  - Pin memory: {training_args.dataloader_pin_memory}")
        logger.info(f"  - Prefetch factor: {training_args.dataloader_prefetch_factor}")

    with tqdm(desc="Initializing trainer", total=1) as pbar:
        trainer_instance = TrainerClass(
            model=hf_trainable_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
            
        )
        pbar.update(1)
        pbar.set_description("Trainer initialized")

    if training_args.label_names is None: trainer_instance.label_names = ["lables"]


    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        logger.info(f"Training configuration: batch_size={training_args.per_device_train_batch_size}, "
                   f"grad_accum_steps={training_args.gradient_accumulation_steps}, "
                   f"max_steps={training_args.max_steps}, epochs={training_args.num_train_epochs}, "
                   f"learning_rate={training_args.learning_rate}")
        
        # Log streaming dataset info
        if streaming:
            logger.info("Using streaming dataset - monitoring first few batches for data flow...")
        
        # Patch previous trainer_state.json to update batch size before resuming
        ckpt = training_args.resume_from_checkpoint
        if ckpt:
            ts_path = os.path.join(ckpt, "trainer_state.json")
            if os.path.exists(ts_path):
                # Load existing state, update batch size, and rewrite file cleanly
                with open(ts_path, "r") as rf:
                    state = json.load(rf)
                state["train_batch_size"] = training_args.per_device_train_batch_size
                with open(ts_path, "w") as wf:
                    json.dump(state, wf, indent=2)
                logger.info(f"Updated train_batch_size in {ts_path} to {training_args.per_device_train_batch_size}")
        
        # Log just before starting training
        logger.info("Initializing training loop - this may take a moment for streaming datasets...")
        
        
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            
        logger.info("Training completed successfully!")
            
        trainer_instance.save_model()
        
        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = trainer_instance.model.t3 if hasattr(trainer_instance.model, 't3') else trainer_instance.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()
        
        output_t3_safetensor_path = Path(CHATTERBOX_PROJECT) / "t3_cfg.safetensors"
        from safetensors.torch import save_file
        
        with tqdm(desc="Saving T3 model weights", total=1) as pbar:
            save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
            pbar.update(1)
            pbar.set_description(f"T3 weights saved to {output_t3_safetensor_path}")
        
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        if original_model_dir_for_copy:
            import shutil
            files_to_copy = ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]
            
            with tqdm(desc="Copying model components", total=len(files_to_copy)+1) as pbar:
                for f_name in files_to_copy:
                    src_path = original_model_dir_for_copy / f_name
                    if src_path.exists(): 
                        pbar.set_description(f"Copying {f_name}")
                        shutil.copy2(src_path, Path(CHATTERBOX_PROJECT) / f_name)
                    pbar.update(1)
                    
                if (original_model_dir_for_copy / "conds.pt").exists():
                    pbar.set_description("Copying conds.pt")
                    shutil.copy2(original_model_dir_for_copy / "conds.pt", Path(CHATTERBOX_PROJECT) / "conds.pt")
                pbar.update(1)
                pbar.set_description("All model components copied")
                
            logger.info(f"Full model components structured in {CHATTERBOX_PROJECT}")

        with tqdm(desc="Saving training metrics", total=3) as pbar:
            metrics = train_result.metrics
            pbar.set_description("Logging metrics")
            trainer_instance.log_metrics("train", metrics)
            pbar.update(1)
            
            pbar.set_description("Saving metrics")
            trainer_instance.save_metrics("train", metrics)
            pbar.update(1)
            
            pbar.set_description("Saving trainer state")
            trainer_instance.save_state()
            pbar.update(1)
            pbar.set_description("Training metrics saved")

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        with tqdm(desc="Running evaluation", total=1) as pbar:
            metrics = trainer_instance.evaluate()
            pbar.update(1)
            pbar.set_description("Evaluation completed")
            
        with tqdm(desc="Saving evaluation metrics", total=2) as pbar:
            pbar.set_description("Logging evaluation metrics")
            trainer_instance.log_metrics("eval", metrics)
            pbar.update(1)
            
            pbar.set_description("Saving evaluation metrics")
            trainer_instance.save_metrics("eval", metrics)
            pbar.update(1)
            pbar.set_description("Evaluation metrics saved")


    logger.info("Finetuning script finished.")


multiprocessing.set_start_method('spawn', force=True)


device_map = {
    'voice_encoder': 'cpu',  # Keep on CPU to save GPU memory
    't3': 'cuda',           # Place on CUDA for training with backpropagation
    's3gen': 'cpu'          # Keep on CPU if not needed on GPU
}



def run():
    model_args = ModelArguments(
        model_config="/kaggle/working/model_path.json",
        cache_dir=None,
        freeze_voice_encoder=True,
        freeze_s3gen=True,
        freeze_text_embeddings=704
    )
    
    data_args = DataArguments(
        dataset_dir="/kaggle/input/punjabi-speech-cleaned",
        preprocessing_num_workers=1,
        text_column_name="text",
        audio_column_name="audio",
        max_text_len=256,
        max_speech_len=800,
        audio_prompt_duration_s=3.0,
        ignore_verifications=False
    )
    
    training_args = CustomTrainingArguments(
        output_dir="checkpoints/punjabi_run",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=4,
        fp16=True,
        dataloader_num_workers = 1,
        do_train=True,
        do_eval=True,
        dataloader_pin_memory=True if torch.cuda.is_available() and data_args.use_webdataset else False,
        eval_on_start=False,
        use_torch_profiler=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=2,
        fsdp="full_shard",
        fsdp_offload_params=True
    )
    
    # Use preprocessing_num_workers as dataloader_num_workers if set
    if data_args.preprocessing_num_workers is not None:
        training_args.dataloader_num_workers = data_args.preprocessing_num_workers
    
    
    # run_cleaner(model_args=model_args, data_args=data_args)
    run_training(model_args, data_args, training_args, is_local=True)

if __name__ == "__main__":
    run()