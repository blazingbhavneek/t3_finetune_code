import gc
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from typing import Dict, List, Optional, Union
import logging

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from tqdm import tqdm
from pathlib import Path

from tts import ChatterboxTTS, punc_norm
from models.t3.modules.t3_config import T3Config
from models.s3tokenizer import S3_SR

from tts import ChatterboxTTS 

from models.t3.t3 import T3, T3Cond

import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."}
    )
    dataset_dirs: List[str] = field(
        default_factory=list,
        metadata={"help": "List of paths to multiple dataset directories (e.g., for multi-language training). Each directory should contain JSON and audio files."}
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a metadata file. Used if dataset_name is not provided."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the Hugging Face datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."}
    )
    train_split_name: Optional[str] = field(default="train", metadata={"help": "The name of the training data set split."})
    
    train_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."}
    )
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "The name of the evaluation data set split."})
    text_column_name: str = field(default="text", metadata={"help": "The name of the text column in the HF dataset."})
    audio_column_name: str = field(default="audio", metadata={"help": "The name of the audio column in the HF dataset."})
    max_text_len: int = field(default=256, metadata={"help": "Maximum length of text tokens (including BOS/EOS)."})
    max_speech_len: int = field(default=800, metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."})
    audio_prompt_duration_s: float = field(
        default=3.0, metadata={"help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."}
    )
    eval_split_size: float = field(
        default=0.0005, metadata={"help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help":"Set to true to ignore dataset verifications."}
    )
    lang_split: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the language split to use."}
    )
    lang_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the language split to use."}
    )
    lang_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."}
    )
    lang_paths: List[str] = field(
        default_factory=list,
        metadata={"help": "List of paths corresponding to each language split."}
    )
    use_webdataset: bool = field(
        default=False,
        metadata={"help": "Use webdataset format for optimized streaming and loading of large datasets like Emilia YODAS."}
    )
    webdataset_urls: Optional[str] = field(
        default=None,
        metadata={"help": "URL pattern for webdataset files (e.g., 'https://example.com/data-{000000..001000}.tar'). Used when use_webdataset=True."}
    )
    webdataset_shuffle_buffer: int = field(
        default=1000,
        metadata={"help": "Shuffle buffer size for webdataset streaming. Larger values improve randomness but use more memory."}
    )

# Your SpeechFineTuningDataset class goes here
class SpeechFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: DataArguments,
                 t3_config: T3Config,
                 hf_dataset: Union[HFDataset, List[Dict[str, str]]],
                 is_hf_format: bool,
                 model_dir: str,
                 m_paths: dict = None,
                 device: str = "cpu"):
        # Store raw args
        self.data_args = data_args
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format
        # Path to model checkpoint directory for lazy loading
        self._model_dir = model_dir
        self.m_paths = m_paths
        self._device = device
        # Placeholders for components, will be initialized lazily
        self.chatterbox_model = None
        self.text_tokenizer = None
        self.speech_tokenizer = None
        self.voice_encoder = None
        
        # Sampling and conditioning setup
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(data_args.audio_prompt_duration_s * self.s3_sr)
        # Immediately load model in main process; workers will reload lazily
        self._init_model()
        
    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        if self.is_hf_format:
            item = self.dataset_source[idx]
            # Get text field, with fallback for different column names
            try:
                # HF default
                text = item[self.data_args.text_column_name]
            except KeyError:
                # Emilia Dataset
                if "json" in item and isinstance(item["json"], dict):
                    meta = item["json"]
                    if "text" in meta:
                        text = meta["text"]
                    else:
                        logger.error(f"'text' field not found in JSON metadata. Available JSON keys: {list(meta.keys())}. Skipping.")
                        return None, None
                else:
                    logger.error(f"Text column '{self.data_args.text_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                    return None, None
            except Exception as e:
                logger.error(f"Error loading text for item {idx}: {e}. Skipping.")
                return None, None
                
            # Get audio data, with fallback for different column names
            try:
                # HF default
                audio_data = item[self.data_args.audio_column_name]
            except KeyError:
                # Emilia Dataset
                if "mp3" in item:
                    audio_data = item["mp3"]
                else:
                    for alt in ["audio", "wav"]:
                        if alt in item:
                            logger.warning(f"Column '{self.data_args.audio_column_name}' not found. Using '{alt}' instead.")
                            audio_data = item[alt]
                            break
                    else:
                        logger.error(f"Audio column '{self.data_args.audio_column_name}' not found. Available keys: {list(item.keys())}. Skipping.")
                        return None, None
            
            # Load audio from bytes (streaming), file path, or pre-loaded dict
            if isinstance(audio_data, (bytes, bytearray)):
                import io
                try:
                    wav_array, original_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
                except Exception as e:
                    logger.error(f"Error loading audio bytes for item {idx}: {e}. Skipping.")
                    return None, None
            elif isinstance(audio_data, str):
                wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif isinstance(audio_data, dict) and "array" in audio_data and "sampling_rate" in audio_data:
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            else:
                logger.error(f"Unexpected audio data format for item {idx}: {type(audio_data)}. Skipping.")
                return None, None

            if not isinstance(wav_array, np.ndarray):
                logger.error(f"Audio array is not numpy for item {idx}: {type(wav_array)}. Skipping.")
                return None, None

            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(wav_array, orig_sr=original_sr, target_sr=self.s3_sr)
            else:
                wav_16k = wav_array.copy()
            
            if wav_16k.ndim > 1: wav_16k = librosa.to_mono(wav_16k)
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            item_info_for_log = f"Item {idx} (text: '{text[:30]}...', audio_len: {len(wav_16k)}, audio_dtype: {wav_16k.dtype})"

            return wav_16k, text
        else:
            item = self.dataset_source[idx]
            audio_path = item["audio"]
            text = item["text"]
            try:
                wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
                return wav_16k, text
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                return None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            # Ensure model is loaded (in worker)
            self._init_model()
            speaker_emb_np = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=self.s3_sr)
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(f"Error getting speaker embedding for item {idx}: {e}. Skipping.")
            return None

        normalized_text = punc_norm(text)
        logger.info(f"Normalized text: {normalized_text}")
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[:self.data_args.max_text_len-1]
            text_tokens = torch.cat([text_tokens, torch.tensor([self.chatterbox_t3_config.stop_text_token], device=text_tokens.device)])
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            # Ensure tokenizer is available
            self._init_model()
            raw_speech_tokens_batch, speech_token_lengths_batch = self.speech_tokenizer.forward([wav_16k])
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[:speech_token_lengths_batch.squeeze(0).item()]
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None
            
        speech_tokens = F.pad(raw_speech_tokens, (1, 0), value=self.chatterbox_t3_config.start_speech_token)
        speech_tokens = F.pad(speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token)
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[:self.data_args.max_speech_len-1]
            speech_tokens = torch.cat([speech_tokens, torch.tensor([self.chatterbox_t3_config.stop_speech_token], device=speech_tokens.device)])
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        cond_audio_segment = wav_16k[:self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0 :
            cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward([cond_audio_segment], max_len=self.chatterbox_t3_config.speech_cond_prompt_len)
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                cond_prompt_speech_tokens = torch.zeros(self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long)

        if cond_prompt_speech_tokens.size(0) != self.chatterbox_t3_config.speech_cond_prompt_len:
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len: cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else: cond_prompt_speech_tokens = F.pad(cond_prompt_speech_tokens, (0, target_len - current_len), value=0)
        
        emotion_adv_scalar = 0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return_dict = {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }

        return return_dict

    def _init_model(self):
        if self.chatterbox_model is None:
            with tqdm(desc="Loading ChatterboxTTS components", total=1, leave=False) as pbar:
                pbar.set_description("Loading from specified paths...")
                conds_path = Path(self.m_paths["conds_path"])
                tokenizer_path = Path(self.m_paths["tokenizer_path"])
                s3gen_path = Path(self.m_paths["s3gen_path"])
                t3_path = Path(self.m_paths["t3_path"])
                voice_encoder_path = Path(self.m_paths["voice_encoder_path"])
                logger.info(f"Conds: {conds_path}, {conds_path.exists()}, {conds_path.is_file()}, {conds_path.is_dir()}")
                self.chatterbox_model = ChatterboxTTS.from_specified(
                    voice_encoder_path=voice_encoder_path,
                    t3_path=t3_path,
                    s3gen_path=s3gen_path,
                    tokenizer_path=tokenizer_path,
                    conds_path=conds_path,
                    device_map=self.m_paths.get("device_map", {'voice_encoder': 'cpu', 't3': 'cuda', 's3gen': 'cpu'})
                )
                pbar.set_description("Extracting tokenizers and encoder...")
                self.text_tokenizer = self.chatterbox_model.tokenizer
                self.speech_tokenizer = self.chatterbox_model.s3gen.tokenizer
                self.voice_encoder = self.chatterbox_model.ve
                pbar.update(1)
                pbar.set_description("Model components loaded")

    def __getstate__(self):
        # Drop unpickleable objects; they will be reloaded in each worker
        state = self.__dict__.copy()
        state['chatterbox_model'] = None
        state['text_tokenizer'] = None
        state['speech_tokenizer'] = None
        state['voice_encoder'] = None
        return state

    def __setstate__(self, state):
        # Restore state and reload model
        self.__dict__.update(state)
        self._init_model()



@dataclass
class SpeechDataCollator:
    t3_config: T3Config  # Chatterbox T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning("SpeechDataCollator received no valid features. Returning empty batch.")
            return {}
        features = valid_features
        
        # Log batch formation occasionally
        if hasattr(self, '_batch_count'):
            self._batch_count += 1
        else:
            self._batch_count = 1
            
        if self._batch_count <= 5 or self._batch_count % 50 == 0:
            logger.info(f"Forming batch #{self._batch_count} with {len(features)} samples")

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        # Pad text tokens
        padded_text_tokens = torch.stack([
            F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
            for t in text_tokens_list
        ])  # shape: (B, max_text_len)

        # Pad speech tokens
        padded_speech_tokens = torch.stack([
            F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
            for s in speech_tokens_list
        ])  # shape: (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in features])      # (B,)
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in features])  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in features])             # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in features])  # (B, prompt_len)
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in features])  # (B, 1, 1)
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        # Shift off BOS from padded_text_tokens: new length = max_text_len - 1
        shifted_text = padded_text_tokens[:, 1:].contiguous()  # shape: (B, max_text_len - 1)
        T_text = shifted_text.size(1)

        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        arange_text = torch.arange(T_text, device=shifted_text.device)  # (T_text,)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]  # (B, T_text)

        labels_text = shifted_text.clone()           # (B, T_text)
        labels_text[mask_pad_text] = IGNORE_ID       # set pad/beyond to -100

        # --- Build labels_speech ---
        # Shift off BOS from padded_speech_tokens: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()  # shape: (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)

        # Mask positions t >= (speech_len - 1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        arange_speech = torch.arange(T_speech, device=shifted_speech.device)  # (T_speech,)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]  # (B, T_speech)

        # Mask positions t < prompt_len
        mask_prompt = arange_speech[None] < prompt_len  # (1, T_speech) -> broadcast to (B, T_speech)
        mask_prompt = mask_prompt.expand(batch_size, T_speech)

        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt  # (B, T_speech)

        labels_speech = shifted_speech.clone()          # (B, T_speech)
        labels_speech[mask_speech_total] = IGNORE_ID    # set prompt & pad to -100

        batch_result = {
            "text_tokens": padded_text_tokens, 
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens, 
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,       # (B, max_text_len - 1) masked with -100
            "labels_speech": labels_speech,   # (B, max_speech_len - 1) masked with -100
        }
        
        # Log batch details for first few batches
        if self._batch_count <= 3:
            logger.info(f"Batch #{self._batch_count} details: text_shape={padded_text_tokens.shape}, "
                       f"speech_shape={padded_speech_tokens.shape}, max_text_len={max_text_len}, "
                       f"max_speech_len={max_speech_len}")
        
        return batch_result
    

class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = chatterbox_t3_config.text_tokens_dict_size
        hf_config_instance.speech_tokens_dict_size = chatterbox_t3_config.speech_tokens_dict_size
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = chatterbox_t3_config.speech_cond_prompt_len
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

    def forward(self,
                text_tokens,
                text_token_lens,
                speech_tokens,
                speech_token_lens,
                t3_cond_speaker_emb,
                t3_cond_prompt_speech_tokens,
                t3_cond_emotion_adv,
                labels_text = None,
                labels_speech=None):

        current_t3_cond = T3Cond(
                                speaker_emb=t3_cond_speaker_emb,
                                cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
                                cond_prompt_speech_emb=None,
                                emotion_adv=t3_cond_emotion_adv
                                ).to(device=self.t3.device)

        loss_text, loss_speech, speech_logits = self.t3.loss(
                                t3_cond=current_t3_cond,
                                text_tokens=text_tokens,
                                text_token_lens=text_token_lens,
                                speech_tokens=speech_tokens,
                                speech_token_lens=speech_token_lens,
                                labels_text =labels_text,
                                labels_speech=labels_speech
                                )
        
        total_loss = loss_text + loss_speech

        return total_loss, speech_logits

trainer_instance: Optional[Trainer] = None

class DetailedLoggingCallback(TrainerCallback):
    """Custom callback for detailed training progress logging"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.last_log_time = None
        self.samples_processed = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("Training started - monitoring performance metrics")
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_time = current_time - self.step_start_time
        self.step_times.append(step_time)
        self.samples_processed += args.per_device_train_batch_size * args.gradient_accumulation_steps
        
        # Log detailed progress every few steps
        if state.global_step % 5 == 0 or (current_time - self.last_log_time) >= 60:  # Every 5 steps or 1 minute
            avg_step_time = np.mean(self.step_times[-10:]) if self.step_times else 0
            total_time = current_time - self.start_time
            samples_per_sec = self.samples_processed / total_time if total_time > 0 else 0
            
            # Memory usage
            memory_info = psutil.Process().memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # GPU memory usage if available
            gpu_memory_str = ""
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_str = f", GPU memory: {gpu_memory_mb:.1f}MB"
            
            logger.info(f"Training step {state.global_step}/{args.max_steps if args.max_steps > 0 else 'unknown'}: "
                       f"avg_step_time={avg_step_time:.3f}s, samples_processed={self.samples_processed}, "
                       f"samples/sec={samples_per_sec:.2f}, memory={memory_mb:.1f}MB{gpu_memory_str}")
            
            self.last_log_time = current_time
            
            # Trigger garbage collection periodically to prevent memory buildup
            if state.global_step % 1 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            # Enhanced loss logging with additional context
            current_time = time.time()
            total_time = current_time - self.start_time
            logger.info(f"Training metrics at step {state.global_step}: loss={logs['loss']:.4f}, "
                       f"learning_rate={logs.get('learning_rate', 'N/A')}, "
                       f"total_time={total_time/60:.1f}min")

