from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a json file specifying local paths to models to load."
        },
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."
        },
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_voice_encoder: bool = field(
        default=True, metadata={"help": "Freeze the Voice Encoder."}
    )
    freeze_s3gen: bool = field(
        default=True,
        metadata={"help": "Freeze the S3Gen model (speech token to waveform)."},
    )
    freeze_text_embeddings: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of original text embedding tokens to freeze (e.g., 704 for original vocab size)."
        },
    )


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."
        },
    )
    dataset_dirs: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of paths to multiple dataset directories (e.g., for multi-language training). Each directory should contain JSON and audio files."
        },
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a metadata file. Used if dataset_name is not provided."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    train_split_name: Optional[str] = field(
        default="train", metadata={"help": "The name of the training data set split."}
    )

    train_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."},
    )
    eval_split_name: Optional[str] = field(
        default="validation",
        metadata={"help": "The name of the evaluation data set split."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the text column in the HF dataset."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the audio column in the HF dataset."},
    )
    max_text_len: int = field(
        default=256,
        metadata={"help": "Maximum length of text tokens (including BOS/EOS)."},
    )
    max_speech_len: int = field(
        default=800,
        metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."},
    )
    audio_prompt_duration_s: float = field(
        default=3.0,
        metadata={
            "help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."
        },
    )
    eval_split_size: float = field(
        default=0.0005,
        metadata={
            "help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help": "Set to true to ignore dataset verifications."}
    )
    lang_split: Optional[str] = field(
        default=None, metadata={"help": "The name of the language split to use."}
    )
    lang_path: Optional[str] = field(
        default=None, metadata={"help": "The path to the language split to use."}
    )
    lang_splits: List[str] = field(
        default_factory=list,
        metadata={"help": "List of language splits to use (e.g., ['de', 'fr'])."},
    )
    lang_paths: List[str] = field(
        default_factory=list,
        metadata={"help": "List of paths corresponding to each language split."},
    )
    use_webdataset: bool = field(
        default=False,
        metadata={
            "help": "Use webdataset format for optimized streaming and loading of large datasets like Emilia YODAS."
        },
    )
    webdataset_urls: Optional[str] = field(
        default=None,
        metadata={
            "help": "URL pattern for webdataset files (e.g., 'https://example.com/data-{000000..001000}.tar'). Used when use_webdataset=True."
        },
    )
    webdataset_shuffle_buffer: int = field(
        default=1000,
        metadata={
            "help": "Shuffle buffer size for webdataset streaming. Larger values improve randomness but use more memory."
        },
    )
