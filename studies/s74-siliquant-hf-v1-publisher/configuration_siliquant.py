"""Hugging Face PretrainedConfig for Siliquant bit-packed lattice models (`siliquant-hf-v1`).

Supports `AutoConfig.from_pretrained(..., trust_remote_code=True)` and
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` for both
Qwen2 (`qwen2`) and Llama (`llama`, e.g. SmolLM2) decoder-only architectures.
"""

from __future__ import annotations

import copy
from typing import Any

from transformers import LlamaConfig, PretrainedConfig, Qwen2Config

SILIQUANT_FORMAT_VERSION = "siliquant-hf-v1"
SUPPORTED_BASE_ARCHS = ("qwen2", "llama")
SUPPORTED_ROTATIONS = ("axis", "block_h2", "block_h4", "h16")
SUPPORTED_K_BITS = (2, 3, 4)


class SiliquantConfig(PretrainedConfig):
    """Configuration class for Siliquant bit-packed `A_16` lattice causal LMs."""

    model_type = "siliquant"

    def __init__(
        self,
        base_arch: str = "qwen2",
        base_model_config: dict[str, Any] | None = None,
        siliquant_format_version: str = SILIQUANT_FORMAT_VERSION,
        siliquant_bpw_target: float = 4.5,
        siliquant_default_group_size: int = 128,
        siliquant_recipe: str = "A16_BlockH4",
        siliquant_layer_map: dict[str, dict[str, Any]] | None = None,
        siliquant_cache_dequantized: bool = True,
        siliquant_verify_syndrome_on_load: bool = True,
        base_model_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        if base_arch not in SUPPORTED_BASE_ARCHS:
            raise ValueError(
                f"Unsupported base_arch={base_arch!r}; expected one of {SUPPORTED_BASE_ARCHS}"
            )
        if siliquant_format_version != SILIQUANT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported siliquant_format_version={siliquant_format_version!r}; "
                f"expected {SILIQUANT_FORMAT_VERSION!r}"
            )
        if int(siliquant_default_group_size) not in (32, 128):
            raise ValueError(
                f"Unsupported siliquant_default_group_size={siliquant_default_group_size!r}; "
                "expected 32 or 128"
            )

        self.base_arch = str(base_arch)
        self.siliquant_format_version = str(siliquant_format_version)
        self.siliquant_bpw_target = float(siliquant_bpw_target)
        self.siliquant_default_group_size = int(siliquant_default_group_size)
        self.siliquant_recipe = str(siliquant_recipe)
        self.siliquant_layer_map: dict[str, dict[str, Any]] = (
            copy.deepcopy(siliquant_layer_map) if siliquant_layer_map is not None else {}
        )
        self.siliquant_cache_dequantized = bool(siliquant_cache_dequantized)
        self.siliquant_verify_syndrome_on_load = bool(siliquant_verify_syndrome_on_load)
        self.base_model_name_or_path = base_model_name_or_path

        if base_model_config is None:
            default_cls = Qwen2Config if self.base_arch == "qwen2" else LlamaConfig
            base_cfg_dict = default_cls().to_dict()
        else:
            base_cfg_dict = copy.deepcopy(dict(base_model_config))

        # Remove top-level auto_map / architectures from nested base config to prevent recursion.
        base_cfg_dict.pop("auto_map", None)
        base_cfg_dict.pop("architectures", None)
        self.base_model_config = base_cfg_dict

        # Mirror essential decoder attributes onto top-level config for GenerationMixin & Cache.
        mirrored_keys = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "rope_parameters",
            "hidden_act",
            "use_cache",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "head_dim",
        )
        for key in mirrored_keys:
            if key in self.base_model_config and key not in kwargs:
                kwargs[key] = copy.deepcopy(self.base_model_config[key])

        super().__init__(**kwargs)

        # Ensure tie_word_embeddings matches base_model_config unambiguously.
        if "tie_word_embeddings" in self.base_model_config:
            self.tie_word_embeddings = bool(self.base_model_config["tie_word_embeddings"])

    def to_base_config(self) -> Qwen2Config | LlamaConfig:
        """Instantiate the exact underlying Hugging Face `Qwen2Config` or `LlamaConfig`."""
        cfg_dict = copy.deepcopy(self.base_model_config)
        cfg_dict.pop("auto_map", None)
        cfg_dict.pop("architectures", None)
        cfg_dict["tie_word_embeddings"] = bool(self.tie_word_embeddings)
        if self.base_arch == "qwen2":
            cfg_dict["model_type"] = "qwen2"
            return Qwen2Config(**cfg_dict)
        if self.base_arch == "llama":
            cfg_dict["model_type"] = "llama"
            return LlamaConfig(**cfg_dict)
        raise ValueError(f"Unsupported base_arch={self.base_arch!r}")

    @classmethod
    def from_base_config(
        cls,
        base_config: PretrainedConfig,
        *,
        siliquant_bpw_target: float = 4.5,
        siliquant_default_group_size: int = 128,
        siliquant_recipe: str = "A16_BlockH4",
        siliquant_layer_map: dict[str, dict[str, Any]] | None = None,
        siliquant_cache_dequantized: bool = True,
        siliquant_verify_syndrome_on_load: bool = True,
        base_model_name_or_path: str | None = None,
        dtype_str: str = "bfloat16",
    ) -> "SiliquantConfig":
        """Construct a `SiliquantConfig` from an upstream `Qwen2Config` or `LlamaConfig`."""
        base_arch = str(getattr(base_config, "model_type", "")).lower()
        if base_arch not in SUPPORTED_BASE_ARCHS:
            raise ValueError(
                f"Base config model_type={base_arch!r} is not supported; "
                f"expected one of {SUPPORTED_BASE_ARCHS}"
            )
        base_dict = base_config.to_dict()
        base_dict.pop("auto_map", None)
        base_dict.pop("architectures", None)
        base_dict.pop("_name_or_path", None)
        # Normalize dtype field cleanly for both transformers 4.x and 5.x.
        base_dict["dtype"] = dtype_str
        base_dict["torch_dtype"] = dtype_str

        return cls(
            base_arch=base_arch,
            base_model_config=base_dict,
            siliquant_format_version=SILIQUANT_FORMAT_VERSION,
            siliquant_bpw_target=siliquant_bpw_target,
            siliquant_default_group_size=siliquant_default_group_size,
            siliquant_recipe=siliquant_recipe,
            siliquant_layer_map=siliquant_layer_map or {},
            siliquant_cache_dequantized=siliquant_cache_dequantized,
            siliquant_verify_syndrome_on_load=siliquant_verify_syndrome_on_load,
            base_model_name_or_path=base_model_name_or_path,
            architectures=["SiliquantForCausalLM"],
            auto_map={
                "AutoConfig": "configuration_siliquant.SiliquantConfig",
                "AutoModelForCausalLM": "modeling_siliquant.SiliquantForCausalLM",
            },
            dtype=dtype_str,
            torch_dtype=dtype_str,
        )


SiliquantConfig.register_for_auto_class()
