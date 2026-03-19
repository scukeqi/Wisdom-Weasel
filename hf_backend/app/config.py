"""配置管理模块"""
import json
from pathlib import Path
from typing import Any, Optional
import torch


class Config:
    """从JSON文件加载配置"""
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
    DEFAULT_CONFIG = {
        "model_id": "Qwen/Qwen3-4B",
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": False,
        "adapter_path": None,
        "merge_adapter_on_load": False,
        "model_kwargs": {},
        "prompting": {
            "use_chat_template": True,
            "system_prompt": (
                "你是 Wisdom Weasel 输入法候选生成模型。"
                "请根据历史上下文和当前拼音，输出最自然、最贴合用户意图的中文候选。"
                "只输出候选文本本身，不要解释，不要编号，不要额外标点。"
            )
        },
        "telemetry": {
            "enabled": True,
            "directory": "telemetry"
        },
        "generation": {
            "max_new_tokens_no_constraint": 4,
            "num_beams": 4,
            "num_return_sequences": 4,
            "num_beam_groups": 2,
            "diversity_penalty_no_constraint": 0.4,
            "diversity_penalty_with_constraint": 0.2
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """加载配置文件，不存在则使用默认配置"""
        merged = self.DEFAULT_CONFIG.copy()
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    merged.update(json.load(f))
                print(f"配置已从 {self.config_path} 加载")
            except Exception as e:
                print(f"加载配置失败: {e}，使用默认配置")
        else:
            print(f"配置文件不存在: {self.config_path}，使用默认配置")
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def get_model_id(self) -> str:
        return self.get("model_id")
    
    def get_torch_dtype(self) -> Any:
        torch_dtype = self.get("torch_dtype")
        if torch_dtype in (None, "auto"):
            return torch_dtype
        return getattr(torch, torch_dtype)

    def get_device_map(self) -> str:
        return self.get("device_map")

    def get_trust_remote_code(self) -> bool:
        return bool(self.get("trust_remote_code", False))

    def get_adapter_path(self) -> Optional[str]:
        adapter_path = self.get("adapter_path")
        return str(adapter_path) if adapter_path else None

    def should_merge_adapter_on_load(self) -> bool:
        return bool(self.get("merge_adapter_on_load", False))

    def get_model_kwargs(self) -> dict:
        model_kwargs = self.get("model_kwargs", {})
        return model_kwargs if isinstance(model_kwargs, dict) else {}

    def get_prompting_config(self) -> dict:
        prompting = self.get("prompting", {})
        return prompting if isinstance(prompting, dict) else {}

    def use_chat_template(self) -> bool:
        return bool(self.get_prompting_config().get("use_chat_template", True))

    def get_system_prompt(self) -> str:
        prompting = self.get_prompting_config()
        default = self.DEFAULT_CONFIG["prompting"]["system_prompt"]
        value = prompting.get("system_prompt", default)
        return value if isinstance(value, str) and value.strip() else default

    def get_telemetry_config(self) -> dict:
        telemetry = self.get("telemetry", {})
        return telemetry if isinstance(telemetry, dict) else {}

    def is_telemetry_enabled(self) -> bool:
        return bool(self.get_telemetry_config().get("enabled", True))

    def get_telemetry_directory(self) -> Path:
        configured = self.get_telemetry_config().get("directory", "telemetry")
        return (Path(__file__).parent.parent / configured).resolve()
    
    def get_generation_config(self) -> dict:
        return self.get("generation", {})
    
    def get_max_new_tokens_no_constraint(self) -> int:
        return self.get("generation", {}).get("max_new_tokens_no_constraint", 4)
    
    def get_num_beams(self) -> int:
        return self.get("generation", {}).get("num_beams", 4)
    
    def get_num_return_sequences(self) -> int:
        return self.get("generation", {}).get("num_return_sequences", 4)
    
    def get_num_beam_groups(self) -> int:
        return self.get("generation", {}).get("num_beam_groups", 2)
    
    def get_diversity_penalty_no_constraint(self) -> float:
        return self.get("generation", {}).get("diversity_penalty_no_constraint", 0.4)
    
    def get_diversity_penalty_with_constraint(self) -> float:
        return self.get("generation", {}).get("diversity_penalty_with_constraint", 0.2)
    
    def to_dict(self) -> dict:
        return self.config.copy()


_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
