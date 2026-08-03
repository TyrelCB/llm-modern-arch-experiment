"""ModernLM: a capacity-matched dense modern-architecture baseline for DeepSeek-V4."""
from .config import ModernConfig
from .model import ModernLM, ModernOutput

__all__ = ["ModernConfig", "ModernLM", "ModernOutput"]
