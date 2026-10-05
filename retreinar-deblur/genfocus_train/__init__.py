"""GenRefocus — treino da Stage 1 (DeblurNet), reprodução do paper.

Backbone FLUX-1-dev + LoRA (rank 128), condicionamento OminiControl-style
(`S_t = [X_t ; E(I_in)]`), rectified flow (`v = ε - x_0`). Dados vêm do
Hugging Face (`akcit-pixel/*`) via `data.py`.
"""

__all__ = [
    "config",
    "env",
    "data",
    "math_utils",
    "backbone",
    "models",
    "trainer",
]
