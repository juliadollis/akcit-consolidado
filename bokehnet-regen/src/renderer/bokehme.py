"""Adaptador in-process do BokehMe público (JuewenPeng/BokehMe).

É o renderer `[43]` que o paper cita na Eq. 5 e na Fig. 3(a). A extensão privada dos
autores é **só a Eq. 6** (formato de abertura), confirmado pelo supplement B.2:
*"we then optimized the parameter K using simulator [43]"*.

Três diferenças em relação ao adaptador antigo, todas necessárias:

1. **In-process.** O antigo chamava `subprocess demo.py` uma vez por avaliação de K —
   24 no grid grosso e 16 no fino, cada uma subindo Python, CUDA e dois checkpoints do
   zero. Aqui os modelos são carregados **uma vez** e ficam na GPU.

2. **Sem normalizar a disparidade.** O `demo.py` lê um PNG de disparidade, renormaliza
   para [0,1] e faz `defocus = K*(disp - disp_focus)/defocus_scale`. Como já temos
   profundidade métrica, montamos o tensor `defocus` direto do CoC canônico. Isso
   elimina a quantização de 8 bits do PNG e a viagem de ida e volta pela normalização.

3. **Saída em float.** Quantizar é gravação, não renderização. Um ponto de luz de 255
   espalhado num disco de raio 12 dá 0,56 por pixel, que `astype(np.uint8)` trunca
   para zero — o harness de verificação mediria raio 0.

4. **`demo.py` NÃO é importável.** Ele roda `args = parser.parse_args()` na linha 130,
   no nível de módulo, e em seguida instancia os modelos, carrega os checkpoints e
   executa o demo inteiro. Um `from demo import pipeline` parsearia o nosso `argv`,
   duplicaria os modelos na GPU e escreveria em `outputs/`. Extraímos `pipeline` e
   `gaussian_blur` do fonte por AST e executamos só essas duas definições, num
   namespace controlado. O sha256 do trecho extraído vai na proveniência — prova
   melhor que o commit, porque identifica o corpo exato da função que rodou.

O contrato, verificado no `demo.py`:

    defocus = K * (disp - disp_focus) / defocus_scale
    # e dentro de pipeline():
    classical_renderer(image ** gamma, defocus * defocus_scale)

Logo o `classical_renderer` recebe o **CoC com sinal em pixels**, e `defocus_scale` só
normaliza a entrada da rede — se cancela.

O paper NÃO publica nenhum dos sete parâmetros de `BokehMeConfig`. Eles são congelados
aqui, gravados na proveniência de cada amostra, e o desvio é declarado no texto.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from control.contract import SampleRejected, signed_coc_px


@dataclass(frozen=True)
class BokehMeConfig:
    """Os sete parâmetros que o paper não publica, mais a arquitetura das redes.

    Os defaults de arquitetura vêm do `demo.py` do repositório público e não são
    escolha nossa. Os sete primeiros são: `gamma`, `defocus_scale`, `highlight`,
    `highlight_rgb_threshold`, `highlight_enhance_ratio`, `output` e a versão dos
    checkpoints (via hash na proveniência).
    """

    # --- os que o paper cala sobre ---
    gamma: float = 4.0
    defocus_scale: float = 10.0
    highlight: bool = False
    highlight_rgb_threshold: float = 220.0 / 255.0
    highlight_enhance_ratio: float = 0.4
    #: qual das três saídas usar. `bokeh_pred` é a híbrida clássica+neural do paper.
    output: str = "bokeh_pred"
    gamma_min: float = 1.0
    gamma_max: float = 5.0
    #: Sempre False. O `pipeline` do BokehMe escreve JPEGs intermediários em
    #: `save_root` quando isto é True, e `save_root` não existe fora do demo.
    save_intermediate: bool = False

    # --- arquitetura, defaults do demo.py público ---
    arnet_shuffle_rate: int = 2
    arnet_in_channels: int = 5
    arnet_out_channels: int = 4
    arnet_middle_channels: int = 128
    arnet_num_block: int = 3
    arnet_share_weight: bool = False
    arnet_connect_mode: str = "distinct_source"
    arnet_use_bn: bool = False
    arnet_activation: str = "elu"
    iunet_shuffle_rate: int = 2
    iunet_in_channels: int = 8
    iunet_out_channels: int = 3
    iunet_middle_channels: int = 64
    iunet_num_block: int = 3
    iunet_share_weight: bool = False
    iunet_connect_mode: str = "distinct_source"
    iunet_use_bn: bool = False
    iunet_activation: str = "elu"

    def as_namespace(self) -> Any:
        """Objeto com os atributos que `pipeline(...)` espera do `args` do demo.py."""
        from argparse import Namespace
        return Namespace(**asdict(self))


_VALID_OUTPUTS = {"bokeh_pred", "bokeh_classical", "bokeh_neural"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _Unavailable:
    """Sentinela para nome que o código extraído referencia mas nunca executa.

    Explodir com mensagem clara é melhor que importar uma dependência que não
    precisamos — e melhor ainda que um `None` que viraria `AttributeError` obscuro.
    """

    def __init__(self, name: str, why: str):
        self._name = name
        self._why = why

    def __getattr__(self, attr: str):
        raise RuntimeError(
            f"`{self._name}.{attr}` foi acessado, mas {self._name} {self._why}. "
            "Se este caminho passou a ser usado, trate a dependência de verdade "
            "em vez de contornar."
        )


def _extract_demo_functions(demo_path: Path) -> tuple[dict, str]:
    """Extrai `gaussian_blur` e `pipeline` do `demo.py` SEM executar o módulo.

    O `demo.py` do BokehMe roda `parse_args()` e carrega os modelos no nível de
    módulo, então importá-lo tem três efeitos colaterais inaceitáveis: parseia o
    nosso `argv`, duplica os modelos na GPU e escreve em `outputs/`.

    Aqui o fonte é parseado por AST, só as duas `FunctionDef` necessárias são
    recompiladas, e o resultado roda num namespace montado à mão. Devolve o
    namespace e o sha256 do trecho extraído, que vai na proveniência.
    """
    import ast

    source = demo_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = ("gaussian_blur", "pipeline")
    segments: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"não consegui extrair o fonte de {node.name} em {demo_path}")
            segments[node.name] = segment
    missing = [name for name in wanted if name not in segments]
    if missing:
        raise RuntimeError(
            f"{demo_path} não define {missing}. O upstream mudou: revise o adaptador "
            "em vez de contornar."
        )

    extracted = "\n\n".join(segments[name] for name in wanted)
    import numpy
    import torch
    import torch.nn.functional as torch_functional

    namespace: dict = {
        "torch": torch,
        "F": torch_functional,
        "np": numpy,
        # `cv2` e `save_root` aparecem no `pipeline` SÓ dentro dos ramos
        # `if args.save_intermediate:`, que nunca executam aqui. Em vez de importar
        # cv2 — que no cluster resolve para o opencv completo do ~/.local e quebra
        # com `GLIBC_2.38 not found`, porque o container tem 2.35 — deixamos uma
        # sentinela que explode com mensagem clara se alguém mexer nisso.
        "cv2": _Unavailable("cv2", "só é usado sob save_intermediate, que é sempre False"),
        "save_root": _Unavailable("save_root", "não existe fora do demo.py"),
        "os": __import__("os"),
        "__name__": "bokehme_extracted",
    }
    exec(compile(extracted, f"{demo_path}:extracted", "exec"), namespace)
    return namespace, hashlib.sha256(extracted.encode("utf-8")).hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class BokehMeRenderer:
    """Carrega o BokehMe uma vez e renderiza a partir do CoC canônico.

    Falha alto e cedo se o checkout ou os checkpoints não estiverem no lugar. Não
    existe caminho de fallback: o pipeline antigo tinha um `raise ImportError`
    incondicional que fazia `render_bokeh` cair sempre num gaussiano, e ninguém
    percebeu porque nada media o borrão produzido.
    """

    def __init__(
        self,
        bokehme_dir: str | Path,
        *,
        config: BokehMeConfig | None = None,
        device: str = "cuda",
        arnet_checkpoint: str | Path | None = None,
        iunet_checkpoint: str | Path | None = None,
    ):
        # Validação barata ANTES de importar torch: erro de config tem que aparecer
        # como erro de config, não como "No module named 'torch'".
        self.root = Path(bokehme_dir).expanduser().resolve()
        self.config = config or BokehMeConfig()
        self.device = device

        if self.config.output not in _VALID_OUTPUTS:
            raise ValueError(f"output inválido: {self.config.output!r}; use um de {sorted(_VALID_OUTPUTS)}")
        if not (self.root / "demo.py").is_file():
            raise FileNotFoundError(
                f"{self.root} não parece um checkout do JuewenPeng/BokehMe (falta demo.py)"
            )

        self.arnet_checkpoint = Path(arnet_checkpoint or self.root / "checkpoints" / "arnet.pth")
        self.iunet_checkpoint = Path(iunet_checkpoint or self.root / "checkpoints" / "iunet.pth")
        for ckpt in (self.arnet_checkpoint, self.iunet_checkpoint):
            if not ckpt.is_file():
                raise FileNotFoundError(f"checkpoint do BokehMe não encontrado: {ckpt}")

        import torch  # importado aqui para o módulo ser importável sem GPU
        self._torch = torch

        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            from classical_renderer.scatter import ModuleRenderScatter
            from neural_renderer import ARNet, IUNet
        except ImportError as exc:                      # sem fallback, de propósito
            raise ImportError(
                f"não consegui importar o BokehMe de {self.root}: {type(exc).__name__}: {exc}. "
                "O `classical_renderer.scatter` depende de `cupy` (kernel de scatter em CUDA); "
                "instale com `pip install --target <projeto>/.pydeps cupy-cuda12x` e PREFIXE o "
                "PYTHONPATH. NÃO existe renderer alternativo — o gaussiano de fallback é "
                "exatamente o defeito D4."
            ) from exc

        demo_ns, self.demo_sha256 = _extract_demo_functions(self.root / "demo.py")

        cfg = self.config
        if cfg.save_intermediate:
            raise ValueError("save_intermediate tem que ser False: `save_root` não existe fora do demo.")
        self._pipeline = demo_ns["pipeline"]
        self._classical = ModuleRenderScatter().to(device)
        self._arnet = ARNet(
            cfg.arnet_shuffle_rate, cfg.arnet_in_channels, cfg.arnet_out_channels,
            cfg.arnet_middle_channels, cfg.arnet_num_block, cfg.arnet_share_weight,
            cfg.arnet_connect_mode, cfg.arnet_use_bn, cfg.arnet_activation,
        ).to(device)
        self._iunet = IUNet(
            cfg.iunet_shuffle_rate, cfg.iunet_in_channels, cfg.iunet_out_channels,
            cfg.iunet_middle_channels, cfg.iunet_num_block, cfg.iunet_share_weight,
            cfg.iunet_connect_mode, cfg.iunet_use_bn, cfg.iunet_activation,
        ).to(device)
        # weights_only=False é obrigatório: os checkpoints do BokehMe carregam objetos
        # numpy (`numpy.core.multiarray._reconstruct`), e o torch >= 2.6 mudou o default
        # para True. É seguro AQUI e só aqui porque a origem é o repositório oficial
        # clonado por git e o sha256 de cada arquivo entra na proveniência da amostra —
        # dá para provar depois exatamente qual peso rodou.
        self._arnet.load_state_dict(
            torch.load(self.arnet_checkpoint, map_location=device, weights_only=False)["model"]
        )
        self._iunet.load_state_dict(
            torch.load(self.iunet_checkpoint, map_location=device, weights_only=False)["model"]
        )
        self._arnet.eval()
        self._iunet.eval()

        self._namespace = cfg.as_namespace()
        self.calls = 0

    # -- proveniência ---------------------------------------------------------

    def provenance(self) -> dict:
        """Vai inteiro nos metadados de CADA amostra.

        Sem o commit e os hashes não dá para provar depois com qual renderer um K foi
        calibrado — e K calibrado contra renderer diferente não é a mesma grandeza.
        """
        return {
            "renderer": "bokehme_public",
            "renderer_repo": "JuewenPeng/BokehMe",
            "renderer_commit": _git_commit(self.root),
            "renderer_dir": str(self.root),
            "demo_pipeline_sha256": self.demo_sha256,
            "scatter_py_sha256": _sha256(self.root / "classical_renderer" / "scatter.py"),
            "arnet_sha256": _sha256(self.arnet_checkpoint),
            "iunet_sha256": _sha256(self.iunet_checkpoint),
            "config": asdict(self.config),
            "is_final_label_renderer": True,   # Eq. 5 e Fig. 3(a) usam o [43] público
        }

    # -- renderização ---------------------------------------------------------

    def __call__(
        self,
        aif_bgr: np.ndarray,
        depth_m: np.ndarray,
        focus_disparity: float,
        k_value: float,
    ) -> np.ndarray:
        """Renderiza e devolve BGR **float** em [0, 255], na mesma resolução da entrada.

        `k_value` está na convenção oficial (disparidade em 1/m) e na escala de pixel
        da imagem que chega aqui. Quem reduz resolução para a calibração é responsável
        por converter K junto — ver `renderer.calibration.calibrate_k`.
        """
        torch = self._torch
        if aif_bgr.shape[:2] != depth_m.shape[:2]:
            raise SampleRejected(
                "resolution_invalid", f"aif {aif_bgr.shape[:2]} != depth {depth_m.shape[:2]}"
            )

        # CoC canônico em pixels, direto do contrato. Sem normalizar disparidade.
        coc_px = signed_coc_px(depth_m, focus_disparity, k_value)
        defocus = coc_px / self.config.defocus_scale

        rgb = np.ascontiguousarray(np.asarray(aif_bgr, dtype=np.float32)[..., ::-1]) / 255.0
        image_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        defocus_t = torch.from_numpy(np.ascontiguousarray(defocus, dtype=np.float32))
        defocus_t = defocus_t.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self._pipeline(
                self._classical, self._arnet, self._iunet,
                image_t, defocus_t, self.config.gamma, self._namespace,
            )
        selected = {"bokeh_pred": 0, "bokeh_classical": 1, "bokeh_neural": 2}[self.config.output]
        result = outputs[selected]

        out = result.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        self.calls += 1
        return np.clip(out[..., ::-1] * 255.0, 0.0, 255.0)      # de volta para BGR


def render_fn_from(renderer: BokehMeRenderer):
    """Adapta o renderer à assinatura que `calibration` e `verification` esperam."""
    def render(aif_bgr, depth_m, focus_disparity, k_value):
        return renderer(aif_bgr, depth_m, focus_disparity, k_value)
    return render
