#!/usr/bin/env python3
"""As cinco métricas da Tabela 2 do paper, com a VARIANTE explícita.

POR QUE A VARIANTE É O PONTO CENTRAL DESTE MÓDULO
-------------------------------------------------
O paper reporta "LPIPS [83]", "CLIP-IQA [67]", "MANIQA [76]" e "MUSIQ [29]" sem
dizer qual implementação nem qual checkpoint. O `pyiqa` oferece VÁRIAS para cada
nome, e elas NÃO são intercambiáveis — `lpips` e `lpips+` são redes diferentes,
`clipiqa` e `clipiqa+` têm prompts/cabeças diferentes, e `maniqa` versus
`maniqa-kadid` são treinos em datasets diferentes (KonIQ vs KADID).

O pipeline do time escolheu as variantes "+" (ver
`vision-pipeline/evaluation/src/deblur_evaluator.py:38-54`). Essa escolha é
DEFENSÁVEL mas NÃO CONFIRMADA contra o paper. Consequência prática: os números
absolutos podem não ser comparáveis aos publicados por esse motivo, de forma
independente da resolução de avaliação.

Por isso aqui:
  * a variante de cada métrica é parâmetro, não constante;
  * o default reproduz o pipeline do time (para não inventar um terceiro
    protocolo);
  * a variante EXATA usada vai para o JSON de saída;
  * `--variantes base` permite a análise de sensibilidade num comando.

DEFEITO DO PIPELINE DO TIME QUE ESTE MÓDULO NÃO REPETE
------------------------------------------------------
Em `vision-pipeline/evaluation/src/core/evaluator.py:47-52`, uma métrica que
lança exceção é registrada como **0.0**. Para LPIPS e DISTS, onde menor é
melhor, um erro vira uma nota PERFEITA e melhora a média em silêncio. Aqui um
erro devolve `None`, é contado, e fica visível no resumo.
"""

from __future__ import annotations

# Variantes que o pipeline do time usa. É o default para não criar um terceiro
# protocolo no projeto.
VARIANTES_TIME = {
    "LPIPS": "lpips+",
    "DISTS": "dists",
    "CLIP-IQA": "clipiqa+",
    "MANIQA": "maniqa-kadid",
    "MUSIQ": "musiq",
}

# Variantes BASE, as que o nome do paper sugere ao pé da letra. Servem para
# medir quanto da diferença com o publicado vem da escolha de variante.
VARIANTES_BASE = {
    "LPIPS": "lpips",
    "DISTS": "dists",
    "CLIP-IQA": "clipiqa",
    "MANIQA": "maniqa",
    "MUSIQ": "musiq",
}

CONJUNTOS = {"time": VARIANTES_TIME, "base": VARIANTES_BASE}

# Sentido de cada métrica, como o paper reporta (↓ menor melhor, ↑ maior melhor).
SENTIDO = {
    "LPIPS": "menor", "DISTS": "menor",
    "CLIP-IQA": "maior", "MANIQA": "maior", "MUSIQ": "maior",
}

# Quais exigem ground-truth. As no-reference são calculadas SÓ na predição.
PRECISA_GT = {
    "LPIPS": True, "DISTS": True,
    "CLIP-IQA": False, "MANIQA": False, "MUSIQ": False,
}

ORDEM = ["LPIPS", "DISTS", "CLIP-IQA", "MANIQA", "MUSIQ"]


def criar_metricas(conjunto: str = "time", device: str = "cuda",
                   sobrescrever: dict[str, str] | None = None):
    """Instancia as 5 métricas. Devolve (metricas, detalhes).

    `detalhes` registra a variante e o sentido de cada uma, para ir ao JSON.
    Falha explicitamente se o `pyiqa` não estiver instalado — métrica faltando
    em silêncio é pior que erro.
    """
    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit(
            "ERRO: `pyiqa` não está instalado — é ele que fornece as 5 métricas.\n"
            "  pip install pyiqa\n"
            "No cluster, instale DENTRO do container, com --target num .pydeps do\n"
            "projeto, e prefixe o PYTHONPATH (ver INSTRUCOES_H100.md). Nunca no ~/.local."
        ) from exc

    if conjunto not in CONJUNTOS:
        raise SystemExit(f"conjunto de variantes inválido: {conjunto!r}. Use: {sorted(CONJUNTOS)}")

    variantes = dict(CONJUNTOS[conjunto])
    if sobrescrever:
        variantes.update({k: v for k, v in sobrescrever.items() if v})

    metricas, detalhes = {}, {}
    for nome in ORDEM:
        var = variantes[nome]
        try:
            m = pyiqa.create_metric(var, device=device)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"ERRO: não consegui criar a métrica {nome} (variante {var!r}): {exc}\n"
                "Abortando: uma tabela com métrica faltando não é interpretável."
            ) from exc
        metricas[nome] = m
        detalhes[nome] = {
            "variante_pyiqa": var,
            "conjunto": conjunto,
            "sentido": SENTIDO[nome],
            "precisa_gt": PRECISA_GT[nome],
            "lower_better_pyiqa": bool(getattr(m, "lower_better", SENTIDO[nome] == "menor")),
        }
    return metricas, detalhes


def calcular(metricas, t_pred, t_gt=None) -> dict[str, float | None]:
    """Calcula as métricas para um par. Tensores em [0,1], shape (1,3,H,W).

    Um erro devolve `None` para aquela métrica, NUNCA 0.0 — ver o defeito
    documentado no topo do módulo.
    """
    import torch

    valores: dict[str, float | None] = {}
    for nome, m in metricas.items():
        try:
            with torch.no_grad():
                if PRECISA_GT[nome]:
                    if t_gt is None:
                        raise ValueError(f"{nome} exige ground-truth")
                    v = m(t_pred, t_gt)
                else:
                    v = m(t_pred)
            valores[nome] = float(v.item())
        except Exception as exc:  # noqa: BLE001
            print(f"    AVISO: {nome} falhou neste par: {type(exc).__name__}: {exc}")
            valores[nome] = None
    return valores


def para_tensor(img, device):
    """PIL RGB → tensor (1,3,H,W) em [0,1], que é o range que o pyiqa espera."""
    import numpy as np
    import torch

    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def agregar(por_imagem: list[dict]) -> dict[str, dict]:
    """Média/mediana por métrica, contando explicitamente os None.

    Diferente do pipeline do time, que divide a soma pelo total de pares mesmo
    quando algum par falhou (`core/evaluator.py:113`) — isso enviesa a média.
    Aqui a média é sobre os pares VÁLIDOS e o nº de falhas fica registrado.
    """
    resumo = {}
    for nome in ORDEM:
        vals = [r[nome] for r in por_imagem if r.get(nome) is not None]
        falhas = sum(1 for r in por_imagem if nome in r and r[nome] is None)
        if not vals:
            resumo[nome] = {"media": None, "n_validos": 0, "n_falhas": falhas}
            continue
        ordenados = sorted(vals)
        resumo[nome] = {
            "media": sum(vals) / len(vals),
            "mediana": ordenados[len(ordenados) // 2],
            "min": ordenados[0],
            "max": ordenados[-1],
            "n_validos": len(vals),
            "n_falhas": falhas,
            "sentido": SENTIDO[nome],
        }
    return resumo


def adicionar_args(p) -> None:
    """Adiciona as flags de variante a um ArgumentParser."""
    p.add_argument("--variantes", default="time", choices=sorted(CONJUNTOS),
                   help="'time' = lpips+/clipiqa+/maniqa-kadid (default do pipeline do time); "
                        "'base' = lpips/clipiqa/maniqa")
    for nome in ORDEM:
        p.add_argument(f"--{nome.lower().replace('-', '')}-variante", default=None,
                       help=f"sobrescreve a variante pyiqa de {nome}")


def variantes_dos_args(args) -> dict[str, str]:
    """Extrai as sobrescritas de variante do namespace do argparse."""
    out = {}
    for nome in ORDEM:
        attr = f"{nome.lower().replace('-', '')}_variante"
        v = getattr(args, attr, None)
        if v:
            out[nome] = v
    return out


if __name__ == "__main__":
    print("Variantes disponíveis:\n")
    for conj, d in CONJUNTOS.items():
        print(f"  {conj}:")
        for nome in ORDEM:
            print(f"    {nome:9s} -> {d[nome]:14s} ({SENTIDO[nome]} é melhor, "
                  f"{'full-ref' if PRECISA_GT[nome] else 'no-ref'})")
        print()
    print("As variantes do conjunto 'time' NÃO foram confirmadas contra o paper.")
    print("Ver PROTOCOLO.md, seção 'Variantes de métrica'.")
