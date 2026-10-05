"""Guardas do laço de treino que nenhum teste de unidade alcança.

As três guardas aqui nasceram de defeitos MEDIDOS neste projeto, e todas as três
vivem dentro de `_train_loop`, que só roda com FLUX carregado e `accelerate`
inicializado — caro demais para um teste. Então a verificação é sobre o TEXTO do
módulo. É uma trava fraca, e está declarado que é: ela não prova que a guarda
funciona, só que alguém não a apagou sem querer. O que prova que funciona é o
run.
"""

from __future__ import annotations

import pathlib

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def _fonte(nome: str) -> str:
    return (RAIZ / "genfocus_train" / f"{nome}.py").read_text(encoding="utf-8")


def test_probe_interno_recusa_multi_gpu():
    """O probe interno em multi-GPU é a dessincronia que abortou o treino 2x.

    Ele roda ~5 min só no rank 0; os outros avançam até o `all_reduce` dos
    gradientes e ficam presos — cenário exato do watchdog do NCCL. E a
    `ParadaPorControlabilidade` sobe só no rank 0, que vai parar num
    `wait_for_everyone()` enquanto os demais esperam num `all_reduce`: coletivos
    diferentes, deadlock. A medida correta roda FORA (scripts/monitor_externo.py).
    """
    fonte = _fonte("trainer")
    assert 'probe_every and accelerator.num_processes > 1' in fonte, (
        "sumiu a guarda que recusa `probe_every_steps > 0` com mais de um "
        "processo"
    )
    # e tem de ABORTAR, não degradar para probe desligado
    trecho = fonte.split("probe_every and accelerator.num_processes > 1")[1][:1600]
    assert "raise ValueError" in trecho, "a guarda tem de levantar, não avisar"
    assert "monitor_externo.py" in trecho, (
        "a mensagem precisa apontar o caminho que FUNCIONA, senão o operador "
        "só desliga o probe e fica sem instrumento nenhum"
    )


def test_aviso_de_validacao_so_no_rank_principal():
    """Nos ranks != 0 o `val_loader` é None POR PROJETO.

    Imprimir "validação DESLIGADA" ali é falso. Custou dois dias de README
    errado: a fase 1 validou normalmente e o `best.pt` saiu do step 40.000 com
    `val_loss=0,0602`, mas os três avisos dos ranks 1-3 no log diziam o
    contrário.
    """
    fonte = _fonte("trainer")
    assert (
        "if eval_every and val_loader is None and accelerator.is_main_process:"
        in fonte
    ), "o aviso de validação voltou a rodar em todos os ranks"


def test_bokeh_shape_exige_init_lora_com_mensagem():
    """Sem `--init-lora`, `Path(None)` estoura um TypeError opaco.

    E a mensagem de `run_bokeh_shape_stage`, que explica por que a fase de forma
    precisa do LoRA da fase 2, nunca chega a ser impressa.
    """
    fonte = (RAIZ / "genfocus_train" / "train.py").read_text(encoding="utf-8")
    trecho = fonte.split('if args.command == "bokeh-shape":')[1][:900]
    assert "if not args.init_lora:" in trecho
    assert "SystemExit" in trecho


def test_guarda_de_memoria_compartilhada():
    """`num_workers > 0` com /dev/shm de 64 MB mata o treino com Bus error.

    E mata DEPOIS de carregar o FLUX, de um processo filho, às vezes com status
    de saída limpo. O container da fase 1 subiu com `--shm-size=64g --ipc=host`;
    os de verificação de 2026-09-25 não, e queimaram dez minutos de GPU cada até
    o diagnóstico. A checagem custa um `statvfs`.
    """
    fonte = _fonte("trainer")
    assert "_conferir_memoria_compartilhada" in fonte
    trecho = fonte.split("def _conferir_memoria_compartilhada")[1][:1800]
    assert "/dev/shm" in trecho and "raise RuntimeError" in trecho
    assert "--shm-size" in trecho, (
        "a mensagem precisa dar a flag que resolve, senão o operador só baixa "
        "`num_workers` e perde vazão sem saber por quê"
    )
    # e tem de ser CHAMADA, não só definida
    assert "_conferir_memoria_compartilhada(config)" in fonte
