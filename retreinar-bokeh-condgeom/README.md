# retreinar-bokeh-condgeom — condicionamento geométrico na BokehNet

Árvore da **campanha v2** de condicionamento geométrico. Clone de
`../retreinar-bokeh` (o treino corrigido da BokehNet) mais o pacote `geocond/`.

Independente: não altera `../retreinar-bokeh`, que é o que está treinando agora
na h100n1, nem `../genrefocus_deblurnet_paper/`, que é o treino antigo.

> **Leia primeiro:** `PLANO_CONDGEOM.md` — a pergunta, o desenho dos
> experimentos, o cronograma e os riscos.
> **Antes de reaproveitar qualquer número antigo:** `referencia_antiga/LEIA.md`.

---

## Por que esta árvore começa do zero

A campanha anterior (03–09/09/2026) mediu contra modelos treinados com o **mapa
de defocus errado** — profundidade linear normalizada por imagem, onde a
inferência oficial usa disparidade métrica absoluta. O condicionamento
geométrico modula exatamente esse sinal, então toda linha de base, toda
constante calibrada e todo veredito daquela campanha mediram outra coisa.

O que sobrevive é o que não é resultado de experimento: a física, a curvatura
verificada em esfera/plano/cilindro, e o diagnóstico da densidade de token. Está
tudo **reimplementado** em `geocond/` contra o contrato novo, e requalificado.
Nada é importado da quarentena.

---

## Estado

| | |
|---|---|
| contrato do release | `metric_disparity_official_v1` |
| rota A | 69.700 amostras · 1.700 cenas · **verificado**: 210.809 arquivos, 0 faltando |
| rota B oficial · rota C | 13.761 · 15.423 (4,2% de K censurado) |
| BokehNet fase 1 | rodando na h100n1, step ~13.4K/40K |
| fase 2 (= condição A) | ainda não existe; é o que destranca a campanha |

---

## Layout

```
PLANO_CONDGEOM.md         a campanha: pergunta, condições, cronograma, riscos
proposta_wallisson.pdf    o documento de origem
geocond/
  contrato.py             CANAIS, AmostraGeo, PilhaGeo, G1_IDX/G2_IDX — a fonte da verdade
  constants.py            as constantes de normalização, SEM default (são medidas)
  signals.py              os 6 canais, de disparidade a (6,H,W) em [0,1]
  dataloader.py           a ponte com resize/crop/flip
  loss_weight.py          oclusão em pixel → peso por token, limiarizado
  ebleed.py               a métrica de vazamento (§5.4 da proposta)
  jobs/                   calibração das constantes, bancada de E_bleed
  tests/                  geometrias de propriedade fechada + os canais
genfocus_train/           o treino, com a fiação do geo
slurm/                    h100n3 (SLURM + Singularity) — ver slurm/README_n3.md
referencia_antiga/        a campanha invalidada. NÃO importar, NÃO reaproveitar número
```

## Duas máquinas, dois papéis

| | h100n1 | h100n3 |
|---|---|---|
| como roda | **Docker**, sem SLURM | **SLURM + Singularity** |
| papel | treino base da BokehNet | **bancada do condgeom** |
| GPUs | 8 (4 no treino) | 3 gerenciadas pelo SLURM, de 8 |
| armadilhas | `--user $(id -u):$(id -g)` sempre | GPU4 com defeito: selecionar por **UUID** |

## Rodar os testes

```bash
python3 -m pytest tests/ -q          # a suíte do treino
python3 -m pytest geocond/tests/ -q  # os canais e a perda
```

## Regras que custaram tempo de GPU

Estão em `../INSTRUCOES_H100.md` e em `PLANO_CONDGEOM.md` §12. As inegociáveis:
nunca excluir job, nunca apagar nada no cluster sem perguntar, `--time` sempre
alto, download do Hub é opt-in explícito, `rsync` sempre com excludes.
