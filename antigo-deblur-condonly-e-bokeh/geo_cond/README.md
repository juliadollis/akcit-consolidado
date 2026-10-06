# Condicionamento geométrico da BokehNet

Pasta isolada. Enquanto a campanha não decidir, **nada aqui é importado pelo
`genfocus_train`**: a integração no dataloader e no trainer é um passo explícito
e posterior. Assim dá para escrever e testar sem risco de mexer no caminho que
produziu os modelos já treinados.

Contexto: `../PLANO_CONDICIONAMENTO_GEOMETRICO.md`.
Números que fundamentam as decisões: `../AUDITORIA_DADOS_ROTAS_BC.md`.

## O que já existe

| arquivo | o que é | estado |
|---|---|---|
| `constants.py` | `GeoConstants`, as 8 constantes fixas de normalização | pronto, **não calibrado** |
| `signals.py` | os 6 canais, de `depth01` a `(6, H, W)` em [0,1] | pronto |
| `loss_weight.py` | mapa de oclusão em pixel para peso por token, e a perda ponderada | pronto |
| `tests/geometrias.py` | cenas sintéticas de geometria conhecida | pronto |
| `tests/test_signals.py` | 28 testes | passando |
| `tests/test_loss_weight.py` | 15 testes | passando |

```bash
.venv/bin/python -m pytest tests/ -q      # do diretorio geo_cond, 43 testes
```

O venv local existe porque este Mac não tem numpy, torch nem PIL. Ele está no
`.gitignore` e não vai para o cluster; lá o ambiente já tem tudo.

## O que os testes garantem

**Curvatura correta, não a de Monge.** Esfera de raio R devolve `K = 1/R²` com
erro abaixo de 2%, para R de 0,5 a 5 m. Plano inclinado e cilindro devolvem 0.
É o mesmo teste que o `RETESTE_CURVATURA.md` usou para mostrar que a formulação
de `geometry_maps.principal_curvatures` devolve ~10 onde o valor verdadeiro é
0,25, e é ela que gerou as figuras do documento de proposta.

**Invariância a resolução.** Mesma cena a 96 e a 192 px dá o mesmo `K` quando o
`fx` acompanha, e o mesmo gradiente de primeira ordem quando `px_per_unit`
acompanha. Há um teste que faz o contrário de propósito, com o `fx` sem correção,
e exige que o resultado dê ERRADO: é a guarda contra alguém esquecer o fator
`512/min(W,H)` do dataloader.

**Ordenação dos tokens.** Um bloco de 16x16 pixels aceso acende exatamente um
token, no índice esperado, usando o `_pack_latents` REAL do diffusers. Um erro de
ordenação aqui não levanta exceção e custaria 60K steps.

**Max contra média.** Uma borda de 1 px dentro de um bloco de 16x16 mantém a
amplitude cheia com max e cai para 1/16 com média. Está no teste em número.

**Continuidade da ablação.** `lambda_o = 0` reproduz numericamente a perda atual,
senão a condição A' não isolaria nada.

## O que falta

1. **Calibrar as 8 constantes** sobre o conjunto de treino. `constants.py` não
   tem default nenhum, de propósito: um número plausível silencioso reintroduz a
   classe de defeito que a auditoria encontrou.
2. **Decidir `field`**: default é `"inverse"` (`u = 1/Z`), por física. `"depth"`
   existe para a diferença ser ablacionável.
3. **Trabalhos de dado da fase F0b**: Depth Pro na rota c, sentinela do `z_max`,
   gate de quantização, correção do `fx` pelo resize.
4. **Integração**: gancho em `prepare_aligned_bokeh`, chave `geo_map` nos dois
   `__getitem__`, branch em `models.py`, config nos quatro lugares, paridade de
   inferência.

## Uma nota sobre o `field`

O default `"inverse"` não é estética. O raio do círculo de confusão é linear em
`1/Z`, e o termo de anisotropia da expansão é `eps = gamma ||grad D|| / (Z² c)`,
com `grad(1/Z) = -grad D / Z²`. Logo a magnitude que governa a anisotropia é
`||grad u||`, não `||grad D||`.

Com `||grad D||` o fundo distante domina o sinal, que é justamente onde o borrão
é mais uniforme: um degrau de 1 m para 20 m dá 19 em `D` e 0,95 em `u`; um de
20 m para 40 m dá 20 em `D`, **maior**, e 0,025 em `u`. Só em `u` a ordenação é a
opticamente correta. Há um teste que mede exatamente essa razão.
