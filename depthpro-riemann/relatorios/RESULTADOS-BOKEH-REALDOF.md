# Bokeh no RealDOF — a profundidade afinada gera bokeh melhor?

Rota C do `bokehnet-regen`, adaptada para aceitar o RealDOF (50 imagens).
Renderiza bokeh a partir de uma all-in-focus mais um mapa de profundidade, e
compara com a fotografia com bokeh real.

Três fontes de profundidade × dois modos de calibração, mais dois controles sem
alinhamento. n = 50 em todas.

---

## 1. As condições

| fonte | calibração | SSIM média±dp | SSIM mediana | PSNR média±dp |
|---|---|---|---|---|
| DepthPro base | busca | 0,8904 ± 0,0688 | 0,9005 | 26,75 ± 2,60 |
| Bgrad s0 | busca | 0,8917 ± 0,0694 | 0,9043 | 26,77 ± 2,61 |
| Bgeod s0 | busca | 0,8926 ± 0,0686 | 0,9052 | 26,77 ± 2,60 |
| DepthPro base | K fixo | 0,8103 ± 0,1041 | 0,8306 | 25,01 ± 2,72 |
| Bgrad s0 | K fixo | 0,8163 ± 0,1037 | 0,8391 | 25,11 ± 2,74 |
| Bgeod s0 | K fixo | 0,8173 ± 0,1031 | 0,8357 | 25,13 ± 2,74 |

K fixo = 266,782, a mediana dos K que a busca escolheu na condição base.

## 2. O contraste pareado

O desvio entre imagens (0,07–0,10) é dez vezes maior que a diferença entre
braços, então o único teste legível é o pareado por imagem.

| comparação | ΔSSIM média±dp | vence em | ΔPSNR | t pareado |
|---|---|---|---|---|
| Bgrad − base, busca | +0,00133 ± 0,00954 | 24/50 | +0,013 dB | +0,99 |
| Bgeod − base, busca | +0,00218 ± 0,00866 | 27/50 | +0,020 dB | +1,78 |
| Bgrad − base, K fixo | +0,00605 ± 0,01998 | 28/50 | +0,096 dB | +2,14 |
| Bgeod − base, K fixo | +0,00705 ± 0,01914 | 30/50 | +0,123 dB | +2,60 |

---

## 3. A leitura

**Não há efeito que sustente a hipótese.** Sob a busca de K, nada: t < 2 nos dois
braços, e cada um ganha em cerca de metade das imagens. Sob K fixo os dois passam
de t = 2,01, mas com quatro comparações a barreira sobe para ~2,68 e nem o Bgeod
chega lá. O tamanho do efeito é +0,007 de SSIM contra um desvio entre imagens de
0,10.

A leitura compatível com os números é "os afinados são levemente melhores quando
não se pode reajustar K", não "geram bokeh melhor".

**O alinhamento afim não fazia o que eu disse que fazia.** Eu justifiquei o
alinhamento dizendo que sem ele a diferença no bokeh seria deriva de escala. Sob
a busca de K isso é falso, e o motivo é algébrico:

```
CoC = K·(disp − D_focus),   D_focus = mediana(disp na região)
disp → a·disp + b  ⟹  D_focus → a·D_focus + b     (mediana é afim-equivariante)
⟹ CoC → K·a·(disp − D_focus)
```

O deslocamento `b` cancela exatamente e a escala `a` é absorvida por K. Medido:
`K_alinhado/K_cru` tem mediana 1,335 (Bgrad) e 1,308 (Bgeod), que é `1/a` com `a`
mediano 0,744 e 0,755. Excluindo as amostras censuradas no teto, a diferença
entre alinhar e não alinhar é **+1,3e-5 ± 5,5e-5**. Zero numérico.

O alinhamento continua necessário, por dois outros motivos: torna os K
**comparáveis entre braços** (sem ele a tabela reportaria K mediano 194 contra
267 como se fossem a mesma grandeza), e sob K fixo é a condição de existência da
comparação — um K compartilhado sobre profundidade 1,7× fora de escala mediria
escala, não forma.

**O que o ajuste afim revela:** R² mediano de 0,968 (mínimo 0,61) entre a
disparidade dos braços e a da base. A relação não é perfeitamente afim, e esse
resíduo **é** a diferença de forma que o experimento queria isolar. Ela existe;
só não aparece no SSIM do bokeh.

---

## 4. Decisões de método que valem registro

**Alinhamento em disparidade, não em profundidade.** Medido, não assumido: R²
0,907 contra 0,876 (Bgrad) e 0,878 contra 0,828 (Bgeod), com a disparidade
ganhando em 10 de 10 casos da sonda. Coerente com o renderizador, que vive em
disparidade (`CoC = K·(1/z − D_focus)`).

**Os pesos não são métricos.** O treino alinhava por transformação afim dentro da
loss, então a escala absoluta nunca entrou. A disparidade sai em 0,574 da base
(faixa 0,271–0,843), com fator variável por imagem.

**Os checkpoints foram mesclados com a base** antes de carregar, porque o
`create_model_and_transforms` carrega com `strict=True` e `raise KeyError`. Dos
1119 tensores, 433 vêm do nosso `best.pt` (0 inesperados, 0 shape errado) e 686
dos troncos DINOv2 congelados. Das 433, só **77 mudaram de verdade**, com |δ|
máximo de 0,0141.

---

## 5. O que estes números NÃO sustentam

**3 amostras batem no teto de K = 960** na condição base. O RealDOF tem desfoque
muito mais forte que a RealBokeh, e a resolução é ~2320 px: os K ficam na casa
das centenas, não dos 3,6–36 da âncora original da rota C.

**O K fixo está longe do ótimo em quase toda imagem.** A busca escolhe K entre
39,9 e 960; um valor único não serve a essa dispersão. A condição existe para
separar o que vem do ajuste do que vem da profundidade, não para ser realista.

**Uma amostra teve 14,9% dos pixels presos no piso** de disparidade (z = 10.000 m,
teto do DepthPro) depois da afim. Nas demais é ~0. Gravado em `fracao_no_piso`.

**Resolução heterogênea** (5 resoluções nas 50 linhas) dá ~1,4% de espalhamento
no K em pixel. Não contamina a comparação, que é pareada por imagem.

**`k_source` continua `eq5_ssim_sweep` mesmo no modo K fixo** — `dataio.KSource` é
vocabulário fechado de outra pessoa e não foi editado. Quem ler tem que olhar
`k_search.k_mode`, que diz `k_fixo`. Estes diretórios são instrumento de
experimento, não release.

---

Artefatos em `/raid/user_juliadollis/julia_docker/caminho_a_rotac/`.
Proveniência idêntica nas 8 condições (`pipeline_source_sha256` 2bb75ca002fb1537…).
