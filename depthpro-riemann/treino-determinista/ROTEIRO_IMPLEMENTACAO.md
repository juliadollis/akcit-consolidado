# Roteiro de Implementação na H100 — Dataset, Instalação e Execução

Documento de apoio para **orientar a pessoa que vai implementar**. Cobre duas coisas:
1. **Como criar os batches do dataset** (quantas imagens em cada fase e por quê).
2. **Passo a passo operacional**: onde salvar tudo, o que ajustar no código, o que testar.

> Este roteiro é para VOCÊ (que vai orientar). Cada seção diz o que a pessoa deve fazer e
> o que verificar. Os comandos assumem a versão em container; há notas para a versão sem
> container onde muda algo.

---

# PARTE 1 — Estratégia do Dataset (os "batches")

## 1.1 Por que não usar o Hypersim inteiro

O Hypersim tem ~77 mil imagens (~150 GB+). Usar tudo desde o início é desperdício: para
**fine-tuning das cabeças** de um modelo já forte (DepthPro), o ganho satura muito antes
de usar o dataset completo. A estratégia certa é uma **escada de três tamanhos**, cada um
com um propósito distinto.


## 1.1-B Baixar SEM ter espaço para o dataset inteiro (download em streaming)

**Se não há disco para os ~150 GB do Hypersim**, use o `download_hypersim_stream.py`. Ele
baixa **uma cena por vez**, extrai só o RGB + profundidade, converte para `rgb/` + `depth/`,
e **apaga o zip antes da próxima cena**. O pico de disco é "uma cena (~1–2 GB) + o batch
convertido (pequeno)" — nunca o dataset todo.

**Duas formas de escolher o que baixar (você controla):**

Por NÚMERO de imagens (conveniência — baixa cenas até atingir o alvo):
```bash
docker compose run --rm trainer python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/ablation/train --n-images 3000 --limit-scenes 60
```

Por LISTA de cenas (controle e reprodutibilidade — todos baixam as mesmas):
```bash
docker compose run --rm trainer python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/ablation/train \
  --scenes ai_001_001 ai_001_002 ai_001_003 ai_002_001 ai_002_002
```

**Separar treino e validação (cenas diferentes!):** use `--start-index` para o val pegar
cenas de outra faixa, OU liste cenas explícitas distintas:
```bash
# treino: primeiras cenas | val: cenas a partir da 200ª (não se sobrepõem)
docker compose run --rm trainer python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/ablation/train --n-images 3000 --limit-scenes 60
docker compose run --rm trainer python scripts/download_hypersim_stream.py \
  --out-root /data/hypersim/ablation/val --n-images 500 --start-index 200 --limit-scenes 20
```

**✓ Verificação:** `ls /data/hypersim/ablation/train/rgb | wc -l` retorna ~3000.

> ⚠️ **REDE:** este download acessa a Apple (`docs-assets.developer.apple.com`), que exige
> **internet aberta** — diferente do `docker build`, que só usa GitHub/PyPI. Se o ambiente
> Docker tiver rede restrita, rode este passo com `--network host` OU num host com acesso
> livre. **A lógica de extração/conversão foi validada; o download real da Apple NÃO pôde
> ser testado no desenvolvimento.** No primeiro uso, teste com UMA cena e confira:
> ```bash
> docker compose run --rm trainer python scripts/download_hypersim_stream.py \
>   --out-root /tmp/teste1 --scenes ai_001_001
> ```
> Se gerar pares em `/tmp/teste1/rgb` e `/depth`, está funcionando — siga em frente.

> **Nem toda cena ai_VVV_NNN existe** (há lacunas no dataset). O script tolera isso: uma
> cena inexistente retorna 404, é pulada, e ele segue para a próxima. Por isso o
> `--limit-scenes` é uma folga de segurança (tente um valor maior que o mínimo teórico).

## 1.2 Os três batches (tamanhos recomendados)

| Batch | Nº de imagens (treino) | Validação | Para quê | Tempo de GPU |
|---|---|---|---|---|
| **smoke** | 40–60 | 20 | Só provar que o pipeline roda ponta a ponta | minutos |
| **ablation** | 2.000–4.000 | 400–600 | Escolher a melhor configuração de perda (39 configs) | ~1 dia |
| **full** | 15.000–25.000 | 1.500–2.500 | Treino final da configuração campeã | horas–1 dia |

**Racional de cada número:**

- **smoke (40–60):** o objetivo é *só* verificar que dados, modelo e GPU conversam. Não
  importa a qualidade — importa não dar erro. Poucas dezenas bastam e roda em minutos.

- **ablation (2.000–4.000):** aqui você compara 39 configurações de perda. O que importa é
  o **ranking relativo** entre elas ser confiável, não o desempenho absoluto. Alguns
  milhares de imagens dão sinal estatístico suficiente para dizer "a config X é melhor que
  a Y" sem gastar dias em cada uma das 39. Ir além de ~4.000 aqui raramente muda o ranking
  e multiplica o tempo por 39.

- **full (15.000–25.000):** só a configuração campeã é treinada aqui, por mais épocas e com
  3 seeds. Como é uma única configuração, você pode dar mais dados. Acima de ~25.000 o
  ganho para fine-tuning de cabeças tende a saturar; se quiser, pode usar o dataset todo,
  mas o custo/benefício cai.

> **Regra prática de proporção treino/validação:** ~10% do tamanho do treino para
> validação, com **cenas diferentes** (nunca as mesmas cenas nos dois — senão a validação
> fica otimista e mente para você).

## 1.3 Como criar os batches (o script `make_subset.py`)

Primeiro converta o Hypersim uma vez para o layout `rgb/` + `depth/` (isso é o
`prepare_hypersim.py`). Depois, o `make_subset.py` recorta os três tamanhos SEM duplicar
arquivos (usa symlink), de forma **reprodutível** (o mesmo `--n` dá sempre o mesmo
recorte).

**Fluxo completo dos dados:**
```
Hypersim bruto (.hdf5)
      │  prepare_hypersim.py  (uma vez, converte tudo ou um --max)
      ▼
/data/hypersim_full/{train,val}/  (layout rgb/ + depth/)
      │  make_subset.py  (recorta os tamanhos)
      ├──►  /data/hypersim/smoke/{train,val}
      ├──►  /data/hypersim/ablation/{train,val}
      └──►  /data/hypersim/full/{train,val}
```

## 1.4 Duas formas de montar os batches

**Forma A — converter tudo primeiro, depois recortar (recomendada se tem disco):**
```bash
# 1) Converter um bom volume do Hypersim de uma vez (ex.: 30k imagens)
python scripts/prepare_hypersim.py --hypersim-root /data/hypersim_raw \
    --out-root /data/hypersim_full/train --max 30000
python scripts/prepare_hypersim.py --hypersim-root /data/hypersim_raw_val \
    --out-root /data/hypersim_full/val --max 3000

# 2) Recortar os três batches (symlink, não duplica)
python scripts/make_subset.py --src /data/hypersim_full/train --dst /data/hypersim/smoke/train    --n 60   --seed-stride
python scripts/make_subset.py --src /data/hypersim_full/val   --dst /data/hypersim/smoke/val      --n 20   --seed-stride
python scripts/make_subset.py --src /data/hypersim_full/train --dst /data/hypersim/ablation/train --n 3000 --seed-stride
python scripts/make_subset.py --src /data/hypersim_full/val   --dst /data/hypersim/ablation/val   --n 500  --seed-stride
python scripts/make_subset.py --src /data/hypersim_full/train --dst /data/hypersim/full/train     --n 20000 --seed-stride
python scripts/make_subset.py --src /data/hypersim_full/val   --dst /data/hypersim/full/val       --n 2000  --seed-stride
```

**Forma B — converter só o que precisa (se disco é limitado):**
Use o `--max` do `prepare_hypersim.py` diretamente para cada batch, apontando para pastas
finais separadas. Mais simples, mas reconverte imagens repetidas entre batches.
```bash
python scripts/prepare_hypersim.py --hypersim-root /data/hypersim_raw \
    --out-root /data/hypersim/ablation/train --max 3000
# ...e assim por diante para cada batch/split.
```

**Forma C — download em streaming (recomendada se NÃO há disco para o dataset):**
Pula o passo de baixar tudo. O `download_hypersim_stream.py` baixa cena a cena e já
entrega no layout final. Veja a seção **1.1-B** acima. É a opção certa quando o
armazenamento é o gargalo.

> **Recomendação:** se tem disco sobrando, Forma A (converter tudo uma vez e recortar).
> **Se o disco é limitado, Forma C (streaming)** — baixa direto o tamanho que precisa.

---

# PARTE 2 — Roteiro Operacional (passo a passo para a pessoa)

## 2.1 Onde salvar cada coisa (estrutura de diretórios no host)

Defina uma raiz de trabalho, por exemplo `/data`. A estrutura final fica:

```
/data/
├── hypersim_raw/            ← Hypersim bruto baixado (o .hdf5 original)
├── hypersim_full/           ← convertido (rgb/ + depth/) — pool completo
│   ├── train/{rgb,depth}/
│   └── val/{rgb,depth}/
└── hypersim/                ← os batches recortados
    ├── smoke/{train,val}/
    ├── ablation/{train,val}/
    └── full/{train,val}/

/models/
└── checkpoints/depth_pro.pt ← pesos do modelo (baixados)

<pasta-do-projeto>/runs/     ← resultados (o container escreve aqui)
```

**Regra de ouro dos volumes (container):** o container enxerga
`/data`, `/models` e `/workspace/runs`. No host, esses três apontam para pastas suas via
`docker-compose.yml`. **A pessoa só precisa garantir que os dados estão nessas três pastas
do host** que o compose mapeia.

## 2.2 O que precisa ser ajustado no código

**Resposta curta: quase nada no código-fonte.** O caminho do dataset NÃO é fixo no código
— ele é passado por argumento de linha de comando (`--train-root` e `--val-root`). Então
trocar de batch é só trocar o caminho no comando, sem editar `.py`.

O que a pessoa ajusta, em ordem de importância:

1. **Os caminhos nos volumes do `docker-compose.yml`** (uma vez):
   ```yaml
   volumes:
     - ./runs:/workspace/runs
     - /data:/data          # ← raiz dos dados no host
     - /models:/models      # ← raiz dos modelos no host
   ```

2. **Os argumentos `--train-root` / `--val-root`** em cada comando, para apontar ao batch
   da fase atual. Exemplos:
   - smoke:    `--train-root /data/hypersim/smoke/train    --val-root /data/hypersim/smoke/val`
   - ablation: `--train-root /data/hypersim/ablation/train --val-root /data/hypersim/ablation/val`
   - full:     `--train-root /data/hypersim/full/train     --val-root /data/hypersim/full/val`

3. **(Só se necessário) o `--focal-px` do `prepare_hypersim.py`**: o padrão serve para o
   Hypersim na resolução usual. Se as imagens vierem em outra resolução, esse valor da
   focal precisa ser conferido. Para o Hypersim padrão, não mexa.

4. **(Só se o modelo reclamar) as substrings de camada em `riemann/model.py`**: ver 2.5.

> **Nenhum outro arquivo precisa ser editado no fluxo normal.** Batch, resolução, épocas,
> pesos da loss — tudo é argumento de comando.

## 2.3 Sequência exata de execução (a ordem que a pessoa deve seguir)

**Etapa 0 — Infra (uma vez):**
1. Instalar NVIDIA Container Toolkit (README_CONTAINER seção 2).
2. `docker build -t riemann-depthpro:latest .`
   **✓ testar:** o build termina com "Geometria OK".

**Etapa 1 — Preparar dados e pesos (uma vez):**
3. Criar pastas do host: `/data`, `/models`, `./runs`.
4. Baixar pesos: `download_weights.py --out-dir /models`.
   **✓ testar:** `ls -lh /models/checkpoints/depth_pro.pt` (~1.5 GB).
5. **Obter os dados** — escolha UMA via:
   - **Com disco sobrando:** baixar o Hypersim bruto, converter com `prepare_hypersim.py`
     e recortar com `make_subset.py` (Formas A/B, Parte 1).
   - **Sem disco para tudo (recomendado):** usar `download_hypersim_stream.py` para baixar
     direto o tamanho de cada batch (seção 1.1-B). Não precisa dos passos 6–7.
   **✓ testar (streaming):** primeiro rode com UMA cena e confira os pares:
   `... download_hypersim_stream.py --out-root /tmp/teste1 --scenes ai_001_001`
6. (Só na via "com disco") Converter com `prepare_hypersim.py`.
7. (Só na via "com disco") Recortar os batches com `make_subset.py`.
   **✓ testar:** `ls /data/hypersim/ablation/train/rgb | wc -l` retorna ~3000.

**Etapa 2 — Validar o pipeline (minutos):**
8. Sanidade: `test_geometry.py` → **✓** "Geometria OK".
9. Smoke test no batch smoke (2 épocas, poucas imagens):
   ```bash
   docker compose run --rm trainer python scripts/run_ablation.py \
     --train-root /data/hypersim/smoke/train --val-root /data/hypersim/smoke/val \
     --checkpoint /models/checkpoints/depth_pro.pt \
     --out-dir /workspace/runs/smoke \
     --variant heads --epochs 2 --filter B0 B1
   ```
   **✓ testar:** cria `runs/smoke/ablation_results.csv` sem erro.
   **✓ CONFERIR A LINHA:** `[Modelo] ... params treináveis=XXX,XXX` — se for 0 ou
   minúsculo, ver 2.5 ANTES de prosseguir.

**Etapa 3 — Experimento principal (horas–dias):**
10. Ablação completa no batch ablation (as 39 configs), variante heads:
    ```bash
    docker compose run -d --rm --name abl_heads trainer python scripts/run_ablation.py \
      --train-root /data/hypersim/ablation/train --val-root /data/hypersim/ablation/val \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir /workspace/runs/ablation_heads \
      --variant heads --epochs 40 --batch-size 8 --size 512
    ```
11. (Opcional) Repetir com `--variant heads_lora --out-dir .../ablation_lora`.
12. Optuna no batch ablation:
    ```bash
    docker compose run -d --rm --name optuna trainer python scripts/run_optuna.py \
      --train-root /data/hypersim/ablation/train --val-root /data/hypersim/ablation/val \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir /workspace/runs/optuna \
      --variant heads --trials 200 --epochs-per-trial 20 --batch-size 8 --size 512
    ```
    **✓ testar:** ao fim, `runs/optuna/term_presence.json` mostra quais termos dominam.

**Etapa 4 — Treino final (horas):**
13. Pegar os melhores pesos (da ablação/Optuna) e treinar a campeã no batch **full**:
    ```bash
    docker compose run -d --rm --name champ trainer python scripts/train_single.py \
      --train-root /data/hypersim/full/train --val-root /data/hypersim/full/val \
      --checkpoint /models/checkpoints/depth_pro.pt \
      --out-dir /workspace/runs/champion \
      --variant heads --berhu 0.7 --normal 0.9 --gauss 0.45 \
      --epochs 100 --seeds 3 --batch-size 8 --size 512 --export-curvature
    ```
    **✓ testar:** `runs/champion/seed_0/best.pt` existe e `curvature_maps/` tem PNGs.

## 2.4 Como acompanhar treinos longos

Os comandos das etapas 3 e 4 usam `-d` (rodam destacados). Para acompanhar:
```bash
docker logs -f abl_heads      # segue o log ao vivo
docker ps                     # ver o que está rodando
```
O CSV da ablação é escrito **a cada config concluída** — dá para abrir e ver o progresso
parcial a qualquer momento em `runs/ablation_heads/ablation_results.csv`.

## 2.5 O único ponto do código que pode exigir ajuste

O modelo identifica as camadas a treinar por NOME. Se a versão do DepthPro instalada usar
nomes diferentes, o número de parâmetros treináveis sai errado. **Sintoma:** a linha
`[Modelo] ... params treináveis=0` (ou um número absurdamente pequeno) no primeiro run.

**O que fazer:** abrir `riemann/model.py`, na função `_build`, e ajustar a lista
`head_substr` para conter os nomes reais das cabeças de decodificação daquela versão
(um `print` dos nomes dos módulos ajuda a descobrir). **Só isto** — o resto do código não
muda. Se `params treináveis` já vier num valor grande e plausível (milhões), está tudo
certo e não precisa tocar em nada.

## 2.6 Versão SEM container (se for o caso)

Se a pessoa rodar sem Docker, muda só o começo: em vez de `docker build`, ela cria um
ambiente conda e instala (`README.md`, seção de instalação), incluindo o DepthPro. Depois
disso, **todos os comandos são idênticos**, tirando o prefixo `docker compose run --rm
trainer` — ela chama `python scripts/...` direto. Os caminhos passam a ser os do host real
(não `/data`, mas o caminho verdadeiro), e não há volumes a configurar.

---

# PARTE 3 — Checklist rápido (para você conferir com a pessoa)

- [ ] NVIDIA Container Toolkit instalado (`nvidia-smi` dentro de container funciona)
- [ ] Imagem construída, build terminou com "Geometria OK"
- [ ] `/models/checkpoints/depth_pro.pt` existe (~1.5 GB)
- [ ] Hypersim convertido: `hypersim_full/{train,val}/{rgb,depth}` com pares
- [ ] Batches criados: smoke / ablation / full, cada um com train e val
- [ ] `docker-compose.yml` com os caminhos do host corretos nos volumes
- [ ] Smoke test rodou e gerou CSV
- [ ] **Linha `params treináveis=` com valor grande e plausível** (não 0)
- [ ] Ablação rodando (ou concluída) → CSV com as configs
- [ ] Optuna concluído → `term_presence.json`
- [ ] Campeã treinada → `best.pt` + `curvature_maps/`

## O que pedir de volta ao final
1. `runs/ablation_heads/ablation_results.csv` (e `_lora` se rodou)
2. `runs/optuna/pareto_front.json` e `term_presence.json`
3. `runs/champion/seed_*/summary.json`
4. Exemplos de `runs/champion/curvature_maps/*.png`
5. A linha `[Modelo] ... params treináveis=...` do primeiro run
