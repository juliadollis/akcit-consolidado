import os
import io
import gc
import cv2
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from datasets import load_dataset
from huggingface_hub import HfApi, list_repo_files
from dotenv import load_dotenv

import depth_pro
from diffusers import FluxPipeline
from Genfocus.pipeline.flux import Condition, generate, seed_everything
from evaluation import CloudBokehEvaluator

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Mapeamento dos datasets de entrada originais
COLUNA_IMAGEM_BLUR = "image_blur"    # Imagem Bokeh Alvo (Gabarito para o SSIM)
COLUNA_IMAGEM_FOCUS = "image_focus"  # Imagem Nítida AIF (Conteúdo de entrada para o FLUX)
COLUNA_NOME = "file_name_base"
COLUNA_MASCARA = "foreground_mask"   # usada quando o dataset ja traz a mascara

MAX_COC = 100.0
DIR_MAPAS_TEMP = "./temp_depth_maps"

# Plano de foco: "mascara" = Eq. 4 do paper (mediana da disparidade dentro da
# mascara do objeto saliente); "centro" = comportamento antigo (pixel central),
# mantido apenas para medir o efeito da correcao.
PLANO_FOCO = os.environ.get("PLANO_FOCO", "mascara").strip().lower()


def resize_and_pad_image(img: Image.Image, target_long_side: int) -> Image.Image:
    """
    Redimensiona uma imagem mantendo a proporção e ajusta as dimensões para múltiplos de 16.
    """
    w, h = img.size
    if target_long_side and target_long_side > 0:
        target_max = int(target_long_side)
        if w >= h:
            new_w = target_max
            scale = target_max / w
            new_h = int(h * scale)
        else:
            new_h = target_max
            scale = target_max / h
            new_w = int(w * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        final_w = (new_w // 16) * 16
        final_h = (new_h // 16) * 16
        final_w = max(final_w, 16)
        final_h = max(final_h, 16)
        left = (new_w - final_w) // 2
        top = (new_h - final_h) // 2
        right = left + final_w
        bottom = top + final_h
        return img.crop((left, top, right, bottom))

    final_w = ((w + 15) // 16) * 16
    final_h = ((h + 15) // 16) * 16
    if final_w == w and final_h == h:
        return img
    return img.resize((final_w, final_h), Image.LANCZOS)


def converter_pil_para_bytes(img: Image.Image) -> bytes:
    """Converte um objeto PIL em buffer binário PNG para colunas Parquet."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def calcular_ssim_pil(img1: Image.Image, img2: Image.Image) -> float:
    """Calcula o SSIM estrutural entre duas imagens PIL redimensionadas."""
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.LANCZOS)
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))
    score, _ = ssim(arr1, arr2, channel_axis=2, full=True, data_range=255)
    return float(score)


# =============================================================================
# FASE 1: DEPTH PRO GLOBAL (Otimizado com Checkpoint do Hub)
# =============================================================================
def fase1_gerar_mapas(dados_validacao, arquivos_no_hub, tamanho_lote, device):
    """
    Roda o Depth Pro de forma isolada na VRAM para extrair os mapas de profundidade.
    Inteligentemente pula imagens cujos lotes finais já existem no Hub de destino.
    """
    print("\n[FASE 1] Inicializando Depth Pro para extração de mapas físicos...")
    os.makedirs(DIR_MAPAS_TEMP, exist_ok=True)
    
    depth_model, depth_transform = depth_pro.create_model_and_transforms()
    depth_model.eval().to(device)
    
    for idx, linha in enumerate(dados_validacao):
        numero_do_lote_atual = (idx // tamanho_lote) + 1
        nome_parquet_lote = f"data/validation_part_{numero_do_lote_atual:03d}.parquet"
        
        # Otimização crucial: Não gasta GPU calculando profundidade de lotes já processados
        if nome_parquet_lote in arquivos_no_hub:
            continue
            
        nome_arquivo = linha.get(COLUNA_NOME) or f"val_{idx:05d}.png"
        caminho_mapa = os.path.join(DIR_MAPAS_TEMP, f"{nome_arquivo}_depth.npy")
        
        if os.path.exists(caminho_mapa):
            continue
            
        try:
            dado_img_focus = linha[COLUNA_IMAGEM_FOCUS]
            if isinstance(dado_img_focus, dict) and "bytes" in dado_img_focus:
                raw_img_focus = Image.open(io.BytesIO(dado_img_focus["bytes"]))
            elif isinstance(dado_img_focus, bytes):
                raw_img_focus = Image.open(io.BytesIO(dado_img_focus))
            else:
                raw_img_focus = dado_img_focus
                
            raw_img_focus = raw_img_focus.convert("RGB")
            clean_input = resize_and_pad_image(raw_img_focus, 0) # Mantém tamanho original múltiplo de 16
            w, h = clean_input.size
            
            img_t = depth_transform(clean_input).to(device)
            with torch.no_grad():
                pred = depth_model.infer(img_t, f_px=None)
                
            depth_arr = pred["depth"].cpu().numpy().squeeze()
            
            # Redimensionamento preciso do mapa para as dimensões exatas de entrada via PIL
            depth_img = Image.fromarray(depth_arr).resize((w, h), Image.BILINEAR)
            depth_arr = np.array(depth_img)
            
            np.save(caminho_mapa, depth_arr)

        except Exception as e:
            print(f"  [Aviso] Falha ao extrair mapa do índice {idx}: {e}")

    del depth_model
    del depth_transform
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[FASE 1] Concluída com sucesso! Memória da GPU totalmente liberada para o FLUX.")


# =============================================================================
# FASE 1b: MASCARA DE PRIMEIRO PLANO (BiRefNet) — necessaria para a Eq. 4
# =============================================================================
# Paper Eq. 4: D_focus = mediana da profundidade DENTRO da mascara do objeto
# saliente, obtida com o BiRefNet. A inferencia oficial do paper
# (Inference_bokehNet.py:113-121) implementa exatamente isso quando recebe
# --mask:  `disp_focus = float(np.median(disp[mask_np > 0]))`.
# Sem mascara ela cai no PIXEL CENTRAL, que e um FALLBACK, nao o metodo.
def fase1b_gerar_mascaras(dados_validacao, arquivos_no_hub, tamanho_lote, device):
    """Gera a mascara de primeiro plano de cada imagem AIF com o BiRefNet."""
    print("\n[FASE 1b] Inicializando BiRefNet para as mascaras da Eq. 4...")
    os.makedirs(DIR_MAPAS_TEMP, exist_ok=True)

    from torchvision import transforms

    # NOTA: sem `token=` — o transformers 5.0.0.dev0 deste ambiente levanta
    # TypeError ("multiple values for keyword argument 'token'"). O BiRefNet e
    # publico, entao o token nao e necessario.
    #
    # DEGRADACAO CONTROLADA: o BiRefNet exige `kornia`, que NAO esta na imagem
    # (foi instalado a parte com --no-deps e entra por PYTHONPATH, para nao
    # deixar o pip mexer no transformers/diffusers/peft). Se ele nao carregar,
    # avisamos alto e seguimos com o pixel central — a corrida continua valida,
    # so deixa de ser fiel a Eq. 4. Sem isto, qualquer runner que nao monte o
    # PYTHONPATH extra (a fila dos keepers, por exemplo) quebraria.
    try:
        from transformers import AutoModelForImageSegmentation
        seg = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        seg.eval().to(device)
    except Exception as e:
        print(f"[FASE 1b] ATENCAO: BiRefNet indisponivel ({type(e).__name__}: {e}).", flush=True)
        print("[FASE 1b] Sem mascara o plano de foco cai no PIXEL CENTRAL, que NAO e a Eq. 4.", flush=True)
        print("[FASE 1b] Para corrigir, monte o PYTHONPATH com o kornia instalado a parte.", flush=True)
        return
    tf_seg = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for idx, linha in enumerate(dados_validacao):
        numero_do_lote_atual = (idx // tamanho_lote) + 1
        if f"data/validation_part_{numero_do_lote_atual:03d}.parquet" in arquivos_no_hub:
            continue
        nome_arquivo = linha.get(COLUNA_NOME) or f"val_{idx:05d}.png"
        caminho_mask = os.path.join(DIR_MAPAS_TEMP, f"{nome_arquivo}_mask.npy")
        if os.path.exists(caminho_mask):
            continue
        try:
            dado = linha[COLUNA_IMAGEM_FOCUS]
            if isinstance(dado, dict) and "bytes" in dado:
                img = Image.open(io.BytesIO(dado["bytes"]))
            elif isinstance(dado, bytes):
                img = Image.open(io.BytesIO(dado))
            else:
                img = dado
            img = img.convert("RGB")
            clean_input = resize_and_pad_image(img, 0)
            w, h = clean_input.size

            with torch.no_grad():
                saida = seg(tf_seg(clean_input).unsqueeze(0).to(device))[-1].sigmoid()
            prob = saida[0, 0].float().cpu().numpy()
            prob_img = Image.fromarray((prob * 255.0).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            )
            np.save(caminho_mask, np.array(prob_img))
        except Exception as e:
            print(f"  [Aviso] Falha ao extrair mascara do indice {idx}: {e}")

    del seg
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[FASE 1b] Mascaras prontas. VRAM liberada.")


# =============================================================================
# FASE 2: PIPELINE FLUX + BUSCA BINÁRIA + LVCORR
# =============================================================================
def rodar_inferencia_em_lotes_hf(
    repo_id_entrada: str,
    padrao_validacao: str,
    repo_id_saida: str,
    tamanho_lote: int = 50,
    caminho_completo_lora: str = "pesos_oficiais/bokehNet.safetensors",
    token_hf: str = None,
    long_side: int = 512,
    limite: int = 0,
    k_escala: float = 1.0,
) -> None:
    """ADAPTADO (2026-08-19) para a comparacao de 3 modelos. Mudancas:
      - `caminho_completo_lora` aceita None/"none" = SEM LoRA (linha de base,
        FLUX.1-dev cru recebendo as mesmas 2 condicoes);
      - `long_side` virou parametro (era fixo em 0 = resolucao original). Com 3
        modelos x busca binaria x LVCorr, a resolucao original ficaria inviavel
        em tempo; 512 e a resolucao em que o nosso modelo foi treinado. Todos os
        3 modelos rodam IGUAL, entao a comparacao ENTRE ELES continua justa.
        Para bater com os numeros PUBLICADOS do paper, use long_side=0.
      - `limite` corta o numero de imagens (0 = todas).
      - `k_escala` (2026-08-19) multiplica o K da bisseccao E do sweep de LVCorr.
        MOTIVO: na 1a rodada os 3 modelos escolheram best_k no PISO da faixa
        (2.0, 2.0 e 1.0 num intervalo 1..100). Otimo encostado no limite =
        BUSCA SATURADA: ate o menor K ja produz blur demais, ou seja, nenhum
        modelo estava operando na faixa para a qual foi treinado, e a comparacao
        entre eles fica enviesada. Com k_escala=0.01 a busca cobre K de 0.01 a
        1.0. O ALGORITMO nao muda (bisseccao inteira 1..100), so a escala.
        A LVCorr NAO e afetada pelo fator: Pearson e invariante a escala linear.
    Nada da busca binaria, do sweep de K ou das metricas foi alterado.
    """
    steps = 28
    temp_parquet_local = "temp_lote_processamento.parquet"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Dispositivo detectado: {device} | Tipo de dado principal: {dtype}")

    sem_lora = caminho_completo_lora is None or str(caminho_completo_lora).lower() in ("", "none")
    if sem_lora:
        print("[info] SEM LoRA: linha de base (FLUX.1-dev cru, mesmas 2 condicoes).")
        lora_dir = lora_weight_name = None
    else:
        if not os.path.exists(caminho_completo_lora):
            print(f"[Erro] Pesos do LoRA de Bokeh não encontrados em: {caminho_completo_lora}")
            return
        lora_dir = os.path.dirname(caminho_completo_lora)
        lora_weight_name = os.path.basename(caminho_completo_lora)

    print("Mapeando integridade do repositório remoto de destino...")
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id_saida, repo_type="dataset", exist_ok=True, token=token_hf)
        arquivos_no_hub = list_repo_files(repo_id=repo_id_saida, repo_type="dataset", token=token_hf)
    except Exception as e:
        print(f"[Aviso] Não foi possível verificar repositório remoto: {e}")
        arquivos_no_hub = []

    print(f"Puxando dataset do Hub: {repo_id_entrada}...")
    try:
        caminho_direto = f"hf://datasets/{repo_id_entrada}/{padrao_validacao}"
        dataset_hf = load_dataset("parquet", data_files={"validation": caminho_direto}, token=token_hf)
        dados_validacao = dataset_hf["validation"]
    except Exception as e:
        print(f"[Erro] Falha ao carregar dataset do Hugging Face: {e}")
        return
        
    if limite and limite > 0:
        dados_validacao = dados_validacao.select(range(min(limite, len(dados_validacao))))
    total_imagens = len(dados_validacao)
    print(f"Total de imagens na validação: {total_imagens}")
    
    # Executa as Fases 1 e 1b isoladas antes de instanciar o FLUX
    fase1_gerar_mapas(dados_validacao, arquivos_no_hub, tamanho_lote, device)
    if PLANO_FOCO == "mascara" and COLUNA_MASCARA not in (dados_validacao.column_names or []):
        fase1b_gerar_mascaras(dados_validacao, arquivos_no_hub, tamanho_lote, device)
    print(f"[info] plano de foco = {PLANO_FOCO!r}", flush=True)

    print("\n[FASE 2] Carregando a pipeline principal do FLUX...")
    pipe_flux = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=token_hf)
    if device == "cuda":
        pipe_flux.to("cuda")
        pipe_flux.enable_attention_slicing()
        if hasattr(pipe_flux, "vae"):
            pipe_flux.vae.enable_tiling()

    if sem_lora:
        print("Pulando o acoplamento do LoRA (linha de base sem treino).")
        ADAPTER = None
    else:
        print("Acoplando Adaptador LoRA (BokehNet)...")
        try:
            pipe_flux.load_lora_weights(lora_dir, weight_name=lora_weight_name, adapter_name="bokeh")
            pipe_flux.set_adapters(["bokeh"])
        except Exception as e:
            print(f"[Erro] Falha ao acoplar LoRA do BokehNet: {e}")
            return
        ADAPTER = "bokeh"

    # Função interna para renderização modularizada de um determinado K
    def renderizar_k(k_val, disp_minus_focus, clean_input_aif, w, h, no_tiled):
        defocus_abs = np.abs(k_val * disp_minus_focus)
        defocus_t = torch.from_numpy(defocus_abs).unsqueeze(0).float()
        cond_map = (defocus_t / MAX_COC).clamp(0, 1).repeat(3, 1, 1).unsqueeze(0)

        cond_img = Condition(clean_input_aif, ADAPTER)
        cond_dmf = Condition(cond_map, ADAPTER, [0, 0], 1.0, No_preprocess=True)

        seed_everything(42)
        gen = torch.Generator(device=device).manual_seed(1234)

        with torch.no_grad():
            bokeh_img = generate(
                pipe_flux, height=h, width=w,
                prompt="an excellent photo with a large aperture",
                num_inference_steps=steps,
                conditions=[cond_img, cond_dmf],
                guidance_scale=1.0, kv_cache=False, generator=gen,
                NO_TILED_DENOISE=no_tiled,
            ).images[0]
        return bokeh_img

    lista_linhas_saida = []
    # Nome do parquet do lote que esta PENDENTE de upload. Existe por causa da
    # retomada: se o upload falha e a retentativa gravasse sob o numero do lote
    # corrente, ficaria um buraco na numeracao, e uma rodada futura, que pula
    # pelo nome do arquivo, geraria de novo as imagens do lote que faltou.
    nome_parquet_pendente = None
    
    for idx, linha in enumerate(tqdm(dados_validacao, desc="Gerando Pipeline Bokeh + SSIM Search")):
        
        numero_do_lote_atual = (idx // tamanho_lote) + 1
        nome_parquet_lote = f"data/validation_part_{numero_do_lote_atual:03d}.parquet"
        
        if nome_parquet_lote in arquivos_no_hub:
            continue
            
        nome_arquivo = linha.get(COLUNA_NOME) or f"val_{idx:05d}.png"
        
        try:
            # 1. Carregamento resiliente da imagem de Foco (AIF / Content input)
            dado_img_focus = linha[COLUNA_IMAGEM_FOCUS]
            if isinstance(dado_img_focus, dict) and "bytes" in dado_img_focus:
                raw_img_focus = Image.open(io.BytesIO(dado_img_focus["bytes"]))
            elif isinstance(dado_img_focus, bytes):
                raw_img_focus = Image.open(io.BytesIO(dado_img_focus))
            else:
                raw_img_focus = dado_img_focus
            raw_img_focus = raw_img_focus.convert("RGB")
            clean_input_aif = resize_and_pad_image(raw_img_focus, long_side)
            w, h = clean_input_aif.size
            force_no_tile = min(w, h) < 512

            # 2. Carregamento da imagem Desfocada Real (Target / Gabarito)
            dado_img_blur = linha[COLUNA_IMAGEM_BLUR]
            if isinstance(dado_img_blur, dict) and "bytes" in dado_img_blur:
                raw_img_blur = Image.open(io.BytesIO(dado_img_blur["bytes"]))
            elif isinstance(dado_img_blur, bytes):
                raw_img_blur = Image.open(io.BytesIO(dado_img_blur))
            else:
                raw_img_blur = dado_img_blur
            raw_img_gabarito = resize_and_pad_image(raw_img_blur.convert("RGB"), long_side)

            # 3. Carregar mapa da Fase 1 e montar a disparidade
            caminho_mapa = os.path.join(DIR_MAPAS_TEMP, f"{nome_arquivo}_depth.npy")
            depth_arr = np.load(caminho_mapa).astype(np.float32)

            # CORRECAO 1 (2026-08-26) — REDIMENSIONAR O MAPA PARA (w, h).
            # A inferencia oficial faz isto (Inference_bokehNet.py:92 e :106):
            #     depth_arr = cv2.resize(depth_arr, (w, h), INTER_LINEAR)
            # O pipeline do time nao fazia. A fase 1 grava o mapa no tamanho
            # ORIGINAL da foto (resize_and_pad_image(img, 0)), enquanto o laco
            # roda com long_side=512. Resultado: a condicao de imagem entrava no
            # VAE a 512 px e a condicao de defocus a 2000 px, com contagens de
            # token e position ids incompativeis. Isso invalidava TODA corrida
            # com long_side>0 — e explica por que o peso OFICIAL pontuava pior
            # que a linha de identidade a 512 px e melhorava muito em long_side=0
            # (unico caso em que os dois tamanhos coincidiam por acidente).
            if depth_arr.shape[:2] != (h, w):
                depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_LINEAR)

            safe_depth = np.where(depth_arr > 0.0, depth_arr, np.finfo(np.float32).max)
            disp = 1.0 / safe_depth

            # CORRECAO 2 (2026-08-26) — PLANO DE FOCO PELA Eq. 4 DO PAPER.
            # Eq. 4: D_focus = mediana da profundidade na mascara do objeto
            # saliente (BiRefNet). O pipeline usava disp[h//2, w//2], o PIXEL
            # CENTRAL, que na inferencia oficial e so o fallback de quando nao
            # ha nem mascara nem ponto. Verificado em imagem: com o centro, o
            # plano de foco caia no fundo e o modelo borrava o objeto que o alvo
            # tinha em foco.
            tx, ty = w // 2, h // 2
            disp_focus = None
            origem_foco = "centro"
            if PLANO_FOCO == "mascara":
                mask_arr = None
                # (a) mascara vinda do proprio dataset, se existir
                if COLUNA_MASCARA in (dados_validacao.column_names or []):
                    try:
                        dm = linha[COLUNA_MASCARA]
                        if isinstance(dm, dict) and dm.get("bytes"):
                            mimg = Image.open(io.BytesIO(dm["bytes"]))
                        elif isinstance(dm, bytes):
                            mimg = Image.open(io.BytesIO(dm))
                        else:
                            mimg = dm
                        if mimg is not None:
                            mask_arr = np.array(mimg.convert("L").resize((w, h), Image.NEAREST))
                            origem_foco = "mascara_dataset"
                    except Exception:
                        mask_arr = None
                # (b) senao, a mascara do BiRefNet gerada na fase 1b
                if mask_arr is None:
                    caminho_mask = os.path.join(DIR_MAPAS_TEMP, f"{nome_arquivo}_mask.npy")
                    if os.path.exists(caminho_mask):
                        mp = np.load(caminho_mask).astype(np.float32)
                        if mp.shape[:2] != (h, w):
                            mp = cv2.resize(mp, (w, h), interpolation=cv2.INTER_LINEAR)
                        mask_arr = mp
                        origem_foco = "birefnet"
                if mask_arr is not None:
                    validos = disp[mask_arr > 127]
                    if validos.size > 0:
                        disp_focus = float(np.median(validos))
                    else:
                        origem_foco = "centro(mascara_vazia)"
            elif PLANO_FOCO == "alvo_nitido":
                # ORACULO: le o plano de foco na regiao que continua NITIDA no
                # ALVO real. Por definicao e onde a camera focou, e por ser lido
                # no proprio mapa do Depth Pro nao depende da escala metrica
                # dele — que medimos errada em 40% das cenas do RealBokeh.
                # Usa o alvo, entao e ORACULO, no mesmo estatuto que a busca
                # binaria por K do paper (que tambem otimiza contra o alvo).
                # Aplicado igual para todos os modelos.
                g = cv2.cvtColor(np.array(raw_img_gabarito), cv2.COLOR_RGB2GRAY).astype(np.float64)
                nitidez = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_64F)).astype(np.float32), (0, 0), 7)
                limiar = float(np.percentile(nitidez, 90))
                sel = nitidez >= limiar
                if sel.any():
                    disp_focus = float(np.median(disp[sel]))
                    origem_foco = "alvo_nitido"
            if disp_focus is None:
                disp_focus = float(disp[ty, tx])
            print(f"  [foco] {nome_arquivo}: origem={origem_foco} disp_focus={disp_focus:.5f}", flush=True)
            disp_minus_focus = disp - np.float32(disp_focus)

            # --- ETAPA A: Varredura Fixa para o cálculo futuro do LVCorr ---
            imagens_lvcorr = {}
            for k_fixo in [1.0 * k_escala, 5.0 * k_escala, 10.0 * k_escala, 15.0 * k_escala]:
                img_gerada_lv = renderizar_k(k_fixo, disp_minus_focus, clean_input_aif, w, h, force_no_tile)
                rotulo = int(round(k_fixo / k_escala)) if k_escala else int(k_fixo)
                imagens_lvcorr[f"image_k{rotulo:02d}"] = converter_pil_para_bytes(img_gerada_lv)

            # --- ETAPA B: Busca Binária Rigorosa (7 passos discretos para K=1 a 100) ---
            low, high = 1, 100
            best_k = 1
            best_ssim = -1.0
            best_img_pil = None

            while low <= high:
                mid = (low + high) // 2
                
                img_mid = renderizar_k(mid * k_escala, disp_minus_focus, clean_input_aif, w, h, force_no_tile)
                ssim_mid = calcular_ssim_pil(img_mid, raw_img_gabarito)
                
                if ssim_mid > best_ssim:
                    best_ssim, best_k, best_img_pil = ssim_mid, mid * k_escala, img_mid

                if low == high:
                    break

                # Avalia o gradiente gerando o ponto adjacente (mid + 1)
                img_next = renderizar_k((mid + 1) * k_escala, disp_minus_focus, clean_input_aif, w, h, force_no_tile)
                ssim_next = calcular_ssim_pil(img_next, raw_img_gabarito)
                
                if ssim_next > best_ssim:
                    best_ssim, best_k, best_img_pil = ssim_next, (mid + 1) * k_escala, img_next

                if ssim_mid < ssim_next:
                    low = mid + 1   # O pico está para a direita
                else:
                    high = mid - 1  # O pico está para a esquerda

            # Compilação dos dados binários estruturados
            bytes_img_best_k = converter_pil_para_bytes(best_img_pil)
            bytes_img_real_bokeh = converter_pil_para_bytes(raw_img_gabarito)

            lista_linhas_saida.append({
                "file_name_base": str(nome_arquivo),
                "image_real_bokeh": bytes_img_real_bokeh,
                "image_best_k": bytes_img_best_k,
                "best_k_value": float(best_k),
                "ssim_score": float(best_ssim),
                "image_k01": imagens_lvcorr["image_k01"],
                "image_k05": imagens_lvcorr["image_k05"],
                "image_k10": imagens_lvcorr["image_k10"],
                "image_k15": imagens_lvcorr["image_k15"],
            })

            # 4. Trigger de fechamento e Upload do Lote Parquet
            # `>=`, e nao `==`: se o upload de um lote falhar, o `except` la
            # embaixo engole a excecao e a lista NAO e zerada. Com igualdade
            # exata o gatilho nunca mais fecha (11 != 10), e a rodada segue
            # gerando por horas sem gravar nada, ate a ultima imagem. Foi o que
            # aconteceu com o `kfix` no LF-repro em 2026-09-04: o lote 10 falhou
            # no Hub e as 2 h seguintes de GPU nao viraram nada. Com `>=`, a
            # proxima imagem ja tenta subir o lote acumulado de novo.
            if len(lista_linhas_saida) >= tamanho_lote or (idx + 1) == total_imagens:
                if nome_parquet_pendente is None:
                    nome_parquet_pendente = nome_parquet_lote
                print(f"\n[Lote {numero_do_lote_atual}] Serializando e subindo Parquet estruturado para o Hub"
                      f" ({len(lista_linhas_saida)} linhas em {nome_parquet_pendente})...")
                
                df_saida = pd.DataFrame(lista_linhas_saida)
                
                schema = pa.schema([
                    pa.field("file_name_base", pa.string()),
                    pa.field("image_real_bokeh", pa.binary()),
                    pa.field("image_best_k", pa.binary()),
                    pa.field("best_k_value", pa.float32()),
                    pa.field("ssim_score", pa.float32()),
                    pa.field("image_k01", pa.binary()),
                    pa.field("image_k05", pa.binary()),
                    pa.field("image_k10", pa.binary()),
                    pa.field("image_k15", pa.binary()),
                ])
                
                # Metadados estritos para que o Hugging Face renderize os visualizadores de imagens nativamente
                metadados_hf = {
                    b"huggingface": b'{"info": {"features": {'
                                    b'"file_name_base": {"dtype": "string", "_type": "Value"},'
                                    b'"image_real_bokeh": {"_type": "Image"},'
                                    b'"image_best_k": {"_type": "Image"},'
                                    b'"best_k_value": {"dtype": "float32", "_type": "Value"},'
                                    b'"ssim_score": {"dtype": "float32", "_type": "Value"},'
                                    b'"image_k01": {"_type": "Image"},'
                                    b'"image_k05": {"_type": "Image"},'
                                    b'"image_k10": {"_type": "Image"},'
                                    b'"image_k15": {"_type": "Image"}'
                                    b'}}}'
                }
                
                tabela_arrow = pa.Table.from_pandas(df_saida, schema=schema.with_metadata(metadados_hf), preserve_index=False)
                pq.write_table(tabela_arrow, temp_parquet_local)
                
                api.upload_file(
                    path_or_fileobj=temp_parquet_local,
                    path_in_repo=nome_parquet_pendente,
                    repo_id=repo_id_saida,
                    repo_type="dataset",
                    token=token_hf
                )
                print(f"Lote {numero_do_lote_atual} integrado e salvo com sucesso.")
                
                lista_linhas_saida = []
                nome_parquet_pendente = None
                if os.path.exists(temp_parquet_local):
                    os.remove(temp_parquet_local)

        except Exception as e:
            print(f"\n[ERRO] Falha crítica no processamento do índice {idx} ({nome_arquivo}): {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nPipeline completo finalizado. Todos os lotes de validação estão no Hugging Face Hub.")


def main():

    avaliador  = CloudBokehEvaluator()

    load_dotenv()

    LF_BOKEH = {
        "repo_entrada": "akcit-pixel/DDPD" ,
        "repo_saida": "AKCITPixel3/bokeh-net-infer" ,
        "padrao_validacao" : "data/validation-*.parquet"
    }

   
    datasets = [LF_BOKEH]

    PESOS_LORA = "pesos_oficiais/bokehNet.safetensors"
    TAMANHO_DO_LOTE = 50 
    TOKEN_HF = os.getenv("HF_TOKEN")
    BOKEH_METRICS_REPO = os.getenv("BOKEHNET_METRICS_REPO")

    for dataset in datasets:
      
        print(f"PROCESSANDO PIPELINE DE EVAL BOKEH: {dataset['repo_entrada']}")
      
        
        rodar_inferencia_em_lotes_hf(
            repo_id_entrada=dataset['repo_entrada'],
            padrao_validacao=dataset['padrao_validacao'], 
            repo_id_saida=dataset['repo_saida'],
            tamanho_lote=TAMANHO_DO_LOTE,
            caminho_completo_lora=PESOS_LORA,
            token_hf=TOKEN_HF
        )

        avaliador.run_pipeline(
            hf_datasets=dataset['repo_saida'],
            output_repo_id=BOKEH_METRICS_REPO,
            model_name="Bokeh-net",
            hf_split="validation",
            hf_token=TOKEN_HF
        )

if __name__ == "__main__":
    main()