"""Sobe um modelo (.safetensors) pro HuggingFace Hub.

Usos:
  # 1) TESTE de envio (cria o repo e sobe um arquivinho de sanidade):
  python scripts/hf_upload.py --test --repo-id genrefocus-deblurnet-ddpd

  # 2) Upload do modelo apontando o arquivo direto:
  python scripts/hf_upload.py \
      --file outputs/deblurnet_ddpd/deblur/deblur.safetensors \
      --repo-id genrefocus-deblurnet-ddpd

  # 3) Upload resolvendo o arquivo a partir do config (output_dir/deblur/deblur.safetensors):
  python scripts/hf_upload.py --config configs/train_ddpd.yaml \
      --repo-id genrefocus-deblurnet-ddpd

Requer HF_TOKEN no ambiente. Se --repo-id não tiver "/", o namespace do seu
usuário (via whoami) é prefixado automaticamente.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _token() -> str:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not tok:
        print("ERRO: defina HF_TOKEN no ambiente.", file=sys.stderr)
        sys.exit(1)
    return tok


def _resolve_repo_id(repo_id: str, token: str) -> str:
    if "/" in repo_id:
        return repo_id
    from huggingface_hub import whoami

    user = whoami(token=token)["name"]
    return f"{user}/{repo_id}"


def _resolve_file(args) -> Path:
    if args.file:
        return Path(args.file)
    if args.config:
        # importa só aqui pra o --test não depender do pacote de treino
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from genfocus_train.config import load_config

        cfg = load_config(args.config)
        return Path(cfg.runtime.output_dir) / "deblur" / "deblur.safetensors"
    raise SystemExit("Informe --file, --config ou --test.")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload de modelo pro HuggingFace Hub")
    p.add_argument("--repo-id", required=True, help="ex.: genrefocus-deblurnet-ddpd (sem '/' = usa seu namespace)")
    p.add_argument("--file", default=None, help="caminho do .safetensors")
    p.add_argument("--config", default=None, help="resolve o arquivo a partir do output_dir do config")
    p.add_argument("--path-in-repo", default=None, help="nome do arquivo no repo (default: basename)")
    p.add_argument("--private", action="store_true", help="cria o repo como privado")
    p.add_argument("--test", action="store_true", help="só testa: cria o repo e sobe um arquivo de sanidade")
    p.add_argument("--also", nargs="*", default=[], help="arquivos extras pra subir junto (ex.: effective_config.yaml)")
    args = p.parse_args()

    token = _token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = _resolve_repo_id(args.repo_id, token)

    print(f"[hf] repo: {repo_id} (private={args.private})")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=args.private)

    if args.test:
        # sobe um arquivinho pra confirmar token + permissão de escrita
        probe = Path("/tmp/_hf_upload_probe.txt")
        probe.write_text("genrefocus upload probe ok\n", encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(probe),
            path_in_repo="UPLOAD_TEST.txt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[hf] TESTE OK -> https://huggingface.co/{repo_id}/blob/main/UPLOAD_TEST.txt")
        print("[hf] (pode apagar esse UPLOAD_TEST.txt depois; o repo já está pronto)")
        return

    f = _resolve_file(args)
    if not f.is_file():
        raise SystemExit(f"ERRO: arquivo não encontrado: {f}")

    name = args.path_in_repo or f.name
    print(f"[hf] subindo {f}  ({f.stat().st_size / 1e6:.1f} MB)  ->  {name}")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="model",
    )

    for extra in args.also:
        ep = Path(extra)
        if ep.is_file():
            print(f"[hf] subindo extra {ep.name}")
            api.upload_file(
                path_or_fileobj=str(ep),
                path_in_repo=ep.name,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print(f"[hf] WARN: extra não encontrado, pulando: {ep}")

    print(f"[hf] PRONTO -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
