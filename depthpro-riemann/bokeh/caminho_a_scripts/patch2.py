import sys
from pathlib import Path
p = Path("/raid/user_juliadollis/julia_docker/bokehnet-regen/scripts/run_route_c.py")
t = p.read_text(encoding="utf-8")

alvo = '''    print(enumeration_summary(pares, log))

    pares = _aplica_teto(pares, args)
    pares = _aplica_fatia(pares, args)

    if args.pilot:
        pares = sample_pairs_for_pilot(pares, limit=args.pilot, seed=args.seed)
        print(f"[fonte] piloto: {len(pares)} pares de "
              f"{len({p.scene_id for p in pares})} cenas (seed {args.seed}).")

    pares = order_pairs_for_sequential_read(pares, index)

    saida = Path(args.output_dir)
    loader = RealDOFImageLoader('''

novo = '''    print(enumeration_summary(pares, log))

    # `_aplica_teto` NÃO é chamado aqui, e isso é decisão e não esquecimento. Ele
    # ordena por `par.level` para espaçar as aberturas uniformemente, e o RealDOF não
    # tem nível: é um par por cena, 50 de 50. Chamá-lo levantaria `AttributeError` com
    # o default `--max-levels-per-scene 4`, o que faria a fonte inteira depender de
    # quem lembrar de passar `0` na linha de comando.
    if args.max_levels_per_scene:
        print("[fonte] --max-levels-per-scene ignorado: o RealDOF tem um par por cena "
              "(50 pares, 50 cenas), então não há série de aberturas a desbastar.")
    pares = _aplica_fatia(pares, args)

    if args.pilot:
        pares = sample_pairs_for_pilot(pares, limit=args.pilot, seed=args.seed)
        print(f"[fonte] piloto: {len(pares)} pares de "
              f"{len({p.scene_id for p in pares})} cenas (seed {args.seed}).")

    pares = order_pairs_for_sequential_read(pares, index)

    saida = Path(args.output_dir)
    loader = RealDOFImageLoader('''

n = t.count(alvo)
if n == 1:
    p.write_text(t.replace(alvo, novo, 1), encoding="utf-8")
    print("ok: _aplica_teto removido do caminho do realdof")
elif novo in t:
    print("ja aplicado")
else:
    print(f"ERRO: alvo aparece {n}x"); sys.exit(1)
