from pathlib import Path
p = Path("/raid/user_juliadollis/julia_docker/bokehnet-regen/scripts/run_route_c.py")
t = p.read_text(encoding="utf-8")
alvo = '''    1. **Sem teto de níveis.** Um par por cena, então `--max-levels-per-scene` não teria
       o que cortar. `_aplica_teto` é chamado do mesmo jeito e vira no-op — chamá-lo
       mantém o log do run com a mesma forma nas três fontes.'''
novo = '''    1. **Sem teto de níveis.** Um par por cena, então `--max-levels-per-scene` não teria
       o que cortar — e `_aplica_teto` não é nem chamado, porque ele ordena por
       `par.level` e o `RealDOFPair` não tem esse campo. Ver o comentário no corpo.'''
assert t.count(alvo) == 1, t.count(alvo)
p.write_text(t.replace(alvo, novo, 1), encoding="utf-8")
print("docstring corrigido")
