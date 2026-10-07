from pathlib import Path
p = Path("/raid/user_juliadollis/julia_docker/bokehnet-regen/scripts/run_route_c.py")
t = p.read_text(encoding="utf-8")
novo = '''    1. **Sem teto de níveis.** Um par por cena, então `--max-levels-per-scene` não teria
       o que cortar. `_aplica_teto` é chamado do mesmo jeito e vira no-op — chamá-lo
       mantém o log do run com a mesma forma nas três fontes.'''
alvo = '''    1. **Sem teto de níveis.** Um par por cena, então `--max-levels-per-scene` não teria
       o que cortar — e `_aplica_teto` não é nem chamado, porque ele ordena por
       `par.level` e o `RealDOFPair` não tem esse campo. Ver o comentário no corpo.'''
assert t.count(alvo) == 1
p.write_text(t.replace(alvo, novo, 1), encoding="utf-8")
print("revertido: as 8 condicoes compartilham um pipeline_source_sha256")
