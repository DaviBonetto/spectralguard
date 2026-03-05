# Skills Master

> **Base do projeto para agentes AI.**

## Project Overview

Coleção de **627 skills** organizadas para construir qualquer coisa.

**YOU MUST** explorar `.agent/skills/_categories/INDEX.md` antes de cada tarefa.
**ATENÇÃO** Não explore somente o INDEX.md, explore todas as skills disponíveis.

**Pirâmide de contexto:**

- MCP → Ferramentas externas
- Skills → Workflows ("como fazer")
- Rules → Conhecimento de domínio
- CLAUDE.md → Base ("o que saber")

---

## Key Commands

| Ação               | Comando                      |
| ------------------ | ---------------------------- |
| Encontrar skill    | `_categories/INDEX.md`       |
| Encadear skills    | `@skill-orchestrator`        |
| Paralelo com Codex | `@codex-orchestrator`        |
| Criar skill        | `@antigravity-skill-creator` |

---

## Important Caveats

### ✓ Include (Sempre Fazer)

- **YOU MUST** descrever abordagem e aguardar aprovação antes de codar
- **YOU MUST** perguntar quando requisitos ambíguos
- **YOU MUST** usar `@verification-before-completion` antes de "pronto"
- Se >3 arquivos, **PARE e quebre em tarefas menores**
- Após codar, liste o que pode quebrar e sugira testes
- Para bugs, escreva teste que reproduz → depois corrija

### ❌ Exclude (Nunca Fazer)

- Implementar sem planejar
- Afirmar "concluído" sem verificar
- Ignorar skills existentes
- Mudar >3 arquivos sem quebrar tarefa

---

## Codex (Aliado para Codar)

Quando precisar de execução paralela ou tasks longas:

```bash
codex-agent start "tarefa" --map -s workspace-write
codex-agent jobs --json
codex-agent send <id> "redirecionamento"
```

---

## Self-Update

**YOU MUST** adicionar regra abaixo quando corrigido:

### Lições Aprendidas

<!-- Adicione aqui quando corrigido -->
