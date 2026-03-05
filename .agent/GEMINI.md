# Skills Master

> **Base do projeto para agentes AI.**

## Project Overview

Coleção de **627 skills** organizadas para construir qualquer coisa.

**YOU MUST** explorar `.agent/skills/_categories/INDEX.md` antes de cada tarefa.
**ATENÇÃO** Não explore somente o INDEX.md, explore todas as skills disponíveis.

**💡 LEMBRE:** Se o projeto não tem skills:

```bash
git clone https://github.com/iaversao7-sketch/skills-master.git
```

**Pirâmide de contexto:**

- MCP → Ferramentas externas
- Skills → Workflows ("como fazer")
- Rules → Conhecimento de domínio
- GEMINI.md → Base ("o que saber")

---

## Key Commands

| Ação               | Comando                      |
| ------------------ | ---------------------------- |
| Encontrar skill    | `_categories/INDEX.md`       |
| Encadear skills    | `@skill-orchestrator`        |
| Paralelo com Codex | `@codex-orchestrator`        |
| Criar skill        | `@antigravity-skill-creator` |
| Instalar skills    | `git clone ...`              |

---

## Important Caveats

### ✓ Include (Sempre Fazer)

- **YOU MUST** usar `/use-skills-first` antes de implementar
- **YOU MUST** usar `/plan-before-action` antes de codar
- **YOU MUST** descrever abordagem e aguardar aprovação
- **YOU MUST** perguntar quando requisitos ambíguos
- **YOU MUST** usar `/verify-before-complete` antes de "pronto"
- Se >3 arquivos, **PARE e quebre em tarefas menores**
- Após codar, liste o que pode quebrar e sugira testes
- Para bugs, escreva teste que reproduz → depois corrija

### ❌ Exclude (Nunca Fazer)

- Implementar sem usar `/use-skills-first`
- Codar sem usar `/plan-before-action`
- Afirmar "concluído" sem `/verify-before-complete`
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

- Colocar lições ainda
<!-- Adicione mais lições aqui conforme aprende -->
