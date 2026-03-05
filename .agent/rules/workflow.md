# Superpowers Workflow

> Workflow completo para desenvolvimento com skills.

---

## 1. BRAINSTORMING

```
@brainstorming
└ Refina ideias, explora alternativas
└ Salva: docs/plans/YYYY-MM-DD-<topic>-design.md
```

---

## 2. USING-GIT-WORKTREES

```
@using-git-worktrees
└ Cria workspace isolado em branch nova
└ Roda setup e verifica testes passando
```

---

## 3. WRITING-PLANS

```
@writing-plans
└ Quebra em tasks de 2-5 minutos
└ Cada task tem: arquivo, código, verificação
└ Salva: docs/plans/YYYY-MM-DD-<feature>.md
```

---

## 4. EXECUTING-PLANS / SUBAGENT-DRIVEN-DEVELOPMENT

```
@executing-plans / @subagent-driven-development
└ Executa tasks em batches
└ Despacha sub-agentes para cada task
└ Review em 2 estágios (spec + quality)
```

---

## 5. TEST-DRIVEN-DEVELOPMENT

```
@test-driven-development
└ RED: Escreve teste, vê falhar
└ GREEN: Código mínimo pra passar
└ REFACTOR: Melhora código
```

---

## 6. VERIFICATION-BEFORE-COMPLETION

```
@verification-before-completion
└ Verifica tudo antes de marcar como pronto
```

---

## Diagrama Visual

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERPOWERS WORKFLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. @brainstorming                                          │
│     └─► docs/plans/YYYY-MM-DD-<topic>-design.md             │
│                          │                                   │
│                          ▼                                   │
│  2. @using-git-worktrees                                    │
│     └─► Branch isolada + setup + testes                     │
│                          │                                   │
│                          ▼                                   │
│  3. @writing-plans                                          │
│     └─► docs/plans/YYYY-MM-DD-<feature>.md                  │
│                          │                                   │
│                          ▼                                   │
│  4. @executing-plans / @subagent-driven-development         │
│     └─► Batches + Sub-agentes + Review 2 estágios           │
│                          │                                   │
│                          ▼                                   │
│  5. @test-driven-development                                │
│     └─► RED → GREEN → REFACTOR                              │
│                          │                                   │
│                          ▼                                   │
│  6. @verification-before-completion                         │
│     └─► Verificar antes de "pronto"                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Regras Importantes

### ✓ Sempre Fazer

- Salvar designs em `docs/plans/`
- Verificar testes antes de continuar
- Tasks de 2-5 minutos máximo
- Review em 2 estágios

### ❌ Nunca Fazer

- Pular etapas do workflow
- Implementar sem plano
- Marcar pronto sem verificar
