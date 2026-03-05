---
description: Verificar skills disponíveis antes de implementar
---

# Workflow: Usar Skills Primeiro

Use este workflow **ANTES** de implementar qualquer funcionalidade.

## Passos

### 1. Explorar Skills PRIMEIRO

**ANTES de qualquer implementação:**

```markdown
1. Ler `.agent/skills/_categories/INDEX.md`
2. Identificar categorias relevantes
3. Explorar skills específicas da categoria
4. Verificar se alguma skill resolve o problema
```

### 2. Verificar Categorias Principais

Sempre verifique estas categorias primeiro:

- **Backend** - APIs, servidores, databases
- **Frontend** - UI, React, Next.js
- **DevOps** - Deploy, CI/CD, Docker
- **Testing** - Testes, QA, verificação
- **AI Engineering** - LLMs, agents, RAG
- **Code Review** - Revisão, qualidade
- **Debugging** - Debug, troubleshooting

### 3. Usar a Skill Correta

Se encontrar uma skill relevante:

- ✅ **USE A SKILL** seguindo suas instruções
- ✅ Mencione qual skill está usando
- ✅ Siga o workflow da skill

Se NÃO encontrar skill relevante:

- ⚠️ Mencione que verificou as skills
- ⚠️ Liste quais categorias explorou
- ⚠️ Só então implemente manualmente

### 4. Exploração Completa

**NÃO explore apenas o INDEX.md!**

Você DEVE:

- ✅ Ler o INDEX.md para visão geral
- ✅ Abrir as categorias relevantes
- ✅ Ler os SKILL.md das skills promissoras
- ✅ Verificar exemplos e recursos das skills

### 5. Lembrete de Instalação

**Se o projeto não tem skills instaladas:**

```powershell
& "C:\Users\Davib\OneDrive\Área de Trabalho\Skills Master\install-skills.ps1"
```

## Exemplo de Uso

### ❌ ERRADO

```
User: "Preciso criar uma API REST"
Agent: "Vou criar com Express..."
```

### ✅ CORRETO

```
User: "Preciso criar uma API REST"
Agent: "Deixe-me verificar as skills disponíveis..."
Agent: [Explora backend-patterns, api-design-principles]
Agent: "Encontrei a skill 'backend-patterns'. Vou usar ela..."
```

## Regra de Ouro

**Skills PRIMEIRO, implementação manual DEPOIS.**

**Sempre, sempre, sempre explore as skills!**
