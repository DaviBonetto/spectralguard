---
description: Planejar e obter aprovação antes de implementar
---

# Workflow: Planejar Antes de Agir

Use este workflow **SEMPRE** antes de escrever código.

## Passos

### 1. Entender o Problema

**ANTES de planejar:**

- ✅ Leia completamente a solicitação do usuário
- ✅ Identifique requisitos ambíguos
- ✅ **PERGUNTE** se algo não estiver claro
- ❌ **NUNCA** assuma requisitos

### 2. Criar o Plano

**Descreva sua abordagem:**

```markdown
## Plano de Implementação

### Objetivo

[O que será feito]

### Abordagem

1. [Passo 1]
2. [Passo 2]
3. [Passo 3]

### Arquivos Afetados

- [arquivo1.py] - [o que será mudado]
- [arquivo2.js] - [o que será mudado]

### Riscos/Considerações

- [Possível problema 1]
- [Possível problema 2]

### Skills Usadas

- [@skill-name] - [para que]
```

### 3. Aguardar Aprovação

**PARE e aguarde:**

- ✅ Mostre o plano ao usuário
- ✅ Pergunte: "Posso prosseguir com esta abordagem?"
- ✅ Aguarde confirmação
- ❌ **NÃO comece a codar sem aprovação**

### 4. Implementar

**Só após aprovação:**

- ✅ Siga o plano aprovado
- ✅ Se precisar desviar, **PARE e pergunte**
- ✅ Mantenha o usuário informado

## Regras Especiais

### Se >3 Arquivos

**PARE IMEDIATAMENTE** e:

1. Quebre em tarefas menores
2. Priorize com o usuário
3. Implemente uma tarefa por vez

### Se Requisitos Ambíguos

**PERGUNTE:**

- "Você prefere A ou B?"
- "Qual comportamento esperado em caso X?"
- "Devo incluir Y também?"

### Se Descobrir Complexidade Inesperada

**PARE e informe:**

- "Descobri que X é mais complexo porque Y"
- "Sugiro replanejar para Z"
- "Posso continuar ou prefere outra abordagem?"

## Exemplo de Uso

### ❌ ERRADO

```
User: "Adicione autenticação"
Agent: [Começa a criar arquivos auth.js, middleware.js, etc]
```

### ✅ CORRETO

```
User: "Adicione autenticação"
Agent: "Vou planejar a implementação de autenticação.

## Plano de Implementação
...

Posso prosseguir?"
```

## Regra de Ouro

**Planejar é rápido. Refazer é caro.**

**SEMPRE planeje, SEMPRE aguarde aprovação, NUNCA assuma.**
