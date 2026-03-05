---
trigger: always_on
description: Documentar aprendizados e atualizar GEMINI.md automaticamente
---

# Rule: Lições Aprendidas

## Quando Aplicar

**SEMPRE** ao finalizar uma sessão ou tarefa complexa, especialmente se:

- Encontrou um erro/bug
- Descobriu uma solução não óbvia
- Aprendeu algo sobre o projeto/stack
- Corrigiu um problema causado por falta de conhecimento

## O Que Fazer

### 1. Identificar o Aprendizado

Pergunte a si mesmo:

- O que deu errado?
- Por que deu errado?
- Como foi resolvido?
- Como evitar no futuro?

### 2. Documentar no GEMINI.md

**VOCÊ DEVE** adicionar na seção "Lições Aprendidas" do `GEMINI.md`:

```markdown
### [Título do Aprendizado] - [Data]

**Problema:**

- Descrição do que aconteceu

**Causa:**

- Por que aconteceu

**Solução:**

- Como foi resolvido

**Prevenção:**

- Como evitar no futuro
```

### 3. Informar o Usuário

Após adicionar ao `GEMINI.md`, informe:

```
✅ Lição aprendida documentada em GEMINI.md:
- [Resumo breve do aprendizado]
```

## Exemplos de Lições Aprendidas

### Encoding de Caminhos no Windows - 2026-02-07

**Problema:**

- Caminhos com caracteres especiais (Á, é) causavam erros em PowerShell

**Causa:**

- Encoding UTF-8 não era preservado em strings hardcoded

**Solução:**

- Usar `$MyInvocation.MyCommand.Path` para detectar caminho automaticamente

**Prevenção:**

- Nunca usar caminhos hardcoded
- Sempre usar variáveis de ambiente ou detecção automática

---

### Symlinks vs Copy no Windows - 2026-02-07

**Problema:**

- Symlinks não eram criados mesmo com comando correto

**Causa:**

- Symlinks no Windows precisam de permissões de administrador

**Solução:**

- Usar `Copy-Item` em vez de `New-Item -ItemType SymbolicLink`

**Prevenção:**

- Para máxima compatibilidade, preferir cópia de arquivos
- Usar symlinks só quando necessário e documentar requisito de admin

---

## Regra de Ouro

**Toda vez que resolver um problema não trivial, documente!**

Isso economiza tempo no futuro e ajuda outros desenvolvedores.
