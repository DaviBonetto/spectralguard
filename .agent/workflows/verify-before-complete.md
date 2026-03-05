---
description: Verificar trabalho antes de marcar como completo
---

# Workflow: Verificação Antes de Completar

Use este workflow ANTES de dizer "pronto", "finalizado" ou "concluído".

## Passos Obrigatórios

### 1. Verificar Arquivos Salvos

Confirme que todos os arquivos modificados foram salvos:

- Verifique arquivos abertos no editor
- Procure por indicadores de "não salvo" (\*)

### 2. Rodar Testes (se existirem)

Execute os testes do projeto:

**Python:**

```bash
pytest
# ou
python -m unittest
```

**Node.js:**

```bash
npm test
# ou
npm run test
```

**C++:**

```bash
make test
# ou
ctest
```

Se os testes falharem, **NÃO** marque como completo.

### 3. Verificar Lint/Formatação

Execute ferramentas de qualidade de código:

**Python:**

```bash
ruff check .
# ou
pylint .
```

**Node.js:**

```bash
npm run lint
# ou
eslint .
```

**C++:**

```bash
clang-format --dry-run -Werror *.cpp
```

### 4. Revisar Mudanças

Liste os arquivos modificados:

```bash
git status
# ou
git diff --name-only
```

Confirme que:

- [ ] Todas as mudanças são intencionais
- [ ] Não há arquivos temporários commitados
- [ ] Não há código comentado desnecessário

### 5. Listar Possíveis Problemas

**Pense criticamente:**

- O que pode quebrar com essas mudanças?
- Há edge cases não testados?
- Há dependências que podem causar problemas?
- A documentação está atualizada?

**Liste explicitamente** os riscos identificados.

### 6. Sugerir Testes Adicionais

Com base nos riscos identificados, sugira:

- Testes unitários que deveriam ser adicionados
- Cenários de teste manual
- Validações que o usuário deveria fazer

### 7. Confirmação Final

Só marque como "completo" se:

- [ ] Todos os arquivos salvos
- [ ] Testes passando (ou não existem testes)
- [ ] Lint sem erros críticos
- [ ] Mudanças revisadas
- [ ] Riscos identificados e comunicados
- [ ] Sugestões de teste fornecidas

**Se algum item falhar, corrija antes de finalizar!**
