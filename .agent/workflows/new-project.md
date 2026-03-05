---
description: Inicializar novo projeto com Skills Master
---

# Workflow: Iniciar Novo Projeto

Este workflow configura um novo projeto com todas as skills e estrutura básica.

## Passos

### 1. Instalar Skills Master

```powershell
& "C:\Users\Davib\OneDrive\Área de Trabalho\Skills Master\install-skills.ps1"
```

Aguarde confirmação de que as skills foram copiadas.

### 2. Criar Estrutura Básica

Pergunte ao usuário:

- Qual o nome do projeto?
- Qual a linguagem/stack principal? (Python, Node.js, C++, etc.)
- É um projeto web, CLI, biblioteca, ou outro?

### 3. Criar .gitignore

Com base na stack escolhida, crie um `.gitignore` apropriado:

- Python: `__pycache__/`, `*.pyc`, `venv/`, `.env`
- Node.js: `node_modules/`, `.env`, `dist/`
- C++: `*.o`, `*.exe`, `build/`

### 4. Criar README.md

Crie um README básico com:

- Nome do projeto
- Descrição breve
- Como instalar/rodar
- Estrutura de diretórios

### 5. Criar task.md Inicial

Crie `.gemini/antigravity/brain/[conversation-id]/task.md` com:

- Objetivo principal do projeto
- Primeiras tarefas a fazer
- Checklist inicial

### 6. Configurar Ambiente (se aplicável)

Dependendo da stack:

- Python: Criar `venv`, instalar dependências
- Node.js: `npm init`, instalar dependências
- C++: Configurar CMake ou Makefile

### 7. Verificação Final

Confirme que:

- [ ] Skills instaladas em `.agent/skills/`
- [ ] `GEMINI.md` presente em `.agent/`
- [ ] `.gitignore` criado
- [ ] `README.md` criado
- [ ] `task.md` criado
- [ ] Ambiente configurado (se aplicável)

**Projeto pronto para começar!** 🚀
