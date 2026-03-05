---
description: Atualizar skills em todos os projetos
---

# Workflow: Atualizar Skills Master

Use este workflow quando atualizar o Skills Master e quiser propagar as mudanças para todos os projetos.

## Passos

### 1. Identificar Projetos com Skills

Procure por diretórios que têm `.agent/skills/`:

```powershell
Get-ChildItem -Path "C:\Users\Davib\OneDrive\Área de Trabalho" -Recurse -Directory -Filter ".agent" | Where-Object { Test-Path (Join-Path $_.FullName "skills") }
```

Ou manualmente liste os projetos que você sabe que usam Skills Master.

### 2. Confirmar Projetos a Atualizar

Mostre a lista de projetos encontrados e pergunte:

- Quais projetos devem ser atualizados?
- Há algum projeto que deve ser pulado?

### 3. Fazer Backup (Opcional mas Recomendado)

Para cada projeto, sugira criar um commit antes de atualizar:

```bash
cd [projeto]
git add .
git commit -m "Backup antes de atualizar Skills Master"
```

### 4. Executar Atualização

Para cada projeto confirmado, rode:

```powershell
cd "[caminho-do-projeto]"
& "C:\Users\Davib\OneDrive\Área de Trabalho\Skills Master\install-skills.ps1"
```

### 5. Verificar Atualização

Para cada projeto atualizado, confirme:

- [ ] Skills copiadas com sucesso
- [ ] `GEMINI.md` atualizado
- [ ] Nenhum erro durante a cópia

### 6. Testar Projeto (se crítico)

Para projetos importantes, sugira:

- Rodar testes: `npm test` / `pytest` / etc.
- Verificar se nada quebrou
- Fazer um commit das atualizações

### 7. Resumo Final

Mostre um resumo:

```
Projetos atualizados: X
Projetos pulados: Y
Erros encontrados: Z

Lista de projetos atualizados:
- [Projeto 1]
- [Projeto 2]
- ...
```

### 8. Próximos Passos

Sugira ao usuário:

- Testar os projetos críticos manualmente
- Fazer commit das mudanças em cada projeto
- Verificar se alguma skill nova precisa de configuração adicional

**Atualização concluída!** 🔄
