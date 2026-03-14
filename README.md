# Проверим, что все файлы на месте
ls -la

# Если нужно, создадим .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
venv/
env/
.env
.DS_Store
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.vscode/
.idea/
*.log
EOF

# Инициализируем git
git init
git add .
git status  # Проверьте, что все нужные файлы добавлены
git commit -m "Initial commit: URL Shortener Service"

# Создайте репозиторий на GitHub и подключите:
git remote add origin https://github.com/ВАШ_ЛОГИН/project-3.git
git branch -M main
git push -u origin main
