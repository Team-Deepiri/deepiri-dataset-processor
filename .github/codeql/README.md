# CodeQL Setup

This repository uses GitHub CodeQL for Python security analysis.

## Workflow Location

- `.github/workflows/codeql.yml`

## Config Location

- `.github/codeql/codeql-config.yml`

## Workflow

```yaml
name: CodeQL

on:
  pull_request:
    branches: [main, dev]
  push:
    branches: [main, dev]

permissions:
  actions: read
  contents: read
  security-events: write

jobs:
  analyze:
    name: Analyze (python)
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python
          config-file: ./.github/codeql/codeql-config.yml

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
```

## CodeQL Config

```yaml
# Exclude generated/build/runtime artifact paths.
paths-ignore:
  - "**/node_modules/**"
  - "**/dist/**"
  - "**/build/**"
  - "**/coverage/**"
  - "**/logs/**"
  - "**/*.min.js"
  - "**/.venv/**"
  - "**/__pycache__/**"
  - "**/.pytest_cache/**"
  - "**/.mypy_cache/**"
```

## Notes

- Language is `python` because this is a Python project.
- The workflow references the config file from `.github/codeql/codeql-config.yml`.