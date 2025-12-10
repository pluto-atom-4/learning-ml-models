# GitHub Copilot Configuration for learning-ml-models

**Date:** December 9, 2025 (Last Updated)  
**Project:** learning-ml-models  
**Status:** Markdown Format – Enhanced & Optimized  
**Environment:** Git Bash on Windows

---

## 🎯 Quick Instructions for Copilot

> ### ⚡ PRIMARY INSTRUCTION
> 
> **ALWAYS wrap your entire response in a fenced code block labeled `markdown`.** This prevents the chat pane from rendering, keeping the content **raw and unrendered** so I can copy-paste directly into Jupyter notebooks.
>
> #### Format Example:
> ````markdown
> # Your markdown content here
> 
> Some text with **bold** and *italic*.
> 
> ```python
> # Nested Python code block
> import pandas as pd
> df = pd.read_csv('data.csv')
> ```
> ````
>
> ✨ **Result:** I copy the entire block (including outer fence) → paste into Jupyter **Markdown** cell → rendered immediately. No chat pane rendering noise!

### Default Output Behavior (2 Documents Maximum)

> ### 📋 RULE: Generate **EXACTLY 2 documents** by default
>
> **Default Documents:**
> 1. **Plan Document** – Structured breakdown with clear steps/methodology
> 2. **Summary Document** – Key findings, results, conclusions, and actionable takeaways
>
> **Generate Additional Documents ONLY IF:**
> - You explicitly request them (e.g., "Also create a quick reference guide")
> - The task inherently requires more docs to be complete (rare cases)
> - Never assume—always ask implicitly through your request
>
> **Examples of valid additional requests:**
> - "Also create a quick reference guide"
> - "Generate an examples document too"
> - "Include a troubleshooting guide as well"
>
> **When in doubt:** Stick to Plan + Summary only. Conciseness is valued.

---

## Overview

This document outlines the recommended configuration and instructions for using GitHub Copilot with the `learning-ml-models` project. This configuration is designed for **Git Bash on Windows** and includes setup instructions, virtual environment management, project installation, and development workflows.

> **Note:** This file is a project-local helper for humans and automation. GitHub Copilot editor extensions may or may not automatically read/obey it.

---

## Python Virtual Environment Setup

### Virtual Environment Name
- **Directory:** `.venv`

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Virtual Environment

> **Important:** On Windows with Git Bash, the venv activation script lives under `.venv/Scripts/activate`. Use forward slashes or Unix-style paths inside Git Bash.

```bash
source .venv/Scripts/activate
```

---

## Project Installation

### Using pip (Recommended)

#### Installation Method
The project uses `pip` with `pyproject.toml` for dependency management (PEP 517/518).

#### Install Project in Editable Mode

```bash
pip install -e .
```

#### Install Project with Development Dependencies

```bash
pip install -e ".[dev]"
```

#### Alternative: PEP 517 Build-Isolated Install

```bash
pip install .
```

> **Note:** Adjust the `[dev]` extras name to match the extras defined in your project's `pyproject.toml`.

### Using Poetry (Alternative)

If you prefer Poetry for dependency management, use these commands instead:

#### Install Dependencies

```bash
poetry install
```

#### Enter Poetry Shell

```bash
poetry shell
```

---

## Jupyter Lab Configuration

### Run Jupyter Lab

```bash
jupyter lab --port 8888 --no-browser --NotebookApp.token=''
```

### Configuration Details

| Setting | Value |
|---------|-------|
| **Port** | 8888 |
| **Browser Auto-Open** | Disabled (`--no-browser`) |
| **Token Authentication** | Disabled (empty token) |

> **Customization:** Adjust `--NotebookApp.token` or other flags based on your authentication needs.

---

## Copilot Output Configuration

### Output Directory

All Copilot-generated markdown files should be saved to:

```
generated/docs-copilot/
```

### Document Naming Convention

Use the following filename template for all generated documents:

```
{{YYYYMMDD}}-{{document-type}}-{{slug}}.md
```

**Template Variables:**
- `{{YYYYMMDD}}` – Current date (e.g., `20251209`)
- `{{document-type}}` – One of: `plan`, `summary`, `guide`, `examples`, `troubleshooting`, `reference`
- `{{slug}}` – URL-safe descriptive slug (e.g., `data-analysis`, `neural-networks`)

### Default Documents (Plan & Summary Only)

By default, generate **only 2 documents**:

1. **Plan Document:** `{{YYYYMMDD}}-plan-{{slug}}.md`
   - **Purpose:** Multi-step breakdown of the task/question
   - **Content:** Clear methodology, step-by-step approach, key considerations
   - **Example:** `20251209-plan-data-analysis.md`

2. **Summary Document:** `{{YYYYMMDD}}-summary-{{slug}}.md`
   - **Purpose:** Key findings, results, conclusions, and actionable takeaways
   - **Content:** Main results, insights, recommendations, next steps
   - **Example:** `20251209-summary-data-analysis.md`

### Generate Additional Documents Only When Requested

If you explicitly request additional documents, generate them with appropriate naming:

- `{{YYYYMMDD}}-guide-{{slug}}.md` – Step-by-step guides or tutorials
- `{{YYYYMMDD}}-examples-{{slug}}.md` – Code examples and sample implementations
- `{{YYYYMMDD}}-reference-{{slug}}.md` – Quick reference cards or checklists
- `{{YYYYMMDD}}-troubleshooting-{{slug}}.md` – Common issues and solutions

> **Rule:** Unless you explicitly say "Also create..." or "Generate...", stick to Plan + Summary only. This keeps documentation concise and focused.

---

## Setup Hooks

### Pre-Execution Hook

Ensures the output directory exists before running tasks:

```bash
mkdir -p generated/docs-copilot
```

### Post-Execution Hook

Moves temporary output files to the target directory:

```bash
if [ -f ./copilot_output.md ]; then
  ts=$(date +%Y%m%d)
  slug="copilot-output"
  mv ./copilot_output.md generated/docs-copilot/${ts}-${slug}.md
fi
```

---

## Quick Start Examples

### Example 1: Setup with pip (Recommended)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
pip install -e ".[dev]"
jupyter lab --port 8888 --no-browser
```

### Example 2: Setup with Poetry

```bash
poetry install
poetry shell
jupyter lab --port 8888 --no-browser
```

---

## Important Notes & Assumptions

- ⚠️ This configuration targets **Git Bash only** on Windows. PowerShell and CMD entries were intentionally removed.
- 🔧 On Windows with Git Bash, the venv activation script is typically available at `.venv/Scripts/activate` (use `source .venv/Scripts/activate`).
- 📦 If you prefer a Unix-style `.venv/bin` layout, create the venv inside WSL or adjust accordingly.
- 🏗️ The `pyproject` section assumes your project lists dependencies in `pyproject.toml` and declares a build backend.
- 🔄 If using Poetry, change `pyproject.using` to `poetry` in the original configuration.
- 🎛️ Adjust `pip install` extras (e.g., `[dev]`) to match the extras defined in your project.

---

## Copilot Response Format Instructions

> 🎯 **PRIMARY INSTRUCTION:** Wrap ALL responses in a fenced code block labeled `markdown`. This prevents chat rendering and provides raw, copy-paste-ready content for Jupyter notebooks.

### Rule 1: Always Wrap in Outer Markdown Code Block

**ALWAYS** wrap your entire response in triple backticks with `markdown` label:

````markdown
```markdown
# Your complete response goes here
All content, including nested code blocks, goes inside this outer fence.

See rules below for how to format markdown, Python, and bash content.
```
````

**Why?** This keeps the chat pane from rendering your response, allowing direct copy-paste into Jupyter **Markdown** cells without reformatting.

### Rule 2: Nested Content Formatting

Inside the outer markdown block, use appropriate nested code fences for different content types:

#### For Markdown Content (Default)

**✅ DO:** Provide raw markdown inside the outer fence:

````markdown
```markdown
# My Heading

This is regular markdown content with **bold** and *italic* text.

- List item 1
- List item 2

| Column A | Column B |
|----------|----------|
| Value 1  | Value 2  |
```
````

**❌ DON'T:** Don't use extra nested markdown fences; the outer fence is sufficient.

#### For Python Code Blocks

When providing Python code for Jupyter, use nested `python` code block:

````markdown
```markdown
Here is the Python code:

```python
import pandas as pd
import numpy as np

def calculate_mean(data):
    return np.mean(data)

df = pd.read_csv('data.csv')
print(df.head())
```

Run this code in a Jupyter **Code** cell.
```
````

#### For Bash/Shell Commands

When providing shell commands, use nested `bash` code block:

````markdown
```markdown
Run these commands in your Git Bash terminal:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
jupyter lab --port 8888 --no-browser
```
```
````

#### For Configuration Files (YAML, JSON, TOML)

Use appropriate nested language block:

````markdown
```markdown
Here is the YAML configuration:

```yaml
dependencies:
  - numpy
  - pandas
  - scikit-learn
```
```
````

### Rule 3: Copy-Paste Workflow for Jupyter

#### For Markdown Cells:
1. Copilot provides response wrapped in outer `markdown` code block
2. Create a new **Markdown** cell in Jupyter
3. Copy the entire outer block (including the outer fence with backticks)
4. Paste into the Markdown cell
5. Run the cell to render

#### For Code Cells:
1. Copilot provides response with nested Python code block
2. Find the inner ```python...``` block
3. Create a new **Code** cell in Jupyter
4. Copy-paste just the Python code (with or without inner backticks)
5. Run the cell to execute

### Rule 4: Response Type Summary Table

| Content Type | Outer Wrapper | Inner Format | Jupyter Destination | Paste Method |
|------------|---------------|--------------|---------------------|--------------|
| **Markdown text** | ✅ `markdown` fence | Raw text | **Markdown** cell | Entire outer block |
| **Python code** | ✅ `markdown` fence | ```python block | **Code** cell | Inner code only |
| **Shell commands** | ✅ `markdown` fence | ```bash block | **Code** cell | Inner commands |
| **Configuration** | ✅ `markdown` fence | ```yaml/json block | **Markdown** or **Code** | As-is |

### Rule 5: Special Cases

#### Multiple Code Blocks in One Response
If providing multiple code blocks (e.g., Step 1, Step 2, Step 3), nest them all inside the same outer markdown fence:

````markdown
```markdown
# Implementation Steps

**Step 1: Setup**

```bash

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```
```
````

### Copy-Paste Workflow for Jupyter

#### For Markdown Cells:
1. ✅ Copilot provides response wrapped in outer `markdown` code block
2. ✅ Create a new **Markdown** cell in Jupyter
3. ✅ Copy the entire outer block (including the outer fence with backticks)
4. ✅ Paste into the Markdown cell
5. ✅ Run the cell to render the markdown

#### For Code Cells:
1. ✅ Copilot provides response with nested Python code block inside markdown wrapper
2. ✅ Find the inner ```python...``` code block
3. ✅ Create a new **Code** cell in Jupyter
4. ✅ Copy-paste the Python code block (with or without backticks; Jupyter Code cells accept both)
5. ✅ Run the cell to execute the code

### Summary

| Content Type | Response Format | Copy-Paste Destination |
|-------------|--------|----------------------|
| Markdown text | Wrapped in outer ```markdown fence | Jupyter **Markdown** cell (paste entire block) |
| Python code | Nested ```python block inside markdown wrapper | Jupyter **Code** cell (extract inner block) |
| Shell commands | Nested ```bash block inside markdown wrapper | Jupyter **Code** cell or Terminal |
| Configuration | Nested ```yaml block inside markdown wrapper | Jupyter **Markdown** or **Code** cell |

---

**Key Principle:** 🎯 ALL responses are wrapped in an outer markdown code block to prevent chat rendering. This gives you raw, unrendered content that's ready to copy-paste into Jupyter.

---

## Directory Structure Reference

```
learning-ml-models/
├── .github/
│   └── copilot-instructions.md      # This file (Markdown format)
├── generated/
│   ├── docs-copilot/                # Output directory for generated docs
│   ├── data/
│   │   └── raw/                     # Raw datasets (CSV, etc.)
│   ├── images/                      # Generated visualizations
│   └── models/                      # Trained ML models (PKL, etc.)
├── learning_modules/
│   ├── 01-explore-analyze-data/
│   ├── 02-train-evaluate-regression/
│   ├── 03-train-evaluate-classification/
│   ├── 04-train-evaluate-clustering/
│   └── 05-train-evaluate-deep-learning/
├── scripts/
│   ├── setup.sh
│   ├── repair_notebooks.py
│   └── smoke_test_pytorch.py
├── src/
│   └── learning_ml_models.egg-info/
├── pyproject.toml
├── README.md
└── .venv/                           # Virtual environment (git-ignored)
```

---

## Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Poetry Documentation:** https://python-poetry.org/docs/
- **Jupyter Lab Documentation:** https://jupyterlab.readthedocs.io/
- **pyproject.toml Specification:** https://www.python.org/dev/peps/pep-0621/

---

**Last Updated:** December 9, 2025 (Enhanced & Optimized)  
**Format Version:** Markdown v2.0  
**Git Bash on Windows:** ✅ Verified

