# GitHub Copilot Configuration for learning-ml-models

**Date:** December 8, 2025  
**Project:** learning-ml-models  
**Status:** Migrated from YAML to Markdown format

---

## 🎯 Quick Instructions for Copilot

> **PRIMARY INSTRUCTION:** When you ask me a question, I will respond by wrapping the entire response in a fenced code block labeled `markdown`. This prevents the chat pane from rendering the content, keeping it raw so you can copy-paste directly into Jupyter.
>
> **Format Example:**
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
> ✨ This means you can copy the entire response (including the outer triple backticks) and paste it into your Jupyter **Markdown** cell without any reformatting!

### Default Output Behavior

> **IMPORTANT:** Generate **only 2 documents** by default:
> 1. **Plan Document** – Multi-step breakdown of the task/question
> 2. **Summary Document** – Key findings, results, or conclusions
>
> Generate additional documents (guides, examples, references, etc.) **only when you explicitly request them** (e.g., "Also create a quick reference guide").
>
> **Exception:** If the task naturally requires more than 2 documents to be complete and useful, use your judgment to include them, but prioritize keeping output concise.

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

### Default Output Documents

By default, generate **only 2 documents** unless specifically requested:

1. **Plan Document:** `{{YYYYMMDD}}-plan-{{slug}}.md`
   - Multi-step breakdown of the task/question
   - Clear step-by-step approach
   - Example: `20251208-plan-data-analysis.md`

2. **Summary Document:** `{{YYYYMMDD}}-summary-{{slug}}.md`
   - Key findings, results, or conclusions
   - Actionable takeaways
   - Example: `20251208-summary-data-analysis.md`

### Filename Template

```
{{YYYYMMDD}}-{{document-type}}-{{slug}}.md
```

### Generate Additional Documents Only When Requested

If you explicitly request additional documents (e.g., "Also create a quick reference guide"), then generate them with appropriate naming:
- `{{YYYYMMDD}}-guide-{{slug}}.md`
- `{{YYYYMMDD}}-examples-{{slug}}.md`
- `{{YYYYMMDD}}-troubleshooting-{{slug}}.md`

> **Note:** Unless you say "Also create..." or "Generate additional...", stick to Plan + Summary only.

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

## Copilot Output Instructions

> 🎯 **PRIMARY INSTRUCTION:** Wrap ALL responses in a fenced code block labeled `markdown`. This prevents chat rendering and provides raw, copy-paste-ready content for Jupyter notebooks.

### Response Format Rules

#### 0. Wrap Everything in a Markdown Code Block

**ALWAYS** wrap your entire response in triple backticks with `markdown` label:

````markdown
```markdown
# Your complete response goes here
All content, including nested code blocks, goes inside this outer fence.

See rules below for how to format markdown, Python, and bash content.
```
````

This keeps the chat pane from rendering your response, allowing the user to copy the entire block directly into a Jupyter **Markdown** cell.

#### 1. For Markdown Content (Default)

**DO:** Inside the outer markdown block, provide raw markdown content:

```markdown
# My Heading

This is regular markdown content with **bold** and *italic* text.

- List item 1
- List item 2
```

**DON'T:** Don't add extra code block wrappers around markdown (the outer fence is enough):

```markdown
```
# Incorrect nesting
```
```

#### 2. For Python Code Blocks

When providing Python code meant for Jupyter notebooks, format it as a nested code block inside the markdown wrapper:

````markdown
```markdown
Here is some Python code:

```python
import pandas as pd

def example_function():
    return "Hello, World!"
```

Then the user can copy this code block and paste it into a Jupyter **Code** cell.
```
````

#### 3. For Bash/Shell Commands

When providing shell commands, format them as a nested code block inside the markdown wrapper:

````markdown
```markdown
Here are the bash commands:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e .
```

Run these in your Git Bash terminal.
```
````

#### 4. For Jupyter Notebook Cells

**Markdown Cell:** The user copies the entire outer markdown block (including outer fence) and pastes into a Jupyter **Markdown** cell:

````markdown
```markdown
# Data Analysis Results

Key findings:
- Point 1
- Point 2

| Column | Value |
|--------|-------|
| A      | 1     |
| B      | 2     |
```
````

**Code Cell:** The user extracts the inner Python code block (or copies it with backticks) and pastes into a Jupyter **Code** cell:

````markdown
```markdown
Here is the code to run:

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
│   └── copilot.yaml                 # Original YAML configuration
├── generated/
│   ├── docs-copilot/                # Output directory for generated docs
│   ├── data/raw/                    # Raw datasets
│   └── models/                      # Trained models
├── learning_modules/
│   ├── 01-explore-analyze-data/
│   ├── 02-train-evaluate-regression/
│   ├── 03-train-evaluate-classification/
│   ├── 04-train-evaluate-clustering/
│   └── 05-train-evaluate-deep-learning/
├── scripts/
├── src/
├── pyproject.toml
└── README.md
```

---

## Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Poetry Documentation:** https://python-poetry.org/docs/
- **Jupyter Lab Documentation:** https://jupyterlab.readthedocs.io/
- **pyproject.toml Specification:** https://www.python.org/dev/peps/pep-0621/

---

**Last Updated:** December 8, 2025  
**Format Version:** Markdown v1.0

