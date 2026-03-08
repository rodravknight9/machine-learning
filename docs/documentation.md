# Documentation (MkDocs)

This folder uses **MkDocs** with the **Material** theme to serve all Markdown files under `docs/` in a single site. Implementation stays under `./machine-learning` per project rules. MkDocs requires the doc sources to live in a **child directory** of the config file (e.g. `docs/`), not the same directory.

## Quick start

1. **Install dependencies** (from the `machine-learning` folder):

   ```bash
   pip install -r requirements_docs.txt
   ```

2. **Serve the docs locally** (from the `machine-learning` folder):

   ```bash
   mkdocs serve
   ```

   Then open **http://127.0.0.1:8000** in your browser.

3. **Build static site** (optional):

   ```bash
   mkdocs build
   ```

   Output goes to the `site/` directory. Add `site/` to `.gitignore` if you don't want to commit the build.

## Adding new Markdown files

1. Add your `.md` file under `machine-learning/docs/` (use **snake_case** for filenames; use **kebab-case** for new folders under `docs/`).
2. Edit **`mkdocs.yml`** in the `machine-learning` folder and add an entry under `nav`, for example:

   ```yaml
   nav:
     - Home: index.md
     - Your Section:
       - Your New Doc: your-folder/your_file.md
   ```

3. Run `mkdocs serve` again; the new page will appear in the sidebar.

## What's included

The `nav` in `mkdocs.yml` currently includes:

- **Home** — `index.md`
- **Overview** — `overview.md`
- **Data preprocessing and visualization** — class doc and techniques summary (snake_case)
- **Linear regression** — comparison of 1D closed-form vs gradient descent

All paths in `nav` are relative to the `docs/` folder. We pin MkDocs to 1.x (`mkdocs<2`) so the Material theme keeps working.
