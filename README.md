# Training and Scaling Language Models

Markdown source for the course website **Training and Scaling Language Models: From First
Principles to Efficient Serving**, designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).

## Local development

```bash
uv sync
uv run mkdocs serve
```

Open <http://127.0.0.1:8000/llm-course/>.

## Validation

```bash
uv run mkdocs build --strict
```

The generated static website is written to `site/`.

## Publishing

The workflow in `.github/workflows/pages.yml` builds the website from the
locked `uv` environment and publishes it through GitHub Pages on pushes to
`main`. Before the first deployment, configure **Settings → Pages → Source** as
**GitHub Actions** in the GitHub repository.

