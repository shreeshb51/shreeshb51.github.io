# shreeshb51.github.io — Portfolio Setup Guide

A complete record of how this portfolio was built, deployed, and how to maintain it.

---

## Overview

This is a static research portfolio site built with [Quarto](https://quarto.org), hosted for free on GitHub Pages. The site is scoped to Geometric Deep Learning (GDL) and Computational Optimal Transport (COT).

**Live site:** https://shreeshb51.github.io

**Stack:**

- Quarto (site generator)
- GitHub Pages (hosting)
- GitHub Actions (auto-deploy on push)
- Zed (editor)
- GitHub Desktop (version control)

---

## One-Time Setup

### 1. Install Quarto
Download and install the `.pkg` from https://quarto.org.

### 2. Create the project
```bash
quarto create project website shreeshb51.github.io
cd shreeshb51.github.io
```

### 3. Create folder structure
```bash
mkdir -p projects catw papers .github/workflows
```

### 4. Create GitHub repo
On GitHub.com, create a new repo named exactly `shreeshb51.github.io`.

### 5. Add to GitHub Desktop
Drag the project folder into GitHub Desktop and add it as a local repo.

---

## First-Time Deployment

Standard push via GitHub Desktop will fail until the `gh-pages` branch exists. Bootstrap it once locally:

### 1. Set remote URL with token
Generate a Personal Access Token on GitHub (Settings → Developer Settings → Tokens (classic) → `repo` scope). Then, in terminal run:

```bash
git remote set-url origin https://shreeshb51:YOUR_TOKEN@github.com/shreeshb51/shreeshb51.github.io.git
```

### 2. Run first publish
```bash
quarto publish gh-pages
```

### 3. Reset remote URL (remove token from URL)
```bash
git remote set-url origin https://github.com/shreeshb51/shreeshb51.github.io.git
```

**Never leave a token in the remote URL.** Revoke and regenerate immediately if accidentally exposed.

### 4. Enable GitHub Pages
Repo → Settings → Pages → Source: **Deploy from a branch** → `gh-pages` → `/ (root)` → Save.

### 5. Enable workflow write permissions
Repo → Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save.

---

## Annual Token Renewal & Maintenance

### Token Renewal (every 1 year)
1. GitHub → Settings → Developer Settings → Personal access tokens → Fine-grained tokens → Generate new token (`repo` scope, 1 year expiry)
2. Store token in a password manager — **never paste it in code or terminal while inside the repo directory**
3. Set remote URL with token temporarily:
   ```bash
   git remote set-url origin https://shreeshb51:YOUR_TOKEN@github.com/shreeshb51/shreeshb51.github.io.git
   ```
4. Run `quarto publish gh-pages`
5. Immediately reset remote URL:
   ```bash
   git remote set-url origin https://github.com/shreeshb51/shreeshb51.github.io.git
   ```
6. Delete the token from hand/notes; GitHub auto-revokes if exposed

### If push is blocked (secret scanning violation)
- Visit the unblock URL shown in the terminal error and allow the secret
- Retry `quarto publish gh-pages`
- Revoke the exposed token immediately and generate a fresh one

### What NOT to do
- Never leave token in remote URL after publishing
- Never commit `_site/` — it contains rendered HTML that may embed the remote URL with token
- Never commit to `gh-pages` branch manually — Quarto manages it

### `.gitignore` must include
```
/.quarto/
**/*.quarto_ipynb
_site/
```

---

## Workflow

1. Create a new subfolder inside `projects/`, `catw/`, or `papers/`
2. Add `index.qmd` with frontmatter and content
3. Place asset files in the same subfolder
4. Run `quarto preview` to verify locally
5. GitHub Desktop → commit on `main` → push
6. GitHub Actions auto-deploys in ~3 minutes

**Never commit to the `gh-pages` branch.** It is auto-managed by Quarto. Discard any changes GitHub Desktop shows on that branch.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
