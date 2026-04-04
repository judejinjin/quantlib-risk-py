# Migrating QuantLib-Risks-Py to Your Own GitHub Repos

## Overview

You have **6 repos** to deal with — 1 parent + 3 submodules + 2 standalone clones:

| Repo | Current upstream | Tracking |
|---|---|---|
| **QuantLib-Risks-Py** | `auto-differentiation/QuantLib-Risks-Py` | parent repo |
| **xad** | `auto-differentiation/xad` | git submodule at `lib/xad` |
| **QuantLib-Risks-Cpp** | `auto-differentiation/QuantLib-Risks-Cpp` | git submodule at `lib/QuantLib-Risks-Cpp` |
| **QuantLib** | `lballabio/QuantLib` | git submodule at `lib/QuantLib` |
| **forge** | `da-roth/forge` | standalone clone at `lib/forge` (not a submodule) |
| **xad-forge** | `da-roth/xad-forge` | standalone clone at `lib/xad-forge` (not a submodule) |

---

## 1. Create the GitHub repos

On GitHub (github.com → **New repository**), create **5 empty repos** under your account (no README/license/gitignore — completely empty):

```
your-username/QuantLib-Risks-Py
your-username/xad
your-username/QuantLib-Risks-Cpp
your-username/forge
your-username/xad-forge
```

You do **not** need to fork `lballabio/QuantLib` unless you plan to modify QuantLib itself. The submodule can keep pointing to the upstream. If you do want your own copy, create `your-username/QuantLib` too.

---

## 2. Push the standalone repos (forge, xad-forge)

These are plain git clones — push them to your new repos:

```bash
# forge
cd /home/jude/QuantLib-Risks-Py/lib/forge
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/forge.git
git push -u origin main

# xad-forge
cd /home/jude/QuantLib-Risks-Py/lib/xad-forge
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/xad-forge.git
git push -u origin main
```

This keeps the original as `upstream` so you can fetch future changes.

---

## 3. Push the submodule repos (xad, QuantLib-Risks-Cpp)

Same pattern, but these are in detached-HEAD state — create a branch first:

```bash
# xad
cd /home/jude/QuantLib-Risks-Py/lib/xad
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/xad.git
git checkout -b main          # create branch from detached HEAD
git push -u origin main

# QuantLib-Risks-Cpp
cd /home/jude/QuantLib-Risks-Py/lib/QuantLib-Risks-Cpp
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/QuantLib-Risks-Cpp.git
git checkout -b main
git push -u origin main
```

**(Optional) QuantLib** — only if you need your own fork:

```bash
cd /home/jude/QuantLib-Risks-Py/lib/QuantLib
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/QuantLib.git
git checkout -b main
git push -u origin main
```

---

## 4. Update `.gitmodules` to point to your repos

```bash
cd /home/jude/QuantLib-Risks-Py
```

Edit `.gitmodules` to change the URLs:

```ini
[submodule "lib/xad"]
    path = lib/xad
    url = https://github.com/YOUR-USERNAME/xad.git

[submodule "lib/QuantLib-Risks-Cpp"]
    path = lib/QuantLib-Risks-Cpp
    url = https://github.com/YOUR-USERNAME/QuantLib-Risks-Cpp.git

[submodule "lib/QuantLib"]
    path = lib/QuantLib
    url = https://github.com/lballabio/QuantLib.git   # keep as-is, or change if forked
```

Then sync the submodule config:

```bash
git submodule sync
```

This updates the internal `.git/config` to match `.gitmodules`.

---

## 5. Push the parent repo (QuantLib-Risks-Py)

```bash
cd /home/jude/QuantLib-Risks-Py
git remote rename origin upstream
git remote add origin https://github.com/YOUR-USERNAME/QuantLib-Risks-Py.git

# Stage the .gitmodules change
git add .gitmodules
git commit -m "Point submodules to own GitHub repos"

git push -u origin main
```

---

## 6. Verify a fresh clone works

Test from a clean directory:

```bash
cd /tmp
git clone --recurse-submodules https://github.com/YOUR-USERNAME/QuantLib-Risks-Py.git
cd QuantLib-Risks-Py

# Verify submodules populated
ls lib/xad/README*
ls lib/QuantLib-Risks-Cpp/README*
ls lib/QuantLib/README*

# Clone forge & xad-forge (not submodules, so manual)
git clone https://github.com/YOUR-USERNAME/forge.git lib/forge
git clone https://github.com/YOUR-USERNAME/xad-forge.git lib/xad-forge
```

---

## 7. (Optional) Add forge and xad-forge as submodules

Currently `lib/forge` and `lib/xad-forge` are standalone clones, not tracked by the parent repo. If you want them managed as submodules (so `--recurse-submodules` clones them automatically):

```bash
cd /home/jude/QuantLib-Risks-Py

# Remove the existing directories first
rm -rf lib/forge lib/xad-forge

# Add as submodules
git submodule add https://github.com/YOUR-USERNAME/forge.git lib/forge
git submodule add https://github.com/YOUR-USERNAME/xad-forge.git lib/xad-forge

git commit -m "Add forge and xad-forge as submodules"
git push origin main
```

---

## 8. (Optional) Pull future updates from upstream

For any repo where you kept `upstream`:

```bash
git fetch upstream
git merge upstream/main    # or upstream/master for QuantLib
```

---

## Quick reference: all repos and their remotes after migration

| Repo | `origin` (yours) | `upstream` (original) |
|---|---|---|
| QuantLib-Risks-Py | `YOUR-USERNAME/QuantLib-Risks-Py` | `auto-differentiation/QuantLib-Risks-Py` |
| lib/xad | `YOUR-USERNAME/xad` | `auto-differentiation/xad` |
| lib/QuantLib-Risks-Cpp | `YOUR-USERNAME/QuantLib-Risks-Cpp` | `auto-differentiation/QuantLib-Risks-Cpp` |
| lib/QuantLib | `YOUR-USERNAME/QuantLib` (or keep upstream) | `lballabio/QuantLib` |
| lib/forge | `YOUR-USERNAME/forge` | `da-roth/forge` |
| lib/xad-forge | `YOUR-USERNAME/xad-forge` | `da-roth/xad-forge` |
