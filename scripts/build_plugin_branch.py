#!/usr/bin/env python3
"""build_plugin_branch -- build the `plugin` branch: only the files the Claude Code plugin runs.

    python3 scripts/build_plugin_branch.py                 # commit the plugin tree to the local `plugin` branch
    python3 scripts/build_plugin_branch.py --stage DIR     # only write the plugin tree to DIR (no git)

Anthropic's plugin directory reads one folder of one branch and scans everything in it. On `main` that
folder is the whole repository: the research and benchmark code, the tests and the docs, some of which
call model endpoints with keys from the environment. None of it runs as part of the plugin. The `plugin`
branch holds what the plugin does run -- the manifests, the hooks, the MCP server, the slash commands, the
`tracepack` modules those load -- plus the README, the license and the changelog, so the listing and its
scans see exactly the plugin.

How the branch is built:

* **The file set is explicit.** `PLUGIN_FILES` is the list; the modules in it are the ones the hook, MCP
  and non-bench CLI paths load (traced 2026-10-19 by running every one of them). Two checks keep the list
  honest after the code changes: a static pass refuses any `tracepack` import in a staged file that is
  not staged (bar the research-only imports listed in `RESEARCH_ONLY`), and a smoke run executes
  `tracepack doctor` and the PreCompact hook from the staged copy.
* **The README is rewritten, never edited by hand.** The `<img>` and `<picture>` blocks become Markdown
  images (the directory shows bundled images only in that form; the dark variant is dropped), links to
  files that are not on the branch point at `main` on GitHub, and two sentences say where the rest lives.
  Every rewrite asserts that it matched, so a README change cannot make one silently stop applying.
* **Git is plumbing only.** The tree is written through a temporary index and committed with
  `commit-tree` on top of the current `plugin` tip; the working tree and the checked-out branch are never
  touched. An unchanged tree makes no commit. Nothing is pushed: the script prints the push command.
  The commit identity comes from git's environment (GIT_AUTHOR_NAME, GIT_COMMITTER_EMAIL, ...) or config.

Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRANCH = "plugin"
MAIN_URL = "https://github.com/oklen/TracePack/blob/main/"

# What the plugin runs. Modules: every `tracepack` module that the hooks (`hook pre-compact`,
# `hook session-start`), the MCP server (initialize, tools/list, recall, expand, status) and the CLI
# commands other than `bench` load.
PLUGIN_FILES = [
    ".claude-plugin/plugin.json",
    ".claude-plugin/marketplace.json",
    ".mcp.json",
    "hooks/hooks.json",
    "commands/last.md",
    "commands/recall.md",
    "commands/status.md",
    "bin/tracepack.py",
    "tracepack/__init__.py",
    "tracepack/cli.py",
    "tracepack/hooks.py",
    "tracepack/mcp_server.py",
    "tracepack/session.py",
    "tracepack/userwords.py",
    "tracepack/redact.py",
    "tracepack/selftest.py",
    "tracepack/adapters/__init__.py",
    "tracepack/adapters/base.py",
    "tracepack/adapters/claude_code.py",
    "tracepack/core/__init__.py",
    "tracepack/core/assembler.py",
    "tracepack/core/closure.py",
    "tracepack/core/excerpt.py",
    "tracepack/core/graph.py",
    "tracepack/core/router.py",
    "tracepack/core/schema.py",
    "tracepack/core/textmatch.py",
    "tracepack/core/values.py",
    "LICENSE",
    "CHANGELOG.md",
]
README = "README.md"                    # staged as a rewritten copy, see transform_readme()

# Lazy imports in staged files that only research entry points reach: `tracepack.pack()` and
# `make_condenser()` in tracepack/__init__.py, and `tracepack bench` in tracepack/cli.py. The plugin
# never calls them; on the `plugin` branch those entry points are absent on purpose.
RESEARCH_ONLY = {
    "tracepack/__init__.py": ("tracepack.adapters.openhands", "tracepack.adapters.pi", "tracepack.adapters.registry",
                              "tracepack.pilot", "tracepack.baselines"),
    "tracepack/cli.py": ("tracepack.bench",),
}


class BuildError(Exception):
    pass


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise BuildError(msg)


# --------------------------------------------------------------------------- README for the branch

_PICTURE = re.compile(r'<picture>\s*<source [^>]*>\s*<img src="(assets/[^"]+)" alt="([^"]*)"[^>]*>\s*</picture>')
_IMG = re.compile(r'<img src="(assets/[^"]+)" alt="([^"]*)"[^>]*>')
_MD_IMAGE = re.compile(r'!\[[^\]]*\]\(([^)\s]+)\)')
_LINK = re.compile(r'((?<!\\)\]\(|href="|src=")([^)"\s]+)')                    # not an escaped \] in alt text
_RESEARCH = "This repository also holds the benchmark"
_RESEARCH_BRANCH = "The repository's main branch also holds the benchmark"
_DEVELOPMENT = "## Development\n\n"
_DEVELOPMENT_NOTE = ("Development happens on the `main` branch, which also holds the tests, the docs and the research\n"
                     "code. This `plugin` branch is built from it by `scripts/build_plugin_branch.py`, and the commands\n"
                     "below are for a checkout of `main`.\n\n")


def _md_image(path: str, alt: str) -> str:
    alt = " ".join(alt.split()).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return "![%s](%s)" % (alt, path)


def transform_readme(text: str) -> str:
    """The README as the `plugin` branch shows it (images and links rewritten, two notes added)."""
    text, n = _PICTURE.subn(lambda m: _md_image(m.group(1), m.group(2)), text)   # light image only
    require(n >= 1, "README: no <picture> block to convert (did the results figure move?)")
    text, n = _IMG.subn(lambda m: _md_image(m.group(1), m.group(2)), text)
    require(n >= 1, "README: no <img> to convert (did the demo GIF move?)")
    require("<img" not in text and "<picture" not in text, "README: an <img> or <picture> the script does not know")
    require(text.count(_RESEARCH) == 1, "README: the privacy sentence about the research code changed")
    text = text.replace(_RESEARCH, _RESEARCH_BRANCH)
    require(text.count(_DEVELOPMENT) == 1, "README: the Development heading changed")
    return text.replace(_DEVELOPMENT, _DEVELOPMENT + _DEVELOPMENT_NOTE)


def readme_images(text: str) -> list:
    return sorted({p for p in _MD_IMAGE.findall(text) if not re.match(r"[a-z]+://", p)})


def point_links_at_main(text: str, staged: set) -> str:
    """Relative links to files that are not on the branch go to the same file on `main`."""
    def fix(m):
        target = m.group(2)
        path = target.split("#", 1)[0]
        if not path or re.match(r"[a-z]+:", target) or path in staged:
            return m.group(0)
        require(os.path.exists(os.path.join(ROOT, path)), "README links to %s, which is not in the repository" % path)
        return m.group(1) + MAIN_URL + target
    return _LINK.sub(fix, text)


# --------------------------------------------------------------------------- staging and checks

_IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+(tracepack(?:\.\w+)*|\.+(?:\w+(?:\.\w+)*)?)[ \t]+import[ \t]+([\w \t,()*]+)"
                     r"|import[ \t]+(tracepack(?:\.\w+)*))", re.M)


def _resolve(rel: str, target: str) -> str:
    """`from ..core import x` in tracepack/adapters/base.py -> tracepack.core (absolute names pass through)."""
    if not target.startswith("."):
        return target
    package = os.path.dirname(rel).replace("/", ".")      # tracepack/adapters/base.py -> tracepack.adapters
    dots = len(target) - len(target.lstrip("."))
    parts = package.split(".")[:len(package.split(".")) - (dots - 1)]
    return ".".join(parts + ([target.lstrip(".")] if target.lstrip(".") else []))


def _on_branch(stage: str, name: str) -> bool:
    parts = name.split(".")
    return os.path.exists(os.path.join(stage, *parts) + ".py") or \
        os.path.exists(os.path.join(stage, *parts, "__init__.py"))


def check_imports(stage: str) -> None:
    """Every `tracepack` import in a staged module resolves to a staged module, bar RESEARCH_ONLY.

    Lazy imports inside functions count too: a path the smoke run does not reach is still checked."""
    missing = []
    for rel in PLUGIN_FILES:
        if not rel.endswith(".py"):
            continue
        src = open(os.path.join(stage, rel), encoding="utf-8").read()
        modules = re.findall(r"__import__\(\s*[\"'](tracepack[\w.]*)[\"']", src)    # these must be modules
        maybe = []                                     # `from m import y`: y is a module or a name in m
        for g_from, g_names, g_imp in _IMPORT.findall(src):
            if g_imp:
                modules.append(g_imp)
                continue
            base = _resolve(rel, g_from)
            modules.append(base)
            maybe += ["%s.%s" % (base, n) for n in re.findall(r"\w+", g_names) if n != "as"]
        for name in modules + maybe:
            if _on_branch(stage, name):
                continue
            if name in maybe and _on_branch(stage, name.rsplit(".", 1)[0]):
                continue                               # a name defined in a staged module, not a module
            if any(name == r or name.startswith(r + ".") for r in RESEARCH_ONLY.get(rel, ())):
                continue
            missing.append("%s imports %s" % (rel, name))
    require(not missing, "modules the plugin imports are not on the branch:\n  " + "\n  ".join(sorted(set(missing))))


def smoke(stage: str) -> None:
    """Run `tracepack doctor` and the PreCompact hook from the staged copy, in a throwaway home."""
    with tempfile.TemporaryDirectory() as home:
        env = dict(os.environ, HOME=home, CLAUDE_CONFIG_DIR=os.path.join(home, "cfg"),
                   TRACEPACK_DATA=os.path.join(home, "data"), PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        launcher = os.path.join(stage, "bin", "tracepack.py")
        r = subprocess.run([sys.executable, launcher, "doctor", "--cwd", home], env=env, cwd=home,
                           capture_output=True, text=True, timeout=120)
        require(r.returncode == 0, "tracepack doctor failed on the staged copy:\n%s%s" % (r.stdout, r.stderr))
        r = subprocess.run([sys.executable, launcher, "hook", "pre-compact"], env=env, cwd=home, timeout=60,
                           input='{"session_id": "smoke", "trigger": "manual"}', capture_output=True, text=True)
        require(r.returncode == 0 and "TracePack notes" in r.stdout,
                "the PreCompact hook printed no notes from the staged copy:\n%s%s" % (r.stdout, r.stderr))
    for dirpath, dirnames, _ in os.walk(stage):            # PYTHONDONTWRITEBYTECODE should leave none
        for d in [d for d in dirnames if d == "__pycache__"]:
            shutil.rmtree(os.path.join(dirpath, d))


def stage_plugin(stage: str) -> list:
    """Write the plugin tree into the empty folder `stage`; return the staged paths."""
    require(os.path.isdir(stage) and not os.listdir(stage), "%s must be an empty folder" % stage)
    for rel in PLUGIN_FILES:
        src = os.path.join(ROOT, rel)
        require(os.path.isfile(src) and not os.path.islink(src), "missing or not a regular file: %s" % rel)
        os.makedirs(os.path.dirname(os.path.join(stage, rel)) or stage, exist_ok=True)
        shutil.copy2(src, os.path.join(stage, rel))      # copy2 keeps bin/tracepack.py executable
    readme = transform_readme(open(os.path.join(ROOT, README), encoding="utf-8").read())
    images = readme_images(readme)
    for rel in images:
        src = os.path.join(ROOT, rel)
        require(os.path.isfile(src), "README shows %s, which does not exist" % rel)
        os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True)
        shutil.copy2(src, os.path.join(stage, rel))
    staged = set(PLUGIN_FILES) | set(images) | {README}
    readme = point_links_at_main(readme, staged)
    with open(os.path.join(stage, README), "w", encoding="utf-8") as fh:
        fh.write(readme)
    check_imports(stage)
    smoke(stage)
    on_disk = sorted(os.path.relpath(os.path.join(d, f), stage) for d, _, fs in os.walk(stage) for f in fs)
    require(on_disk == sorted(staged), "the staged folder holds files the build did not put there: %s"
            % sorted(set(on_disk) ^ staged))
    return on_disk


# --------------------------------------------------------------------------- git (plumbing only)

def git(*args, env=None, cwd=ROOT, check=True) -> str:
    r = subprocess.run(["git"] + list(args), cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise BuildError("git %s failed:\n%s" % (" ".join(args), r.stderr.strip()))
    return r.stdout.strip() if r.returncode == 0 else ""


def commit_branch(stage: str, staged: list) -> int:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    require(branch == "main", "check out main first (HEAD is %s)" % branch)
    sources = PLUGIN_FILES + [README] + [s for s in staged if s.startswith("assets/")]
    for rel in sources:                                 # the branch must match committed main exactly
        require(git("ls-files", "--error-unmatch", "--", rel, check=False) == rel, "%s is not committed on main" % rel)
    dirty = git("status", "--porcelain", "--", *sources)
    require(not dirty, "commit these on main first:\n" + dirty)
    head = git("rev-parse", "--short", "HEAD")
    git_dir = git("rev-parse", "--absolute-git-dir")
    with tempfile.TemporaryDirectory() as tmp:
        # A temporary index and the staged folder as the work tree: the real index, working tree and
        # checked-out branch are never touched.
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"), GIT_DIR=git_dir, GIT_WORK_TREE=stage)
        git("add", "--all", "--force", "--", ".", env=env, cwd=stage)     # --force: no ignore rule may drop a file
        tree = git("write-tree", env=env, cwd=stage)
    parent = git("rev-parse", "--verify", "--quiet", "refs/heads/%s^{commit}" % BRANCH, check=False)
    if parent and git("rev-parse", parent + "^{tree}") == tree:
        print("%s is already up to date with main %s (%s); nothing to commit." % (BRANCH, head, parent[:7]))
        return 0
    commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", "plugin: build from main %s" % head)
    git("update-ref", "-m", "build_plugin_branch", "refs/heads/%s" % BRANCH, commit, parent or "0" * 40)
    print("%s -> %s (from main %s, %d files)" % (BRANCH, commit[:7], head, len(staged)))
    print("push it with:  git push origin %s" % BRANCH)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the `plugin` branch: only the files the Claude Code plugin runs.")
    p.add_argument("--stage", default="", help="only write the plugin tree to this empty folder (no git)")
    a = p.parse_args(argv)
    try:
        if a.stage:
            os.makedirs(a.stage, exist_ok=True)
            staged = stage_plugin(a.stage)
            print("staged %d files in %s" % (len(staged), a.stage))
            return 0
        with tempfile.TemporaryDirectory() as stage:
            staged = stage_plugin(stage)
            return commit_branch(stage, staged)
    except BuildError as e:
        print("build_plugin_branch: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
