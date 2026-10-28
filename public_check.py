"""
One-shot pre-publication check for a git repository.

Run from anywhere inside the repo:   python public_check.py

Prints PASS / WARN / FAIL for each check and exits with code 1 if anything
FAILED. Read-only: it only runs git commands and reads files. Do not commit
this file (it skips itself, but it does not belong in the public repo).

Checks:
  1  working tree clean, nothing unpushed
  2  tracked files: no data/weights/archives/secrets files, no venv or data folders,
     no notebooks, no oversized files
  3  the same file rules across ALL history, not just the latest commit
  4  secrets in every line ever added (key formats, password/token assignments)
  5  personal machine paths (C:\\Users\\..., POSIVIEW) in history
  6  commit messages with AI attribution trailers
  7  author/committer emails visible in history
  8  repository size
  9  .gitignore covers data, weights and the venv
 10  README present, says synthetic + not affiliated; LICENSE present;
     no roll numbers in tracked files
"""

import os
import re
import subprocess
import sys

FAIL, WARN, PASS = "FAIL", "WARN", "PASS"
results = []

BAD_EXT = {".csv", ".tsv", ".pth", ".pt", ".ckpt", ".h5", ".onnx", ".pkl", ".pickle",
           ".npy", ".npz", ".zip", ".tar", ".gz", ".7z", ".env", ".pem", ".key",
           ".sqlite", ".db"}
BAD_NAMES = {"kaggle.json", ".env"}
BAD_DIRS = {"data", "claimguard-env", ".venv", "venv", "env", "__pycache__"}
MAX_FILE_MB = 5
MAX_REPO_MB = 50

SECRET_PATTERNS = [
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("OpenAI/Anthropic-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("Kaggle-style key", re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"')),
    ("hard-coded credential", re.compile(
        r"""(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*['"][^'"\s]{8,}['"]""", re.I)),
]
PATH_PATTERNS = [re.compile(r"[A-Za-z]:\\Users\\", re.I), re.compile(r"POSIVIEW", re.I)]
TRAILER = re.compile(r"co-authored-by:|claude-session:|generated with \[?claude", re.I)
SELF = "public_check.py"


def git(*args, check=True):
    out = subprocess.run(["git", "-c", "core.quotepath=off", *args],
                         capture_output=True)
    if check and out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8", "replace").strip())
    return out.stdout.decode("utf-8", "replace"), out.returncode


def report(status, title, detail=None):
    results.append(status)
    print(f"[{status}] {title}")
    for line in (detail or [])[:8]:
        print(f"         {line}")
    if detail and len(detail) > 8:
        print(f"         ... and {len(detail) - 8} more")


def bad_file(path):
    parts = path.replace("\\", "/").split("/")
    name = parts[-1].lower()
    ext = os.path.splitext(name)[1]
    why = []
    if ext in BAD_EXT:
        why.append(f"extension {ext}")
    if name in BAD_NAMES:
        why.append("credentials file")
    if any(p.lower() in BAD_DIRS for p in parts[:-1]):
        why.append("data/venv folder")
    if ext == ".ipynb":
        why.append("notebook (outputs can leak data)")
    return why


def main():
    try:
        root, _ = git("rev-parse", "--show-toplevel")
    except Exception:
        print("Not inside a git repository.")
        sys.exit(2)
    os.chdir(root.strip())
    print(f"Checking repository at {os.getcwd()}\n")

    # 1 working tree / pushed
    status, _ = git("status", "--porcelain")
    dirty = [l for l in status.splitlines() if SELF not in l]
    if dirty:
        report(WARN, "Uncommitted changes in the working tree", dirty)
    else:
        report(PASS, "Working tree clean")
    up, rc = git("rev-parse", "--abbrev-ref", "@{u}", check=False)
    if rc != 0:
        report(WARN, "No upstream branch set; cannot tell if everything is pushed")
    else:
        ahead, _ = git("rev-list", "--count", "@{u}..HEAD")
        if int(ahead.strip()) > 0:
            report(WARN, f"{ahead.strip()} commit(s) not pushed yet")
        else:
            report(PASS, f"Everything pushed to {up.strip()}")

    # 2 tracked files
    tracked = [f for f in git("ls-files")[0].splitlines() if f]
    hits = [f"{f}  ({', '.join(bad_file(f))})" for f in tracked if bad_file(f)]
    big = []
    for f in tracked:
        try:
            if os.path.getsize(f) > MAX_FILE_MB * 1024 * 1024:
                big.append(f"{f}  ({os.path.getsize(f) / 1048576:.1f} MB)")
        except OSError:
            pass
    jpgs = [f for f in tracked if f.lower().endswith((".jpg", ".jpeg"))]
    report(FAIL if hits else PASS, "Tracked files: no data, weights, archives, secrets, venv, notebooks", hits)
    report(WARN if big else PASS, f"Tracked files: none over {MAX_FILE_MB} MB", big)
    if jpgs:
        report(WARN, f"{len(jpgs)} JPEG image(s) tracked; confirm they are not dataset or bill images",
               jpgs)

    # 3 history file rules
    names, _ = git("log", "--all", "--name-only", "--pretty=format:")
    hist = sorted({n for n in names.splitlines() if n.strip()})
    hist_hits = [f"{f}  ({', '.join(bad_file(f))})" for f in hist if bad_file(f)]
    report(FAIL if hist_hits else PASS,
           "History: no data, weights, archives, secrets, venv or notebooks ever committed", hist_hits)

    # 4 and 5: scan every added line in history
    patch, _ = git("log", "-p", "--all", "--no-color")
    secrets, paths = [], []
    current = ""
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            current = line.split(" b/")[-1]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if current.endswith(SELF):
            continue
        body = line[1:]
        for label, rx in SECRET_PATTERNS:
            if rx.search(body):
                secrets.append(f"{label} in {current}: {body.strip()[:70]}")
        for rx in PATH_PATTERNS:
            if rx.search(body):
                paths.append(f"{current}: {body.strip()[:70]}")
    report(FAIL if secrets else PASS, "History: no API keys, tokens or hard-coded credentials",
           sorted(set(secrets)))
    report(WARN if paths else PASS, "History: no personal machine paths (C:\\Users\\..., POSIVIEW)",
           sorted(set(paths)))

    # 6 attribution trailers
    msgs, _ = git("log", "--all", "--format=%h %s%n%b%x1e")
    trailers = []
    for entry in msgs.split("\x1e"):
        if TRAILER.search(entry):
            trailers.append(entry.strip().splitlines()[0] if entry.strip() else "?")
    report(WARN if trailers else PASS, "Commit messages: no AI attribution trailers", trailers)

    # 7 emails
    emails, _ = git("log", "--all", "--format=%ae%n%ce")
    uniq = sorted({e.strip() for e in emails.splitlines() if e.strip()})
    private = [e for e in uniq if not e.endswith("users.noreply.github.com")]
    if private:
        report(WARN, "Emails visible in commit history (public once the repo is public)", private)
    else:
        report(PASS, "Only GitHub no-reply emails in history")

    # 8 size
    count, _ = git("count-objects", "-v")
    kv = dict(l.split(": ") for l in count.splitlines() if ": " in l)
    mb = (int(kv.get("size", 0)) + int(kv.get("size-pack", 0))) / 1024
    report(FAIL if mb > MAX_REPO_MB else PASS, f"Repository size {mb:.1f} MB (limit {MAX_REPO_MB} MB)")

    # 9 .gitignore
    gi = open(".gitignore", encoding="utf-8", errors="replace").read() if os.path.exists(".gitignore") else ""
    need = {"data": r"(^|/)data/?", "weights (*.pth)": r"\*\.pth", "venv": r"(env|venv)"}
    missing = [k for k, rx in need.items() if not re.search(rx, gi, re.M)]
    report(FAIL if missing else PASS, ".gitignore covers data, weights and the venv",
           [f"missing: {m}" for m in missing])

    # 10 README / licence / roll numbers
    readme = next((f for f in tracked if f.lower() in ("readme.md", "readme.rst", "readme.txt")), None)
    if not readme:
        report(FAIL, "README present")
    else:
        text = open(readme, encoding="utf-8", errors="replace").read().lower()
        problems = []
        if "synthetic" not in text:
            problems.append("does not say the data is synthetic")
        if "not affiliated" not in text and "no affiliation" not in text:
            problems.append("no 'not affiliated with NHA / Government of India' note")
        report(WARN if problems else PASS, "README says synthetic data and not affiliated", problems)
    licence = [f for f in tracked if f.lower().startswith(("license", "licence"))]
    report(PASS if licence else WARN, "LICENSE file present" if licence
           else "No LICENSE file (without one, all rights are reserved; fine if intended)")
    rolls = []
    for f in tracked:
        if f.endswith(SELF) or not f.lower().endswith((".md", ".py", ".txt", ".json", ".html", ".tex")):
            continue
        try:
            for i, l in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
                if re.search(r"BECOB\d{2}", l):
                    rolls.append(f"{f}:{i}")
        except OSError:
            pass
    report(WARN if rolls else PASS, "No student roll numbers in tracked files", rolls)

    print()
    f, w = results.count(FAIL), results.count(WARN)
    if f:
        print(f"RESULT: NOT READY. {f} FAIL, {w} WARN. Fix the FAILs first.")
        sys.exit(1)
    print(f"RESULT: no blockers. {w} WARN to review before going public." if w
          else "RESULT: all checks passed.")


if __name__ == "__main__":
    main()
