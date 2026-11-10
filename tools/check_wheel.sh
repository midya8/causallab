#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$root"
python - "$tmp/expected.json" <<'PY'
import json,subprocess,sys
commands=['demo','multiworld-demo','bounds-demo','graph-demo','backdoor-demo','frontdoor-demo','frontdoor-effect-demo']
expected={c:json.loads(subprocess.check_output([sys.executable,'-m','causallab',c]))for c in commands}
json.dump(expected,open(sys.argv[1],'w'))
PY
python -m pip wheel --no-deps --no-build-isolation . -w "$tmp/wheels"
python -m venv "$tmp/venv"
"$tmp/venv/bin/pip" install --no-deps "$tmp"/wheels/*.whl
mkdir "$tmp/outside"
cd "$tmp/outside"
"$tmp/venv/bin/python" "$root/tools/verify_installed.py" "$tmp/expected.json" "$root"
