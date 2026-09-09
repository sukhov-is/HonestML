"""Stop hook: architecture gate — the turn cannot end with broken layer contracts.

One file serves both drivers. lint-imports judges the contracts declared in
.importlinter — core independence, inward-pointing layers, use-cases apart from
adapters — and exits 1 on a broken one, but only exit 2 blocks and returns stderr
to the model, hence this wrapper. The first
failure continues the turn with the lint output; a turn already continued once by
this hook exits successfully, so a stuck contract cannot loop the Stop gate.
"""

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    data = json.load(sys.stdin)
    if data.get("stop_hook_active"):
        return 0
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["uv", "run", "lint-imports"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
