#!/usr/bin/env bash
set -euo pipefail

# Setup a fresh Conda environment for MCS MEA analysis and verify installs.
# Usage:
#   bash scripts/setup_mcs_env.sh [env_name]
#
# Defaults:
#   env_name = mcs_mea_env

ENV_NAME="${1:-mcs_mea_env}"

echo "[setup] Target Conda environment: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] 'conda' not found on PATH. Please install Miniconda/Anaconda and retry." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="${ROOT_DIR}/environment_mcs.yml"

if [[ ! -f "${ENV_YML}" ]]; then
  echo "[error] Missing environment file: ${ENV_YML}" >&2
  exit 1
fi

echo "[setup] Creating or updating env from ${ENV_YML} ..."
set +e
conda env create -n "${ENV_NAME}" -f "${ENV_YML}" 2>/dev/null
CREATE_RC=$?
set -e
if [[ ${CREATE_RC} -ne 0 ]]; then
  echo "[setup] Env exists; updating instead..."
  conda env update -n "${ENV_NAME}" -f "${ENV_YML}"
fi

echo "[setup] Registering Jupyter kernel (optional) ..."
conda run -n "${ENV_NAME}" python - <<'PY'
try:
    import sys
    from ipykernel import kernelspec
    print("Python:", sys.executable)
    print("Registering IPython kernel ...")
except Exception as e:
    print("(skip) ipykernel not available:", e)
PY

echo "[verify] Checking key packages ..."
conda run -n "${ENV_NAME}" python - <<'PY'
import sys
print('Python:', sys.executable)

def check(modname):
    try:
        m = __import__(modname)
        ver = getattr(m, '__version__', 'unknown')
        print(f'{modname}: OK ({ver})')
        return True
    except Exception as e:
        print(f'{modname}: FAIL ({e})')
        return False

ok = True
ok &= check('numpy')
ok &= check('pandas')
ok &= check('h5py')

# Try both common names for the MCS package
mcs_ok = False
for name in ('McsPyDataTools', 'mcspydatatools'):
    try:
        m = __import__(name)
        print(f'{name}: OK ({getattr(m, "__version__", "unknown")})')
        mcs_ok = True
        break
    except Exception as e:
        print(f'{name}: FAIL ({e})')

if not mcs_ok:
    print('WARNING: MCS tools not importable. Try:')
    print('  conda activate', '"' + sys.prefix + '"')
    print('  pip install McsPyDataTools')

PY

echo
echo "[done] To use the environment:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/run_mcs_scan.py"

