#!/bin/bash
# Run a command inside the PyLith 5.0.2 binary bundle with a clean environment (the bundle's setup.sh is bash-only and
# must be sourced from its own folder; the conda env of the notebooks must not leak in).
# Usage: bash pylith/pylith_env.sh <workdir> <command...>        e.g. bash pylith/pylith_env.sh "$DATA" pylith --version
# The PyLith 5.0.2 bundle (macOS arm64) is installed at $PYLITH_BUNDLE (default below).
P=${PYLITH_BUNDLE:-$HOME/tools/external/pylith/pylith-5.0.2-macOS-12.0-arm64}
WORKDIR="$1"; shift
exec env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin TERM=dumb bash -c "cd '$P' && source setup.sh >/dev/null 2>&1; cd '$WORKDIR' && $*"
