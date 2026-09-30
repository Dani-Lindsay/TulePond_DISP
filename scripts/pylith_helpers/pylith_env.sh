#!/bin/bash
# Run a command inside the PyLith 5.0.2 environment with a clean env (setup.sh is bash-only and must be sourced
# from its own folder). Usage: bash pylith_env.sh <workdir> <command...>
P=$HOME/tools/external/pylith/pylith-5.0.2-macOS-12.0-arm64
WORKDIR="$1"; shift
exec env -i HOME="$HOME" PATH=/usr/bin:/bin:/usr/sbin:/sbin TERM=dumb bash -c "cd '$P' && source setup.sh >/dev/null 2>&1; cd '$WORKDIR' && $*"
