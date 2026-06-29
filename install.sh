#!/bin/sh
# install.sh — thin wrapper; delegates to scripts/install-quipu-global.sh.
set -eu
exec "$(dirname "$0")/scripts/install-quipu-global.sh" "$@"
