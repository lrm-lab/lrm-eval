#!/usr/bin/env bash
set -euo pipefail
umask 077

wildclaw=false
opencode=false
for component in "$@"; do
    case "$component" in
        --wildclaw) wildclaw=true ;;
        --opencode) opencode=true ;;
        *) printf 'Usage: bash setup.sh [--wildclaw] [--opencode]\n' >&2; exit 2 ;;
    esac
done

framework_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
assets=${TIANXI_BENCH_ASSETS:-"$framework_dir/../bundles"}
bundle=${TIANXI_BENCH_BUNDLE:-"$HOME/.cache/tianxi-harbor/bundle"}
bundle_parent=$(dirname -- "$bundle")
mkdir -p -- "$bundle_parent"

exec 9>"$bundle_parent/.bundle-setup.lock"
flock 9
if [[ ! -d "$bundle" ]]; then
    staging=$(mktemp -d "$bundle_parent/.bundle.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    tar -xf "$assets/pi-openclaw-20261120-core.tar" -C "$staging"
    mv -- "$staging" "$bundle"
    trap - EXIT
fi

bash "$bundle/install/setup.sh"
if [[ ! -d "$bundle/datasets/gaia" ]]; then
    staging=$(mktemp -d "$bundle/datasets/.gaia.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    tar -xf "$assets/gaia-validation-20261124.tar" -C "$staging"
    mv -- "$staging/gaia" "$bundle/datasets/"
    rmdir "$staging"
    trap - EXIT
fi
if $wildclaw && [[ ! -d "$bundle/datasets/wildclaw-harbor" ]]; then
    staging=$(mktemp -d "$bundle/datasets/.wildclaw.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    zstd -qdc "$assets/wildclawbench-20261120-tasks.tar.zst" | tar -xf - -C "$staging"
    mv -- "$staging/datasets/wildclaw-harbor" "$bundle/datasets/"
    rmdir "$staging/datasets" "$staging"
    trap - EXIT
fi

if $opencode && [[ ! -x "$bundle/runtime/opencode-1.19.3/opencode" ]]; then
    archive="$assets/opencode-linux-x64-1.19.3.tgz"
    printf '%s  %s\n' 71d67513eb4078a7ff8eb96806b1d5edcd6533a72b311409b2e88bad4188dac6 "$archive" |
        sha256sum --check --status
    staging=$(mktemp -d "$bundle/runtime/.opencode.XXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    tar -xzf "$archive" -C "$staging" \
        --strip-components=2 package/bin/opencode
    test -x "$staging/opencode"
    mv -T -- "$staging" "$bundle/runtime/opencode-1.19.3"
    trap - EXIT
fi

printf 'Bundle ready at %s\n' "$bundle"
