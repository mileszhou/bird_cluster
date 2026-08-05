#!/usr/bin/env bash
#
# verify-export.sh — compare an export tree against its source tree.
#
# Compares by relative path *stem* (directory + basename without extension), so
# DSC_0042.CR3 in the source matches DSC_0042.jpg in the export. Reports what is
# missing, what is unexpected, and which directories came out empty.
#
# Read-only: touches nothing in either tree.
#
# Usage:
#   ./verify-export.sh SRC DST [-s ext] [-d ext] [-q]
#     -s ext   source extension filter (default: any)
#     -d ext   dest extension filter   (default: any)
#     -q       summary counts only, no path listings
#
# Example:
#   ./verify-export.sh ~/Pictures/RAW ~/exports/jpg -s CR3 -d jpg
#
set -euo pipefail

SRC="${1:-}"; DST="${2:-}"
shift 2 2>/dev/null || { echo "usage: $0 SRC DST [-s ext] [-d ext] [-q]" >&2; exit 2; }

SRC_EXT=""; DST_EXT=""; QUIET=0
while getopts "s:d:q" opt; do
  case "$opt" in
    s) SRC_EXT="$OPTARG" ;;
    d) DST_EXT="$OPTARG" ;;
    q) QUIET=1 ;;
    *) exit 2 ;;
  esac
done

[ -d "$SRC" ] || { echo "no such directory: $SRC" >&2; exit 1; }
[ -d "$DST" ] || { echo "no such directory: $DST" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# List relative stems under a tree.
#   $1 = root, $2 = extension filter ("" means any)
# sed strips the leading "./" and then the final ".ext" suffix.
list_stems() {
  local root="$1" ext="$2"
  if [ -n "$ext" ]; then
    ( cd "$root" && find . -type f -iname "*.${ext}" -print )
  else
    ( cd "$root" && find . -type f -print )
  fi \
    | sed -e 's|^\./||' -e 's|\.[^./]*$||' \
    | sort -u
}

list_stems "$SRC" "$SRC_EXT" >"$WORK/src"
list_stems "$DST" "$DST_EXT" >"$WORK/dst"

# comm -23 = only in file1; comm -13 = only in file2
comm -23 "$WORK/src" "$WORK/dst" >"$WORK/missing"
comm -13 "$WORK/src" "$WORK/dst" >"$WORK/extra"

# Directories that exist in the export but contain no files at any depth.
( cd "$DST" && find . -type d -print ) | sed 's|^\./||' | sort >"$WORK/dst_dirs"
: >"$WORK/empty_dirs"
while IFS= read -r d; do
  [ -z "$d" ] && continue
  if [ -z "$(find "$DST/$d" -type f -print -quit 2>/dev/null)" ]; then
    printf '%s\n' "$d" >>"$WORK/empty_dirs"
  fi
done <"$WORK/dst_dirs"

# Suspiciously small outputs — a failed export often still writes a stub.
( cd "$DST" && find . -type f -size -1k -print ) | sed 's|^\./||' | sort >"$WORK/tiny"

n_src=$(wc -l <"$WORK/src")
n_dst=$(wc -l <"$WORK/dst")
n_missing=$(wc -l <"$WORK/missing")
n_extra=$(wc -l <"$WORK/extra")
n_empty=$(wc -l <"$WORK/empty_dirs")
n_tiny=$(wc -l <"$WORK/tiny")

printf 'source:  %6d files%s\n' "$n_src" "${SRC_EXT:+ (*.$SRC_EXT)}"
printf 'export:  %6d files%s\n' "$n_dst" "${DST_EXT:+ (*.$DST_EXT)}"
printf '\n'
printf 'missing from export: %6d\n' "$n_missing"
printf 'unexpected in export:%6d\n' "$n_extra"
printf 'empty directories:   %6d\n' "$n_empty"
printf 'files under 1k:      %6d\n' "$n_tiny"

if [ "$QUIET" -eq 0 ]; then
  show() {   # $1 = heading, $2 = file
    [ -s "$2" ] || return 0
    printf '\n== %s ==\n' "$1"
    sed 's/^/  /' "$2"
  }
  show "MISSING (in source, absent from export)" "$WORK/missing"
  show "UNEXPECTED (in export, absent from source)" "$WORK/extra"
  show "EMPTY DIRECTORIES" "$WORK/empty_dirs"
  show "TINY FILES (<1k)" "$WORK/tiny"
fi

# Non-zero exit when anything is wrong, so this can gate a pipeline:
#   ./verify-export.sh SRC DST -q || echo "export incomplete"
[ "$n_missing" -eq 0 ] && [ "$n_empty" -eq 0 ] && [ "$n_tiny" -eq 0 ]