#!/usr/bin/env bash
# list_struct_grouped.sh
#
# Recursive tree listing with two features:
#  (A) File-collapse per directory: if > FILE_THRESHOLD regular files, show only first FILE_SHOW_FIRST + ellipsis.
#      (Folders are not counted in the >10 rule.)
#  (B) Structure-grouping across siblings: at each directory level, if multiple immediate subfolders share the
#      *same subtree structure* (up to SIGN_MAX_DEPTH), print ONE representative fully, then summarize the rest
#      as compact "same structure" lines. This makes repetition obvious without collapsing everything.
#
# Requirements: bash 4+, GNU find, sha1sum (or md5sum fallback).
# Includes hidden files/dirs. Sorting is lexicographic (LC_ALL=C).
set -Eeuo pipefail

# ---------- Tunables (override via env) ----------
FILE_THRESHOLD=${FILE_THRESHOLD:-10}     # collapse files if > this many
FILE_SHOW_FIRST=${FILE_SHOW_FIRST:-3}    # when collapsing files, show this many

STRUCT_THRESHOLD=${STRUCT_THRESHOLD:-4}  # group when >= this many siblings share identical structure
STRUCT_SHOW_FIRST=${STRUCT_SHOW_FIRST:-4} # how many sibling names to list after representative

SIGN_MAX_DEPTH=${SIGN_MAX_DEPTH:-8}      # structural signature depth (dirs + lightweight file markers)
# -------------------------------------------------

start="${1:-.}"

# Global sets
declare -A SKIP      # paths we decided not to recurse (grouped siblings other than representative)
declare -A PRINTED   # paths already printed explicitly (representatives) to avoid duplicates

sorted_immediate_dirs() {
  local p="$1"
  find "$p" -mindepth 1 -maxdepth 1 -type d -printf '%f\0' 2>/dev/null | LC_ALL=C sort -z
}

sorted_immediate_files() {
  local p="$1"
  find "$p" -mindepth 1 -maxdepth 1 -type f -printf '%f\0' 2>/dev/null | LC_ALL=C sort -z
}

# Compute a structural signature for $1 up to SIGN_MAX_DEPTH.
# Signature captures:
#   - all directory relative paths (D:<rel>/)
#   - for each directory, either the exact filenames if count <= FILE_THRESHOLD, else a marker F+:<rel> (no filenames)
signature_for() {
  local root="$1"
  local maxd="$SIGN_MAX_DEPTH"

  # Gather all dirs (relative), sorted.
  local -a rel_dirs=()
  mapfile -t rel_dirs < <(LC_ALL=C find "$root" -maxdepth "$maxd" -type d -printf '%P/\n' 2>/dev/null | LC_ALL=C sort)

  local out=""
  for rel in "${rel_dirs[@]}"; do
    out+="D:${rel}"$'\n'
  done

  # For each dir, summarize its immediate files.
  for rel in "${rel_dirs[@]}"; do
    local abs="$root/${rel%/}"
    local -a files=()
    mapfile -d '' -t files < <(find "$abs" -mindepth 1 -maxdepth 1 -type f -printf '%f\0' 2>/dev/null | LC_ALL=C sort -z)
    local n="${#files[@]}"
    if (( n == 0 )); then
      :
    elif (( n > FILE_THRESHOLD )); then
      out+="F+:${rel}"$'\n'
    else
      for fn in "${files[@]}"; do
        out+="F:${rel}${fn}"$'\n'
      done
    fi
  done

  local hash
  hash="$(printf '%s' "$out" | sha1sum 2>/dev/null | awk '{print $1}')"
  if [[ -z "${hash:-}" ]]; then
    hash="$(printf '%s' "$out" | md5sum 2>/dev/null | awk '{print $1}')"
  fi
  printf '%s' "$hash"
}

# At directory $dir, detect sibling groups with identical structure signatures.
# For each group with size >= STRUCT_THRESHOLD:
#  - print a header,
#  - print ONE representative fully,
#  - list other siblings succinctly,
#  - mark the other siblings in SKIP to avoid expanding them later.
group_repeated_structures() {
  local dir="$1" indent="$2"

  local -a parents=()
  mapfile -d '' -t parents < <(sorted_immediate_dirs "$dir")
  ((${#parents[@]})) || return 0

  declare -A SIG_OF         # abs path -> signature
  declare -A COUNT_BY_SIG   # signature -> count

  for p in "${parents[@]}"; do
    local abs="$dir/$p"
    local sig
    sig="$(signature_for "$abs")"
    SIG_OF["$abs"]="$sig"
    COUNT_BY_SIG["$sig"]=$(( ${COUNT_BY_SIG["$sig"]:-0} + 1 ))
  done

  for sig in "${!COUNT_BY_SIG[@]}"; do
    local cnt="${COUNT_BY_SIG[$sig]}"
    if (( cnt >= STRUCT_THRESHOLD )); then
      # Collect members of this signature group.
      local -a members=()
      for p in "${parents[@]}"; do
        local abs="$dir/$p"
        if [[ "${SIG_OF[$abs]}" == "$sig" ]]; then
          members+=("$p")
        fi
      done
      # Sort members; choose representative.
      IFS=$'\n' read -r -d '' -a members_sorted < <(printf '%s\n' "${members[@]}" | LC_ALL=C sort && printf '\0')
      local rep="${members_sorted[0]}"
      local rep_abs="$dir/$rep"

      echo "${indent}  [structure ×${cnt}] showing '${rep}/' as representative"
      PRINTED["$rep_abs"]=1

      # Print representative fully (recurse)
      list_dir "$rep_abs" "  ${indent}"

      # Mark others to skip and list a few
      local shown=0
      local to_show="$STRUCT_SHOW_FIRST"
      for ((i=1; i<${#members_sorted[@]}; i++)); do
        local other="${members_sorted[i]}"
        local other_abs="$dir/$other"
        SKIP["$other_abs"]=1
        if (( shown < to_show )); then
          echo "${indent}    like: ${other}/ (same structure)"
          ((shown++))
        fi
      done
      local remaining=$(( cnt - 1 - shown ))
      if (( remaining > 0 )); then
        echo "${indent}    ... (+${remaining} more with same structure)"
      fi
    fi
  done
}

list_files_with_collapse() {
  local dir="$1" indent="$2"
  local -a files=()
  mapfile -d '' -t files < <(sorted_immediate_files "$dir")
  local n="${#files[@]}"
  if (( n > FILE_THRESHOLD )); then
    for ((i=0; i<FILE_SHOW_FIRST; i++)); do
      echo "${indent}  ${files[i]}"
    done
    local remaining=$(( n - FILE_SHOW_FIRST ))
    echo "${indent}  ... (+${remaining} more files)"
  else
    for f in "${files[@]}"; do
      echo "${indent}  ${f}"
    done
  fi
}

list_dir() {
  local dir="$1" indent="$2"

  # Header
  if [[ -z "$indent" ]]; then
    echo "./"
  else
    echo "${indent}${dir##*/}/"
  fi

  # Optional: group repeated structures among immediate subdirs, printing reps inline.
  group_repeated_structures "$dir" "$indent"

  # Now collect immediate subdirs for normal recursion (skip grouped others & already-printed reps).
  local -a dirs=()
  mapfile -d '' -t dirs < <(sorted_immediate_dirs "$dir")
  for d in "${dirs[@]}"; do
    local sub="$dir/$d"
    if [[ -n "${SKIP[$sub]:-}" ]] || [[ -n "${PRINTED[$sub]:-}" ]]; then
      continue
    fi
    list_dir "$sub" "  ${indent}"
  done

  # Finally, list files with per-directory collapse rule.
  list_files_with_collapse "$dir" "$indent"
}

# Normalize start for stable SKIP/PRINTED keys
if command -v realpath >/dev/null 2>&1; then
  start="$(realpath -s "$start")"
fi

list_dir "$start" ""
