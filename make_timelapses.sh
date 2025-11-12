#!/usr/bin/env bash
# make_timelapses.sh
# Given a root directory, find subdirectories that contain only image files and
# build a 60fps timelapse for each (mp4 or gif).
#
# Supported image extensions: jpg jpeg png bmp webp tif tiff
# Output naming: <subfolder>.mp4 or <subfolder>.gif placed in the same directory as the image set.
#
# Usage:
#   ./make_timelapses.sh ROOT_DIR [--format mp4|gif] [--fps 60] [--force] [--dry-run]
#
# Options:
#   --format mp4|gif    Output container (default: mp4)
#   --fps N             Frames per second (default: 60)
#   (Automatically recurses all nested subdirectories; no depth limit)
#   --force             Rebuild even if output exists and is newer than sources
#   --dry-run           Show what would be done without invoking ffmpeg
#   --min N             Minimum number of images required (default: 2)
#   --sort name|mtime   Sorting mode for images (default: name)
#   --help              Show help
#
# Examples:
#   ./make_timelapses.sh data/images --format gif --fps 30 --recursive
#   ./make_timelapses.sh /data/root --dry-run
set -euo pipefail
IFS=$'\n\t'

ROOT=""
FORMAT="mp4"
FPS=60
FORCE=0
DRYRUN=0
MIN_IMAGES=2
SORT_MODE="name"

log() { printf "[%s] %s\n" "$(date +'%H:%M:%S')" "$*"; }
err() { printf "[ERROR] %s\n" "$*" >&2; }
usage() { grep '^#' "$0" | sed -e 's/^# //'; exit 0; }

have_ffmpeg() {
  command -v ffmpeg >/dev/null 2>&1
}

is_image_ext() {
  local f="$1"; f=${f##*.}; f=${f,,}
  case "$f" in jpg|jpeg|png|bmp|webp|tif|tiff) return 0;; *) return 1;; esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --format) FORMAT="$2"; shift 2 ;;
      --fps) FPS="$2"; shift 2 ;;
      --force) FORCE=1; shift ;;
      --dry-run) DRYRUN=1; shift ;;
      --min) MIN_IMAGES="$2"; shift 2 ;;
      --sort) SORT_MODE="$2"; shift 2 ;;
      --help|-h) usage ;;
      *)
        if [[ -z "$ROOT" ]]; then ROOT="$1"; shift; else err "Unexpected arg: $1"; exit 1; fi
      ;;
    esac
  done
  if [[ -z "$ROOT" ]]; then err "ROOT_DIR required"; usage; fi
  if [[ ! -d "$ROOT" ]]; then err "ROOT_DIR '$ROOT' not found"; exit 1; fi
  if [[ "$FORMAT" != "mp4" && "$FORMAT" != "gif" ]]; then err "--format must be mp4 or gif"; exit 1; fi
}

collect_candidate_dirs() {
  find "$ROOT" -type d ! -path "$ROOT" -print
}

# Determine if a directory contains only images (and at least MIN_IMAGES)
is_image_only_dir() {
  local d="$1"
  local files=("$d"/*)
  local count=0
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    if is_image_ext "$f"; then
      ((count++))
    else
      # Encountered non-image file -> not pure image dir
      return 1
    fi
  done
  (( count >= MIN_IMAGES )) || return 1
  return 0
}

# Build sorted list of image files
list_images() {
  local d="$1"
  if [[ "$SORT_MODE" == "mtime" ]]; then
    find "$d" -maxdepth 1 -type f \( $(printf -- '-iname *.%s -o ' jpg jpeg png bmp webp tif tiff | sed 's/ -o $//') \) -printf '%T@ %p\n' | sort -n | awk '{print $2}'
  else
    find "$d" -maxdepth 1 -type f \( $(printf -- '-iname *.%s -o ' jpg jpeg png bmp webp tif tiff | sed 's/ -o $//') \) -print | sort
  fi
}

needs_rebuild() {
  local out="$1"; shift
  local first_img="$1"
  if (( FORCE )); then return 0; fi
  [[ -f "$out" ]] || return 0
  # If output newer than all sources skip
  local newest_src="$(stat -c %Y "$first_img")"
  for img in "$@"; do
    local t="$(stat -c %Y "$img")"
    if (( t > newest_src )); then newest_src=$t; fi
  done
  local out_mtime="$(stat -c %Y "$out")"
  if (( out_mtime >= newest_src )); then return 1; else return 0; fi
}

build_mp4() {
  local out="$1"; shift
  local dir="$1"; shift
  local pattern_type="$1"; shift
  local input_pattern="$1"; shift
  local fps="$1"; shift
  local extra_args=("-y" "-hide_banner" "-loglevel" "error" "-framerate" "$fps" "-pattern_type" "$pattern_type" "-i" "$input_pattern" "-c:v" "libx264" "-pix_fmt" "yuv420p" "-vf" "fps=$fps" "$out")
  if (( DRYRUN )); then
    log "DRY-RUN mp4 ffmpeg ${extra_args[*]}"
  else
    ffmpeg "${extra_args[@]}"
  fi
}

build_gif() {
  local out="$1"; shift
  local dir="$1"; shift
  local pattern_type="$1"; shift
  local input_pattern="$1"; shift
  local fps="$1"; shift
  local palette="$dir/.palette.png"
  if (( DRYRUN )); then
    log "DRY-RUN gif (palettegen + paletteuse)"
    return
  fi
  ffmpeg -y -hide_banner -loglevel error -framerate "$fps" -pattern_type "$pattern_type" -i "$input_pattern" -vf "fps=$fps,scale=iw:-1:flags=lanczos,palettegen" "$palette"
  ffmpeg -y -hide_banner -loglevel error -framerate "$fps" -pattern_type "$pattern_type" -i "$input_pattern" -i "$palette" -lavfi "fps=$fps,scale=iw:-1:flags=lanczos [x]; [x][1:v] paletteuse" "$out"
  rm -f "$palette"
}

main() {
  parse_args "$@"
  have_ffmpeg || { err "ffmpeg not found in PATH"; exit 1; }
  log "Root: $ROOT | format: $FORMAT | fps: $FPS | force: $FORCE | dry-run: $DRYRUN | min: $MIN_IMAGES | sort: $SORT_MODE | depth: unlimited"
  local dirs=( $(collect_candidate_dirs) )
  [[ ${#dirs[@]} -gt 0 ]] || { err "No subdirectories found"; exit 1; }

  for d in "${dirs[@]}"; do
    if is_image_only_dir "$d"; then
      local images=( $(list_images "$d") )
      [[ ${#images[@]} -ge MIN_IMAGES ]] || continue
      local out_ext="$FORMAT"
      local out="$d.$out_ext"
      if ! needs_rebuild "$out" "${images[@]}"; then
        log "Skip (up-to-date): $out"
        continue
      fi
      # Use first file extension pattern to build glob
      local first="${images[0]}"
      local ext="${first##*.}"; ext=${ext,,}
      local pattern_type="glob"
      local glob_pattern="$d/*.${ext}"
      # Check numeric sequence possibility (e.g., frame%05d.jpg)
      # If filenames share prefix and vary by number we can switch to sequence pattern for robustness
      if printf '%s\n' "${images[@]}" | grep -Eq '_[0-9]+\.'; then
        # Extract prefix from first
        local base="$(basename "$first")"
        local prefix="${base%%_[0-9]*}.${ext}" # Not reliable for all cases, keep glob.
      fi
      log "Building $FORMAT: $out (images: ${#images[@]})"
      if [[ "$FORMAT" == "mp4" ]]; then
        build_mp4 "$out" "$d" "$pattern_type" "$glob_pattern" "$FPS"
      else
        build_gif "$out" "$d" "$pattern_type" "$glob_pattern" "$FPS"
      fi
    else
      log "Skip (not pure images): $d"
    fi
  done
  log "Done"
}

main "$@"
