#!/bin/bash
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Script: build_kernel_vm.sh
# Description: Builds a Docker image from the target env subfolder (uv-based)
#              and registers a Jupyter kernel that runs inside the container.
#              Pass --remove to undo everything (kernel, wrapper, image).
#
# Usage:   bash environments/build_kernel_vm.sh <env-subfolder> [--remove]
# Example: bash environments/build_kernel_vm.sh blueprint-env
#          bash environments/build_kernel_vm.sh blueprint-env --remove
# -----------------------------------------------------------------------------

set -euo pipefail

info()  { echo -e "\e[32m[INFO]\e[0m  $1"; }
warn()  { echo -e "\e[33m[WARN]\e[0m  $1"; }
error() { echo -e "\e[31m[ERROR]\e[0m $1"; exit 1; }

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
[ $# -lt 1 ] && error "Usage: $0 <env-subfolder> [--remove]  (e.g. blueprint-env)"

ENV_SUBFOLDER="$1"
REMOVE=false
[ "${2:-}" = "--remove" ] && REMOVE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/$ENV_SUBFOLDER"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

[ -d "$ENV_DIR" ]         || error "Env folder not found: $ENV_DIR"
[ -f "$ENV_DIR/env.yml" ] || error "No env.yml found in $ENV_DIR"

# -----------------------------------------------------------------------------
# Extract image / kernel name from env.yml  (reads the 'name:' field)
# -----------------------------------------------------------------------------
IMAGE_NAME=$(grep '^name:' "$ENV_DIR/env.yml" | awk '{print $2}')
[ -z "$IMAGE_NAME" ] && error "Could not read 'name:' from $ENV_DIR/env.yml"

KERNEL_NAME="$IMAGE_NAME"
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"
VENV_DIR="$HOME/.local/share/virtualenvs/$KERNEL_NAME"

# -----------------------------------------------------------------------------
# --remove: undo everything and exit
# -----------------------------------------------------------------------------
if $REMOVE; then
  info "Removing all traces of '$IMAGE_NAME' ..."

  # 1. Jupyter kernel spec
  if jupyter kernelspec list 2>/dev/null | grep -q "^  $KERNEL_NAME "; then
    jupyter kernelspec remove "$KERNEL_NAME" -y
    info "Jupyter kernel '$KERNEL_NAME' removed."
  else
    warn "Jupyter kernel '$KERNEL_NAME' not found — skipping."
  fi

  # 2. Python wrapper / fake venv
  if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
    info "Python wrapper removed: $VENV_DIR"
  else
    warn "Python wrapper dir not found — skipping."
  fi

  # 3. Docker image (force-remove any stopped containers using it first)
  if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    STALE_CONTAINERS=$(docker ps -a --filter "ancestor=$IMAGE_NAME" -q)
    if [ -n "$STALE_CONTAINERS" ]; then
      docker rm -f $STALE_CONTAINERS &>/dev/null || true
      info "Removed stopped containers using '$IMAGE_NAME'."
    fi
    docker rmi "$IMAGE_NAME"
    info "Docker image '$IMAGE_NAME' removed."
  else
    warn "Docker image '$IMAGE_NAME' not found — skipping."
  fi

  info ""
  info "✅ Removal complete for '$IMAGE_NAME'."
  exit 0
fi

info "Script dir  : $SCRIPT_DIR"
info "Env dir     : $ENV_DIR"
info "Project root: $PROJECT_ROOT"

[ -f "$ENV_DIR/Dockerfile" ] || error "No Dockerfile found in $ENV_DIR"

KERNEL_DISPLAY_NAME="Python ($KERNEL_NAME)"

info "Image / kernel name: $IMAGE_NAME"

# -----------------------------------------------------------------------------
# Build Docker image from the env subfolder
# -----------------------------------------------------------------------------
info "Building Docker image '$IMAGE_NAME' from $ENV_DIR ..."
docker build -t "$IMAGE_NAME" "$ENV_DIR" || error "Docker build failed."
info "Base image '$IMAGE_NAME' built successfully."

# -----------------------------------------------------------------------------
# Add ipykernel on top (thin layer, cached after first run)
# -----------------------------------------------------------------------------
info "Ensuring ipykernel is available in '$IMAGE_NAME' ..."
docker build -t "$IMAGE_NAME" -f - "$ENV_DIR" <<DOCKERFILE
FROM $IMAGE_NAME
RUN uv pip install --system --no-cache ipykernel
DOCKERFILE
info "ipykernel layer done."

# -----------------------------------------------------------------------------
# Register Jupyter kernel
# The kernel.json argv calls 'docker run' transparently — no port forwarding,
# no jupyter server needed. Jupyter spawns the container per kernel session.
# -----------------------------------------------------------------------------
info "Registering Jupyter kernel '$KERNEL_DISPLAY_NAME' at $KERNEL_DIR ..."

mkdir -p "$KERNEL_DIR"

cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "docker", "run", "--rm", "-i",
    "--user", "$(id -u):$(id -g)",
    "-v", "{connection_file}:/tmp/connection.json",
    "-v", "$PROJECT_ROOT:$PROJECT_ROOT",
    "-w", "$PROJECT_ROOT",
    "--network", "host",
    "$IMAGE_NAME",
    "python", "-m", "ipykernel_launcher", "-f", "/tmp/connection.json"
  ],
  "display_name": "$KERNEL_DISPLAY_NAME",
  "language": "python"
}
EOF

info "Kernel spec written to $KERNEL_DIR/kernel.json:"
cat "$KERNEL_DIR/kernel.json"

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
info ""
info "Registered kernels:"
jupyter kernelspec list

# -----------------------------------------------------------------------------
# Create a Python wrapper so VS Code "Python: Select Interpreter" can find it
# The wrapper forwards all calls (python -c, python script.py, etc.) into the
# container, making it behave like a normal interpreter from VS Code's point
# of view.  The directory is symlinked as .venv in the project root so VS Code
# auto-discovers it without any changes to settings.json.
# -----------------------------------------------------------------------------
VENV_BIN="$VENV_DIR/bin"
WRAPPER="$VENV_BIN/python"

PYTHON_VERSION=$(docker run --rm "$IMAGE_NAME" python -c "import sys; print('%d.%d.%d' % sys.version_info[:3])")

# Helper: write the wrapper shim + pyvenv.cfg into a given directory
_write_env_dir() {
  local dir="$1"
  local bin="$dir/bin"
  mkdir -p "$bin"

  cat > "$bin/python" <<'PYSHIM'
#!/bin/bash
# Auto-generated by build_kernel_vm.sh — forwards python calls into Docker.
# ~/.vscode-server is mounted so VS Code's interpreter-discovery scripts are
# accessible inside the container. No -i flag so subprocess calls work without
# a TTY (required for VS Code's "Python: Select Interpreter" discovery).
VSCODE_SERVER="$HOME/.vscode-server"
MOUNTS="-v \"$PWD:$PWD\" -w \"$PWD\""
[ -d "$VSCODE_SERVER" ] && MOUNTS="$MOUNTS -v \"$VSCODE_SERVER:$VSCODE_SERVER\""
eval exec docker run --rm $MOUNTS --network host __IMAGE_NAME__ python '"$@"'
PYSHIM
  sed -i "s|__IMAGE_NAME__|$IMAGE_NAME|g" "$bin/python"
  chmod +x "$bin/python"
  ln -sf python "$bin/python3"

  # pyvenv.cfg — VS Code requires this file to treat the directory as a venv.
  # 'home' must point to a directory that contains a binary named 'python'.
  cat > "$dir/pyvenv.cfg" <<PYCFG
home = /usr/bin
implementation = CPython
version = $PYTHON_VERSION
virtualenv = $IMAGE_NAME (Docker)
prompt = ($KERNEL_NAME)
PYCFG
}

info "Creating Python wrapper in VENV_DIR: $VENV_DIR ..."
_write_env_dir "$VENV_DIR"

info "Python wrapper registered."
info ""
info "✅ Setup complete!"
info "   Image      : $IMAGE_NAME"
info "   Kernel     : $KERNEL_DISPLAY_NAME"
info "   Interpreter: $VENV_DIR/bin/python"
info ""
info "To activate the interpreter in VS Code:"
info "  1. CTRL+SHIFT+P → 'Developer: Reload Window'"
info "  2. CTRL+SHIFT+P → 'Python: Select Interpreter'"
info "     → '$KERNEL_NAME' will appear in the list — no manual path needed."
info ""
info "Select 'Python ($KERNEL_NAME)' as your kernel in VS Code or Jupyter."
