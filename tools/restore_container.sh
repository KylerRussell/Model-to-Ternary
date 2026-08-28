#!/bin/bash
# Restore everything this container loses on restart.
#
# WHY THIS EXISTS. The container's /usr layer is ephemeral: /home and /tmp survive a restart, but every
# apt package and every hand-copied library is wiped. We have now rediscovered that twice, once piece by
# piece from confusing errors (torch reporting "no NVIDIA driver" while nvidia-smi worked; a training
# run dying with rc=127 because /usr/bin/time vanished and the bash BUILTIN `time` masked it in the
# check). This script makes recovery one command instead of an archaeology session.
#
# WHAT SURVIVES (do not reinstall):  ~/Documents/Model-to-Ternary (repo, .venv, py-spy, data/, output*/),
#   ~/.cache/huggingface, ~/_scratch_naive27b, ~/.ssh.
# WHAT DOES NOT: /tmp. An earlier version of this comment claimed /tmp scratchpads survive -- they do
#   NOT. Two long measurement runs were lost to that assumption (a 12-arm, 11.5 h sweep with zero arms
#   completed). Put anything a long run needs to survive under ~/Documents/Model-to-Ternary/output_*
#   (gitignored and durable), never /tmp.
# WHAT RESETS: /usr/lib CUDA userspace libs, apt packages (time, rsync, numactl), tailscale + its state.
#
#   bash tools/restore_container.sh          # restore everything
#   bash tools/restore_container.sh --check  # report only
set -u
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
D=/usr/lib/x86_64-linux-gnu
SP=${SCRATCH:-/tmp/container_restore}
ok=0; fixed=0; manual=0
say()  { printf '  %-34s %s\n' "$1" "$2"; }
need() { [ "$CHECK" = "1" ] && { say "$1" "MISSING (--check: not fixing)"; return 1; }; return 0; }

echo "=== container restore ==="

# ── 1. CUDA userspace driver libs ────────────────────────────────────────────────────────────────
# nvidia-smi works without these (it uses libnvidia-ml), so the symptom is torch reporting
# "Found no NVIDIA driver" while device_count() still returns 2. The version MUST match the running
# kernel module exactly; apt's 550.163.01 does NOT work against a 550.144.03 module.
if [ -e "$D/libcuda.so.1" ]; then
  say "libcuda.so.1" "ok"; ok=$((ok+1))
elif need "libcuda.so.1"; then
  VER=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
  echo "  installing CUDA userspace libs for kernel module $VER ..."
  mkdir -p "$SP" && cd "$SP"
  curl -sSL -o nv.run "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/NVIDIA-Linux-x86_64-$VER.run" \
    && sh nv.run --extract-only --target nvx >/dev/null 2>&1
  for f in libcuda libnvidia-ptxjitcompiler libnvidia-nvvm libcudadebugger; do
    [ -f "$SP/nvx/$f.so.$VER" ] && sudo cp "$SP/nvx/$f.so.$VER" "$D/$f.so.$VER"
  done
  sudo ln -sf "$D/libcuda.so.$VER" "$D/libcuda.so.1";  sudo ln -sf "$D/libcuda.so.1" "$D/libcuda.so"
  sudo ln -sf "$D/libnvidia-ptxjitcompiler.so.$VER" "$D/libnvidia-ptxjitcompiler.so.1"
  sudo ln -sf "$D/libnvidia-nvvm.so.$VER" "$D/libnvidia-nvvm.so.4"
  sudo ln -sf "$D/libcudadebugger.so.$VER" "$D/libcudadebugger.so.1"
  sudo ldconfig
  cd - >/dev/null; say "libcuda.so.1" "restored ($VER)"; fixed=$((fixed+1))
fi

# ── 2. apt packages. `time` is the sneaky one: bash has a BUILTIN `time`, so `command -v time`
# succeeds while /usr/bin/time is absent — and scripts using `/usr/bin/time -f` die with rc=127.
# gh: the git credential helper is configured to use it, so without it even `git ls-remote` fails
# with "could not read Username". The TOKEN survives in ~/.config/gh (that is under /home); only
# the binary is wiped, so reinstalling is the entire fix — no re-login needed.
# --reinstall AND post-verify, both essential. The container reset wipes /usr/bin but leaves dpkg's
# database intact, so a plain `apt-get install` says "already newest version" and exits 0 while the
# binary is still missing. This script reported "numactl restored" for a numactl that did not exist;
# the NUMA-bound sweep then died instantly with exitcode 127 and looked like a torchrun problem.
# Trust the probe, never apt's exit code.
for pkg in time rsync numactl gh; do
  case $pkg in time) probe=/usr/bin/time;; *) probe=/usr/bin/$pkg;; esac
  if [ -x "$probe" ]; then say "$pkg" "ok"; ok=$((ok+1)); continue; fi
  sudo apt-get install -y -q --reinstall "$pkg" >/dev/null 2>&1
  if [ -x "$probe" ] || command -v "$pkg" >/dev/null 2>&1; then say "$pkg" "restored"; fixed=$((fixed+1))
  else say "$pkg" "FAILED — apt exited 0 but $probe is still missing"; manual=$((manual+1)); fi
done

# ── 3. tailscale (needed to reach the old Arch box). No systemd here, so tailscaled must be started
# detached with setsid or it dies with the launching shell — the "it kills itself after 10s" symptom.
if pgrep -x tailscaled >/dev/null 2>&1; then
  say "tailscaled" "running"; ok=$((ok+1))
elif command -v tailscaled >/dev/null 2>&1; then
  if need "tailscaled"; then
    sudo setsid nohup tailscaled --state=/var/lib/tailscale/tailscaled.state \
         --socket=/var/run/tailscale/tailscaled.sock >/tmp/tailscaled.log 2>&1 </dev/null &
    sleep 6
    if tailscale status >/dev/null 2>&1; then say "tailscaled" "started"; fixed=$((fixed+1))
    else say "tailscaled" "started, NEEDS 'sudo tailscale up' (state was lost)"; manual=$((manual+1)); fi
  fi
else
  say "tailscale" "NOT INSTALLED - 'sudo apt-get install -y tailscale' then 'sudo tailscale up' (interactive)"
  manual=$((manual+1))
fi

# ── 4. verify the thing that actually matters ────────────────────────────────────────────────────
cd "$(dirname "$0")/.." || exit 1
if [ -x .venv/bin/python ]; then
  .venv/bin/python - <<'PY' 2>/dev/null || say "torch CUDA" "FAILED - check libcuda"
import torch, sys
assert torch.cuda.is_available(), "cuda not available"
a = torch.randn(512, 512, device="cuda:0", dtype=torch.bfloat16); (a @ a).float().mean().item()
print(f"  {'torch CUDA':<34s} ok ({torch.cuda.device_count()} GPUs, real matmul)")
PY
else say "venv" "MISSING (unexpected - it lives in /home)"; fi

echo "=== $ok ok, $fixed restored, $manual need manual steps ==="
