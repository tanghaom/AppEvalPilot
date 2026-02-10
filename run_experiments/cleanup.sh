#!/bin/bash
# ============================================================
# cleanup.sh - 跑测前/后自动清理残留进程和临时文件
# 用法: bash cleanup.sh          (清理并显示状态)
#       bash cleanup.sh --quiet  (静默模式)
# ============================================================
set -e

QUIET=${1:-""}

log() {
    [ "$QUIET" != "--quiet" ] && echo "$@"
}

log "🧹 开始清理残留进程..."

# ── 0. 提高系统 inotify 限制（50 worker × Chrome 会超出默认 128 限制）──
sysctl -w fs.inotify.max_user_instances=8192 2>/dev/null || true
sysctl -w fs.inotify.max_user_watches=1048576 2>/dev/null || true
log "  inotify.max_user_instances=$(cat /proc/sys/fs/inotify/max_user_instances)"

# ── 1. 杀掉 run_test 相关 python 进程 ──
pkill -9 -f "run_test.py" 2>/dev/null || true
sleep 1

# ── 2. 杀掉所有 Chrome 进程 ──
pkill -9 -f chrome 2>/dev/null || true
killall -9 chrome google-chrome-stable google-chrome 2>/dev/null || true
sleep 1

# ── 3. 杀掉 Xvfb ──
killall -9 Xvfb 2>/dev/null || true
sleep 1

# ── 4. 杀掉 AT-SPI 服务 ──
killall -9 at-spi-bus-launcher at-spi2-registryd 2>/dev/null || true
sleep 1

# ── 5. 杀掉 D-Bus session bus（保留 system bus） ──
# system bus 通常由 message+ 用户运行，排除它
ps aux | grep "dbus-daemon" | grep -v grep | grep -v "^message" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
pkill -9 -f "dbus-launch" 2>/dev/null || true
sleep 1

# ── 6. 释放 GPU 显存（杀掉所有占用 GPU 的 python 进程） ──
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        pid=$(echo "$pid" | tr -d ' ')
        [ -z "$pid" ] && continue
        cmd=$(ps -p "$pid" -o cmd --no-headers 2>/dev/null || echo "unknown")
        log "  GPU 进程: PID $pid ($cmd)"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
fi

# ── 7. 清理临时文件 ──
rm -rf /tmp/chrome_test2_* 2>/dev/null || true
rm -rf /tmp/chrome_debug_* 2>/dev/null || true
rm -rf /tmp/modelscope_worker_* 2>/dev/null || true
# 注意：不清理共享模型缓存 .cache/modelscope，避免重复下载

# ── 8. 显示清理后状态 ──
log ""
log "✅ 清理完毕"
log "───────────────────────────────────"
log "  Chrome:   $(ps aux | grep -i chrome | grep -v grep | wc -l)"
log "  Xvfb:     $(ps aux | grep Xvfb | grep -v grep | wc -l)"
log "  AT-SPI:   $(ps aux | grep at-spi | grep -v grep | wc -l)"
log "  D-Bus:    $(ps aux | grep dbus | grep -v grep | wc -l)"
log "  总进程:   $(ps aux | wc -l)"
if command -v nvidia-smi &>/dev/null; then
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    log "  GPU 进程: $gpu_procs"
fi
log "  内存:     $(free -h | grep Mem | awk '{print $3 "/" $2}')"
log "───────────────────────────────────"

