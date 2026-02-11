# Changelog

## [2026-02-10] Linux 跑测稳定性优化 & a11y 双模式支持

> 分支: `merge-vf-critic-with-linux`  
> 基于 VF_critic_action 分支合并 Linux 跑测支持后的系列优化

---

### 🆕 新增功能

#### 1. Accessibility Tree 双模式支持 (`AT-SPI` / `CDP`)
- **AT-SPI 模式**（默认）：通过 D-Bus → AT-SPI 总线获取完整桌面级无障碍树，坐标为屏幕绝对坐标，与 `pyautogui` 直接兼容
- **CDP 模式**（可选）：通过 Chrome DevTools Protocol 获取浏览器内部无障碍树，无需 D-Bus/AT-SPI 系统服务，更轻量
- 通过 `run_config.yaml` 中 `a11y_mode: atspi | cdp` 配置切换
- 新增 `CDPElementProcessor` 类（`device_controller.py`）

#### 2. 跑测进程管理
- **cleanup.sh**：跑测前/后自动清理残留进程（Chrome / Xvfb / AT-SPI / D-Bus / GPU 占用进程）及临时目录
- **卡死检测**：主进程 20 分钟无新结果 → 判定 worker 卡死，强制退出并保存已完成结果
- **增量保存**：每 2 分钟自动保存已完成结果到 Excel，防止异常退出丢失数据
- **信号处理**：`Ctrl+C` 优雅终止所有 worker 进程组

#### 3. GPU 资源管理
- 支持 `cuda_devices: [0, 1, 2, 3]` 配置多 GPU
- Worker 按 round-robin 策略分配 GPU，避免全部挤在同一张卡
- OCR 模型共享缓存：主进程预下载到 `.cache/modelscope`，worker 启动时复制，避免重复下载

#### 4. 系统限制自动调优
- 自动提升 `fs.inotify.max_user_instances` → 8192
- 自动提升 `fs.inotify.max_user_watches` → 1048576
- 解决 50+ worker 并行时 Chrome 启动失败（`inotify_init() failed`）

#### 5. atoms.dev 登录自动化
- 提示词注入默认测试账号（`press_test8@mgx.dev / 123456`），agent 遇到 atoms.dev 登录/注册页时直接用已有账号登录，不走注册+邮箱验证码流程
- 提示词指引 agent 在 "Authorize Application" 页面点击 "Allow"，在 "Save password?" 弹窗点击 "Never"
- Chrome Preferences 禁用密码保存弹窗（`password_manager.enabled=false`）+ `--password-store=basic` 参数

#### 6. LLM 返回格式兜底
- `run_test.py` 中 `result["0"]` 可能是 dict 或 str（LLM 输出格式不稳定）
- 新增 `isinstance(res, dict)` 类型检查，str 时直接提取 Pass/Fail，不再抛 `'str' object has no attribute 'get'`
- 避免因格式问题导致任务被强制判 0 分（实测影响 3 个任务，其中 2 个丢分）

#### 7. OpenCV Qt 崩溃防护
- 设置 `QT_QPA_PLATFORM=offscreen` 环境变量，防止 `opencv-python` 自带的 Qt 插件在 Xvfb 环境下初始化失败导致 worker 进程直接崩溃
- 安装 `opencv-python-headless` 替代带 Qt 的 `opencv-python`（双重防护）

---

### 🔧 优化改进

#### run_test.py
| 改动 | 说明 |
|------|------|
| `result_queue.get(timeout=30)` | 轮询间隔从 600s → 30s，配合 worker 存活检查 |
| worker 存活检测 | 所有 worker 退出 → 立即结束；20min 无结果 → 强制退出 |
| 增量保存 | 每 120s 保存一次 Excel，防止数据丢失 |
| 进程组管理 | `os.setsid()` + `os.killpg()` 确保子进程全部回收 |
| OCR 缓存 | 共享模型目录 → worker 本地复制 → 各自独立 GPU |
| `QT_QPA_PLATFORM=offscreen` | 防止 cv2 Qt 插件在 Xvfb 下崩溃，worker 不再因 Qt 初始化失败而意外退出 |

#### device_controller.py
| 改动 | 说明 |
|------|------|
| `pyautogui` 懒加载 | 延迟到实际使用时 import，避免 `DISPLAY` 未设置的警告 |
| `PCController` 新增参数 | `a11y_mode` / `remote_debugging_port` |
| `CDPElementProcessor` | 新增 CDP 模式无障碍树采集器 |
| AT-SPI 错误降级 | `pyatspi` 不可用时降为 DEBUG 级日志 |

#### window_utils.py
| 改动 | 说明 |
|------|------|
| `--remote-allow-origins=*` | 允许 CDP WebSocket 连接 |
| `--password-store=basic` | 禁用 Chrome 密码保存弹窗 |
| Preferences | `password_manager.enabled=false`、`credentials_enable_service=false` 禁用密码保存提示 |

#### eval_runner.py / osagent.py
| 改动 | 说明 |
|------|------|
| 参数透传 | `a11y_mode` / `remote_debugging_port` / `use_ocr` 从配置 → AppEvalRole → OSAgent → PCController |
| 默认登录提示 | `add_info` 顶部注入 `[CRITICAL]` 级别的 atoms.dev 登录账密 + Authorize/Save password 处理指引 |

#### prompts/osagent.py
| 改动 | 说明 |
|------|------|
| PC_prompt.hints | 注入默认测试账密（`press_test8@mgx.dev / 123456`），指示 agent 直接登录而非注册 |

---

### 📝 文档

- **`docs/a11y_tree_design.md`**：a11y tree 改造技术文档
  - 架构设计（AT-SPI vs CDP 对比）
  - 实现细节与配置说明
  - CDP 坐标漂移问题调试记录（§8）
  - 已知限制与决策依据

---

### 🐛 已知问题 & 决策

| 问题 | 状态 | 说明 |
|------|------|------|
| CDP 坐标漂移 | ⚠️ 已知 | CDP 返回视口相对坐标，与 `pyautogui` 屏幕绝对坐标差 ~149px（Chrome UI 高度），导致点击偏移。已决策默认 AT-SPI |
| `serial_per_url` 大 group | ✅ 已缓解 | 最大 URL group 23 个任务 × ~38s ≈ 15min，20min 超时足够覆盖 |
| OCR 每 worker 独立加载 | ⚠️ 已知 | 每个 worker 加载 ~1GB GPU 显存，通过多卡分散缓解 |
| OpenCV Qt 崩溃 | ✅ 已修复 | 见下方详细分析 |

#### OpenCV Qt 崩溃问题详细分析

**现象**：开启 OCR 后（`use_ocr=True`），30 个 worker 只完成 37/457 个任务后全部退出。

**错误日志**：
```
QObject::moveToThread: Current thread (0x368b3d60) is not the object's thread (0x130a5850).
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized.
```

**根因**：`opencv-python`（Anaconda 4.7.0）自带 Qt 插件。Qt 后端采用**懒加载**，import cv2 时不会立即初始化，只在触发 Qt GUI 相关操作时才加载 `xcb` 插件连接 X display。在 Xvfb 环境下 `xcb` 不兼容，导致进程直接 crash。

**为什么之前没有这个问题**：AT-SPI 之前 `use_ocr=False`，worker 进程不会 import cv2，Qt 从未被加载。

**为什么还能完成部分任务**：Qt 是懒初始化的，不是 import 时就崩。只有当 agent 的某个 action 触发了 Qt GUI 初始化（如 `pyperclip.copy()` 间接触发剪贴板 / 字体渲染）时才崩溃。崩溃前已完成的任务结果正常保存。

| cv2 操作 | 触发 Qt? | 结果 |
|----------|---------|------|
| `cv2.imread()` / `cv2.resize()` / `cv2.imencode()` | ❌ 纯数组运算 | 正常 |
| `cv2.imshow()` / Qt clipboard / 字体渲染 | ✅ 触发 Qt GUI | 💥 Xvfb 下崩溃 |

**修复方案**（双重防护）：
1. `QT_QPA_PLATFORM=offscreen`：即使 Qt 被触发也使用 offscreen 后端，不连接 X display
2. `opencv-python-headless`：完全不携带 Qt 插件（但 Anaconda 的旧版 cv2 优先级更高，所以方案 1 是必须的）

---

### 🗂 提交历史

| Commit | 日期 | 说明 |
|--------|------|------|
| `70e05f2` | 2026-02-10 | feat: 优化跑测稳定性 - a11y双模式/进程清理/GPU分配/卡死检测 |
| `84a8cd7` | 2026-02-08 | chore: 移除 test_run.sh |
| `329f296` | 2026-02-08 | fix: 移除硬编码 CONFIG_ROOT |
| `cbf4c21` | 2026-02-08 | chore: 移除开发测试文件 |
| `37d1ec7` | 2026-02-08 | chore: 移除 config_gemini3_flash.yaml |
| `4beb1ec` | 2026-02-08 | fix: Linux Chrome 测试问题修复 |
| `cf3cc2b` | 2026-02-06 | feat: 合并 VF_critic_action 与 Linux 支持 |

---

### 📋 配置示例 (`run_config.yaml`)

```yaml
cuda_devices: [0, 1, 2, 3]   # 可用 GPU 列表
tasks: 0                       # 0=全部
workers: 30                    # 并行 worker 数
serial_per_url: true           # 同一网站串行

remote:
  a11y_mode: atspi             # atspi | cdp
  log_dir_prefix: experiment_name
  result_excel: results.xlsx
```

