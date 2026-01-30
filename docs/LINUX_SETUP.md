
## 1. 环境要求

- 操作系统：Linux（Ubuntu 20.04+ 推荐）
- Python：3.10+
- Conda：用于环境管理
- Chrome/Chromium：用于网页测试
- Xvfb：虚拟显示（headless 环境需要）

---

## 2. 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n appeval python=3.10 -y

# 激活环境
conda activate appeval
```

---

## 3. 安装项目依赖

```bash
# 进入项目目录
cd /path/to/AppEvalPilot

# 安装 uv（快速 pip 替代）
pip install uv

# 安装依赖
uv pip install -r requirements.txt

# 安装 appeval 包（开发模式）
uv pip install -e .

# 可选：安装 OCR 和图标检测增强功能
# uv pip install -e .[ultra]
```

---

## 4. 安装浏览器

AppEvalPilot 需要 Chrome 或 Chromium 来执行网页测试。

### 方式 1：安装 Chromium（推荐）

```bash
# Ubuntu/Debian
sudo apt install chromium-browser

# 或使用 snap
sudo snap install chromium
```

### 方式 2：安装 Google Chrome

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
```

### 验证安装

```bash
# 检查 Chrome 路径
which chromium-browser google-chrome google-chrome-stable 2>/dev/null
```

支持的 Chrome 路径（自动检测）：
- `/usr/bin/google-chrome`
- `/usr/bin/google-chrome-stable`
- `/usr/bin/chromium`
- `/usr/bin/chromium-browser`
- `/snap/bin/chromium`
- `/opt/google/chrome/chrome`

---

## 5. 安装和配置 Xvfb（虚拟显示）

在没有物理显示器的服务器上，需要 Xvfb 提供虚拟 X11 显示。

### 安装 Xvfb

```bash
sudo apt install xvfb
```

### 启动 Xvfb

```bash
# 启动虚拟显示（分辨率 1920x1080，24 位色深）
nohup Xvfb :99 -screen 0 1920x1080x24 > /tmp/xvfb.log 2>&1 &

# 验证运行状态
ps aux | grep Xvfb | grep -v grep
```

### 设置 DISPLAY 环境变量

```bash
export DISPLAY=:99
```

建议将此行添加到 `~/.bashrc` 或运行脚本中。

---

## 6. 配置 LLM API

复制配置模板并填入 API 信息：

```bash
cp config/config2.yaml.example config/config2.yaml
```

编辑 `config/config2.yaml`：

```yaml
llm:
  api_type: "openai"
  model: "claude-3-5-sonnet-v2"
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"

case_generator:
  api_type: "openai"
  model: "claude-3-5-sonnet-v2"
  base_url: "https://your-api-endpoint/v1"
  api_key: "your-api-key"
```

## 7. 运行测试

```bash
cd /root/zhijieliu/AppEvalPilot
conda activate appeval
python main_parallel.py
```

