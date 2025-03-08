# AppEvalPilot

## 项目简介

AppEvalPilot是一个应用评估自动化工具，旨在简化应用程序的测试、评估和分析流程。通过集成多种自动化技术，帮助开发者和测试人员更高效地完成应用评估工作。

## 功能特点

- 自动化测试管理
- 操作系统代理集成
- 服务部署支持
- 可扩展的评估框架

## 安装方法

### 前提条件

- Python 3.9+
- 必要的依赖包

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/tanghaom/AppEvalPilot.git
cd AppEvalPilot

# 安装appeval
pip install -e .

# 编辑config/config2.yaml文件设置llm模型
```

## 使用方法

### 基本使用

```bash
# 运行主程序
python main.py

# 启动服务
python scripts/server.py
```


## 项目结构

```
AppEvalPilot/
├── main.py                           # 主程序入口
├── appeval/                          # 核心模块
│   ├── roles/                        # 角色定义
│   │   ├── appeval.py                # 自动化测试角色
│   │   └── osagent.py                # 操作系统代理
│   ├── actions/                      # 动作定义
│   │   ├── screen_info_extractor.py  # 屏幕信息提取
│   │   ├── test_generator.py         # 测试用例生成
│   │   └── reflection.py             # 反思
│   ├── tools/                        # 工具定义
│   │   ├── chrome_debugger.py        # 浏览器调试工具
│   │   ├── icon_detect.py            # 图标检测及描述工具
│   │   ├── device_controller.py      # 设备控制工具
│   │   └── ocr.py                    # ocr识别工具
│   └── utils/                        # 工具函数
├── scripts/                          # 脚本文件
│   ├── server.py                     # 部署服务脚本
│   └── test_server.py                # 测试服务脚本
├── data/                             # 数据文件
└── config/                           # 配置文件
```

## 配置说明

项目使用`config/config2.yaml`文件存储配置信息，包括：

- llm模型
- base_url
- api_key

## 许可证

此项目基于MIT许可证 - 详情请查看 LICENSE 文件
