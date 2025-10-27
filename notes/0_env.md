# 环境相关
*按照之前的习惯，都是使用conda进行环境管理和依赖安装
尝试使用uv进行环境管理和依赖安装：
uv兼容pip的常用命令(加上uv前缀即可)

```bash
# 创建虚拟环境
## 在当前目录创建虚拟环境（默认 .venv）
uv venv

## 指定环境名称（如创建名为 myenv 的环境）
uv venv myenv

# 激活虚拟环境
source .venv/bin/activate

# 管理依赖
## 初始化 pyproject.toml（自动生成项目配置）
uv init

## 安装依赖并添加到 pyproject.toml（生产环境）
uv add requests

## 安装开发依赖（如 pytest，仅在开发时需要）
uv add --dev pytest

## 根据 pyproject.toml 安装所有依赖
uv sync


```
