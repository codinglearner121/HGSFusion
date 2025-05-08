# ubuntu环境初始化

## 1. 更新系统

首先，确保系统的软件包是最新的。

运行以下命令会更新所有的系统软件包，很常用。

```bash
sudo apt update && sudo apt upgrade -y
```

- `apt update`：更新软件包索引。
- `apt upgrade`：升级所有已安装的软件包到最新版本。

---

## 2. 安装常用工具

安装一些常用的开发工具和实用程序。

```bash
sudo apt install -y curl wget git vim build-essential net-tools software-properties-common
```

- `curl` 和 `wget`：用于从命令行下载文件。
- `git`：版本控制工具。
- `vim`：文本编辑器（如果你喜欢其他编辑器，如 `nano` 或 `emacs`，可以替换）。
- `build-essential`：包含编译软件所需的工具（如 `gcc` 编译器）。
- `net-tools`：包含网络相关的工具（如 `ifconfig`）。
- `software-properties-common`：用于管理 PPA（个人包存档）。

---

## 3. 配置 Git

#### 正常配置

安装了Git后，建议配置用户名和邮箱，以便在提交代码时使用。

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### 设置网络代理（非必须）

使用git命令行工具配置Git代理

- 设置HTTP和HTTPS代理：
  
  ```
  git config --global http.proxy http://127.0.0.1:1080
  git config --global https.proxy https://127.0.0.1:1080
  ```

- SOCKS5代理：
  
  ```
  git config --global http.proxy "socks5://127.0.0.1:7890"
  git config --global https.proxy "socks5://127.0.0.1:7890"
  ```

注意这里的端口号需要根据clash或v2ray中的实际代理设置进行调整。
格式为 协议名://代理服务器IP:代理端口号

取消已有的代理设置，可以使用以下命令：

```
git config --global --unset http.proxy
git config --global --unset https.proxy
```

通过以下命令查看配置：

```bash
git config --list
```

---

## 4. 设置时区

确保系统的时区设置正确。

```bash
sudo timedatectl set-timezone Asia/Shanghai
```

你可以通过以下命令查看当前时区：

```bash
timedatectl
```

---

## 5. 下载并安装 Miniconda

Miniconda 是 Anaconda 的轻量级版本，包含了 `conda` 包管理器和 Python 环境管理工具。它比完整的 Anaconda 更小且更适合大多数开发需求。

#### 步骤 1: 下载 Miniconda 安装脚本

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

#### 步骤 2: 运行安装脚本

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

阅读许可协议，按 `Enter` 键继续，最后输入 `yes` 同意许可协议。

*是否将 $Miniconda$ 添加到 PATH*

安装程序会询问是否要将 Miniconda 添加到 `.bashrc` 文件中。建议选择 `yes`，这样每次打开终端时都会自动激活 `conda`。

#### 步骤 3: 激活 Conda

运行以下命令可以重启终端，现在请运行该命令激活 `conda`：

```bash
source ~/.bashrc
```

你可以通过以下命令检查 `conda` 是否安装成功：

```bash
conda --version
```

如果显示 `conda` 的版本号，则表示安装成功。

#### 自学常用的conda命令

`conda update conda`：更新至最新版本，也会更新其它相关包 。

`conda --version`：用于查看当前conda的版本

`conda env list`：显示所有已经创建的环境 。

`conda create -n 环境名 python=x.x`：创建一个新的虚拟环境，并指定Python版本 。

`conda activate 环境名`：激活指定的虚拟环境 。

`conda list`：列举当前活跃环境下的所有包 。

`conda list -n 环境名`：列举指定环境下的所有包 。

`conda update --all`：更新所有包 。

`conda info`：查看conda环境详细信息 。

`conda install <package_name>`：这是最常用的命令之一，用于安装指定的包。例如，`conda install numpy` 会安装 NumPy 包 。

`conda install -c channel <package_name>`：这个命令用于从特定的通道（channel）安装包。例如，`conda install -c conda-forge package_name` 会从 conda-forge 通道安装指定的包 。*conda-forge是最常见最好用的安装通道*。

`conda deactivate`：退出当前的虚拟环境 。

`conda remove -n 环境名 --all`：删除指定的虚拟环境 。

---

## 6. 清理无用的包

清理不再需要的软件包和缓存。

```bash
sudo apt autoremove -y
sudo apt clean
```
