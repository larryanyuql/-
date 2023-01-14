---
{"dg-publish":true,"permalink":"/jax-reax-ff/","tags":["gardenEntry"]}
---


# JAX-ReaxFF

> **参考链接**
> 
> [CentOS7安装NVIDIA显卡驱动 - GolLong - 博客园](https://www.cnblogs.com/gollong/p/12655424.html)
> 
> [CentOS下的CUDA安装和使用指南]([CentOS下的CUDA安装和使用指南 - 腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/1879466#:~:text=CentOS%E4%B8%8B%E7%9A%84CUDA%E5%AE%89%E8%A3%85%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97%20%E5%8F%91%E5%B8%83%E4%BA%8E2021-09-18%2000%3A15%3A04%20%E9%98%85%E8%AF%BB%201.3K%200,%E5%BC%95%E8%A8%80%EF%BC%9A%E6%9C%AC%E6%96%87%E5%AE%89%E8%A3%85%20CUDA%20%E4%B8%BB%E8%A6%81%E7%94%A8%E4%BA%8E%E5%9C%A8%20GPU%20%E4%B8%8A%E8%AE%AD%E7%BB%83%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%EF%BC%8C%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80%E4%B8%BAPython%EF%BC%8C%E4%B8%8EC%2FC%2B%2B%E4%B8%8D%E5%90%8C%EF%BC%8C%E4%BD%BF%E7%94%A8%20Anaconda%20%E5%AE%89%E8%A3%85%E5%BE%88%E6%96%B9%E4%BE%BF%EF%BC%8C%E6%B2%A1%E6%9C%89%E5%8C%85%E7%AE%A1%E7%90%86%E7%9A%84%E5%86%B2%E7%AA%81%E3%80%82))
> 
>   引言：本文安装`CUDA`主要用于在[GPU](https://cloud.tencent.com/product/gpu?from=10680)上训练深度学习模型，编程语言为Python，与C/C++不同，使用`Anaconda`安装很方便，没有包管理的冲突。
> 
> ***Note the `cudatoolkit` distributed by `conda-forge` is missing `ptxas`, which JAX requires. You must therefore either install the `cuda-nvcc` package from the `nvidia` channel, or install CUDA on your machine separately so that `ptxas` is in your path. The channel order above is important (`conda-forge` before `nvidia`). We are working on simplifying this.***
> 
>  [linux # centos # 安装cuda - 简书](https://www.jianshu.com/p/375245691cf2)

# 一、nvidia驱动安装

## 预、查询命令

### 1. Linux查看显卡信息：（ps：若找不到lspci命令，可以安装 yum install pciutils）

```shell
lspci | grep -i vga
```

### 2. 使用nvidiaGPU可以：

```shell
lspci | grep -i nvidia
```

### 3. 查看显卡驱动

```shell
cat/proc/driver/nvidia/version
```

## 前提准备

### 1. 安装依赖环境：

```shell
yum installkernel-devel gcc -y
```

### 2. 检查内核版本和源码版本，保证一致

```shell
ls /boot | grepvmlinu
rpm -aq | grepkernel-devel
```

### 3. 屏蔽系统自带的nouveau

查看命令：

```shell
lsmod | grep nouveau
```

修改dist-blacklist.conf文件：

```shell
vim /lib/modprobe.d/dist-blacklist.conf
```

将nvidiafb注释掉:

```shell
#blacklist nvidiafb
```

然后添加以下语句：

```shell
blacklist nouveau
options nouveaumodeset=0
```

### 4. 重建initramfsimage步骤

```shell
mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
dracut /boot/initramfs-$(uname -r).img $(uname -r)
```

### 5. 修改运行级别为文本模式

```shell
systemctl set-default multi-user.target
```

### 6. 重新启动

```shell
reboot
```

## 本地安装

### 在NVIDIA官网下载驱动

网址：[NVIDIA 驱动程序下载](https://www.nvidia.cn/Download/index.aspx?lang=cn)

```shell
chmod +xNVIDIA-Linux-x86_64-440.64.run
./NVIDIA-Linux-x86_64-440.64.run
```

**如果报错 unable to find the kernel source tree for the currently running kernel.........，使用下面命令安装，3.10.0-1062.18.1.el7.x86_64需要改成自己的目录**

```shell
./NVIDIA-Linux-x86_64-440.64.run --kernel-source-path=/usr/src/kernels/3.10.0-1062.18.1.el7.x86_64 -k $(uname -r)
```

nvidia-smi

# 二、独立安装CUDA

### 1. 确定CUDA toolkit的版本

+ CUDA toolkit对nvidia的版本有要求， 可参见[Release Notes :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)中的CUDA Driver部分的说明。

+ 查看系统和内核的要求  ，参见https://docs.nvidia.com/cuda/archive/9.1/cuda-installation-guide-linux/index.html中[System Requirements](https://docs.nvidia.com/cuda/archive/9.1/cuda-installation-guide-linux/index.html#system-requirements)部分的说明。

### 2. 获取CUDA toolkit下载地址:

CUDA toolkit 下载地址: https://developer.nvidia.com/cuda-toolkit-archive  

### 3. 安装

```shell
sh cuda_9.1.85_387.26_linux.run
nvcc --version
```

# 三、安装CuDNN(https://developer.nvidia.cn/zh-cn/cudnn)

# 四、JAX: Autograd and XLA [GitHub - google/jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more](https://github.com/google/jax)

```shell
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# 五、JAX-ReaxFF[GitHub - cagrikymk/JAX-ReaxFF: JAX-ReaxFF: A Gradient Based Framework for Extremely Fast Optimization of Reactive Force Fields](https://github.com/cagrikymk/JAX-ReaxFF)

```shell
git clone https://github.com/cagrikymk/Jax-ReaxFF
cd Jax-ReaxFF
pip install .
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.0+cuda11.cudnn805-cp38-none-manylinux2010_x86_64.whl
```
