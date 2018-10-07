---
title: 问题解决记录
date: 2018-07-25 22:57:12
tags: [记录]
category: [技术]
---

## 系统问题
### Docker

**1.Ubuntu 下安装 `docker` 使用非 `sudo` 命令的问题**：

```shell
Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.38/images/json: dial unix /var/run/docker.sock: connect: permission denied
```
解决方法：
>the error message tells you that your current user can’t access the docker engine, because you’re lacking permissions to access the unix socket to communicate with the engine.
**Temporary solution**
Use the `sudo` command to execute the commands with elevated permissions every time.
**Permanent (suggested) solution**
Add the current user to the docker group. This can be achieved by typing
```
sudo usermod -a -G docker $USER
```
**You have to log out and log in again** for the group membership to take effect.

<!--more-->

**2.Docker 换源**

新版的 `Docker` 使用 `/etc/docker/daemon.json（Linux）` 或者 `%programdata%\docker\config\daemon.json（Windows）` 来配置 Daemon 。

请在该配置文件中加入（没有该文件的话，请先建一个）：

```
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
```
完成上述配置后，执行如下命令即可：

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```



### 替换及重置Homebrew默认源
替换 `brew.git`:

```
cd "$(brew --repo)"
git remote set-url origin https://mirrors.ustc.edu.cn/brew.git
```
替换 `homebrew-core.git`:

```
cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
git remote set-url origin https://mirrors.ustc.edu.cn/homebrew-core.git
```
替换 `Homebrew Bottles` 源: 参考:[替换 Homebrew Bottles 源](https://lug.ustc.edu.cn/wiki/mirrors/help/homebrew-bottles)

在中科大源失效或宕机时可以： 
1. [使用清华源设置参考](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)。
2. 切换回官方源：

重置 `brew.git`:

```
cd "$(brew --repo)"
git remote set-url origin https://github.com/Homebrew/brew.git
```

重置 `homebrew-core.git`:

```
cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
git remote set-url origin https://github.com/Homebrew/homebrew-core.git
```
注释掉bash配置文件里的有关Homebrew Bottles即可恢复官方源。 重启bash或让bash重读配置文件。

### Xcode 高亮和提示失效
Xcode 在长时间运行之后，会出现一个问题，那就是 例如 return 不再高亮显示了。
重启Xcode 重启机器 也不能解决和个问题。
后来找了一些资料 解决了这个问题

1. 由于 DerivedData 问题
    - 关闭项目，Xcode 设置，选择 Localtions 点击 Derived Data 的的箭头进入 DerivedData 目录
    - 直接进入 /Users/jingwenzheng/Library/Developer/Xcode/DerivedData 目录，删除这里面所有的文件
    - 重启Xcode
2. 由于 pch 文件的问题
    - 把.pch里的内容全部注释掉，clean掉项目里的内容，把.pch里的注释去掉，编译。代码高亮，语法提示功能都回来了。


## VSCode
[mac vscode Python配置](https://segmentfault.com/a/1190000012322533)
[使用vs(visual studio code)写python代码遇到的import requests失败问题](https://blog.csdn.net/ever_now_future/article/details/79075581)
## TensorFlow
[源码编译安装TensorFlow](https://www.jianshu.com/p/1d305ac47d61)
[TensorFlow如何充分使用所有CPU核数，提高TensorFlow的CPU使用率，以及Intel的MKL加速](http://nooverfit.com/wp/tensorflow%E5%A6%82%E4%BD%95%E5%85%85%E5%88%86%E4%BD%BF%E7%94%A8%E6%89%80%E6%9C%89cpu%E6%A0%B8%E6%95%B0%EF%BC%8C%E6%8F%90%E9%AB%98tensorflow%E7%9A%84cpu%E4%BD%BF%E7%94%A8%E7%8E%87%EF%BC%8C%E4%BB%A5/)
[Tensorflow并行计算：多核(multicore)，多线程(multi-thread)，图分割(Graph Partition)](https://blog.csdn.net/rockingdingo/article/details/55652662)
