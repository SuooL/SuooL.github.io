title: 树莓派折腾指南之使用 SSR 实现科学上网
date: 2018-01-09 19:50:57
tags: [树莓派]
category: [硬件]
---

# 树莓派折腾指南之使用 SSR 实现科学上网
p## 预备工作
首先确定你的 SSR 混淆加密方法，如果是 salsa20 或 chacha20 或 chacha20-ietf 就需要编译安装 libsodium 这个库。
其 github 地址是 [libsodium](https://github.com/jedisct1/libsodium/) ，编译安装的步骤如下：

<!--more-->

首先安装 build-essential 软件包，其作用是提供编译程序必须软件包的列表信息，也即是，编译程序有了这个软件包它才能确定头文件在哪，才知道库函数在哪，还会下载依赖的软件包，最后才组成一个开发环境。
安装命令是：

```shell
sudo apt install build-essential
```

安装完成后，就是下载编译 libsodium 库了，首先你要去它的[发布页](https://github.com/jedisct1/libsodium/releases)，确定当前最新的发布版本，替代我下面的当前的 VERSION 。主要命令如下：

打开终端，首先定义两个变量，在命令行中输入：

```shell
VERSION='1.0.15'
CPUNUM=`cat /proc/cpuinfo | grep 'processor' | wc -l`
```
然后回到当前根目录，下载及解压源码包：

```shell
cd ~
wget -c "https://github.com/jedisct1/libsodium/releases/download/${VERSION}/libsodium-${VERSION}.tar.gz"  # 慢慢等
tar xzf libsodium-${VERSION}.tar.gz
```
目录压栈，并进行编译安装，

```shell
pushd libsodium-${VERSION}
./configure
make -j${CPUNUM}
sudo make install
sudo ldconfig
```
出栈目录，并删除编译目录。

```shell
popd
rm -rf libsodium-${VERSION}*
```

关于 pushd，popd 等目录操作命令，具体可以参见 [方便的目录切换——dirs、pushd、popd命令](http://www.361way.com/pushd/1118.html)。

## 编译 SSR 及基本配置
SSR 当前的 Python 版本的项目地址是 [shadowsocksr](https://github.com/shadowsocksr-backup/shadowsocksr) 。

首先安装 m2crypto，git，supervisor 等相关依赖软件包：

```shell
sudo apt install m2crypto git supervisor libevent-dev
```

下载编译如下命令：

```shell
git clone -b manyuser https://github.com/shadowsocksr-backup/shadowsocksr
cd ~/shadowsocksr
bash initcfg.sh
```
等待完成之后，编辑配置文件，在命令行输入：

```shell
sudo nano user-config.json
```
具体参数的意义及作用，参照 [config.json](https://github.com/ssrbackup/shadowsocks-rss/wiki/config.json) 。

local_address 和 local_port 不需要修改，其余的按照 Wiki 修改，基本上配置好： server 填服务器地址
server_port 填服务器端口
password 密码
method 加密方法
obfs 混淆插件
protocol 协议插件
等关键参数，具体看你的 SSR 服务端的设置。

编辑好后，运行一下命令测试一下是否能运行

```shell
python shadowsocks/local.py
```
没报错的话 Ctrl+C 结束继续下一步，如果有报错继续改配置文件。

## 协议转换代理 
因为 SS 系的协议走的是 Socks5 协议，对于 Terminal 的 get,wget 等走 Http 协议的地方是无能为力的，所以需要转换成 Http 代理。
此时可以选择使用 [redsocks](https://github.com/darkk/redsocks) 来实现。

下载编译的命令如下:

```shell
cd ~
git clone https://github.com/darkk/redsocks
pushd redsocks
make
cp redsocks.conf.example redsocks.conf
sed -i "s/ip = example.org/ip = 127.0.0.1/" redsocks.conf
popd
```

## SSR 开机自启

首先新建一个用户专门用来运行 SSR 和 redsocks 的账号：

```shell
 sudo useradd -M -s /sbin/nologin ss
```

然后使用 supervisor 守护进程和开机自启，直接将下面的命令全部粘贴进入命令行回车即可。

```shell
sudo bash -c "cat > /etc/supervisor/conf.d/ss.conf" <<EOF
[program:ss]
directory = /home/pi/shadowsocksr/shadowsocks
command = python /home/pi/shadowsocksr/shadowsocks/local.py
autostart = true
autorestart = true
startsecs = 10
startretries = 36
user = ss
redirect_stderr = true
stdout_logfile = /dev/null
EOF

sudo bash -c "cat > /etc/supervisor/conf.d/redsocks.conf" <<EOF
[program:redsocks]
directory = /home/pi/redsocks
command = /home/pi/redsocks/redsocks -c /home/pi/redsocks/redsocks.conf
autostart = true
autorestart = true
startsecs = 10
startretries = 36
user = ss
redirect_stderr = true
stdout_logfile = /dev/null
EOF

sudo supervisorctl update
```

上面的配置把 log 全部扔掉了，如果有需要的可以自己改 stdout_logfile 项。

## 设置 iptables 实现 nat 转发

主要步骤如下，
首先创建一个叫 REDSOCKS 的链

```shell
sudo iptables -t nat -N REDSOCKS
```

然后进行基本的规则定制，首先忽略服务器的地址，下面的 server_ip 填服务器 ip 或域名，这里和 SSR 配置文件里的 server 值要一样

```shell
sudo iptables -t nat -A REDSOCKS -d server_ip -j RETURN
```

然后忽略本地地址

```shell
sudo iptables -t nat -A REDSOCKS -d 0.0.0.0/8 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 10.0.0.0/8 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 127.0.0.0/8 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 169.254.0.0/16 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 172.16.0.0/12 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 192.168.0.0/16 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 224.0.0.0/4 -j RETURN
sudo iptables -t nat -A REDSOCKS -d 240.0.0.0/4 -j RETURN
```

除了除上面之外的所有流量都转发到 socks 的本地端口：

```shell
sudo iptables -t nat -A REDSOCKS -p tcp -j REDIRECT --to-ports 12345
```

之后应用上面的规则,将 OUTPUT 出去的 tcp 流量全部经过 REDSOCKS 链

```shell
sudo iptables -t nat -A OUTPUT -p tcp -j REDSOCKS
```
保存规则

```shell
sudo bash -c "iptables-save > /etc/iptables.up.rules"
```

## 开机自动加载规则

```shell
sudo bash -c "cat > /etc/network/if-pre-up.d/iptables" <<EOF
#!/bin/bash
/sbin/iptables-restore < /etc/iptables.up.rules
EOF
```

添加执行权限

```shell
sudo chmod +x /etc/network/if-pre-up.d/iptables
```

使用

```shell
curl myip.ipip.net
```

可以查看当前的ip，如果是你的服务器 IP 就表示可以了。
比如：
![](http://7xjsv3.com1.z0.glb.clouddn.com/15154976353553.jpg)


## 关闭全局代理
执行下面的命令

```
# 下面的server_ip填服务器ip或域名，和上面添加的值要一样
sudo iptables -t nat -D REDSOCKS -d server_ip -j RETURN
sudo iptables -t nat -D REDSOCKS -d 0.0.0.0/8 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 10.0.0.0/8 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 127.0.0.0/8 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 169.254.0.0/16 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 172.16.0.0/12 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 192.168.0.0/16 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 224.0.0.0/4 -j RETURN
sudo iptables -t nat -D REDSOCKS -d 240.0.0.0/4 -j RETURN
sudo iptables -t nat -D REDSOCKS -p tcp -j REDIRECT --to-ports 12345
sudo iptables -t nat -D OUTPUT -p tcp -j REDSOCKS
```

## 其他

更多关 iptables 的学习可以去 [iptables应用：NAT、数据报处理、清空iptables规则](https://lesca.me/archives/iptables-nat-mangle-clear-rules.html)

# NEXT
1. 实现了科学上网之后，下一步就可以 Google Assistant 的安装和试用，把你的树莓派变身成为一个 Google Home
2. 接入 HomeKit，连入小米家具全家桶，让树莓派变身成为你的智能家居中枢。

