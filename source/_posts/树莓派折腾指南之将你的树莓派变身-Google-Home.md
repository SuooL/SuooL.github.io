title: 树莓派折腾指南之将你的树莓派变身 Google Home
date: 2018-01-11 00:20:12
tags: [树莓派]
category: [硬件]
---
# 将你的树莓派变身 Google Home
目前的智能化已经对人们的生活影响越来越深刻，从智能音箱鼻祖 Amazon Echo 发布之后，智能数字助理就进入了人们的生活当中，也成为人们生活的一部分，特别是其与智能家居配合之后，以前科幻电影中的场景，目前已经渐渐成为现实。
<!--more-->

当前国内的厂家也有发布类似的产品，比如小米的小爱同学，阿里的天猫精灵等智能助理产品，智能家居加上智能助理中枢，将是物联网真正进入人们平常生活的最深刻的体现。

而且 Google 和 Amazon 都发布了自己语音助手的 SDK ，今天就来看一下如何接入 
`Google Assistant SDK` ，来体验一下树莓派版本的 `Google Home` 。


## 准备工作

### 材料准备

硬件：
	
* Raspberry Pi
* USB 麦克风或者免驱声卡搭配 3.5mm 接口麦克风
* 3.5mm 接口的扬声器

软件：
	
* Google 账户
* 科学上网环境

### 基本步骤
1. 硬件准备和网络连接
2. 配置测试声音设备
3. 配置开发者项目及相关设置
4. 安装 SDK 和示例项目
5. 注册你的硬件设备
6. 运行示例代码
7. 其他

## 步骤一：硬件准备和网络连接
1. 已经安装好系统并且搭建好科学上网环境的树莓派
2. 连接你的麦克风和扬声器到树莓派，麦克风你可以购买一个免驱声卡，连接你的 3.5 mm  麦克风，比如下面这样的，几块钱就能买到。树莓派有一个 3.5mm 的耳机插口，你的扬声器可以接入此接口。
	![](http://7xjsv3.com1.z0.glb.clouddn.com/15155927429534.jpg)

3. 将你的树莓派连接网络
4. ssh 远程连接你的树莓派

## 步骤二：配置测试声音设备
### 第一步：确认你的录音和播放设备正常工作
1. 在命令行输入以下命令，按下 `Ctrl+C` 来结束。

```
speaker-test -t wav
```

如果你没有听到任何声音，请检查你的扬声器和树莓派连接是否正常，或者扬声器是不是声音调的太小了。同时你可以通过输入以下命令
```
sudo raspi-config
```
选择
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155932467591.jpg)
然后选择
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155932800447.jpg)
将你的声音输出从耳机插口输出
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155933119751.jpg)

2. 录制一个简短的声音片段，如果你这时出现了问题，请转到下面的第二步。

```
arecord --format=S16_LE --duration=5 --rate=16000 --file-type=raw out.raw
```
3. 播放刚刚录制的音频来检查录制效果。

```
aplay --format=S16_LE --rate=16000 out.raw
```
调整你的录制和播放设备，通过下面这个命令：

```
alsamixer
```
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155937585522.jpg)

如果你的录制和播放设备工作正常，那么这一步的声音设备的配置就完成了，如果你出现了错误，那么可以继续下面的步骤尝试修复问题。

### 第二步：找到你的录制和播放设备。
通过下面这个命令来显示你的声音捕捉设备列表，找到你的麦克风，并记录下设备卡号和设备号。
```
arecord -l
```
![录音设备](http://7xjsv3.com1.z0.glb.clouddn.com/15155941154890.jpg)
如上图所示的 `card num`  是1，`device num` 为 0。

通过下面这个命令来显示你的声音播放设备列表，记录下你的 3.5 mm  接口的设备卡号和设备号，3.5mm 设备的特征是带有 `Analog` 或者 `bcm2835 ALSA` (不是 bcm2835 IEC958/HDMI) 字段。

```
aplay -l
```
![播放设备](http://7xjsv3.com1.z0.glb.clouddn.com/15155944050157.jpg)

### 第三步
在你的 /home/pi 目录下创建一个名为 .asoundrc 文件，可以直接输入

```
cd ~
sudo nano .asoundrc
```
粘贴以下内容到你的终端编辑器内：

```
pcm.!default {
  type asym
  capture.pcm "mic"
  playback.pcm "speaker"
}
pcm.mic {
  type plug
  slave {
    pcm "hw:<card number>,<device number>"
  }
}
pcm.speaker {
  type plug
  slave {
    pcm "hw:<card number>,<device number>"
  }
}
```
其中的两处 `<card number>,<device number>` ，分别是刚刚让你记录的麦克风和扬声器的设备卡号和设备号，比如刚刚我展示的图片，我的是内容是
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155947400239.jpg)

如果你有一个 HDMI 的声音输出并且连接了耳机接口，那么请按照刚刚上面 `sudo raspi-config` 的设置将你的声音输出强制从耳机接口输出。

然后重复步骤一看看是不是依然存在问题，如果依然存在问题，可以尝试更换你的声音录制和播放设备。

## 步骤三：配置开发者项目及相关设置
Google 的开发者项目允许你的设备能够访问 `Google Assistant API`，为了能够访问这个 API，需要进行以下的配置工作。
首先，进入 Google 的 Cloud Platform Console，在你的[项目页面](https://console.cloud.google.com/project)，创建一个项目或者选一个已经存在的项目。
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155950828092.jpg)

然后允许你刚刚所选的项目使用 `Google Assistant API`，点击以下链接[开通 Google Assistant API](https://console.developers.google.com/apis/api/embeddedassistant.googleapis.com/overview)
点击启用即可。
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155953311656.jpg)

然后通过下面的步骤创建一个客户端认证 ID 凭证：
1. [创建客户端 ID](https://console.developers.google.com/apis/credentials/oauthclient)
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155954695217.jpg)
2. 输入产品名称及其他可选信息，点击保存即可。
![输入名称](http://7xjsv3.com1.z0.glb.clouddn.com/15155955232716.jpg)
3. 应用类型选择其他，并输入一个名称。
![类型](http://7xjsv3.com1.z0.glb.clouddn.com/15155956528728.jpg)
4. 点击创建后弹出一个提醒的窗口，这里可以直接点击确定关闭它。
5. 此时出现这样的界面：
![列表](http://7xjsv3.com1.z0.glb.clouddn.com/15155957656082.jpg)
点在下载图标，下载这个 json 文件。名字类似与`client_secret_client-id.json`

最后上面下载的文件必须放到 pi 用户的 Downloads 目录下面以授权使 `Google Assistant SDK` 示例项目正常使用，不要重命名这个文件。
可以通过以下方法来将刚刚下载的文件传到 Downloads 目录。
1. 在你的电脑新打开一个终端，**不要 ssh 连接 pi**，输入以下命令：

```
scp ~/Downloads/client_secret_具体你的文件名.json pi@raspberry-pi-ip-address:/home/pi/
```
然后输入密码。
2. 如果你是 Windows，你可以使用 FileZilla 等 ftp 工具，如下:
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155961850035.jpg)
打开 Downloads 目录，将你的文件拖入即可。
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155962460662.jpg)

为了使用 Google Assistant 你必须要共享一些活动数据给 Google，授予 `Google Assistant` 一定的权限，否则你运行项目的时候，她会一直跟你说，她需要权限才能跟你说话。
打开[活动控制页面](https://myaccount.google.com/activitycontrols)
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155964264654.jpg)
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155964345681.jpg)
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155964424810.jpg)

## 步骤四：安装 SDK 和示例项目
Google 推荐在 Python 的虚拟环境运行该项目，避免对系统环境造成影响，具体运行以下命令：

```
sudo apt-get update
sudo apt-get install python3-dev python3-venv 
python3 -m venv env
env/bin/python -m pip install --upgrade pip setuptools
source env/bin/activate
```
上面的最后一条命令是激活虚拟环境，然后安装依赖软件包：

```
sudo apt-get install portaudio19-dev libffi-dev libssl-dev
```

使用 pip 来安装最新版本的依赖包：

```
python -m pip install --upgrade google-assistant-library
python -m pip install --upgrade google-assistant-sdk[samples]
```

生成授权资格
首先安装及升级授权工具
```
python -m pip install --upgrade google-auth-oauthlib[tool]
```
然后在命令行输入一下命令

```
google-oauthlib-tool --scope https://www.googleapis.com/auth/assistant-sdk-prototype \
          --save --headless --client-secrets /home/pi/Downloads/client_secret_你的文件名.json
```
然后你应该会在命令行中看到一个网址
`Please visit this URL to authorize this application: https://...`

将这个网址完整的 copy 到浏览器中，登陆你的 Google 账号，然后点击授权允许，你会在浏览器中看到一行代码，类似`4/XXXX`，将这行代码 copy 到命令行`Enter the authorization code:`的后面
如果授权成功，那么你会在类似下面的响应。
`credentials saved: /path/to/.config/google-oauthlib-tool/credentials.json`

## 步骤五：注册你的硬件设备
使用 Google 的注册工具，首先设备名最好是数字和字母的组合，首字段必须是数字或字母。
使用下面的命令格式注册：

```
googlesamples-assistant-devicetool register-model --manufacturer 生产者 \
          --product-name 产品名 [--description my-product-description] \
          --type device-type [--trait 支持的特性] --model 我的设备
```
上面的我的设备这个必须是一个唯一的名字，所以你可以使用你的项目 ID 作为前缀，下面的是 Google 的一个示例命令：

```
googlesamples-assistant-devicetool register-model --manufacturer "Assistant SDK developer" \
          --product-name "Assistant SDK light" --type LIGHT --model my-model
```
然后使用以下命令向服务器查询你刚刚创建的设备：

```
googlesamples-assistant-devicetool get --model my-model
```
比如下面
![](http://7xjsv3.com1.z0.glb.clouddn.com/15155975614463.jpg)

## 步骤六：运行示例代码
现在你就可以准备运行示例项目来和你的树莓派对话了，首先运行下面的命令：

```
googlesamples-assistant-hotword --project_id 你的项目 iD --device_model_id  上一步创建的设备
```
然后对着你的麦克风可以说话询问了，注意要是英文。激活的关键词是 `Ok Google` 或 `Hey Google`，你可以在终端中看到对话开始和结束的提示。

可以试着问问她你所在的城市的天气，让她唱首歌，让她叫你起床等等。

## 步骤七： 其他
 你还可以增加一个 LED 灯来为你的对话添加一些特性，通过 LED 灯提示对话开始等，也可以扩展这个项目及你的树莓派。具体的可以参考[扩展 Google Assistant](https://developers.google.com/assistant/sdk/guides/library/python/extend/install-hardware)。
 
 

以上就是变身 Google Home 的基本过程，下一步，我将我的树莓派变身成为智能家居中枢，让 Siri 为我开关灯！

