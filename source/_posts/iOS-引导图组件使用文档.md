title: iOS 引导图组件使用文档
date: 2016-12-05 10:48:04
tags: [iOS,组件文档]
category: [技术]
---
# iOS 引导图组件使用文档
## 说明
### 实际效果

* 引导图实际显示效果截图如下:
<!--more-->

![引导图示例.png](http://7xjsv3.com1.z0.glb.clouddn.com/2016120576244Simulator Screen Shot Dec 5, 2016, 9.43.40 AM.png)

* 引导图的动态显示效果如下视频：
[引导图 Live Video](https://appetize.io/embed/29xabc89e2u0j8uh3y0dkkk0yg?device=iphone5s&scale=75&orientation=portrait&osVersion=9.3)

### 使用场景

当 APP 第一次安装启动或者大版本更新第一次启动的时候，运营有新的引导图介绍 APP 的重点功能或更新的，需要使用引导图完成。

### 基本介绍

本引导图组件基于 [ZWIntroductionViewController](https://github.com/squarezw/ZWIntroductionViewController)修改。

只要设置引导图组件的 cover images ，初始化后将其加入到APPDelegate 的window即可， 对 APP 侵入性小，使用灵活。

## 使用文档
* 首先是滑动的需求：
在组件的`ZWIntroductionViewController`类中的`- (void)scrollViewDidScroll:(UIScrollView *)scrollView`方法中几行关键代码：

```
    // 当滑动到最后一个的时候,禁止左滑动,只能通过点击按钮来去除引导图
    CGFloat x = (_pageControl.numberOfPages-1) * YYScreenSize().width;
    
    if (scrollView.contentOffset.x>x) {
        [scrollView setContentOffset:CGPointMake(x, scrollView.contentOffset.y) animated:NO];
    }
    
    // 第一个的时候不能右滑动
    if (scrollView.contentOffset.x<=0.0) {
        [scrollView setContentOffset:CGPointMake(0.0, scrollView.contentOffset.y) animated:NO];
    }
```

以上的代码是控制在引导图第一页的时候不能向右滑动，在最后一页的时候不能通过继续左滑结束引导，只能通过点击按钮进入 APP 中。
（如果没有以上需求，将代码注释即可。）
* 其次是指示器颜色和进入按钮的需求：
如果对于`pageControl`的指示器颜色有特殊的需求，比如需要指示器的颜色及进入 APP 的按钮样式，需要修改`ZWIntroductionViewController`类中的`- (void)viewDidLoad`方法，指示器颜色修改具体的代码如下：

```
self.pageControl.pageIndicatorTintColor = [UIColor grayColor];
self.pageControl.currentPageIndicatorTintColor = GlobalColor;
```

按钮相关的修改如下：

```
[self.enterButton setTitle:NSLocalizedString(@"立即体验", nil) forState:UIControlStateNormal];
[self.enterButton setTitleColor:GlobalColor forState:UIControlStateNormal];
self.enterButton.layer.borderWidth = 1.0;
self.enterButton.layer.cornerRadius = 20.0;
self.enterButton.layer.borderColor = GlobalColor.CGColor;
```

* 最后是使用的规范：
下面是一个使用示例：

```
- (void)isFirstLaunch{
    if(![[NSUserDefaults standardUserDefaults] boolForKey:@"firstStart_V2"]){
		  // 如果是第一次启动，设置对应的版本标示
        [[LSUserDefaults sharedInstance]setNewVeisonCode:@"1"];
        [[NSUserDefaults standardUserDefaults] setBool:YES forKey:@"firstStart_V2"];
		  // 设置引导图的图片
        NSArray *coverImageNames = @[@"leadPage_1", @"leadPage_2", @"leadPage_3",@"leadPage_4"];
		  // 初始化引导图组件
        self.introductionView = [[ZWIntroductionViewController alloc] initWithCoverImageNames:coverImageNames backgroundImageNames:nil];
		  // 添加组件
        [self.window addSubview:self.introductionView.view];
        __weak AppDelegate *weakSelf = self;
		  // 引导完成后的回调
        self.introductionView.didSelectedEnter = ^() {
            [weakSelf.introductionView.view removeFromSuperview];
            weakSelf.introductionView = nil;
        };
    }
}
```

整个使用的过程如上代码所示。

*以上就是引导图组件的文档说明*