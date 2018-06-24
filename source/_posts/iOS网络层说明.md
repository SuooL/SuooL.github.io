title: iOS网络层说明
date: 2016-11-30 16:44:02
tags: [iOS,组件文档]
category: [技术]
---
# iOS 网络层文档

## 说明

iOS 的网络层使用 [YTKNetwork](https://github.com/yuantiku/YTKNetwork)作为网络层底层架构，在  [YTKNetwork](https://github.com/yuantiku/YTKNetwork)的`YTKRequest`类和具体的业务请求层之间架设了一个中间业务类`LSBaseRequest`，所有具体的 API 请求都继承于此类。
 <!--more-->
 
关于 [YTKNetwork](https://github.com/yuantiku/YTKNetwork) 
> YTKNetwork主要用于请求的发送及回调处理，YTKNetwork 的基本的思想是把每一个网络请求封装成对象。使用 YTKNetwork，每一个请求都需要继承 YTKRequest 类，通过覆盖父类的一些方法来构造指定的网络请求。

这里为了避免我们具体业务类的相关回调逻辑与 YTKNetwork 耦合过高，所以封装了`LSBaseRequest`来处理请求回调的公共逻辑。

下面来一一说明。

## LSBaseRequest 类的设计

![iOS Network.png](http://7xjsv3.com1.z0.glb.clouddn.com/2016113096998iOS Network.png)

继承关系及中间层的 `BaseRequest`  的设计如上图所示。

### 基本的原理和字段说明：
首先一个请求的回调都是直接一个 JSON 的字符串，网络层的底层可以将这个字符串转化为`NSDictionary`类型的数据，这个是请求回调的第一次处理得到的数据。处理的具体过程如下：

```
// 此处是在网络请求完成的回调方法内拿到responseString的回调字符串的处理
NSData *data = [self.responseString dataUsingEncoding:NSUTF8StringEncoding];
if (data) {
	NSMutableDictionary *dictionary = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
  // ...
}else{
  DDLogError(@"返回内容为空.");
  // ... 
}
```

所以，如果 API 的请求没有特殊的要求，可能直接拿到这个 `data`数据来提供给 ViewController 使用。

但是，一般的请求并不会简单的直接要求`NSMutableDictionary`的 `data`数据，ViewController 希望能够直接拿到其需要的数据类型。而展示数据一般也只需要两种类型，分别是展示一个列表数据和展示一个实体模型的数据。所以业务请求 API 能够自己处理回调数据的“深加工”是View 和 ViewController希望看到的。因此，这里直接给 `BaseRequest`添加了两个属性，分别是泛型的回调列表数据和泛型的实体模型数据，而`NSDictionary`类型的`resultDic`属性是第一次处理的结果。

根据以上的需求，具体的业务请求的 API 都会具有三种附加的状态属性，分别是该请求是否需要登录，该请求是否需要一个列表的回调数据，该请求是否需要一个实体模型的回调数据。在初始化具体的 API 请求的时候就会对这三个属性进行初始化。

具体的对应数属性是：

```
@property (nonatomic, assign) BOOL isLogin;  // 是否需登录
@property (nonatomic, assign) BOOL isRetAry; // 是否需要返回数组
@property (nonatomic, assign) BOOL isRetObj; // 是否需要返回实体 model
```

所以`BaseRequest`的对外暴露的方法也只有三个，此处甚至可以减少到两个，即是所有的 API 请求自己处理自己的初始化，而不是像上面的使用中间类的初始化方法。

而 API 的初始化方法中，需要传入该 API 请求需要的参数及其相关的状态属性。请求参数会根据请求的登录状态属性来决定是否添加`AccessToken 参数`。

在具体的发送 API 请求的时候，会根据以上三个状态属性来确定如何发送请求，具体的处理逻辑如下：

```
- (void)startRequest{
    sendCount ++; // 请求重发的次数
    BOOL isOnline = [[LSUserDefaults sharedInstance] isOnLine];
    if (_isLogin && isOnline) {  // 需要登录且目前用户已经登录
        [self start];
    }
    
    if(_isLogin && !isOnline) { // 需要登录但目前用户并没有登录
        self.failure(@{@"code" : NoLogin});
    }
    
    if (!_isLogin) {         // 不需要登录，直接登录
        [self start];
    }
}
```

请求完成之后的公共逻辑处理在`requestCompleteFilter`和`requestFailedFilter`回调方法中实现，这两个回调方法是 YTKNetwork 中暴露出来给请求结果回调处理使用的。

而一旦请求发送成功，获得回调数据之后，回调数据统一处理逻辑会根据 API 初始化携带的状态来加工回调数据，具体如下：

```
- (void)requestCompleteFilter{
    NSData *data = [self.responseString dataUsingEncoding:NSUTF8StringEncoding];
    if (data) {
		  // 初次加工数据
        NSMutableDictionary *dictionary = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
        NSEnumerator *enumerator = [dictionary keyEnumerator];
        NSString *key = @"code";
        int code = [dictionary[key] intValue];
        BOOL isExist = NO;
        while ((key = [enumerator nextObject])) {
            isExist = YES;
        }
        if (isExist) {
            switch (code) {
                case 200:
                    if (_isRetAry) { // 如果要求返回的是数组，则在具体的 API 中再次深加工数据，并将加工之后的数据回调
                        self.successArray(self.resultData);
                    }else if(_isRetObj) // 如果要求返回的是实体，则在具体的 API 中再次深加工数据，并将加工之后的数据回调
                        self.successObject(self.model);
                    else // 如果要求返回没有额外的要求，则在数据回调中直接将初加工数据回调
                        self.success(dictionary);
                    break;
					  // 请求成功但是获取的不是预期的成功数据
                case 801:{
                    // 访问令牌失效，重新获取
                    [self requestTokenWithRetArray:_isRetAry retObj:_isRetObj];
                }
                    break;
                case 802:
                    [[PMStore store] removeUserDataAndLogOff];
                    self.failure(dictionary);
				  // 其他情况
            }
        }
        else{
            self.failure(@{@"msg" : @"服务暂时不可用，请稍后重试"});
        }
    }else{
		  // 请求成功但是没有获取到数据
        DDLogWarn(@"self.success(nil)");
        self.failure(@{@"msg" : @"服务暂时不可用，请稍后重试"});;
    }

}
```

具体的深加工过程，在具体的 API 类中再做介绍。

请求发送失败的公共处理逻辑：

```
- (void)requestFailedFilter {
    NSData *data = [self.responseString dataUsingEncoding:NSUTF8StringEncoding];
    if (data) {
        NSMutableDictionary *dictionary = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
        self.failure(dictionary);
    }else{
        DDLogError(@"返回内容为空.");
        self.failure(@{@"msg" : @"服务暂时不可用，请稍后重试"});
    }
}
```

此处在请求成功发送并获取回调之后，会出现一种特殊的情况（上面 Code 为801 的情况）：如果该请求需要登录状态，即是会携带`AccessToken`来发送请求，但是请求的结果是服务端提示`AccessToken`过期，此时不能立即回调该结果给 API 的调用者，而是要发送一个`LSServerTokenAPI`请求来尝试跟服务端请求换取新的`AccessToken`，如果服务端成功的返回了新的`AccessToken`，则应该让该请求重新携带新的`AccessToken`重发请求数据。如果没有获取到新的`AccessToken`，则此时应该失败回调给调用者，告诉调用者该用户已经掉线，需要重新登录才能继续请求。
此处的具体处理逻辑代码如下：

```
- (void)requestTokenWithRetArray:(BOOL)isAry retObj:(BOOL)isObj{
    NSMutableDictionary *requrestDic = [NSMutableDictionary dictionaryWithDictionary:self.requestArgument];
    // accessToken 过期
    LSServerTokenAPI *accessAPI = [[LSServerTokenAPI alloc]init];
    [accessAPI startWithCompletionBlockWithSuccess:^(YTKBaseRequest *request) {   // 此处是直接使用的 YTKRequest 的回调方法
        // you can use self here, retain cycle won't happen
        NSLog(@"succeed");
        NSData *data = [request.responseString dataUsingEncoding:NSUTF8StringEncoding];
        NSMutableDictionary *dictionary = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
        NSDictionary *dic = (NSDictionary *)dictionary[@"data"][@"result"];
        HNUserModel *user = [HNUserModel mj_objectWithKeyValues:dic];
        if (user.accessToken) {   // 如果换取到新的 Token
            [[PMStore store]saveDataWithUserInfo:user];
            requrestDic[@"accessToken"] = user.accessToken;
            self.baseRequestArgument = requrestDic;
            if (sendCount > 3) {
                self.failure(@{@"msg" : @"重发系统错误，请稍后重试"});  // 重发最多三次，多余三次的直接回调失败
                return;
            }else{
               [self startRequest];  // 重发该请求
            }
        }else{
            [[PMStore store] removeUserDataAndLogOff];
            self.failure(@{@"code" : NoLogin});  // 没有获取到新的 Token，回调数据登录失效
        }
    } failure:^(YTKBaseRequest *request) {
        // you can use self here, retain cycle won't happen
        NSLog(@"failed");
        NSData *data = [request.responseString dataUsingEncoding:NSUTF8StringEncoding];
        if (data) {
            NSMutableDictionary *dictionary = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
            DDLogError(@"发送请求更新 accessToken 失败，返回内容：\n%@",dictionary);
            self.failure(dictionary);
        }else{
            self.failure(@{@"msg" : @"刷新 Token 系统错误，请稍后重试"});
        }
    }];
}
```

这里需要注意的是为了避免无限的循环调用，`LSServerTokenAPI `本身不是`LSBaseRequest`类型的，而是和`LSBaseRequest`一样都是直接继承自`YTKRequest`的，因此此处的`LSServerTokenAPI `请求的回调不会再次经过`LSBaseRequest` 在`requestCompleteFilter`和`requestFailedFilter`回调方法中的公共逻辑，而是直接调用的`startWithCompletionBlockWithSuccess:(YTKRequestCompletionBlock)success failure:(YTKRequestCompletionBlock)failure`方法处理数据回调数据。

## operationAPI 类的设计
具体的业务类非常的简单，因此非常多的回调处理逻辑都在`LSBaseRequest`内部完成处理了。
如果需要创建一个新的业务请求 API，只要创建一个继承自`LSBaseRequest`的 API 类，在类的头文件声明中，除特殊情况外，不需要做任何的处理，比如一个典型的请求个人信息的 API 接口声明如下：

```
@interface HNUserInfoAPI : LSBaseRequest
// 此处一般不需要声明任何额外的属性和方法
@end
```

  而具体业务的 API 实现，也只需要配置该 API 的 URL，API 的请求方式即可。
### 额外的数据逻辑处理需求
如果该 API 请求对于回调数据有特殊的加工需求：

* 需要一个列表数据，则会增加下面这些处理逻辑，实现`LSBaseRequest`的`resultData`的获取方法 ，比如请求交易记录的 API ：

```
// 此处是实现LSBaseRequest的属性
- (NSArray *)resultData{
    return [self handleData:self.resultDic];
}
// 对请求的回调数据进行深加工，JSON 转列表
- (NSArray *)handleData:(NSDictionary *)dictionary{
    NSMutableArray *resultArray = [NSMutableArray new];
    if ([dictionary count]) {
        NSDictionary *resultDic = (NSDictionary *)[[dictionary objectForKey:@"data"] objectForKey:@"result"];
        [resultArray addObjectsFromArray:[HNTradeRecordModel mj_objectArrayWithKeyValuesArray:resultDic]];
    }
    return resultArray;
}
```

* 需要一个模型数据，则会增加下面这些处理逻辑， 实现`LSBaseRequest`的`model`的获取方法 ，比如请求个人信息的 API ：

```
- (id)model{
    return [self handleData:self.resultDic];
}
// 对请求的数据进行深加工，JSON 转模型
- (HNUserModel *)handleData:(NSDictionary *)dictionary{
    if ([dictionary count]) {
        [[PMStore store]handleUserDictionary:dictionary resiterJpush:NO];
        NSDictionary *dic = [[dictionary objectForKey:@"data"] objectForKey:@"result"];
        HNUserModel *model = [HNUserModel mj_objectWithKeyValues:dic];
        return model;
    }else
        return NULL;
}
```

* 如果没有额外的数据需求，则上面这些逻辑都不需要。
	
## 具体业务流程说明：

如果需要新增一个新的业务请求类：
1. 新建一个新的基于`LSBaseRequest`类
2. 完成 API 类的基本配置，包括 URL 和请求方式的配置
3. 如果 该 API 类有更多的数据处理需求，按照上面所写的方式来处理

在 ViewController 或者 View 中初始化 API，一个实例如下：

```
- (void)loadDataFromService{
    // API 请求的参数
    NSMutableDictionary *dic = [[NSMutableDictionary alloc]initWithDictionary:@{@"offset":self.offset,
                                                                                @"limit":@"10"}];
	  // 初始化 API 请求，传入需要的参数及状态信息
    HNInquiryListAPI *request = [[HNInquiryListAPI alloc]initWithArgumentValueDictionary:dic isLogin:YES isRetAry:YES isRetObj:NO];
    [request setSuccessArray:^(NSArray *resultData) {
        // 初始化的要求 API 能够对回调数据进行深加工——返回列表，所以这里直接设置列表的 block 回调处理逻辑。
    }];
    [request setFailure:^(NSDictionary *dictionary) {
		  // 请求失败的回调
    }];
    [request startRequest];  // 发送请求
}
```

以上就是完整的网络层架构的说明及具体业务处理的流程。


## 其他说明——图片及文件上传请求
上传图片的 API 不同于其他的业务处理 API，它不需要处理一堆的业务逻辑，而且目前也不需要登录等状态，因此网络层将上传图片的 API 直接独立了出来，直接继承与`YTKRequest`。目前的图片 API 如下，已经封装完成，图片微服务不做大的重构，此 API 也不需要做修改：

声明：

```
#import <YTKRequest.h>

@interface HNUploadImageApi : YTKRequest

- (id)initWithImage:(NSArray *)image;  // 需要上传的图片文件数组

@property (nonatomic, strong) id baseRequestArgument;  // 请求的参数

@end
```

实现部分 

```
#import "HNUploadImageApi.h"

@implementation HNUploadImageApi{
    NSArray *_image;
}

- (id)initWithImage:(NSArray *)image {
    self = [super init];
    if (self) {
        _image = image;  // 初始化图片数组的属性
    } 
    return self;
}

- (YTKRequestMethod)requestMethod {
    return YTKRequestMethodPOST;
}

// 构建请求参数
- (void)setBaseRequestArgument:(id)baseRequestArgument{
    if (baseRequestArgument) {
        _baseRequestArgument = baseRequestArgument;
    }
}

- (id)requestArgument {
    if (_baseRequestArgument) {
        return _baseRequestArgument;
    }
    return nil;
}

- (NSString *)requestUrl {
    return UploadImageAPI;  // 上传 URL 地址
}

// 请求的 body 的构造
- (AFConstructingBlock)constructingBodyBlock {
    return ^(id<AFMultipartFormData> formData) {
        for (int i = 0;i < _image.count; i++){
            NSData *data = UIImageJPEGRepresentation(_image[i], 0.5);
            if ((float)data.length/1024 > 1000) {
                data = UIImageJPEGRepresentation(_image[i], 1024*1000.0/(float)data.length);
            }
            NSString *name = [NSString stringWithFormat:@"image%d.png",i];
            NSString *type = @"image/jpeg";
            [formData appendPartWithFileData:data name:@"image" fileName:name mimeType:type];
        }
    };
}

@end
```

具体构建及发送请求的流程如下，以上传头像为例：

```
HNUploadImageApi *api = [[HNUploadImageApi alloc] initWithImage:@[scaleImage]]; // 初始化请求，并传递需要上传的图片数组
api.baseRequestArgument = @{@"type":@"0"};   // 构造请求的参数
[api startWithCompletionBlockWithSuccess:^(YTKBaseRequest *request){
    NSData *data = [request.responseString dataUsingEncoding:NSUTF8StringEncoding];
    if (data) {
        NSMutableDictionary *dic = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
		  // 初次处理上传成功后的数据回调
    }
} failure:^(YTKBaseRequest *request){
    // 上传失败后是回调
}];
```

