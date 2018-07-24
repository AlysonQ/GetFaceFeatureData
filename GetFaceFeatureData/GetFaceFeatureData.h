//
//  GetFaceFeatureData.h
//  GetFaceFeatureData
//
//  Created by Alyson Chen on 2018/7/4.
//  Copyright © 2018年 Alyson Chen. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import <UIKit/UIKit.h>

@interface GetFaceFeatureData : NSObject
+(NSArray<NSNumber *> *)LoadImageandGetFaceFeature:(UIImage*)image;
+(NSString*)transformToString:(NSData*)data;
+ (NSString*)transformToDecimalWithString:(NSString*)string;
@end
