//
//  GetFaceFeatureData.m
//  GetFaceFeatureData
//
//  Created by Alyson Chen on 2018/7/4.
//  Copyright © 2018年 Alyson Chen. All rights reserved.
//

#import "GetFaceFeatureData.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_utils.h"
using tensorflow::uint8;
namespace {
    class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
    public:
        explicit IfstreamInputStream(const std::string& file_name)
        : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
        ~IfstreamInputStream() { ifs_.close(); }
        
        int Read(void* buffer, int size) {
            if (!ifs_) {
                return -1;
            }
            ifs_.read(static_cast<char*>(buffer), size);
            return (int)ifs_.gcount();
        }
        
    private:
        std::ifstream ifs_;
    };
}  // namespace
bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
    ::google::protobuf::io::CopyingInputStreamAdaptor stream(
    new IfstreamInputStream(file_name));
    stream.SetOwnsCopyingStream(true);
    // TODO(jiayq): the following coded stream is for debugging purposes to allow
    // one to parse arbitrarily large messages for MessageLite. One most likely
    // doesn't want to put protobufs larger than 64MB on Android, so we should
    // eventually remove this and quit loud when a large protobuf is passed in.
    ::google::protobuf::io::CodedInputStream coded_stream(&stream);
    // Total bytes hard limit / warning limit are set to 1GB and 512MB
    // respectively.
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
        << [extension UTF8String] << "' in bundle.";
    }
    return file_path;
}
@implementation GetFaceFeatureData

+(NSArray<NSNumber *> *)LoadImageandGetFaceFeature:(UIImage*)image{
    NSMutableArray *dataArray = [NSMutableArray array];
    
    //Init TensorFlow setting----START
    tensorflow::SessionOptions options;
//    //Alyson ++ -- in order to fix session run crash  20180724
//
//    options.config.set_intra_op_parallelism_threads(1);
//    options.config.set_inter_op_parallelism_threads(1);
//    options.config.add_session_inter_op_thread_pool()->set_num_threads(1);
//    options.config.set_use_per_session_threads(false);
//    //Alyson -- -- in order to fix session run crash  20180724
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
    if (!session_status.ok()) {
        std::string status_string = session_status.ToString();
        LOG(INFO) << "[Alyson log] Session create failed - "<< status_string.c_str();
        return nil;
    }
    std::unique_ptr<tensorflow::Session> session(session_pointer);
    LOG(INFO) << "[Alyson log] Session created.";
    
    tensorflow::GraphDef tensorflow_graph;
    LOG(INFO) << "[Alyson log]Graph created.";
    
    NSString* network_path = FilePathForResourceName(@"20180713-185347", @"pb");
    PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);
    
    LOG(INFO) << "[Alyson log] Creating session.";
    tensorflow::Status s = session->Create(tensorflow_graph);
    if (!s.ok()) {
        LOG(INFO) << "[Alyson log] Could not create TensorFlow Graph: " << s;
        return nil;
    }
    //Init TensorFlow setting----END
    
        const int img_dim = 160;
        const int wanted_channels = 3; //3 for RGB image
        const float input_mean = 128;
        const float input_std = 128;
        //Resize Image
    /*
        CGSize newSize = CGSizeMake(160, 160);
        UIGraphicsBeginImageContextWithOptions(newSize, NO, 0.0);
        [image drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
        UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
        UIGraphicsEndImageContext();*/
        NSData *imageData = UIImagePNGRepresentation(image); //convert image into .png format.
        //UInt8 buff_str[img_dim*img_dim*wanted_channels];
        tensorflow::Tensor image_tensor(
                                        tensorflow::DT_FLOAT,
                                        tensorflow::TensorShape({1,160,160,3}));
        auto image_tensor_data = image_tensor.tensor<float,4>();
        /*
        for (int index = 0; index < imageData.length; index++) {
            NSData *subData = [imageData subdataWithRange:NSMakeRange(index, 1)];
            NSString *dataLenstring = [GetFaceFeatureData transformToString:subData];
            int dataForInt = [[GetFaceFeatureData transformToDecimalWithString:dataLenstring]intValue];
            memcpy(buff_str,[dataLenstring UTF8String], [dataLenstring length]+1);
        }*/
    std::vector<tensorflow::uint8> image_data = LoadImageFromFile(imageData);
        tensorflow::uint8* in = image_data.data() ;
    float* out = image_tensor_data.data();
    int image_height = image.size.height;
    int image_width = image.size.width;
    int image_channels = 4;
    for (int y = 0; y < img_dim; ++y) {
        const int in_y = (y * image_height) / img_dim;
        tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * img_dim * wanted_channels);
        for (int x = 0; x < img_dim; ++x) {
            const int in_x = (x * image_width) / img_dim;
            tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
        
        //Setting input_layers and out_layer name
        std::string bool_input = "phase_train";
        std::string image_input = "input";
        std::string output_name = "embeddings";
        std::vector<tensorflow::Tensor> outputs;
        
        //Feed bool_input for not training model
        tensorflow::Tensor bool_input_tensor(tensorflow::DT_BOOL,tensorflow::TensorShape());
        auto bool_data = bool_input_tensor.scalar<bool>()();
        bool_data=false;
        
        //Run facenet model
        tensorflow::Status run_status = session->Run ({{image_input, image_tensor},{bool_input,bool_input_tensor}},
                                                     {output_name}, {}, &outputs);

    if (!run_status.ok()) {
        LOG(INFO) << "Running model failed: " << run_status;
        tensorflow::LogAllRegisteredKernels();
        session->Close();
        return nil;
    }
        //Get Face Feature data
        tensorflow::Tensor *output = &outputs[0];
        
        auto FaceFeature = output->flat<float>();
        //std::cout << "===== FaceFeature Data =====" << std::endl;
        if(run_status.ok()){
            for (int index = 0; index < FaceFeature.size(); index += 1) {
                const float FaceFeatureValue = FaceFeature(index);
                [dataArray addObject:[NSNumber numberWithFloat:FaceFeatureValue]];
            }
            LOG(INFO) << "[Alyson log] Get Face Feature Data Done !! ";
        }else{
            for (int index = 0; index < FaceFeature.size(); index += 1) {
                [dataArray addObject:[NSNumber numberWithFloat:1]];
            }
            LOG(INFO) << "[Alyson log] Get Face Feature Data ERROR -- ";
        }
        //Get face feature --------------END
    
    return dataArray;
}

+(NSString*)transformToString:(NSData*)data{
    Byte *byte = (Byte* )[data bytes];
    NSString *hexStr=@"";
        if(data.length>0){
        for(int i=0;i<[data length];i++){
            NSString *newHexStr = [NSString stringWithFormat:@"%X",byte[i]&0xFF]; ///16进制数
            if([newHexStr length]==1)
                hexStr = [NSString stringWithFormat:@"%@0%@",hexStr,newHexStr];
            else
                hexStr = [NSString stringWithFormat:@"%@%@",hexStr,newHexStr];
        }
        return hexStr;
    }
    return nil;
}

+ (NSString*)transformToDecimalWithString:(NSString*)string{
    int int_c = 0;
    int charCount = (int)[string length];
    for (int i=0; i< charCount; i++) {
        int int_ch;
        char tempChar = [string characterAtIndex:i];
        if(tempChar >= '0' && tempChar <='9')
            int_ch = (tempChar-48)*powf(16,(charCount-i-1)) ;   //// 0 的Ascll - 48
        else if(tempChar >= 'A' && tempChar <='F')
            int_ch = (tempChar-55)*powf(16,(charCount-i-1)); //// A 的Ascll - 65
        else
            int_ch = (tempChar-87)*powf(16,(charCount-i-1)); //// a 的Ascll - 97
        int_c = int_c+int_ch;
    }
    if (int_c<10) {
        return [NSString stringWithFormat:@"0%d",int_c];
    }else{
        return [NSString stringWithFormat:@"%d",int_c];
    }
}
std::vector<uint8> LoadImageFromFile(NSData * file_data) {
    
    
    NSUInteger len = [file_data length];
    Byte *byteData = (Byte*)malloc(len);
    memcpy(byteData, [file_data bytes], len);
    
    CFDataRef file_data_ref = CFDataCreateWithBytesNoCopy(NULL, byteData,
                                                          file_data.length,
                                                          kCFAllocatorNull);
    CGDataProviderRef image_provider =
    CGDataProviderCreateWithCFData(file_data_ref);
    
    CGImageRef image =  CGImageCreateWithPNGDataProvider(image_provider, NULL, true,
                                                      kCGRenderingIntentDefault); ;
    
    const int width = (int)CGImageGetWidth(image);
    const int height = (int)CGImageGetHeight(image);
    const int channels = 4;
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    const int bytes_per_row = (width * channels);
    const int bytes_in_image = (bytes_per_row * height);
    std::vector<uint8> result(bytes_in_image);
    const int bits_per_component = 8;
    CGContextRef context = CGBitmapContextCreate(result.data(), width, height,
                                                 bits_per_component, bytes_per_row, color_space,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(color_space);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CFRelease(image);
    CFRelease(image_provider);
    CFRelease(file_data_ref);
    
    return result;
}
@end
