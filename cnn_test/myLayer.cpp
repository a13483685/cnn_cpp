#ifndef __MYLAYER_HPP__
#define __MYLAYER_HPP__
struct Param
{
	//1.卷积层超参数
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	//2.池化层超参数
	int pool_stride;
	int pool_width;
	int pool_height;
	//3.全连接层
	int fc_kernels;
};


#endif