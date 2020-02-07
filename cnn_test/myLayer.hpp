#ifndef __MYLAYER_HPP__
#define __MYLAYER_HPP__

struct Param
{
	//卷积层超参数 
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;
	//池化层超参数
	int pool_stride;
	int pool_width;
	int pool_height;
		//全连接层超参数（该层神经元的个数）
	int fc_kernel;
};

#endif