#ifndef __MYLAYER_HPP__
#define __MYLAYER_HPP__
struct Param
{
	//1.����㳬����
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	//2.�ػ��㳬����
	int pool_stride;
	int pool_width;
	int pool_height;
	//3.ȫ���Ӳ�
	int fc_kernels;
};


#endif