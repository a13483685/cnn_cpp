#ifndef __MYLAYER_HPP__
#define __MYLAYER_HPP__

struct Param
{
	//����㳬���� 
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;
	//�ػ��㳬����
	int pool_stride;
	int pool_width;
	int pool_height;
		//ȫ���Ӳ㳬�������ò���Ԫ�ĸ�����
	int fc_kernel;
};

#endif