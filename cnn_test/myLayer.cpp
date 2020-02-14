#include "myLayer.hpp"

using namespace std;
void ConvLayer::init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param)
{
	//F,C,H,W
	int tW = param.conv_width;
	int tH = param.conv_height;
	int tF = param.conv_kernels;
	int tC = inShape[1];
	if (!data[1])
	{
		data[1].reset(new Blob(tF, tC, tW, tH, TRANDN));
		(*data[1]) *= 1e-2;
		(*data[1]).print("data is :");
	}

	if (!data[2])
	{
		data[2].reset(new Blob(tF, 1, 1, 1, TRANDN));
		(*data[2]) *= 1e-2;
		(*data[2]).print("data is :");
	}
	std::cout << "ConvLayer::init_layer" << endl;
	return;
}

void PoolLayer::init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param)
{
	std::cout << "PoolLayer::init_layer ..... OK!!!"<< endl;
	return;
}

void FcLayer::init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param)
{
	//1.��ȡȫ���Ӻ˳ߴ�(F,C,H,W)
	int tF = param.fc_kernel;
	int tC = inShape[1];
	int tH = inShape[2];
	int tW = inShape[3];
	
	//��ʼ��w,b
	if (!data[1])
	{
		data[1].reset(new Blob(tF, tC, tW, tH, TRANDN));
		(*data[1]) *= 1e-2;
	}
	
	if (!data[2])
	{
		data[2].reset(new Blob(tF, 1, 1, 1, TZERO));
		(*data[2]).print("data is :");
	}
	return;
}

void ReluLayer::init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param)
{
	std::cout << "PoolLayer::init_layer ..... OK!!!" << endl;
	return;
}


void ConvLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//1.��ȡ����Blob�ߴ�
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.��ȡ����˳ߴ�
	int tF = param.conv_kernels;   //����˸������ɲ����������õ���		
	int tH = param.conv_height;   //����˸�
	int tW = param.conv_width;    //����˿�  
	int tP = param.conv_pad;        //padding��
	int tS = param.conv_stride;    //��������
	//3.��������ĳߴ�
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + 2 * tP - tH) / tS + 1;    //�����ͼƬ�߶�
	int Wo = (Wi + 2 * tP - tW) / tS + 1;  //�����ͼƬ���
	//4.��ֵ���Blob�ߴ�
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;

	return;
}
void PoolLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//�����Pool��paddingֻΪ1
	//1.��ȡ����Blob�ߴ�
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.��ȡ�ػ��˳ߴ�
	int tH = param.pool_height;    //�ػ��˸�
	int tW = param.pool_width;    //�ػ��˿�  
	int tS = param.pool_stride;      //�ػ��˻�������
	//3.����ػ���ĳߴ�
	int No = Ni;
	int Co = Ci;
	int Ho = (Hi - tH) / tS + 1;    //�����ͼƬ�߶�
	int Wo = (Wi - tW) / tS + 1;  //�����ͼƬ���
	//4.��ֵ���Blob�ߴ�
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void FcLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//1.�������Blob�ߴ�
	int No = inShape[0];                 //��batch��������
	int Co = param.fc_kernel;		   //�ò���Ԫ����
	int Ho = 1;
	int Wo = 1;
	//��200,10,1,1��
	//2.��ֵ
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
void ReluLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//Relu�������������һ����ά��
	outShape.assign(inShape.begin(), inShape.end());//inshape��ֵ�����outShape,�õ��ں���ǳ����
	return;
}