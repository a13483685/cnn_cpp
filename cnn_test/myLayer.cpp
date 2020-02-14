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
	//1.获取全连接核尺寸(F,C,H,W)
	int tF = param.fc_kernel;
	int tC = inShape[1];
	int tH = inShape[2];
	int tW = inShape[3];
	
	//初始化w,b
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
	//1.获取输入Blob尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.获取卷积核尺寸
	int tF = param.conv_kernels;   //卷积核个数（由层名称索引得到）		
	int tH = param.conv_height;   //卷积核高
	int tW = param.conv_width;    //卷积核宽  
	int tP = param.conv_pad;        //padding数
	int tS = param.conv_stride;    //滑动步长
	//3.计算卷积后的尺寸
	int No = Ni;
	int Co = tF;
	int Ho = (Hi + 2 * tP - tH) / tS + 1;    //卷积后图片高度
	int Wo = (Wi + 2 * tP - tW) / tS + 1;  //卷积后图片宽度
	//4.赋值输出Blob尺寸
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;

	return;
}
void PoolLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//这里的Pool层padding只为1
	//1.获取输入Blob尺寸
	int Ni = inShape[0];
	int Ci = inShape[1];
	int Hi = inShape[2];
	int Wi = inShape[3];
	//2.获取池化核尺寸
	int tH = param.pool_height;    //池化核高
	int tW = param.pool_width;    //池化核宽  
	int tS = param.pool_stride;      //池化核滑动步长
	//3.计算池化后的尺寸
	int No = Ni;
	int Co = Ci;
	int Ho = (Hi - tH) / tS + 1;    //卷积后图片高度
	int Wo = (Wi - tW) / tS + 1;  //卷积后图片宽度
	//4.赋值输出Blob尺寸
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}

void FcLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//1.计算输出Blob尺寸
	int No = inShape[0];                 //该batch的样本数
	int Co = param.fc_kernel;		   //该层神经元个数
	int Ho = 1;
	int Wo = 1;
	//（200,10,1,1）
	//2.赋值
	outShape[0] = No;
	outShape[1] = Co;
	outShape[2] = Ho;
	outShape[3] = Wo;
	return;
}
void ReluLayer::calcShape(vector<int>& inShape, vector<int>& outShape, Param& param)
{
	//Relu层的输入和输出是一样的维度
	outShape.assign(inShape.begin(), inShape.end());//inshape的值深拷贝给outShape,用等于号是浅拷贝
	return;
}