#ifndef __MYLAYER_HPP__
#define __MYLAYER_HPP__
#include <iostream>
#include <memory>
#include "myBlob.hpp"
using namespace std;

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


class Layer
{
public:
	//inShape, lname, data_[lname], param.lparams[lname]
	Layer(){};
	virtual ~Layer(){};
	virtual void init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param) = 0;
	virtual void calcShape(vector<int>& inShape, vector<int>& outShape, Param& param) = 0;
};

class ConvLayer :public Layer
{
public:
	ConvLayer(){};
	~ConvLayer(){};
	void init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param);
	void calcShape(vector<int>& inShape, vector<int>& outShape, Param& param);

};

class PoolLayer :public Layer
{
public:
	PoolLayer(){};
	~PoolLayer(){};
	void init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param);
	void calcShape(vector<int>& inShape, vector<int>& outShape, Param& param);

};

class FcLayer :public Layer
{
public:
	FcLayer(){};
	~FcLayer(){};
	void init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param);
	void calcShape(vector<int>& inShape, vector<int>& outShape, Param& param);
};

class ReluLayer :public Layer
{
public:
	ReluLayer(){};
	~ReluLayer(){};
	void init_layer(const vector<int>& inShape, const string& name, vector<shared_ptr<Blob>>& data, const Param& param);
	void calcShape(vector<int>& inShape, vector<int>& outShape, Param& param);
};

#endif