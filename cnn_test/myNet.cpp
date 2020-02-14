#include <iostream>
#include "myNet.hpp"
#include <fstream>
#include <cassert>
#include <cstdio>
#include <json/json.h>
#include "myBlob.hpp"

using namespace std;
void NetParam::readNetParam(std::string file)
{
	ifstream ifs;
	ifs.open(file);
	assert(ifs.is_open());
	Json::Reader reader;
	Json::Value value;
	if (reader.parse(ifs, value))
	{
		if (!value["train"].isNull())
		{
			auto &tparam = value["train"];
			this->lr = tparam["learning rate"].asDouble();
			this->lr_decay = tparam["lr decay"].asDouble();
			this->update = tparam["update method"].asString();
			this->momentum = tparam["momentum parameter"].asDouble();
			this->num_epoches = tparam["num epochs"].asInt();
			this->use_batch = tparam["use batch"].asBool();
			this->batch_size = tparam["batch size"].asInt();
			this->acc_frequence = tparam["evaluate interval"].asInt();
			this->acc_update_lr = tparam["lr update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_turn = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString();
		}
		if (!value["net"].isNull())
		{
			auto &nparam = value["net"];
			for (int i = 0; (int)i < nparam.size(); ++i)
			{
				auto &ii = nparam[i];
				this->layers.push_back(ii["name"].asString());
				this->ltype.push_back(ii["type"].asString());
				if (ii["type"].asString() == "Conv")
				{
					int num = ii["kernel num"].asInt();
					int height = ii["kernel height"].asInt();
					int width = ii["kernel width"].asInt();
					int pad = ii["pad"].asInt();
					int stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].conv_height = height;
					this->lparams[ii["name"].asString()].conv_width = width;
					this->lparams[ii["name"].asString()].conv_pad = pad;
					this->lparams[ii["name"].asString()].conv_stride = stride;
					this->lparams[ii["name"].asString()].conv_kernels = num;
				}
				if (ii["type"].asString() == "Pool")
				{
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].pool_height = height;
					this->lparams[ii["name"].asString()].pool_width = width;
					this->lparams[ii["name"].asString()].pool_stride = stride;
				}
				if (ii["type"].asString() == "Fc")
				{
					int num = ii["kernel num"].asInt();
					this->lparams[ii["name"].asString()].fc_kernel = num;
				
				}
			}
		}
	}
}


void Net::initNet(NetParam& param, vector < shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y)
{
	layers_ = param.layers;
	ltypes_ = param.ltype;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}
	X_train_ = X[0];
	X_val_ = X[1];
	Y_train_ = Y[0];
	Y_train_ = Y[1];
	for (int i = 0; i < (int)layers_.size(); ++i)
	{
		data_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);//用来存储正向运算的数据，一共3个blob 分别存储W,X,B
		diff_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);//用来存储反向运算的数据，求导数据，一共3个blob 分别存储W,X,B
		outShape_[layers_[i]] = vector<int>(4);//输出给下一层的N,C,H,W，缓存
	}

	//初始化层
	shared_ptr<Layer> myLayer(NULL);
	//输入的shape
	vector<int> inShape = 
	{
		param.batch_size,
		X_train_->get_C(),
		X_train_->get_W(),
		X_train_->get_H()
	};
	//根据每一层的类型名来做初始化
	for (int i = 0; i < (int)layers_.size(); ++i)
	{
		string lname = layers_[i];
		string ltype = ltypes_[i];
		
		if (ltype == "Conv")
		{
			myLayer.reset(new ConvLayer);
		}
		
		if (ltype == "Pool")
		{
			myLayer.reset(new PoolLayer);
		}
		if (ltype == "Fc")
		{
			myLayer.reset(new FcLayer);
		}
		if (ltype == "Relu")
		{
			myLayer.reset(new ReluLayer);
		}
		//将结果存储到中unordered_map中
		myLayers_[lname] = myLayer;
		//对每一层进行初始化
		myLayer->init_layer(inShape, lname, data_[lname], param.lparams[lname]);//N,H,W是可以通过json文件制定的，但是通道数不行
		myLayer->calcShape(inShape, outShape_[lname], param.lparams[lname]);
		inShape.assign(outShape_[lname].begin(), outShape_[lname].end());
		cout << lname << "->(" << outShape_[lname][0] << "," << outShape_[lname][1] << "," << outShape_[lname][2] << "," << outShape_[lname][2] << ")" << endl;
	}
	data_["fc1"][1]->print("w 为：");
	data_["fc1"][2]->print("b 为：");

}