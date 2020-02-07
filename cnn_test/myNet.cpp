#include <iostream>
#include "myNet.hpp"
#include <fstream>
#include <cassert>
#include <cstdio>
#include <json/json.h>

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
					this->lparams[ii["name"].asString()].conv_height = height;
					this->lparams[ii["name"].asString()].conv_width = width;
					this->lparams[ii["name"].asString()].conv_stride = stride;
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