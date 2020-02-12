#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include "myBlob.hpp"

using std::unordered_map;
using std::vector;
using std::string;
using std::shared_ptr;


struct NetParam
{
	//学习率
	double lr;
	//学习率衰减系数
	double lr_decay;
	//优化算法
	string update;
	//momentum系数
	double momentum;
	//epoch的次数
	int num_epoches;
	//是否使用mini-batch梯度下降
	bool use_batch;
	//每批次样本个数
	int batch_size;
	//迭代多少次之后测试一次准确度
	int acc_frequence;
	//是否更新学习率
	bool acc_update_lr;
	//是否保存模型快照
	bool snap_shot;
	//每个多少个迭代周期保存一次快照
	int snapshot_interval;
	//是否使用fine-turn方式训练
	bool fine_turn;
	//预训练模型保存的路径
	string preTrainModel;
	//层名
	vector <string> layers;
	//层类别
	vector <string> ltype;
	//无序关联容器
	unordered_map<string, Param> lparams;
	void readNetParam(string file);
};

class Net
{
public:
	void initNet(NetParam& param, vector < shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y);
private:
	//训练集
	shared_ptr<Blob> X_train;
	shared_ptr<Blob> Y_train;
	//验证集
	shared_ptr<Blob> X_val;
	shared_ptr<Blob> Y_val;

	vector<string> layers;
	vector<string> ltyple;
	
	unordered_map<string, vector<shared_ptr<Blob>>> data;
	unordered_map<string, vector<shared_ptr<Blob>>> diff;
};





#endif