#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>

using std::unordered_map;
using std::vector;
using std::string;

struct NetParam
{
	//ѧϰ��
	double lr;
	//ѧϰ��˥��ϵ��
	double lr_decay;
	//�Ż��㷨
	string update;
	//momentumϵ��
	double momentum;
	//epoch�Ĵ���
	int num_epoches;
	//�Ƿ�ʹ��mini-batch�ݶ��½�
	bool use_batch;
	//ÿ������������
	int batch_size;
	//�������ٴ�֮�����һ��׼ȷ��
	int acc_frequence;
	//�Ƿ����ѧϰ��
	bool acc_update_lr;
	//�Ƿ񱣴�ģ�Ϳ���
	bool snap_shot;
	//ÿ�����ٸ��������ڱ���һ�ο���
	int snapshot_interval;
	//�Ƿ�ʹ��fine-turn��ʽѵ��
	bool fine_turn;
	//Ԥѵ��ģ�ͱ����·��
	string preTrainModel;
	//����
	vector <string> layers;
	//�����
	vector <string> ltype;
	//�����������
	unordered_map<string, Param> lparams;
	void readNetParam(string file);
};


#endif