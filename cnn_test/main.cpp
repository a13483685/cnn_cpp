#include<iostream>
#include<string>
#include "myNet.hpp"
#include "myBlob.hpp"

using namespace std;

int main(int argc, char** argv)
{
	string configFile = "./myModel.json";
	NetParam net_param;
	net_param.readNetParam(configFile);
	cout << "learning rate = " << net_param.lr << endl;
	cout << "batch size = " << net_param.batch_size << endl;
	vector<string> layers_ =  net_param.layers;
	vector<string> ltypes_ = net_param.ltype;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << ";" << "ltype = " << ltypes_[i] << endl;
	}
	Blob test_blob(2, 3, 5, 5, TRANDN);
	test_blob.print("Blob 里面的数据是：\n");
	
	system("pause");
}