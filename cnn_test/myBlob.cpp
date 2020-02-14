#include "myBlob.hpp"
#include "cassert"
using namespace std;
using namespace arma;
using std::cout;
using std::endl;


//构造函数是没有返回值类型的
Blob::Blob(const int n, const int c, const int h, const int w, int type) :N_(n), C_(c), H_(h), W_(w)
{
	arma_rng::set_seed_random();//设置伪随机数,，种子是系统时间
	_init(N_, C_, H_, W_, type);
}

void Blob::_init(const int n, const int c, const int h, const int w, int type)
{

	if (type == TONE)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::ones));
		return;
	}
	if (type == TZERO)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::zeros));
		return;
	}
	if (type == TDEFAULT)
	{
		blob_data = vector<cube>(n, cube(h, w, c));
		return;
	}
	if (type == TRANDU)
	{
		for (int i = 0; i < n; ++i)   //生成n个填充了随机值（均匀分布）的cube，堆叠进vector容器
			blob_data.push_back(arma::randu<cube>(h, w, c)); //堆叠
		return;
	}
	if (type == TRANDN)
	{
		for (int i = 0; i < n; ++i)   //生成n个填充了随机值(标准高斯分布）的cube，堆叠进vector容器
			blob_data.push_back(arma::randn<cube>(h, w, c)); //堆叠
		return;
	}

}
void Blob::print(string str)
{
	//assert(!blob_data.empty());
	cout << str << endl;
	for (int i = 0; i < N_; ++i)
	{
		this->blob_data[i].print();
	}

}

cube& Blob::operator[] (int i)
{
	return blob_data[i];
}

Blob Blob::operator *=(double rate)
{
	for (int i = 0; i < blob_data.size(); ++i)
	{
		blob_data[i] *= rate;
	}
	return *this;
}

vector<cube>& Blob::get_data()
{
	return blob_data;
}

Blob Blob::subBlob(int low_idx, int high_idx)
{
	auto size = blob_data.size();
	cout << "size = " << size << endl;
	if (high_idx > low_idx)
	{
		Blob tmp(high_idx - low_idx, C_, H_, W_);
		for (int i = low_idx; i < high_idx; ++i)
		{
			tmp[i - low_idx] = (*this)[i];

		}
		return tmp;
	}
	
}