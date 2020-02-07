#include "myBlob.hpp"
#include "cassert"
using namespace std;
using namespace arma;

//构造函数是没有返回值类型的
Blob::Blob(const int n, const int c, const int h, const int w, int type) :N_(n), C_(c), H_(h), W_(w)
{
	arma_rng::set_seed_random();//设置伪随机数,，种子是系统时间
	_init(n,c,h,w,type);
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
	if (type == TRANDU)
	{
		for (int i = 0; i < n; ++i)
		{
			blob_data.push_back(arma::randu<cube>(h, w, c));
		}
		return;
	}
	if (type == TRANDN)
	{
		for (int i = 0; i < n; ++i)
		{
			blob_data.push_back(arma::randn<cube>(h, w, c));
		}
		return;
	}
}
void Blob::print(string str)
{
	assert(!blob_data.empty());
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

vector<cube>& Blob::get_data()
{
	return blob_data;
}