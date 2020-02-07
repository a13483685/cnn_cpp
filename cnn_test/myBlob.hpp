#ifndef __MYBLOB_HPP__
#define __MYBLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

enum Filltype
{
	TONE = 1,//所有元素都填充为1
	TZERO = 2, //所有元素都填充为0
	TRANDU = 3,//[0,1]区间内均匀分布
	TRANDN = 4,//均值为0方差为1的高斯分布
	TDEFAULT = 5
};
class Blob
{
public:
	Blob() :N_(0), C_(0), H_(0), W_(0)
	{}
	Blob(const int n, const int c, const int h, const int w,int type = TDEFAULT);
	void _init(const int n, const int c, const int h, const int w,int type = TDEFAULT);
	void print(string str = "");
private:
	int N_;
	int C_;
	int H_;
	int W_;
	//一个blob中含有N个cube,拥有的cube的数目是动态的
	vector<cube> blob_data;
	
};



#endif