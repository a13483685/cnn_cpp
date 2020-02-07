#ifndef __MYBLOB_HPP__
#define __MYBLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

enum Filltype
{
	TONE = 1,//����Ԫ�ض����Ϊ1
	TZERO = 2, //����Ԫ�ض����Ϊ0
	TRANDU = 3,//[0,1]�����ھ��ȷֲ�
	TRANDN = 4,//��ֵΪ0����Ϊ1�ĸ�˹�ֲ�
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
	//һ��blob�к���N��cube,ӵ�е�cube����Ŀ�Ƕ�̬��
	vector<cube> blob_data;
	
};



#endif