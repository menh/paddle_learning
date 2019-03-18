#include <iostream>
#include<cstdio>
#include<cstdlib>
#include <pthread.h>
using namespace std;
pthread_mutex_t mutex;
class SingleTon
{
	public: 
		static SingleTon * getInstance();
	private:
		SingleTon();
		//�Ѹ��ƹ��캯����=������Ҳ��Ϊ˽��,��ֹ������
		SingleTon(const SingleTon&);
		SingleTon& operator=(const SingleTon&);
		
		static SingleTon* instance ;
};

SingleTon::SingleTon()
{
	
}

SingleTon::SingleTon(const SingleTon& singleTon)
{
	
}

SingleTon& SingleTon::operator=(const SingleTon&){
	
}

SingleTon* SingleTon::instance = nullptr;

SingleTon* SingleTon::getInstance()
{
	if(instance == nullptr)
	{
		pthread_mutex_lock(&mutex);
		if(instance == nullptr)
		{
			instance = new SingleTon();
		}
		pthread_mutex_unlock(&mutex);
	//	unclock();
	}
	return instance;
}



int main()
{
	SingleTon* singleTon1 = SingleTon::getInstance();
	SingleTon* singleTon2 = SingleTon::getInstance();
	cout << (singleTon1 == singleTon2) <<endl;
	
	return 0;
}
