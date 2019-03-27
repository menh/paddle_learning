#include <iostream>
#include "Matrix.h" 
using namespace std;
class CFnn{
		public:
		CPerception();
		CPerception(int p_iInterationNum,int p_iDim, double p_dRate);
		void train(vector< vector<double> > p_vecInput, vector<double> p_vecLabel);
		int predict(vector<double> p_vecInput);
		void printWeight();
	private:
		void cell(vector< vector<double> > p_vecInput, vector<double> p_vecLabel);
		int forwardPropagation(vector<double> p_vecInput);
		void backPropagation(vector<double> p_vecExample,int p_iPrediceLable, int p_iLabel);
		int activationFunc(double p_dPredictY);
	private:
		vector<double> m_vecWeight;
		double m_dBias;
		int m_iInterationNum;
		int m_iDim;	
		double m_dRate;
};
int main()
{
	
	return 0;
}
