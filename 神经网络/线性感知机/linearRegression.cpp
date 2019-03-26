//https://blog.csdn.net/drilistbox/article/details/79341381

#include <iostream>
#include <vector>
using namespace std;
class CPerception{
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
CPerception::CPerception()
{
	m_vecWeight.clear();
	m_dBias = 0.0;
	m_iInterationNum = 0;
	m_iDim = 0;
	m_dRate = 0.0;
}

CPerception::CPerception(int p_iInterationNum,int p_iDim, double p_dRate)
{
	
	m_iInterationNum = p_iInterationNum;
	m_iDim = p_iDim;
	m_dRate = p_dRate;
	
	m_vecWeight.clear();
	m_dBias = 0.0;
	for(int iDimIndex = 0; iDimIndex < m_iDim; ++iDimIndex)
	{
		m_vecWeight.push_back(0.0);
	}
}

void CPerception::train(vector< vector<double> > p_vecInput, vector<double> p_vecLabel)
{
	for(int iInterationNumIndex = 0; iInterationNumIndex < m_iInterationNum; ++iInterationNumIndex)
	{
		cell(p_vecInput, p_vecLabel);
	}
	
}
void CPerception::cell(vector< vector<double> > p_vecInput, vector<double> p_vecLabel)
{
	for(int iExampleNumIndex; iExampleNumIndex < p_vecInput.size(); ++iExampleNumIndex)
	{
		int ipredictLable = forwardPropagation(p_vecInput[iExampleNumIndex]);
		backPropagation(p_vecInput[iExampleNumIndex], ipredictLable, p_vecLabel[iExampleNumIndex]);
	}
}

int CPerception::forwardPropagation(vector<double> p_vecExample)
{
	double dY = 0.0;
	for(int iDimIndex = 0; iDimIndex < p_vecExample.size(); ++iDimIndex)
	{
		dY += p_vecExample[iDimIndex] * m_vecWeight[iDimIndex];
	}
	int iO = activationFunc(dY);
	return iO;
}

void CPerception::backPropagation(vector<double> p_vecExample,int p_iPrediceLable, int p_iLabel)
{
	{
		for(int iDimIndex = 0; iDimIndex < p_vecExample.size(); ++iDimIndex)
		{
			m_vecWeight[iDimIndex] -= m_dRate * (p_iPrediceLable - p_iLabel) * p_vecExample[iDimIndex];
		}
		m_dBias -= m_dRate * (p_iPrediceLable - p_iLabel);
	}
}

int CPerception::activationFunc(double p_dy)
{
	return p_dy;
}

int CPerception::predict(vector<double> p_vecInput)
{
	return forwardPropagation(p_vecInput);
}

void CPerception::printWeight()
{
	for(int iDimIndex = 0; iDimIndex < m_iDim; ++iDimIndex)
	{
		cout<<m_vecWeight[iDimIndex]<<" ";
	}
	cout<<endl;
}
int main()
{
	CPerception CPerception(100, 2, 0.01);
	
	vector<vector<double>> example;
	vector<double> exampleRow;
	exampleRow.push_back(2);
	exampleRow.push_back(3);
	example.push_back(exampleRow);
	exampleRow.clear();
	exampleRow.push_back(2);
	exampleRow.push_back(0);
	example.push_back(exampleRow);
	exampleRow.clear();
	exampleRow.push_back(0);
	exampleRow.push_back(3);
	example.push_back(exampleRow);
	exampleRow.clear();
	exampleRow.push_back(0);
	exampleRow.push_back(0);
	example.push_back(exampleRow);
	
	vector<double>label;
	label.push_back(3000);
	label.push_back(2000);
	label.push_back(1000);
	label.push_back(100);
	
	CPerception.train(example,label);
	CPerception.printWeight();
	return 0;
 } 
