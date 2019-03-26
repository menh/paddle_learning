#include <iostream>
#include <cmath>
using namespace std;
# define INPUT_LAYER_NODE 2
# define HIDDEN_LAYER_NODE 3
# define HIDDEN_LAYER 3
# define OUTPUT_LAYER_NODE 4
class Node{
	public:
		double value;
		double w[2];
};

class Net{
	public:
		Net();
		double sigmoid(double z);
		double getLoss();
	    void forwardPropagation(double input1, double input2);
		void backPropagation(double T1, double T2);
		void printResult();
	private:
		Node input_layer[2];
		Node hidden_layer[2];
		Node output_layer[2];
		
		double yita = 0.5;
		
		double k1;
		double k2;
		
		double Tg[2];
		double O[2];
};

Net::Net()
{
	k1 = 0.35;
	k2 = 0.60;
	input_layer[0].w[0] = 0.15;
	input_layer[0].w[1] = 0.25;
	input_layer[1].w[0] = 0.20;
	input_layer[1].w[1] = 0.30;
	
	hidden_layer[0].w[0] = 0.40;
	hidden_layer[0].w[1] = 0.50;
	hidden_layer[1].w[0] = 0.45;
	hidden_layer[1].w[1] = 0.55;
}

double Net::sigmoid(double z)
{
	return 1 / (1 + exp(-z));
} 

void Net::forwardPropagation(double input1, double input2)
{
	input_layer[0].value = input1;
	input_layer[1].value = input2;
	

	for(int iHiddenLayerNodeIndex = 0; iHiddenLayerNodeIndex < 2; ++iHiddenLayerNodeIndex)
	{
		double z = 0;	
		for(int iNode = 0; iNode < 2; ++iNode)
		{
			z += input_layer[iNode].value * input_layer[iNode].w[iHiddenLayerNodeIndex];
		}
		z += k1;
		hidden_layer[iHiddenLayerNodeIndex].value = z;
	}
	
	for(int iOutputLayerNodeIndex = 0; iOutputLayerNodeIndex < 2; ++iOutputLayerNodeIndex)
	{
		double z = 0;	
		for(int iNode = 0; iNode < 2; ++iNode)
		{
			z += hidden_layer[iNode].value * hidden_layer[iNode].w[iOutputLayerNodeIndex];
		}
		z += k2;
	//	output_layer[iHiddenLayerNodeIndex].value = z;
	}
}

void Net::backPropagation(double T1, double T2)
{
	Tg[0] = T1;
    Tg[1] = T2;
    for (size_t iNNum = 0; iNNum < 2; iNNum++)//更新输入层权重
    {
        for (size_t wnum = 0; wnum < 2; wnum++)
        {
            double y = hidden_layer[wnum].value;
            input_layer[iNNum].w[wnum] -= yita*((O[0] - T1)*O[0] *(1- O[0])*
                hidden_layer[wnum].w[0] +(O[1] - T2)*O[1] *(1 - O[1])*hidden_layer[wnum].w[1])*
                y*(1- y)*input_layer[iNNum].value;
        }
    }
    for (size_t hNNum = 0; hNNum < 2; hNNum++)//更新隐含层权重
    {
        for (size_t wnum = 0; wnum < 2; wnum++)
        {
            hidden_layer[hNNum].w[wnum]-= yita*(O[wnum] - Tg[wnum])*
                O[wnum] *(1- O[wnum])*hidden_layer[hNNum].value;
        }
    }
}

void Net::printResult()
{
    double loss = getLoss();
    cout << "loss：" << loss << endl;
    cout << "输出1：" << O[0] << endl;
    cout << "输出2：" << O[1] << endl;
}


int main()
{
	return 0;
}
