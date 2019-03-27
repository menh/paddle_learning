#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<memory>
#define SAFE_FREE(p) {if(p) free(p); p = NULL;}
#define SAFE_DELETE(p) {if(p) delete(p); p = NULL;}
#define SAFE_DELETE_ARRAY(p) {if(p) delete[] (p); p = NULL;}
#define SAFE_FCLOSE(p) {if(p) fclose(p); p = NULL;}

class Matrix
{
public:
	Matrix();
	Matrix(int iRow, int iCol, double d = 0.0);//函数参数的默认值 在函数声明时定义，函数定义时不再说明
	Matrix(int iRow, int iCol, int idph, double d = 0.0);
	Matrix(Matrix& A);//一般地，对于有动态分配内存的类，需重写(深)拷贝构造函数
	~Matrix();
public:
	int m_iRow, m_iCol, m_iDph;
	double* m_data;
public:
	Matrix TransForm();  //没有返回引用, 因为这里返回的是新矩阵，不是原来的矩阵自身？？？？？
	Matrix Rot90(int);
	Matrix reshape(int, int);
	double Norm(int);
	Matrix& operator = (Matrix&);
	Matrix& operator = (double);
	//Matrix& operator + (Matrix&);
	//Matrix& operator - (Matrix&);
	//Matrix& operator * (double);
	Matrix& operator +=(Matrix&);
	Matrix& operator +=(double&);
	Matrix& operator -=(Matrix&);
	double& operator( )(int, int);
	double sum();
	//	transForm
};
//双目运算符重载函数应该在类外定义，单目运算符应该在类内定义
Matrix operator * (Matrix&, Matrix&);
Matrix operator + (Matrix&, Matrix&);
Matrix operator - (Matrix&, Matrix&);
Matrix operator * (Matrix&, double);
//在类外定义重载函数时
//这里的函数返回值不能写成引用，引用返回的是地址, 跳出函数体后, 函数中的返回值的储值空间中的内容会析构掉
//导致在函数体外返回的是空地址
//为了获取函数中的计算内容,返回值必须写成按值引用，
Matrix dotProduct(Matrix&, Matrix&);
Matrix conv(Matrix&, Matrix&);
Matrix interConv(Matrix&, Matrix&);

