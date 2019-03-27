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
	Matrix(int iRow, int iCol, double d = 0.0);//����������Ĭ��ֵ �ں�������ʱ���壬��������ʱ����˵��
	Matrix(int iRow, int iCol, int idph, double d = 0.0);
	Matrix(Matrix& A);//һ��أ������ж�̬�����ڴ���࣬����д(��)�������캯��
	~Matrix();
public:
	int m_iRow, m_iCol, m_iDph;
	double* m_data;
public:
	Matrix TransForm();  //û�з�������, ��Ϊ���ﷵ�ص����¾��󣬲���ԭ���ľ���������������
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
//˫Ŀ��������غ���Ӧ�������ⶨ�壬��Ŀ�����Ӧ�������ڶ���
Matrix operator * (Matrix&, Matrix&);
Matrix operator + (Matrix&, Matrix&);
Matrix operator - (Matrix&, Matrix&);
Matrix operator * (Matrix&, double);
//�����ⶨ�����غ���ʱ
//����ĺ�������ֵ����д�����ã����÷��ص��ǵ�ַ, �����������, �����еķ���ֵ�Ĵ�ֵ�ռ��е����ݻ�������
//�����ں������ⷵ�ص��ǿյ�ַ
//Ϊ�˻�ȡ�����еļ�������,����ֵ����д�ɰ�ֵ���ã�
Matrix dotProduct(Matrix&, Matrix&);
Matrix conv(Matrix&, Matrix&);
Matrix interConv(Matrix&, Matrix&);

