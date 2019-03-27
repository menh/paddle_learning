#include "Matrix.h"

Matrix::Matrix()
{
	m_iRow = 0;
	m_iCol = 0;
	m_data = NULL;
}

Matrix::Matrix(int iRow, int iCol, double d)
{
	m_iRow = iRow;
	m_iCol = iCol;
	m_data = new double[m_iRow*m_iCol];
	//m_data = (double*)malloc(m_iRow*m_iCol*sizeof(double));
	//memset(m_data, 1.0, m_iRow*m_iCol*sizeof(double));
	for (int i = 0; i < m_iRow*m_iCol; i++)
		m_data[i] = d;
}

Matrix::Matrix(int iRow, int iCol, int iDph, double d)
{
	m_iRow = iRow;
	m_iCol = iCol;
	m_iDph = iDph;
	m_data = new double[m_iRow*m_iCol*m_iDph];
	for (int i = 0; i < m_iRow*m_iCol; i++)
		m_data[i] = d;
}

Matrix::Matrix(Matrix& A)
{
	if (A.m_iRow < 0 || A.m_iCol < 0) 
	{
		throw "error"; 
	}
	this->m_iRow = A.m_iRow;
	this->m_iCol = A.m_iCol;
	//SAFE_DELETE(this->m_data);
	this->m_data = new double[this->m_iRow*this->m_iCol];
	memcpy(this->m_data, A.m_data, sizeof(double)*m_iRow*m_iCol);
}

Matrix::~Matrix()
{
	//SAFE_FREE(m_data);
	SAFE_DELETE(m_data);
}

Matrix Matrix::TransForm()
{
	Matrix temp(this->m_iCol, this->m_iRow);
	for (int i = 0; i < this->m_iCol; i++)
		for (int j = 0; j < this->m_iRow; j++)
			temp(i, j) = (*this)(j, i);
	return temp;
}

Matrix Matrix::Rot90(int i)
{
	i %= 4;
	if (i == 1)
	{
		Matrix temp(m_iCol, m_iRow);
		for (int i = 0; i < m_iCol; i++)
			for (int j = 0; j < m_iRow; j++)
				temp(i, j) = m_data[(m_iRow - 1 - j)*m_iCol + i];
		return temp;
	}
	else if (i == 2)
	{
		Matrix temp(m_iRow, m_iCol);
		for (int i = 0; i < m_iRow; i++)
			for (int j = 0; j < m_iCol; j++)
				temp(i, j) = m_data[(m_iRow - 1 - i)*m_iCol + m_iCol - 1 - j];
		return temp;
	}
}

Matrix Matrix::reshape(int iNewRow, int iNewCol)
{
	if(iNewRow*iNewCol != m_iCol*m_iRow)
	{
		throw("输入参数有误");
	}
	else
	{
		Matrix temp(iNewRow, iNewCol);
		memcpy(temp.m_data, m_data, iNewRow*iNewCol*sizeof(double));
		return temp;
	}
}

double Matrix::Norm(int iNum)
{
	double sum = 0.0;
	for (int i = 0; i < m_iRow; i++)
		for (int j = 0; j < m_iCol; j++)
		{
			sum += pow(m_data[i*m_iCol +j],iNum);
		}
	return sum;
}

Matrix& Matrix::operator = (Matrix& A)
{
	this->m_iRow = A.m_iRow;
	this->m_iCol = A.m_iCol;
	SAFE_DELETE(this->m_data);
	this->m_data = new double[this->m_iRow*this->m_iCol];
	memcpy(this->m_data, A.m_data, sizeof(double)*m_iRow*m_iCol);
	return *this;
}

Matrix& Matrix::operator = (double d)
{
	for (int i = 0; i < m_iRow; i++)
		for (int j = 0; j < m_iCol; j++)
			m_data[i*m_iCol + j] = d;
	return (*this);
}

Matrix operator + (Matrix& A, Matrix& B)
{
	if (A.m_iRow == B.m_iRow && A.m_iCol == B.m_iCol)
	{
		Matrix temp(A.m_iRow, A.m_iCol, 0.0);
		for (int i = 0; i < A.m_iRow; i++)
			for (int j = 0; j < A.m_iCol; j++)
				temp.m_data[i*A.m_iCol + j] = A.m_data[i*A.m_iCol + j] + B.m_data[i*A.m_iCol + j];
		return temp;
	}
	else
	{
		throw "相加矩阵维数不一致！";
	}
}

Matrix operator - (Matrix& A, Matrix& B)
{
	if (A.m_iRow == B.m_iRow && A.m_iCol == B.m_iCol)
	{
		Matrix temp(A.m_iRow, A.m_iCol, 0.0);
		for (int i = 0; i < A.m_iRow; i++)
			for (int j = 0; j < A.m_iCol; j++)
				temp.m_data[i*A.m_iCol + j] = A.m_data[i*A.m_iCol + j] - B.m_data[i*A.m_iCol + j];
		return temp;
	}
	else
	{
		throw "相减矩阵维数不一致！";
	}
}

Matrix operator * (Matrix& A, double d)
{
	if (A.m_iRow >= 1 && A.m_iCol >= 1)
	{
		Matrix temp(A.m_iRow, A.m_iCol, 0.0);
		for (int i = 0; i < A.m_iRow; i++)
			for (int j = 0; j < A.m_iCol; j++)
				temp.m_data[i*A.m_iCol + j] = A.m_data[i*A.m_iCol + j] * d;
		return temp;
	}
	else
	{
		throw "矩阵维数为0或小于0！";
	}
}

Matrix& Matrix::operator +=(Matrix& A)
{
	if (A.m_iRow == m_iRow && A.m_iCol == m_iCol)
	{
		for (int i = 0; i < m_iRow; i++)
			for (int j = 0; j < m_iCol; j++)
				m_data[i*m_iCol + j] += A.m_data[i*m_iCol + j];
		return *this;
	}
	else
	{
		throw "自减矩阵维数不一致！";
	}
}

Matrix& Matrix::operator +=(double& d)
{
	for (int i = 0; i < m_iRow; i++)
		for (int j = 0; j < m_iCol; j++)
			m_data[i*m_iCol + j] += d;
	return *this;
}

Matrix& Matrix::operator -=(Matrix& A)
{
	if (A.m_iRow == m_iRow && A.m_iCol == m_iCol)
	{
		for (int i = 0; i < m_iRow; i++)
			for (int j = 0; j < m_iCol; j++)
				m_data[i*m_iCol + j] -= A.m_data[i*m_iCol + j];
		return *this;
	}
	else
	{
		throw "自减矩阵维数不一致！";
	}
}

double& Matrix::operator( )(int i, int j)
{
	if (i<m_iRow && j<m_iCol)
	{
		return m_data[i*m_iCol + j];
	}
	else
	{
		throw "下标越界！";
	}
}

double Matrix::sum()
{
	double temp = 0.0;
	for (int i = 0; i < m_iRow; i++)
		for (int j = 0; j < m_iCol; j++)
				temp += m_data[i*m_iCol + j];
	return temp;
}

Matrix operator * (Matrix& A, Matrix& B)
{
	if (A.m_iCol == B.m_iRow)
	{
		Matrix temp(A.m_iRow, B.m_iCol, 0.0);
		for (int i = 0; i < A.m_iRow; i++)
			for (int j = 0; j < B.m_iCol; j++)
				for (int k = 0; k < A.m_iCol; k++)
					temp(i, j) += A(i, k)*B(k, j);
		return temp;
	}
	else
	{
		throw "相乘数组维数不匹配！";
	}
}

Matrix dotProduct(Matrix& A, Matrix& B)
{
	if (A.m_iRow == B.m_iRow && A.m_iCol == B.m_iCol)
	{
		Matrix temp(A.m_iRow, A.m_iCol, 0.0);
		for (int i = 0; i < A.m_iRow; i++)
			for (int j = 0; j < A.m_iCol; j++)
				temp(i, j) = A(i, j)*B(i, j);
		return temp;
	}
	else
	{
		throw "点乘数组维数不一致！";
	}
}

Matrix conv(Matrix& A, Matrix& B)
{
	Matrix temp(A.m_iRow - B.m_iRow + 1, A.m_iCol - B.m_iCol + 1, 0.0);
	for (int i = 0; i < temp.m_iRow; i++)
		for (int j = 0; j < temp.m_iCol; j++)
		{
			for (int i2 = 0; i2 < B.m_iRow; i2++)
				for (int j2 = 0; j2 < B.m_iCol; j2++)
					temp(i, j) += A(i + i2, j + j2) * B(B.m_iRow - 1 - i2, B.m_iCol - 1 - j2);
		}
	return temp;
}

Matrix interConv(Matrix& A, Matrix& B)
{
	//Matrix temp(A.m_iRow, A.m_iCol, 0.0);
	//for (int i = 0; i < temp.m_iRow; i++)
	//	for (int j = 0; j < temp.m_iCol; j++)
	//	{
	//		temp(i, j) = A(i, j)*B(i, j);
	//	}
	Matrix temp(A.m_iRow - B.m_iRow + 1, A.m_iCol - B.m_iCol + 1, 0.0);
	for (int i = 0; i < temp.m_iRow; i++)
		for (int j = 0; j < temp.m_iCol; j++)
		{
			for (int i2 = 0; i2 < B.m_iRow; i2++)
				for (int j2 = 0; j2 < B.m_iCol; j2++)
					temp(i, j) += A(i + i2, j + j2) * B(i2, j2);
		}	
	return temp;
}

