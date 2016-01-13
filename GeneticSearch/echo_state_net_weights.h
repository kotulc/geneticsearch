#ifndef ECHO_STATE_NET_WEIGHTS_H
#define ECHO_STATE_NET_WEIGHTS_H

#include "stdafx.h"


struct EchoStateNetWeights {
	MatrixXf w_in;
	MatrixXf w_reservoir;
	MatrixXf w_out;
	MatrixXf w_back;

	EchoStateNetWeights();
	EchoStateNetWeights(int, int, int, float);
	EchoStateNetWeights(std::vector<MatrixXf> weight_matrices);
	MatrixXf GetRandomSparseMatrix(int row, int col, float connectivity);
	std::vector<MatrixXf> GetWeightMatrices(); //Change to pointers?
};

#endif