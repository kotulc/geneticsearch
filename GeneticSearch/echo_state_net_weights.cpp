// echo_state_net_weights.cpp : Defines the EchoStateNetWeights struct
// Copyright 2013, Clayton Kotualk

#include "echo_state_net_weights.h"


// Constructs the ESN weight struct with empty weight matrices.
EchoStateNetWeights::EchoStateNetWeights() {

	w_in = MatrixXf(); 
	w_reservoir = MatrixXf();
	w_out = MatrixXf();
	w_back = MatrixXf();

}


// Construct the ESN weight struct with randomly initialized matrices of the given dimensions.
EchoStateNetWeights::EchoStateNetWeights(int in_nodes, int res_nodes, int out_nodes, float connectivity) {

	srand(time(NULL));

	w_in = MatrixXf::Random(in_nodes, res_nodes); //use connectivity
	w_reservoir = GetRandomSparseMatrix(res_nodes, res_nodes, connectivity);
	w_out = MatrixXf::Random(in_nodes + res_nodes, out_nodes);
	w_back = MatrixXf::Random(out_nodes, res_nodes);

}


// Construct the ESN weight struct with a n=4 vector of weight matrices. 
// Copies vector index 0 to w_in, index 1 to w_reservoir, index 2 to w_out and index 3 to w_back.
EchoStateNetWeights::EchoStateNetWeights(std::vector<MatrixXf> weight_matrices) {

	w_in = weight_matrices[0];
	w_reservoir = weight_matrices[1];
	w_out = weight_matrices[2];
	w_back = weight_matrices[3];

}


// Returns a float matrix with [connectivity*row*col] randomly generated coefficients 
// where the connectivity parameter is in the interval of (0.0 to 1.0]. 
// Each random float coefficient is generated with the [-1.0,1.0] uniform distribution.
MatrixXf EchoStateNetWeights::GetRandomSparseMatrix(int row, int col, float connectivity) { 

	MatrixXf sparse_matrix = MatrixXf::Zero(row, col);

	std::random_device random_seed;
	std::mt19937 random_generator(random_seed());
	std::uniform_int_distribution<int> idx_distribution(0,sparse_matrix.size()-1);
	std::uniform_real_distribution<float> float_distribution(-1.0,1.0);

	float * matrix_array = sparse_matrix.data();
	int random_coefficients = (int)std::ceil(connectivity*row*col);

	// Seed the zero matrix with the randomly generated coefficients 
	for (int i=0; i<random_coefficients; i++) {
		matrix_array[i] = float_distribution(random_generator); 
	}

	float temp_value = 0;
	int swap_idx = 0;

	// Randomly shuffle the indices of the generated coefficients with the remaining zero coefficients
	for (int i=0; i<random_coefficients; i++) {
		swap_idx = idx_distribution(random_generator);
		temp_value = matrix_array[i];
		matrix_array[i] = matrix_array[swap_idx];
		matrix_array[swap_idx] = temp_value;
	}

	return sparse_matrix;

}


// Returns a copy of all edge weight matrices contained in a vector:
// w_in = index 0, w_reservoir = index 1, w_out = index 2, and w_back = index 3.
std::vector<MatrixXf> EchoStateNetWeights::GetWeightMatrices() {

	std::vector<MatrixXf> weight_matrices;

	weight_matrices.push_back(w_in);
	weight_matrices.push_back(w_reservoir);
	weight_matrices.push_back(w_out);
	weight_matrices.push_back(w_back);

	return weight_matrices;

}