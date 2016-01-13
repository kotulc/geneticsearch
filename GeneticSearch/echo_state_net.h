#ifndef ECHO_STATE_NET_H
#define ECHO_STATE_NET_H

#include "stdafx.h"
#include "echo_state_net_weights.h"


class EchoStateNet {
	private:
	int in_nodes;
	int reservoir_nodes;
	int out_nodes;
	int net_edges;
	int net_states;
	float training_accuracy;
	float validation_accuracy;
	float net_weight;
	float net_mean;
	float net_capacity;
	float net_entropy;
	float net_memory;
	float net_sensitivity;
	float net_connectivity;
	float net_alpha;
	EchoStateNetWeights net_weights;
	VectorXf activation_distribution;

	void ScaleNetWeights();
	void ValidateParameters(float * param_alpha, float * param_connectivity);

	public:
	EchoStateNet(const MatrixXf& training_data, const MatrixXf& training_labels, 
		int res_nodes, float a, float connectivity);
	EchoStateNet(const MatrixXf& training_data, const MatrixXf& training_labels, 
		std::vector<float> params, std::vector<MatrixXf> esnw);
	VectorXf GetSigmoidActivation(VectorXf* propagation_vector);
	VectorXf PropagateSignal(const VectorXf* training_example, 
		VectorXf* actv_reservoir, VectorXf* actv_out, int time_increment);
	void PropagateTrainingData(const MatrixXf& training_set, const MatrixXf& training_labels,
		MatrixXf* state_collection, MatrixXf* output_collection, bool force_output);
	MatrixXf RidgeRegression(const MatrixXf& state_collection, 
		const MatrixXf& output_collection, float lambda);
	MatrixXf TrainNetwork(const MatrixXf& data_set, const MatrixXf& data_labels);
	void CalculateNetMemory(const MatrixXf& training_data, const MatrixXf& training_labels);
	void CalculateNetMetrics(const MatrixXf& training_data, const MatrixXf& training_labels);
	std::vector<MatrixXf> GetWeightMatrices();
	std::vector<float> GetESNParameters();
	float GetNetFitness();
	void OutputNetMetrics(std::ofstream& out_file);
	void SaveNetToFile(std::string file_name);
};

#endif