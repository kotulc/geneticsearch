// echo_state_net.cpp : Defines methods of the EchoStateNet class
// Copyright 2013, Clayton Kotualk

#include "echo_state_net.h"

// Validate the provided parameters 
void EchoStateNet::ValidateParameters(float * param_alpha, float * param_connectivity) {
	*param_connectivity = (*param_connectivity <= 0.0) ? 0.01 : *param_connectivity;
	*param_connectivity = (*param_connectivity > 1.0) ? 1.0 : *param_connectivity;
	*param_alpha = (*param_alpha <= 0.0) ? 0.01 : *param_alpha;
}

// Construct the ESN with randomly initialized weight matrices of the given dimensions
EchoStateNet::EchoStateNet(const MatrixXf& training_data, const MatrixXf& training_labels,
		int res_nodes, float alpha = 0.7, float connectivity = 0.2) {
	ValidateParameters(&alpha, &connectivity);
	in_nodes = training_data.cols();
	reservoir_nodes = res_nodes;
	out_nodes = training_labels.cols();
	net_edges = in_nodes*res_nodes + res_nodes*res_nodes*connectivity + out_nodes*in_nodes + out_nodes*res_nodes*2;
	net_states, net_weight, training_accuracy, validation_accuracy, net_mean, net_capacity, net_entropy, 
		net_memory, net_sensitivity = 0;
	net_connectivity = connectivity;
	net_alpha = alpha;

	activation_distribution = VectorXf::Zero(reservoir_nodes);
	net_weights = EchoStateNetWeights(in_nodes, res_nodes, out_nodes, connectivity);
	ScaleNetWeights();
	CalculateNetMetrics(training_data, training_labels);
}

// Construct the ESN with a vector of parameters and a vector of ESN weight matrices
EchoStateNet::EchoStateNet(const MatrixXf& training_data, const MatrixXf& training_labels, 
		std::vector<float> params, std::vector<MatrixXf> esnw) { //store in/out size
	ValidateParameters(&params[0], &params[1]);
	in_nodes = esnw[0].rows();
	reservoir_nodes = esnw[1].rows();
	out_nodes = esnw[2].cols();
	net_edges = esnw[0].size() + esnw[1].size()*params[1] + esnw[2].size() + esnw[3].size();
	net_states, net_weight, training_accuracy, validation_accuracy, net_mean, net_capacity, net_entropy, 
		net_memory, net_sensitivity = 0;
	net_connectivity = params[1];
	net_alpha = params[0];

	activation_distribution = VectorXf::Zero(reservoir_nodes);
	net_weights = EchoStateNetWeights(esnw);
	ScaleNetWeights();
	CalculateNetMetrics(training_data, training_labels);
}

// Scale the network reservoir weights using the given net_alpha value to (presumably) 
// establish the echo state network property for this network
void EchoStateNet::ScaleNetWeights() {
	Eigen::VectorXcf v_reservoir_eigen_complex = net_weights.w_reservoir.eigenvalues();
	VectorXf v_reservoir_eigen_float = v_reservoir_eigen_complex.real();
	float max_eigen = v_reservoir_eigen_float.maxCoeff(); 
	float min_eigen = v_reservoir_eigen_float.minCoeff() * float(-1.0);
	// Spectral radius is equal to the greatest absolute eigenvalue
	float spectral_radius = (max_eigen > min_eigen) ? max_eigen : min_eigen;
	if (spectral_radius != 0) {
		net_weights.w_reservoir *= float(1/spectral_radius);
	}
	net_weights.w_reservoir *= net_alpha;
}

// Return the sigmoid g, of given matrix x: g = 1 ./ (1 + e.^(-1.*x));
VectorXf EchoStateNet::GetSigmoidActivation(VectorXf* propagation_vector) {
	*propagation_vector *= float(-1.0);
	for (int i=0; i<propagation_vector->size(); i++) {
		// Assign each element of the given matrix corresponding to e .^(-1.*x) 
		float temp = std::pow( float(2.71828182845904523536), (*propagation_vector)(i) );
		(*propagation_vector)(i) = temp;
		if (temp != temp) {
			int j=0;
		}
	}
	VectorXf ones_vector = VectorXf::Ones(propagation_vector->size());
	*propagation_vector += ones_vector;
	*propagation_vector = (ones_vector.cwiseQuotient(*propagation_vector));
	VectorXf sigmoid_activation = *propagation_vector;
	return sigmoid_activation;
}

// Propagate the given signal (training_example) through the activation reservoir and propagate
// the actv_out signal through the feedback connections. Return the sum of signals to the reservoir.
VectorXf EchoStateNet::PropagateSignal(const VectorXf* training_example, VectorXf* actv_reservoir, 
		VectorXf* actv_out, int time_increment = 2) {
	VectorXf propagation_in = net_weights.w_in.transpose() * *training_example;
	VectorXf propagation_reservoir = net_weights.w_reservoir.transpose() * *actv_reservoir;
	VectorXf propagation_back = net_weights.w_back.transpose() * *actv_out;
	VectorXf propagation_out = net_weights.w_out.block(0,0,training_example->size(),out_nodes).transpose() * *training_example;
	propagation_out += net_weights.w_out.block(training_example->size(),0,reservoir_nodes,out_nodes).transpose() * *actv_reservoir;
	*actv_out = GetSigmoidActivation( &propagation_out ); 
	VectorXf propagation_sum = propagation_in + propagation_reservoir + propagation_back;
	*actv_reservoir = GetSigmoidActivation( &propagation_sum );
	return propagation_sum;
}

// Populate the M X (N+K) state_collection Matrix with input (N elements) and reservoir (K elements) 
// node activations after propagating the training data set (M rows) through the network.
void EchoStateNet::PropagateTrainingData(const MatrixXf& training_set, const MatrixXf& training_labels,
		MatrixXf* state_collection, MatrixXf* output_collection, bool force_output = 1) {
	VectorXf activation_reservoir = VectorXf::Zero(reservoir_nodes);
	VectorXf activation_out = VectorXf::Zero(out_nodes);
	VectorXf training_example;

	// Prime the network with the first set of examples <= 100.
	for (int i = 0; i<training_set.rows() && i<100; i++) {
		training_example = training_set.row(i);
		PropagateSignal(&training_example, &activation_reservoir, &activation_out);
		// Introduce teacher forcings
		activation_out = training_labels.row(i);
	}
	// Propagate all training data through the network and build the state_collection matrix. 
	for (int i = 0; i<training_set.rows(); i++) {
		training_example = training_set.row(i);
		PropagateSignal(&training_example, &activation_reservoir, &activation_out);
		state_collection->block(i, 0, 1, training_set.cols()) = training_set.row(i);
		state_collection->block(i, training_set.cols(), 1, activation_reservoir.rows()) = activation_reservoir.transpose();
		// Introduce "teacher" forcings if bool force_output is true
		if (force_output) { 
			activation_out = training_labels.row(i); 
		}
		output_collection->row(i) = activation_out.transpose(); 
	}
}

// Return the matrix of weights W that minimizes the MSE with the following: 
// w_out = (state_corr + lambda*I)^-1 * state_out_xcorr
MatrixXf EchoStateNet::RidgeRegression(const MatrixXf& state_collection, const MatrixXf& output_collection, float lambda = 1) { //test
	MatrixXf state_correlation = state_collection.transpose()*state_collection;
	MatrixXf state_out_xcorrelation = state_collection.transpose()*output_collection;
	MatrixXf lambda_identity = MatrixXf::Identity(state_correlation.rows(), state_correlation.cols());
	lambda_identity *= lambda;
	MatrixXf inverse_term = (state_correlation + lambda_identity);
	MatrixXf w_out = inverse_term.inverse()*state_out_xcorrelation;
	return w_out;
}

// Return the M X (N+K) state collection matrix containing the reservoir activation states
// against the validation data set. Set the network training and validation accuracies.  
MatrixXf EchoStateNet::TrainNetwork(const MatrixXf& data_set, const MatrixXf& data_labels) {
	int validation_rows = (int)data_set.rows()*0.3;
	const MatrixXf validation_set = data_set.block(0, 0, validation_rows, data_set.cols());
	const MatrixXf training_set = data_set.block(validation_rows, 0, data_set.rows() - validation_rows, data_set.cols());
	const MatrixXf validation_labels = data_labels.block(0, 0, validation_rows, data_labels.cols());
	const MatrixXf training_labels = data_labels.block(validation_rows, 0, data_labels.rows() - validation_rows, data_labels.cols());
	
	// state_collection is a Matrix of size M X (N+K); training_set (M) X ( input (N) + reservoir (K) ) node activations
	MatrixXf state_collection = MatrixXf(training_set.rows(), in_nodes + reservoir_nodes);
	MatrixXf output_collection = MatrixXf(training_set.rows(), out_nodes);
	MatrixXf::Index max_out_index, max_label_index;
	int correct_output = 0;
	PropagateTrainingData(training_set, training_labels, &state_collection, &output_collection, 1);
	net_weights.w_out = RidgeRegression(state_collection, training_labels);
	PropagateTrainingData(training_set, training_labels, &state_collection, &output_collection, 0);
	for (int i = 0; i < training_labels.rows(); i++) {
		output_collection.row(i).maxCoeff(&max_out_index);
		training_labels.row(i).maxCoeff(&max_label_index);
		if (max_out_index == max_label_index) { correct_output++; }
	}
	training_accuracy = (float)correct_output / training_labels.rows();

	state_collection = MatrixXf(validation_set.rows(), in_nodes + reservoir_nodes);
	output_collection = MatrixXf(validation_set.rows(), out_nodes);
	correct_output = 0;
	PropagateTrainingData(validation_set, validation_labels, &state_collection, &output_collection, 0);
	for (int i = 0; i < validation_labels.rows(); i++) {
		output_collection.row(i).maxCoeff(&max_out_index);
		validation_labels.row(i).maxCoeff(&max_label_index);
		if (max_out_index == max_label_index) { correct_output++; }
	}
	validation_accuracy = (float)correct_output / validation_labels.rows();
	return state_collection.block(0, validation_set.cols(), state_collection.rows(), reservoir_nodes);
}

void EchoStateNet::CalculateNetMemory(const MatrixXf& training_data, const MatrixXf& training_labels) { //test
	int propagations = 1;
	float signal_change = 1;
	VectorXf prev_prop_sum;
	VectorXf propagation_sum;
	VectorXf prop_difference = VectorXf::Zero(reservoir_nodes);
	VectorXf activation_reservoir = VectorXf::Zero(reservoir_nodes);
	VectorXf activation_out = VectorXf::Zero(out_nodes);
	VectorXf null_signal = VectorXf::Zero(training_data.cols());
	VectorXf training_example = training_data.row(0);
	propagation_sum = PropagateSignal(&training_example, &activation_reservoir, &activation_out); 
	float propagation_start = propagation_sum.sum();
	while (signal_change != 0 && propagations <= 100) {
		prev_prop_sum = propagation_sum;
		propagation_sum = PropagateSignal(&null_signal, &activation_reservoir, &activation_out); 
		activation_out = training_labels.row(0); 
		prop_difference = prev_prop_sum - propagation_sum;
		signal_change = prop_difference.sum();
		propagations++;
	}
	net_memory = (propagation_start - signal_change)/propagations;
}

// Run the provided training examples through the matrix to calculate the network metrics
void EchoStateNet::CalculateNetMetrics(const MatrixXf& training_data, const MatrixXf& training_labels) { //Mean, Memory, sensitivity states=1	
	if (training_data.size() < 1 || training_data.rows() != training_labels.rows()) return;

	std::vector<bool> activations;
	std::map<std::vector<bool>,int> state_map;
	std::map<std::vector<bool>,int>::iterator state_iterator;
	MatrixXf reservoir_activations;
	VectorXb reservoir_bool_actv = VectorXb::Zero(reservoir_nodes); 
	VectorXf actv_reservoir_sum = VectorXf::Zero(reservoir_nodes);

	// Propagate all training data through the network and train the output weights
	reservoir_activations = TrainNetwork(training_data, training_labels);
	// Node activations sum calculated on the distribution of activations
	actv_reservoir_sum += reservoir_activations.colwise().sum();

	// Cast the node activations to 0 or 1 values.
	for (int i = 0; i < reservoir_activations.rows(); i++) { //Test here
		for (int j = 0; j < reservoir_nodes; j++) {
			float temp = std::floor(reservoir_activations(i, j));
			reservoir_bool_actv(j) = (bool)temp;
		}
		// Map the boolean state vector. Insert unique states else increment the state occurence count
		activations.assign(reservoir_bool_actv.data(), reservoir_bool_actv.data() + reservoir_bool_actv.size());
		state_iterator = state_map.find(activations);
		if (state_iterator == state_map.end()) {
			state_map.insert(std::pair<std::vector<bool>, int>(activations, 1));
		}
		else { state_iterator->second += 1; }
	}
	// activation_distribution is the mean firing value for each reservoir nodes
	activation_distribution = actv_reservoir_sum / reservoir_activations.rows();

	// net_sensitivity is the mean firing value of all reservoir nodes
	net_sensitivity = activation_distribution.sum() / activation_distribution.size();

	// net_entropy is calculated sum( -p(x)log(x) ) where x is a unique boolean activation state
	for (state_iterator = state_map.begin(); state_iterator != state_map.end(); state_iterator++) {
		float actv_probability = float(state_iterator->second) / reservoir_activations.rows();
		net_entropy += actv_probability * (std::log(actv_probability) / std::log(2.0));
	}
	net_entropy = -1.0 * net_entropy;

	// net_states is the number of unique reservoir boolean activation states
	net_states = state_map.size();
	net_mean = net_weights.w_reservoir.sum()/net_weights.w_reservoir.size();
	net_weight = net_weights.w_reservoir.sum();
	net_capacity = net_entropy*( (float)net_states/reservoir_nodes );
	CalculateNetMemory(training_data, training_labels);
}

// Returns a copy of the vector of weight matrices that compose the network edge weights
std::vector<MatrixXf> EchoStateNet::GetWeightMatrices() {
	return net_weights.GetWeightMatrices();
}

// Returns the vector of network parameters 
std::vector<float> EchoStateNet::GetESNParameters() {
	std::vector<float> parameters;
	parameters.push_back(net_alpha);
	parameters.push_back(net_connectivity);
	return parameters;
}

// Returns the calculated fitness of the current network
float EchoStateNet::GetNetFitness() { 
	return validation_accuracy;
}

void EchoStateNet::OutputNetMetrics(std::ofstream& out_file) {
	out_file << "Echo State Network Metrics..." << std::endl;
	out_file << "Total reservoir nodes: " << reservoir_nodes << std::endl;
	out_file << "Total network edges: " << net_edges << std::endl;
	out_file << "Total network weight: " << net_weight << std::endl;
	out_file << "Total network states: " << net_states << std::endl;
	out_file << "Training accuracy: " << training_accuracy << std::endl;
	out_file << "Validation accuracy: " << validation_accuracy << std::endl;
	out_file << "Mean edge weight value: " << net_mean << std::endl;
	out_file << "Network capacity: " << net_capacity << std::endl;
	out_file << "Network entropy: " << net_entropy << std::endl;
	out_file << "Network memory: " << net_memory << std::endl;
	out_file << "Network sensitivity: " << net_sensitivity << std::endl;
	out_file << "Network connectivity: " << net_connectivity << std::endl;
	out_file << "Scaling value (alpha): " << net_alpha << std::endl;
	out_file << "Node activation distribution: " << activation_distribution.transpose() << std::endl;
}

void EchoStateNet::SaveNetToFile(std::string file_name) {
	FILE* out_file;
	std::vector<MatrixXf> weights = net_weights.GetWeightMatrices();
	out_file = std::fopen(file_name.c_str(), "ab");

	if (out_file != NULL) { 
		float esn_header[] = {net_alpha, net_connectivity, (float) 3*(weights.size()>0)};
		std::fwrite (esn_header, 1, sizeof(esn_header), out_file);
		for (int i=0; i<weights.size(); i++) {
			float matrix_header[] = {(float) weights[i].rows(), (float) weights[i].cols(), (float) 3*(i+1<weights.size())};
			std::fwrite(matrix_header, 1, sizeof(matrix_header), out_file);
			size_t m_size = sizeof(float)*weights[i].size();
			std::fwrite(weights[i].data(), 1, sizeof(float)*weights[i].size(), out_file);
		}
		std::fclose(out_file);
	}
	else { 
		std::cout << "Error opening output file \"" << file_name << ",\" ESN was not saved." << std::endl; 
	}
}