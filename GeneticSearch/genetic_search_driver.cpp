// genetic_search_driver.cpp : Defines the entry point for the console application.
// Copyright 2013, Clayton Kotualk

#include "genetic_search.h"
#include "stdafx.h"


// Wait for user input before terminating the application
void ExitSearch (int exit_code) {
	std::string s;
	std::cout << "\n\nPress enter to close the application.";
	std::getline(std::cin, s);
	exit(exit_code);
}

MatrixXf LabelVectorToClassMatrix(const VectorXf& label_vector) {
	float min_label = label_vector.minCoeff();
	float max_label = label_vector.maxCoeff();
	int matrix_cols = std::ceil(max_label - min_label + 1);
	int j = 0;
	MatrixXf output_classes = MatrixXf::Zero(label_vector.size(), matrix_cols);
	for (int i=0; i<output_classes.rows(); i++) {
		// Round the vector float value to the nearest int
		j = std::floor(label_vector(i) + 0.5);
		output_classes(i,j) = 1;
	}
	return output_classes;
}

// Return a vector containing all data elements in the given string 
VectorXf ParseString(std::string data_string) {
	VectorXf data_line;
	std::vector<std::string> tokens; 
	std::istringstream iss(data_string);
	std::copy( std::istream_iterator<std::string>(iss),
		std::istream_iterator<std::string>(),
		std::back_inserter< std::vector< std::string > >(tokens) );

	if (tokens.size() > 0 ) {
		data_line = VectorXf(tokens.size());
		for (int i=0; i<tokens.size(); i++) {
			try { data_line(i) = std::stof(tokens[i]); }
			catch (const std::invalid_argument& ia) {
				std::cout << "Error: Unable to cast value, invalid training data type: ";
				ExitSearch(1);
			}
		}
	}
	return data_line;
}

// Return the data matrix containing all data elements in the supplied file.
void ParseDataFile(MatrixXf* data_ptr, MatrixXf* label_ptr, std::string data_file_name) {
	std::string string_line;
	VectorXf data_labels;
	MatrixXf data_matrix; 
	std::cout << "Loading training data..." << std::endl;
	std::ifstream data_file(data_file_name);
	if (data_file.is_open() ) {
		int line_count = 0;
		int data_points = 0;
		while( std::getline(data_file, string_line) ) {
			if( (ParseString(string_line)).size() != 0) {
				line_count++;
				data_points = (ParseString(string_line)).size();
			}
		}
		data_file.clear();
		data_file.seekg(0, std::ios::beg);
		if (data_points <= 1) { fputs("Error: Invalid training data format.",stderr); ExitSearch(1); }
		data_matrix = MatrixXf(line_count, data_points-1);
		data_labels = VectorXf(line_count);
		for (int i=0; i<line_count; i++) {
			std::getline(data_file, string_line);
			VectorXf data_line = ParseString(string_line);
			if (data_line.size() != 0) {
				if (data_line.size() != data_points) { 
					std::cout << "Error: Invalid training data format";
					ExitSearch(1);
				}
				data_matrix.row(i) = data_line.segment(1,data_points-1).transpose();
				data_labels(i) = data_line(0);
			}
		}
		data_file.close();
		*data_ptr = data_matrix; //Make sure they are copied instead of trashed
		*label_ptr = LabelVectorToClassMatrix(data_labels);
		std::cout << "Training data loaded.\n" << std::endl;
	}
	else {
		std::cout << "Error: Could not open data file \""<< data_file_name <<"\"." << std::endl;
		ExitSearch(1);
	}
	return; 
}

// Return a map of parameter pairs from the config file where [parameter_name]=[parameter_value]
std::map<std::string,std::string> ParseConfigFile(std::string config_file_name) {
	std::string string_line;
	std::string parameter_name;
	std::string parameter_value; 
	std::map<std::string,std::string> parameter_map;
	std::ifstream config_file(config_file_name);
	if (config_file.is_open() ) {
		while ( std::getline(config_file, string_line) ) {
			string_line.erase( std::remove_if( string_line.begin(), string_line.end(), ::isspace), string_line.end());
			size_t string_position = string_line.find('=');
			if (string_position != std::string::npos) {
				parameter_name = string_line.substr(0, string_position);
				parameter_value = string_line.substr(string_position+1, string_line.length()-1 );
				parameter_map.insert(std::pair<std::string,std::string>(parameter_name, parameter_value));
			}
		}
		config_file.close();
	}
	else {
		std::cout << "Error: Could not open config file \""<< config_file_name <<"\"." << std::endl;
	}
	return parameter_map; 
}

// Convert a float array to a MatrixXf of the given dimensions
MatrixXf ArrayToMatrix(float * f_array, int row, int col) {
	return  Eigen::Map<MatrixXf>(f_array, row, col);
}

// Read the binary block header, build and append the MatrixXf if the 
// end of the file has not yet been reached. 
void ParseBinaryMatrices(std::vector<MatrixXf>* matrices, FILE* in_file, 
	long file_size, long* file_read, size_t header_length) {
	if (*file_read >= file_size || header_length <= 0) return;
	if (file_size < header_length*sizeof(float)+*file_read) { fputs("Memory error",stderr); ExitSearch(1); }

	// allocate memory to contain the matrix header
	float * buffer = new float[header_length];
	if (buffer == NULL) { fputs("Memory error",stderr); ExitSearch(2); }
	// copy the header into the buffer:
	size_t result = fread(buffer, sizeof(float), header_length, in_file);
	if (result != header_length) { fputs ("Reading error",stderr); ExitSearch(3); }
	*file_read += header_length*sizeof(float);
	int matrix_rows = buffer[0];
	int matrix_cols = buffer[1];

	size_t matrix_size = matrix_rows*matrix_cols;
	float * matrix_buffer = new float[matrix_size];
	if (matrix_buffer == NULL) { fputs("Memory error",stderr); ExitSearch(2); }
	// copy the matrix data into the buffer:
	result = fread(matrix_buffer, sizeof(float), matrix_size,in_file);
	if (result != matrix_size) { fputs ("Reading error",stderr); ExitSearch(3); }
	*file_read += (matrix_size)*sizeof(float);
	MatrixXf parsed_matrix = ArrayToMatrix(matrix_buffer, matrix_rows, matrix_cols);
	matrices->push_back(parsed_matrix);
	free(matrix_buffer);

	size_t header_size = buffer[2];
	free(buffer);

	// Continue parsing the file recursively until the end has been reached
	ParseBinaryMatrices(matrices, in_file, file_size, file_read, header_size);
}

// Returns a pair (network_parameters, network_weights) Read the network 
// header and then retrieve the weight matrices.
std::pair< std::vector<float>, std::vector<MatrixXf> > ParseBinaryParams(FILE * in_file, long file_size, long* file_read) {
	size_t parameter_length = 3;
	if (file_size < parameter_length*sizeof(float)+*file_read) { fputs("Memory error",stderr); ExitSearch(1); }

	// allocate memory to contain the network header block:
	float * buffer = new float[parameter_length];
	if (buffer == NULL) { fputs("Memory error",stderr); ExitSearch(2); }
	// copy the network parameter header into the buffer:
	size_t result = fread(buffer, sizeof(float), parameter_length, in_file);
	if (result != parameter_length) { fputs ("Reading error",stderr); ExitSearch(3); }
	*file_read += parameter_length*sizeof(float);

	std::vector<float> parameters;
	std::vector<MatrixXf> matrices;
	float net_alpha = buffer[0];
	parameters.push_back(net_alpha);
	float net_connectivity = buffer[1];
	parameters.push_back(net_connectivity);
	size_t header_size = buffer[2];
	free(buffer);

	ParseBinaryMatrices(&matrices, in_file, file_size, file_read, header_size);
	return std::pair< std::vector<float>, std::vector<MatrixXf> >(parameters, matrices);
}

// Read the binary file and extract all stored network data 
std::vector< std::pair<std::vector<float>, std::vector<MatrixXf>> > ParseBinaryFile(std::string file_name) {
	FILE * in_file;
	long file_size = 0;
	long file_read = 0;

	std::vector < std::pair<std::vector<float>, std::vector<MatrixXf>> > networks;
	in_file = fopen( file_name.c_str() , "rb" );
	if (in_file==NULL) { fputs("Error: Unable to open binary data file.",stderr); return networks; }
	
	std::cout << "Loading saved network data..." << std::endl;
	// obtain file size:
	fseek(in_file , 0 , SEEK_END);
	file_size = ftell(in_file);
	rewind(in_file);

	while (file_read < file_size) {
		networks.push_back(ParseBinaryParams(in_file, file_size, &file_read));
	}
	std::cout << "Network data loaded.\n" << std::endl;
	fclose(in_file);
	return networks;
}

// Return the mean normalized instance of the given data_matrix
MatrixXf NormalizeMatrixElements(MatrixXf* data_matrix) {
	float range = data_matrix->maxCoeff() - data_matrix->minCoeff();
	float data_matrix_mean = (float)data_matrix->sum() / data_matrix->size();
	MatrixXf mean_data_matrix = MatrixXf::Ones(data_matrix->rows(), data_matrix->cols());
	mean_data_matrix *= data_matrix_mean;
	MatrixXf delta_data_matrix = *data_matrix - mean_data_matrix;
	return delta_data_matrix / range;
}

// Look for the key_value in the string_map. Cast this value to a float else return 0.
float MapStringToFloat(std::map<std::string,std::string>* string_map, std::string key_value) {
	float float_value = 0;
	std::map<std::string,std::string>::iterator map_iterator;
	map_iterator = string_map->find(key_value);
	if ( map_iterator != string_map->end()) { 
		try { float_value = std::stof(map_iterator->second); }
		catch (const std::invalid_argument& ia) {
			std::cout << "Unable to cast " << map_iterator->second << " to float: " << ia.what() << std::endl;
		}
	}
	return float_value;
}

// Look for the GeneticSearch parameters in the given map and cast to the appropriate
// type. Initialize a search instance with these parameters.
void InitializeSearch(std::map<std::string,std::string>* parameter_map, 
	const MatrixXf& training_data, const MatrixXf& training_labels, std::string bin_file_name) {
	std::map<std::string,std::string>::iterator map_iterator;
	std::string log_file_name = "search_log.txt";
	map_iterator = parameter_map->find("log_file_name");
	if ( map_iterator != parameter_map->end()) { log_file_name = map_iterator->second; }
	int population_size = (int)MapStringToFloat(parameter_map, "population_size");
	int max_generations = (int)MapStringToFloat(parameter_map, "max_generations");
	int reservoir_nodes = (int)MapStringToFloat(parameter_map, "reservoir_nodes");
	float new_random_population = MapStringToFloat(parameter_map, "new_random_population");
	float mutation_rate = MapStringToFloat(parameter_map, "mutation_rate");
	float mutation_range = MapStringToFloat(parameter_map, "mutation_range");
	float crossover_rate = MapStringToFloat(parameter_map, "crossover_rate");
	float train_sample = MapStringToFloat(parameter_map, "train_sample");
	GeneticSearch(training_data, training_labels, reservoir_nodes, population_size, max_generations,
		new_random_population, mutation_rate, mutation_range, crossover_rate, train_sample, log_file_name, bin_file_name);
}

// Control the execution of the application
int _tmain(int argc, _TCHAR* argv[])
{
	std::map<std::string,std::string> parameter_map = ParseConfigFile("config.txt");
	std::map<std::string,std::string>::iterator map_iterator;
	std::string bin_file_name = "search_data.bin";
	std::string train_file_name = "training_data.txt";
	bool overwrite_bin = 1;
	map_iterator = parameter_map.find("bin_file_name");
	if ( map_iterator != parameter_map.end()) { bin_file_name = map_iterator->second; }
	map_iterator = parameter_map.find("train_file_name");
	if ( map_iterator != parameter_map.end()) { train_file_name = map_iterator->second; }
	overwrite_bin = (bool)MapStringToFloat(&parameter_map, "overwrite_data");

	std::vector< std::pair<std::vector<float>, std::vector<MatrixXf>> > networks;

	FILE * binary_file;
	binary_file = std::fopen(bin_file_name.c_str(), "r");
	if  (binary_file != NULL) { 
		networks = ParseBinaryFile(bin_file_name);
	}
	if (overwrite_bin == 1) {
		binary_file = std::fopen(bin_file_name.c_str(), "w");
		if (binary_file == NULL) { 
			std::cout << "Data file " << bin_file_name << " not found. A new copy has been created.\n" << std::endl;
		}
		else { std::fclose(binary_file); }
	}
	MatrixXf training_data;
	MatrixXf training_labels;
	ParseDataFile(&training_data, &training_labels, train_file_name);
	training_data = NormalizeMatrixElements(&training_data);
	InitializeSearch(&parameter_map, training_data, training_labels, bin_file_name);
	ExitSearch(0);
	return 0;
}
