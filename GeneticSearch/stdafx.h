// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
// Copyright 2015, Clayton Kotulak

#ifndef STDAFX_H
#define STDAFX_H

#pragma once

//#include "targetver.h"

#include <stdio.h>
#include <math.h>
#include <tchar.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <random>
#include <bitset>
#include <map>
#include <time.h>
#include <vector>
#include <Windows.h>
#include <thread>
#include <mutex>
#include <eigen/Eigen>

typedef Eigen::Matrix<float,Eigen::Dynamic,1> VectorXf;
typedef Eigen::Matrix<bool,Eigen::Dynamic,1> VectorXb;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXf;


#endif