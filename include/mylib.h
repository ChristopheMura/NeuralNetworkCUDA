#ifndef MYLIB_H
#define MYLIB_H

#include <iostream>
#include <vector>
#include <random>
#include <corecrt_math_defines.h>
#include "types.h"

void Run(int choix, std::vector<std::vector<double>>& X, std::vector<int>& y, int N);

NetworkParams Initialisation(int n0, int n1, int n2, int n3);

Activations ForwardPropagation(std::vector<std::vector<double>>& X, NetworkParams& params);

double log_loss(std::vector<std::vector<double>>& A, std::vector<int> y);

Gradients BackPropagation(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params, Activations& activations);

void Update(NetworkParams& params, Gradients& gradients, double learning_rate);

NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, int n2, double learning_rate=0.1, int n_iter=100);

std::vector<int> Predict(std::vector<std::vector<double>>& X, NetworkParams& params);

double Accuracy(std::vector<int>& y, std::vector<int>& y_pred);

void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params);

#endif