#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct NetworkParams
{
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b2;
    std::vector<std::vector<double>> W3;
    std::vector<double> b3;
};

struct Activations
{
    std::vector<std::vector<double>> A1;    // (n1, m)
    std::vector<std::vector<double>> A2;    // (n2, m)
    std::vector<std::vector<double>> A3;    // (n3, m)
    std::vector<std::vector<double>> Z1;    // (n1, m)
    std::vector<std::vector<double>> Z2;    // (n2, m)
    std::vector<std::vector<double>> Z3;    // (n3, m)
};

struct Gradients
{
    std::vector<std::vector<double>> dW1;
    std::vector<double> db1;
    std::vector<std::vector<double>> dW2;
    std::vector<double> db2;
    std::vector<std::vector<double>> dW3;
    std::vector<double> db3;
};

#endif