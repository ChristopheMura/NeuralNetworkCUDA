#include "mylib.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

#include "cuda_kernels.cuh"

void Run(int choix, std::vector<std::vector<double>>& X, std::vector<int>& y, int N)
{
    std::random_device rd;
    std::mt19937 gen(42);

    switch(choix)
    {
        case 1: // Spirale
        {
            std::cout << "Dataset choisi: Spirale entrelacée" << std::endl;
            std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
            for (int i = 0; i < N; i++)
            {
                double t = uniform_dist(gen) * 4 * M_PI; // 2 tours
                double r = t / (4 * M_PI); // rayon croissant
                double noise = 0.1 * uniform_dist(gen);
                
                if (i < N/2)
                {
                    // Spirale classe 0
                    X.push_back({r * std::cos(t) + noise, r * std::sin(t) + noise});
                    y.push_back(0);
                }
                else
                {
                    // Spirale classe 1 (décalée)
                    X.push_back({r * std::cos(t + M_PI) + noise, r * std::sin(t + M_PI) + noise});
                    y.push_back(1);
                }
            }
            break;
        }

        case 2: // XOR
        {
            std::cout << "Dataset choisi: XOR" << std::endl;
            std::normal_distribution<> noise_dist(0.0, 0.1);
            for (int i = 0; i < N; i++)
            {
                if (i < N/4)
                {
                    // Quadrant 1: (1,1) -> classe 0
                    X.push_back({1.0 + noise_dist(gen), 1.0 + noise_dist(gen)});
                    y.push_back(0);
                }
                else if (i < N/2)
                {
                    // Quadrant 2: (-1,1) -> classe 1
                    X.push_back({-1.0 + noise_dist(gen), 1.0 + noise_dist(gen)});
                    y.push_back(1);
                }
                else if (i < 3*N/4)
                {
                    // Quadrant 3: (-1,-1) -> classe 0
                    X.push_back({-1.0 + noise_dist(gen), -1.0 + noise_dist(gen)});
                    y.push_back(0);
                }
                else
                {
                    // Quadrant 4: (1,-1) -> classe 1
                    X.push_back({1.0 + noise_dist(gen), -1.0 + noise_dist(gen)});
                    y.push_back(1);
                }
            }
            break;
        }

        case 3: // Damier
        {
            std::cout << "Dataset choisi: Damier" << std::endl;
            std::uniform_real_distribution<> uniform_dist(-2.0, 2.0);
            std::normal_distribution<> noise_dist(0.0, 0.05);
            for (int i = 0; i < N; i++)
            {
                double x_val = uniform_dist(gen) + noise_dist(gen);
                double y_val = uniform_dist(gen) + noise_dist(gen);
                
                // Damier 2x2
                int x_grid = (x_val > 0) ? 1 : 0;
                int y_grid = (y_val > 0) ? 1 : 0;
                int class_label = (x_grid + y_grid) % 2;
                
                X.push_back({x_val, y_val});
                y.push_back(class_label);
            }
            break;
        }

        case 4: // Cercles concentriques multiples
        {
            std::cout << "Dataset choisi: Cercles concentriques multiples" << std::endl;
            std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
            std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
            std::normal_distribution<> noise_dist(0.0, 0.05);
            
            for (int i = 0; i < N; i++)
            {
                double radius = std::sqrt(uniform_dist(gen)) * 2.0;
                double angle = angle_dist(gen);
                double x_val = radius * std::cos(angle) + noise_dist(gen);
                double y_val = radius * std::sin(angle) + noise_dist(gen);
                
                // Classification par anneaux : 0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0
                int ring = (int)(radius / 0.5);
                int class_label = ring % 2;
                
                X.push_back({x_val, y_val});
                y.push_back(class_label);
            }
            break;
        }

        case 5: // Linéairement séparable
        {
            std::cout << "Dataset choisi: Données linéairement séparables" << std::endl;
            std::normal_distribution<> dist1_x(-1.0, 0.5);
            std::normal_distribution<> dist1_y(-1.0, 0.5);
            std::normal_distribution<> dist2_x(1.0, 0.5);
            std::normal_distribution<> dist2_y(1.0, 0.5);

            for (int i=0; i<N/2; i++)
            {
                X.push_back({dist1_x(gen), dist1_y(gen)});
                y.push_back(0);
            }
            for (int i=0; i<N/2; i++)
            {
                X.push_back({dist2_x(gen), dist2_y(gen)});
                y.push_back(1);
            }
            break;
        }

        case 6: // Croissants de lune
        {
            std::cout << "Dataset choisi: Croissants de lune" << std::endl;
            std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
            std::normal_distribution<> noise_dist(0.0, 0.1);
            
            for (int i = 0; i < N; i++)
            {
                if (i < N/2)
                {
                    // Premier croissant
                    double t = uniform_dist(gen) * M_PI;
                    double r = 1.0;
                    X.push_back({r * std::cos(t) + noise_dist(gen), 
                                r * std::sin(t) + noise_dist(gen)});
                    y.push_back(0);
                }
                else
                {
                    // Deuxième croissant (décalé)
                    double t = uniform_dist(gen) * M_PI;
                    double r = 1.0;
                    X.push_back({r * std::cos(t) + 1.0 + noise_dist(gen), 
                                r * std::sin(t) - 0.5 + noise_dist(gen)});
                    y.push_back(1);
                }
            }
            break;
        }

        case 7: // Cercles concentriques (original)
        {
            std::cout << "Dataset choisi: Cercles concentriques (original)" << std::endl;
            std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
            std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);
            std::normal_distribution<> noise_dist(0.0, 0.1);

            for (int i = 0; i < N; i++)
            {
                double radius = std::sqrt(uniform_dist(gen));
                double angle = angle_dist(gen);
                
                if (i < N/2)
                {
                    radius *= 0.5;
                    X.push_back({radius * std::cos(angle) + noise_dist(gen), 
                                radius * std::sin(angle) + noise_dist(gen)});
                    y.push_back(0);
                }
                else
                {
                    radius = 0.6 + radius * 0.4;
                    X.push_back({radius * std::cos(angle) + noise_dist(gen), 
                                radius * std::sin(angle) + noise_dist(gen)});
                    y.push_back(1);
                }
            }
            break;
        }

        default:
            std::cout << "Choix invalide, utilisation du dataset par défaut (linéairement séparable)" << std::endl;
            Run(5, X, y, N);
            break;
    }
}

NetworkParams Initialisation(int n0, int n1, int n2, int n3)
{
    NetworkParams params;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    // Couche 1: n0 -> n1
    params.W1.resize(n1, std::vector<double>(n0));
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            params.W1[i][j] = dist(gen);
        }
    }
    params.b1.resize(n1, 0.0);

    // Couche 2: n1 -> n2
    params.W2.resize(n2, std::vector<double>(n1));
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            params.W2[i][j] = dist(gen);
        }
    }
    params.b2.resize(n2, 0.0);

    // Couche 3: n2 -> n3
    params.W3.resize(n3, std::vector<double>(n2));
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<n2; j++)
        {
            params.W3[i][j] = dist(gen);
        }
    }
    params.b3.resize(n3, 0.0);

    return params;
}

Activations ForwardPropagation(std::vector<std::vector<double>>& X, NetworkParams& params)
{
    Activations activations;

    int m = X.size();
    int n0 = X[0].size();
    int n1 = params.W1.size();
    int n2 = params.W2.size();
    int n3 = params.W3.size();

    // Initialisation des matrices
    activations.Z1.resize(n1, std::vector<double>(m));
    activations.A1.resize(n1, std::vector<double>(m));
    activations.Z2.resize(n2, std::vector<double>(m));
    activations.A2.resize(n2, std::vector<double>(m));
    activations.Z3.resize(n3, std::vector<double>(m));
    activations.A3.resize(n3, std::vector<double>(m));

    // Calcul Z1 = W1 * X + b1
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.Z1[i][j] = params.b1[i];
            for (int k=0; k<n0; k++)
            {
                activations.Z1[i][j] += params.W1[i][k]*X[j][k];
            }
        }
    }

    // Calcul A1 = sigmoid(Z1)
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.A1[i][j] = 1.0 / (1.0 + std::exp(-activations.Z1[i][j]));
        }
    }

    // Calcul Z2 = W2 * A1 + b2
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.Z2[i][j] = params.b2[i];
            for (int k=0; k<n1; k++)
            {
                activations.Z2[i][j] += params.W2[i][k] * activations.A1[k][j];
            }
        }
    }

    // Calcul A2 = sigmoid(Z2)
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.A2[i][j] = 1.0 / (1.0 + std::exp(-activations.Z2[i][j]));
        }
    }

    // Calcul Z3 = W3 * A2 + b3
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.Z3[i][j] = params.b3[i];
            for (int k=0; k<n2; k++)
            {
                activations.Z3[i][j] += params.W3[i][k] * activations.A2[k][j];
            }
        }
    }

    // Calcul A3 = sigmoid(Z3)
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<m; j++)
        {
            activations.A3[i][j] = 1.0 / (1.0 + std::exp(-activations.Z3[i][j]));
        }
    }

    return activations;
}

double log_loss(std::vector<std::vector<double>>& A, std::vector<int> y)
{
    int m = y.size();
    double cost = 0.0;
    double eps = 1e-15;

    for (int i=0; i<m; i++)
    {
        double ai = std::min(std::max(A[0][i], eps), 1.0 - eps);
        cost += -y[i] * std::log(ai) - (1 - y[i])*std::log(1 - ai);
    }

    return cost / m;
}

Gradients BackPropagation(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params, Activations& activations)
{
    Gradients gradients;

    int m = y.size();
    int n0 = X[0].size();
    int n1 = activations.A1.size();
    int n2 = activations.A2.size();
    int n3 = activations.A3.size();

    // Initialisation des gradients
    gradients.dW1.resize(n1, std::vector<double>(n0, 0.0));
    gradients.db1.resize(n1, 0.0);
    gradients.dW2.resize(n2, std::vector<double>(n1, 0.0));
    gradients.db2.resize(n2, 0.0);
    gradients.dW3.resize(n3, std::vector<double>(n2, 0.0));
    gradients.db3.resize(n3, 0.0);

    // Calcul dZ3 = A3 - y
    std::vector<std::vector<double>> dZ3(n3, std::vector<double>(m));
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<m; j++)
        {
            dZ3[i][j] = activations.A3[i][j] - y[j];
        }
    }

    // Calcul dW3 = (1/m) * dZ3 . A2.T
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<n2; j++)
        {
            gradients.dW3[i][j] = 0.0;
            for (int k=0; k<m; k++)
            {
                gradients.dW3[i][j] += dZ3[i][k] * activations.A2[j][k];
            }
            gradients.dW3[i][j] /= m;
        }
    }

    // Calcul db3 = (1/m) * sum(dZ3)
    for (int i=0; i<n3; i++)
    {
        gradients.db3[i] = 0.0;
        for (int j=0; j<m; j++)
        {
            gradients.db3[i] += dZ3[i][j];
        }
        gradients.db3[i] /= m;
    }

    // Calcul dZ2 = W3.T . dZ3 * A2 * (1 - A2)
    std::vector<std::vector<double>> dZ2(n2, std::vector<double>(m));
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<m; j++)
        {
            double W3T_dZ3 = 0.0;
            for (int k=0; k<n3; k++)
            {
                W3T_dZ3 += params.W3[k][i] * dZ3[k][j];
            }
            dZ2[i][j] = W3T_dZ3 * activations.A2[i][j] * (1 - activations.A2[i][j]);
        }
    }

    // Calcul dW2 = (1/m) * dZ2 . A1.T
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            gradients.dW2[i][j] = 0.0;
            for (int k=0; k<m; k++)
            {
                gradients.dW2[i][j] += dZ2[i][k] * activations.A1[j][k];
            }
            gradients.dW2[i][j] /= m;
        }
    }

    // Calcul db2 = (1/m) * sum(dZ2)
    for (int i=0; i<n2; i++)
    {
        gradients.db2[i] = 0.0;
        for (int j=0; j<m; j++)
        {
            gradients.db2[i] += dZ2[i][j];
        }
        gradients.db2[i] /= m;
    }

    // Calcul dZ1 = W2.T . dZ2 * A1 * (1 - A1)
    std::vector<std::vector<double>> dZ1(n1, std::vector<double>(m));
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<m; j++)
        {
            double W2T_dZ2 = 0.0;
            for (int k=0; k<n2; k++)
            {
                W2T_dZ2 += params.W2[k][i] * dZ2[k][j];
            }
            dZ1[i][j] = W2T_dZ2 * activations.A1[i][j] * (1 - activations.A1[i][j]);
        }
    }

    // Calcul dW1 = (1/m) * dZ1 . X.T
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            gradients.dW1[i][j] = 0.0;
            for (int k=0; k<m; k++)
            {
                gradients.dW1[i][j] += dZ1[i][k] * X[k][j];
            }
            gradients.dW1[i][j] /= m;
        }
    }

    // Calcul db1 = (1/m) * sum(dZ1, axis=1)
    for (int i=0; i<n1; i++)
    {
        gradients.db1[i] = 0.0;
        for (int j=0; j<m; j++)
        {
            gradients.db1[i] += dZ1[i][j];
        }
        gradients.db1[i] /= m;
    }

    return gradients;
}

void Update(NetworkParams& params, Gradients& gradients, double learning_rate)
{
    int n0 = params.W1[0].size();
    int n1 = params.W1.size();
    int n2 = params.W2.size();
    int n3 = params.W3.size();

    // MAJ W1 = W1 - learning_rate * dW1
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n0; j++)
        {
            params.W1[i][j] = params.W1[i][j] - learning_rate * gradients.dW1[i][j];
        }
    }
    // MAJ b1 = b1 - learning_rate * db1
    for (int i=0; i<n1; i++)
    {
        params.b1[i] = params.b1[i] - learning_rate * gradients.db1[i];
    }

    // MAJ W2 = W2 - learning_rate * dW2
    for (int i=0; i<n2; i++)
    {
        for (int j=0; j<n1; j++)
        {
            params.W2[i][j] = params.W2[i][j] - learning_rate * gradients.dW2[i][j];
        }
    }
    // MAJ b2 = b2 - learning_rate * db2
    for (int i=0; i<n2; i++)
    {
        params.b2[i] = params.b2[i] - learning_rate * gradients.db2[i];
    }

    // MAJ W3 = W3 - learning_rate * dW3
    for (int i=0; i<n3; i++)
    {
        for (int j=0; j<n2; j++)
        {
            params.W3[i][j] = params.W3[i][j] - learning_rate * gradients.dW3[i][j];
        }
    }
    // MAJ b3 = b3 - learning_rate * db3
    for (int i=0; i<n3; i++)
    {
        params.b3[i] = params.b3[i] - learning_rate * gradients.db3[i];
    }
}

// NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, int n2, double learning_rate, int n_iter)
// {
//     NetworkParams params;

//     int n0 = X[0].size();
//     int n3 = 1;

//     params = Initialisation(n0, n1, n2, n3);

//     std::vector<double> train_loss;
//     std::vector<double> train_acc;

//     std::cout << "Debut de l'entrainement du reseau de neuronnes a 3 couches..." << std::endl;

//     for (int i=0; i<n_iter; i++)
//     {
//         Activations activations = ForwardPropagation(X, params);

//         double loss = log_loss(activations.A3, y);
//         train_loss.push_back(loss);

//         std::vector<int> y_pred = Predict(X, params);
//         double acc = Accuracy(y, y_pred);
//         train_acc.push_back(acc);

//         if (i%100 == 0)
//         {
//             std::cout << "Iterration " << i << ", Loss: " << loss << ", Accuracy: " << acc << std::endl;
//         }

//         Gradients gradients = BackPropagation(X, y, params, activations);

//         Update(params, gradients, learning_rate);
//     }

//     std::cout << "Entrainement termine !" << std::endl;
//     std::cout << "Loss finale: " << train_loss.back() << std::endl;
//     std::cout << "Accuracy finale: " << train_acc.back() << std::endl;

//     return params;
// }

NetworkParams Neural_network(std::vector<std::vector<double>>& X, std::vector<int>& y, int n1, int n2, double learning_rate, int n_iter)
{
    int n0 = X[0].size();
    int n3 = 1;

    NetworkParams params = Initialisation(n0, n1, n2, n3);

    std::vector<double> train_loss;
    std::vector<double> train_acc;

    // --- Validation CPU vs GPU sur la première itération ---
    {
        Activations act_cpu = ForwardPropagation(X, params);
        Activations act_gpu = ForwardPropagation_GPU(X, params);

        double max_diff = 0.0f;
        for (size_t j = 0; j < act_cpu.A3[0].size(); j++)
        {
            double diff = std::abs(act_cpu.A3[0][j] - act_gpu.A3[0][j]);
            max_diff = std::max(max_diff, diff);
        }

        std::cout << "[Validation] Diff max CPU vs GPU : " << max_diff << std::endl;
        if (max_diff > 1e-10)
        {
            std::cerr << "[ERREUR] GPU et CPU divergent ! Arret." << std::endl;
            return params;
        }
        std::cout << "[Validation] OK - GPU et CPU coherents." << std::endl;
    }

    std::cout << "Debut entrainement..." << std::endl;

    GPUMemory mem = AllocateGPUMemory(n0, n1, n2, n3, (int)X.size());

    for (int i = 0; i < n_iter; i++)
    {
        // Forward propagation sur GPU
        Activations activations = ForwardPropagation_GPU(X, params, mem);

        double loss = log_loss(activations.A3, y);
        train_loss.push_back(loss);

        //std::vector<int> y_pred = Predict(X, params);
        std::vector<int> y_pred;
        for (size_t j = 0; j < activations.A3[0].size(); j++)
            y_pred.push_back(activations.A3[0][j] >= 0.5 ? 1 : 0);
        double acc = Accuracy(y, y_pred);
        train_acc.push_back(acc);

        if ((i % 100) == 0)
        {
            std::cout << "Iter " << i << " | Loss: " << loss << " | Acc: " << acc << std::endl;
        }

        // Back et Update sur CPU
        Gradients gradients = BackPropagation(X, y, params, activations);
        Update(params, gradients, learning_rate);
    }

    FreeGPUMemory(mem);

    std::cout << "Entrainement termine !" << std::endl;
    std::cout << "Loss finale : " << train_loss.back() << std::endl;
    std::cout << "Accuracy finale : " << train_acc.back() << std::endl;

    return params;
}

std::vector<int> Predict(std::vector<std::vector<double>>& X, NetworkParams& params)
{
    Activations activations = ForwardPropagation(X, params);
    std::vector<std::vector<double>>& A3 = activations.A3;

    std::vector<int> prediction(A3[0].size());
    for (size_t i=0; i<A3[0].size(); i++)
    {
        prediction[i] = (A3[0][i] >= 0.5) ? 1 : 0;
    }

    return prediction;
}

double Accuracy(std::vector<int>& y, std::vector<int>& y_pred)
{
    int correct = 0;

    for (size_t i=0; i<y.size(); i++)
    {
        if (y[i] == y_pred[i])
        {
            correct += 1;
        }
    }

    return (double)correct / y.size();
}

void PlotDecisionBoundary(std::vector<std::vector<double>>& X, std::vector<int>& y, NetworkParams& params)
{
    std::vector<double> x0, y0, x1, y1;

    for (size_t i = 0; i < X.size(); i++)
    {
        if (y[i] == 0)
        {
            x0.push_back(X[i][0]);
            y0.push_back(X[i][1]);
        }
        else
        {
            x1.push_back(X[i][0]);
            y1.push_back(X[i][1]);
        }
    }

    plt::scatter(x0, y0, 40.0, {{"color", "yellow"}});
    plt::scatter(x1, y1, 40.0, {{"color", "teal"}});

    double x_min = -2.0, x_max = 2.0;
    double y_min = -2.0, y_max = 2.0;
    int n_points = 400;

    std::cout << "Calcul de la frontière de décision..." << std::endl;
    
    std::vector<double> boundary_x, boundary_y;
    
    for (int i = 0; i < n_points; i++)
    {
        for (int j = 0; j < n_points; j++)
        {
            double x_val = x_min + (x_max - x_min) * i / (n_points - 1);
            double y_val = y_min + (y_max - y_min) * j / (n_points - 1);
            
            std::vector<std::vector<double>> single_point = {{x_val, y_val}};
            Activations activations = ForwardPropagation(single_point, params);
            double prob = activations.A3[0][0];
            
            if (prob >= 0.45 && prob <= 0.55)
            {
                boundary_x.push_back(x_val);
                boundary_y.push_back(y_val);
            }
        }
    }

    std::cout << "Nombre de points de frontière trouvés: " << boundary_x.size() << std::endl;

    if (!boundary_x.empty())
    {
        plt::scatter(boundary_x, boundary_y, 8.0, {{"color", "red"}});
    }

    plt::scatter(x0, y0, 50.0, {{"color", "orange"}});
    plt::scatter(x1, y1, 50.0, {{"color", "darkblue"}});

    std::cout << "Affichage terminé!" << std::endl;
    plt::show();
}

