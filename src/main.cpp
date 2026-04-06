#include "mylib.h"
#include <chrono>

int main(int argc, char** argv)
{
    std::cout << "Lancement programme - Reseau de neuronnes a 3 couches" << std::endl;

    int N = 200;
    int n1 = 8;
    int n2 = 4;

    std::vector<std::vector<double>> X;
    std::vector<int> y;

    int choix;

    do
    {
        std::cout << "\nChoisissez un test:" << std::endl;
        std::cout << "1. Spirale (tres difficile)" << std::endl;
        std::cout << "2. XOR (classique non-lineaire)" << std::endl;
        std::cout << "3. Damier (tres complexe)" << std::endl;
        std::cout << "4. Cercles concentriques multiples" << std::endl;
        std::cout << "5. Donnees linéairement separables (facile)" << std::endl;
        std::cout << "6. Croissants de lune (difficulte moyenne)" << std::endl;
        std::cout << "7. Cercles concentriques (dataset original)" << std::endl;
        std::cout << "Votre choix (1-7): ";

        std::cin >> choix;
    } while (choix < 1 || choix > 7);

    Run(choix, X, y, N);

    auto t1 = std::chrono::high_resolution_clock::now();

    NetworkParams params = Neural_network(X, y, n1, n2, 0.1, 50000);

    auto t2 = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "Temps d'execution: " << duration << " us" << std::endl;

    PlotDecisionBoundary(X, y, params);

    return 0;
}