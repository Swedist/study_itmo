#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <array_size>" << endl;
        return 1;
    }

    int arraySize = atoi(argv[1]);
    vector<int> arrayVec(arraySize);

    srand(time(0));

    // Fill the vector with random numbers
    for (int i = 0; i < arraySize; ++i) {
        arrayVec[i] = rand();
    }

    int maxValue = 0;
    double startTime, endTime;
    cout << setw(7) << "Threads" << setw(12) << "Time (s)" << endl;

    for (int numThreads = 1; numThreads <= 10; ++numThreads) {
        maxValue = 0;
        omp_set_num_threads(numThreads);
        startTime = omp_get_wtime();

#pragma omp parallel for reduction(max : maxValue)
        for (int i = 0; i < arraySize; ++i) {
            if (arrayVec[i] > maxValue) {
                maxValue = arrayVec[i];
            }
        }

        endTime = omp_get_wtime();
        cout << setw(7) << numThreads << setw(12) << fixed << setprecision(5) << endTime - startTime << endl;
    }

    cout << "Max value on large vector: " << maxValue << endl;

    // Checking the correctness of the program on 10 elements
    cout << "Checking correctness on 10 elements: ";
    for (int i = 0; i < 10; ++i) {
        arrayVec[i] = i + 1;
    }
    maxValue = 0;

#pragma omp parallel for reduction(max : maxValue)
    for (int i = 0; i < 10; ++i) {
        if (arrayVec[i] > maxValue) {
            maxValue = arrayVec[i];
        }
    }

    cout << "Max value on check vector (10 expected): " << maxValue << endl;

    return 0;
}
