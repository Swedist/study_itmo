#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

// Function to multiply two matrices
vector<vector<int>> multiplyMatrices(vector<vector<int>> &matrix1, vector<vector<int>> &matrix2) {
    int size = matrix1.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return result;
}

// Function to check if two matrices are equal
bool checkEquality(vector<vector<int>> &matrix1, vector<vector<int>> &matrix2) {
    if (matrix1.size() != matrix2.size())
        return false;
    for (int i = 0; i < matrix1.size(); i++) {
        if (matrix1[i] != matrix2[i])
            return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;
    }

    int matrixSize = stoi(argv[1]);

    // Check correctness on 5x5 matrices
    vector<vector<int>> matrix1 = {
        {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
    vector<vector<int>> matrix2 = {
        {1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}};
    vector<vector<int>> result = multiplyMatrices(matrix1, matrix2);
    vector<vector<int>> expected_result = {{215, 230, 245, 260, 275},
                                           {490, 530, 570, 610, 650},
                                           {765, 830, 895, 960, 1025},
                                           {1040, 1130, 1220, 1310, 1400},
                                           {1315, 1430, 1545, 1660, 1775}};
    if (!checkEquality(result, expected_result)) {
        cout << "Multiplication incorrect!" << endl;
        return 1;
    }
    cout << "Multiplication correct for 5x5 matrices." << endl;

    // Perform performance tests on larger matrices
    vector<vector<int>> largeMatrix1(matrixSize, vector<int>(matrixSize, 2));
    vector<vector<int>> largeMatrix2(matrixSize, vector<int>(matrixSize, 3));

    double startTimeSingle = 0.0, endTimeSingle = 0.0;
    double startTimeMulti = 0.0, endTimeMulti = 0.0;

    // Single thread time
    startTimeSingle = omp_get_wtime();
    multiplyMatrices(largeMatrix1, largeMatrix2);
    endTimeSingle = omp_get_wtime();

    // Multi thread
    cout << setw(7) << "Threads" << setw(12) << "Time (s)" << setw(15) << "Efficiency" << endl;
    for (int numThreads = 1; numThreads <= 10; numThreads++) {
        omp_set_num_threads(numThreads);
        startTimeMulti = omp_get_wtime();
        multiplyMatrices(largeMatrix1, largeMatrix2);
        endTimeMulti = omp_get_wtime();
        double durationSingle = endTimeSingle - startTimeSingle;
        double durationMulti = endTimeMulti - startTimeMulti;
        double efficiency = durationSingle / durationMulti;
        if (numThreads == 1) {
            cout << setw(7) << numThreads << setw(12) << fixed << setprecision(5) << durationMulti << setw(15) << "-"
                 << endl;
        } else {
            cout << setw(7) << numThreads << setw(12) << fixed << setprecision(5) << durationMulti << setw(15)
                 << setprecision(6) << efficiency << endl;
        }
    }
    return 0;
}
