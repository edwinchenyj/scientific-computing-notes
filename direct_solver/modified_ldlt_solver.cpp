#include <iostream>
#include <vector>

using namespace std;

// Function to solve a linear system using an LDLT decomposition
vector<double> ldlt_solve(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();

    // Create the matrices D and L
    vector<double> D(n, 0.0);
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    
    vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
        // Calculate the diagonal element
        D[i] = A[i][i];

        // Update the lower triangular matrix
        for (int k = 0; k < i; k++) {
            D[i] -= L[i][k] * L[i][k] * D[k];
        }
        // Solve the system LDy = b
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j] * D[j];
        }
        y[i] = y[i] / D[i];

        // Calculate the remaining elements of the lower triangular matrix
        for (int j = i + 1; j < n; j++) {
            L[j][i] = A[j][i];
            for (int k = 0; k < i; k++) {
                L[j][i] -= L[j][k] * L[i][k] * D[k];
            }
            L[j][i] /= D[i];
        }
    }

    vector<double> x(n, 0.0);

    // Solve the system Lt x = y
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= L[j][i] * x[j];
        }
    }

    return x;
}

// Sample usage
int main() {
    // Solve the system Ax = b, where A is a 3x3 symmetric matrix and b is a 3x1 vector
    vector<vector<double>> A{{4.0, -1.0, 1.0},
                              {-1.0, 4.0, -2.0},
                              {1.0, -2.0, 4.0}};
    vector<double> b{12.0, -1.0, 5.0};
    vector<double> x = ldlt_solve(A, b);

    // Print the solution
    for (int i = 0; i < x.size(); i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }

    return 0;
}
