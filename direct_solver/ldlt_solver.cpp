#include <iostream>
#include <vector>

using namespace std;

vector<double> ldl(vector<vector<double>> A, vector<double> b) {
    int n = A.size();
    vector<double> x(n, 0.0);
    vector<vector<double>> L(n, vector<double>(n, 0.0));
    vector<double> D(n, 0.0);

    // LDL decomposition
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[i][k] * D[k] * L[i][k];
        }
        D[i] = A[i][i] - sum;
        L[i][i] = 1.0;
        for (int j = i + 1; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[j][k] * D[k] * L[i][k];
            }
            L[j][i] = (A[j][i] - sum) / D[i];
            L[i][j] = 0.0;
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout << L[i][j] << " ";
        }
        cout << endl;
    }

    // Solve linear system
    vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int k = 0; k < i; k++) {
            sum += L[i][k] * y[k];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int k = i + 1; k < n; k++) {
            sum += L[k][i] * x[k];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    return x;
}

int main() {
    vector<vector<double>> A = {{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
    vector<double> b = {4, 3, 2};
    vector<double> x = ldl(A, b);

    cout << "The solution is:\n";
    for (int i = 0; i < x.size(); i++) {
        cout << x[i] << "\n";
    }

    return 0;
}