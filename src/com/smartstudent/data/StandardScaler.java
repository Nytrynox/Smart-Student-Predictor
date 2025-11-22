package com.smartstudent.data;

public class StandardScaler {
    private double[] mean;
    private double[] std;

    public void fit(double[][] X) {
        int n = X.length;
        if (n == 0) { mean = new double[0]; std = new double[0]; return; }
        int d = X[0].length;
        mean = new double[d];
        std = new double[d];
        for (int j = 0; j < d; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) s += X[i][j];
            mean[j] = s / n;
        }
        for (int j = 0; j < d; j++) {
            double var = 0;
            for (int i = 0; i < n; i++) {
                double diff = X[i][j] - mean[j];
                var += diff * diff;
            }
            std[j] = Math.sqrt(var / Math.max(1, n - 1));
            if (std[j] == 0) std[j] = 1.0;
        }
    }

    public double[][] transform(double[][] X) {
        int n = X.length;
        if (n == 0) return new double[0][0];
        int d = X[0].length;
        double[][] Z = new double[n][d];
        for (int i = 0; i < n; i++) {
            double[] row = new double[d];
            for (int j = 0; j < d; j++) {
                row[j] = (X[i][j] - mean[j]) / std[j];
            }
            Z[i] = row;
        }
        return Z;
    }
}
