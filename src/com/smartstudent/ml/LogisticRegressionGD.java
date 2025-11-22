package com.smartstudent.ml;

public class LogisticRegressionGD {
    private double[] w; // includes bias

    private static double sigmoid(double z) {
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    public void fit(double[][] X, double[] y, int epochs, double lr) {
        int n = X.length; if (n == 0) { w = new double[0]; return; }
        int d = X[0].length;
        w = new double[d + 1];
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] grad = new double[d + 1];
            for (int i = 0; i < n; i++) {
                double z = w[d];
                for (int j = 0; j < d; j++) z += w[j] * X[i][j];
                double p = sigmoid(z);
                double err = p - y[i];
                for (int j = 0; j < d; j++) grad[j] += err * X[i][j];
                grad[d] += err;
            }
            for (int j = 0; j < d + 1; j++) w[j] -= lr * grad[j] / n;
        }
    }

    public double[] predictProba(double[][] X) {
        int n = X.length; int d = (w == null ? 0 : w.length - 1);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double z = w[d];
            for (int j = 0; j < d; j++) z += w[j] * X[i][j];
            out[i] = sigmoid(z);
        }
        return out;
    }
}
