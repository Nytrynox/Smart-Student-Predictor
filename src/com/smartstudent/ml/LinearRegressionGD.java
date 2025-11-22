package com.smartstudent.ml;

import java.util.Arrays;

public class LinearRegressionGD {
    private double[] w; // includes bias as last term

    public void fit(double[][] X, double[] y, int epochs, double lr) {
        int n = X.length; if (n == 0) { w = new double[0]; return; }
        int d = X[0].length;
        w = new double[d + 1];
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[] grad = new double[d + 1];
            for (int i = 0; i < n; i++) {
                double pred = predictRow(X[i]);
                double err = pred - y[i];
                for (int j = 0; j < d; j++) grad[j] += err * X[i][j];
                grad[d] += err; // bias
            }
            for (int j = 0; j < d + 1; j++) w[j] -= lr * grad[j] / n;
        }
    }

    private double predictRow(double[] x) {
        double s = w[w.length - 1];
        for (int j = 0; j < w.length - 1; j++) s += w[j] * x[j];
        return s;
    }

    public double[] predict(double[][] X) {
        double[] out = new double[X.length];
        for (int i = 0; i < X.length; i++) out[i] = predictRow(X[i]);
        return out;
    }

    public double[] getWeights() { return Arrays.copyOf(w, w.length); }
}
