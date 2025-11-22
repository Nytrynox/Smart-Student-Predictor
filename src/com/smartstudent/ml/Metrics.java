package com.smartstudent.ml;

public class Metrics {
    public static double mse(double[] yTrue, double[] yPred) {
        double s = 0; int n = yTrue.length;
        for (int i = 0; i < n; i++) {
            double d = yTrue[i] - yPred[i];
            s += d * d;
        }
        return s / n;
    }

    public static double r2(double[] yTrue, double[] yPred) {
        double mean = 0; int n = yTrue.length;
        for (double v : yTrue) mean += v; mean /= n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            double d = yTrue[i] - yPred[i];
            ssRes += d * d;
            double t = yTrue[i] - mean;
            ssTot += t * t;
        }
        return 1.0 - (ssRes / ssTot);
    }

    public static double accuracy(double[] yTrue, double[] yProb, double threshold) {
        int correct = 0, n = yTrue.length;
        for (int i = 0; i < n; i++) {
            int pred = yProb[i] >= threshold ? 1 : 0;
            if ((int)Math.round(yTrue[i]) == pred) correct++;
        }
        return correct / (double)n;
    }

    public static double f1(double[] yTrue, double[] yProb, double threshold) {
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < yTrue.length; i++) {
            int y = (int)Math.round(yTrue[i]);
            int p = yProb[i] >= threshold ? 1 : 0;
            if (p == 1 && y == 1) tp++;
            else if (p == 1 && y == 0) fp++;
            else if (p == 0 && y == 1) fn++;
        }
        double precision = tp + fp == 0 ? 0 : tp / (double)(tp + fp);
        double recall = tp + fn == 0 ? 0 : tp / (double)(tp + fn);
        return (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
    }
}
