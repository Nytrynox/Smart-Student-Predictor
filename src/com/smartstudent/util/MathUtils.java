package com.smartstudent.util;

public class MathUtils {
    public static double sigmoid(double z) {
        if (z >= 0) {
            double ez = Math.exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            double ez = Math.exp(z);
            return ez / (1.0 + ez);
        }
    }

    public static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public static double[] addInPlace(double[] a, double[] b, double scale) {
        for (int i = 0; i < a.length; i++) a[i] += scale * b[i];
        return a;
    }

    public static double[][] trainTestSplit(double[][] X, double[] y, double testSplit, long seed) {
        int n = X.length;
        int testSize = (int) Math.max(1, Math.round(n * testSplit));
        int trainSize = n - testSize;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        java.util.Random r = new java.util.Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        double[][] out = new double[6][];
        double[][] Xtr = new double[trainSize][];
        double[] ytr = new double[trainSize];
        double[][] Xte = new double[testSize][];
        double[] yte = new double[testSize];
        for (int i = 0; i < trainSize; i++) { Xtr[i] = X[idx[i]]; ytr[i] = y[idx[i]]; }
        for (int i = 0; i < testSize; i++) { Xte[i] = X[idx[trainSize + i]]; yte[i] = y[idx[trainSize + i]]; }
        out[0] = new double[]{trainSize};
        out[1] = new double[]{testSize};
        out[2] = ytr;
        out[3] = yte;
        // Pack Xtr and Xte as 1D flattened arrays not needed; keep separate methods.
        return new double[][]{ ytr, yte };
    }
}
