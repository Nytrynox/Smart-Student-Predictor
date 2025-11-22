package com.smartstudent.data;

import java.util.Random;

public class DataSet {
    public final double[][] X; // shape: [nSamples][nFeatures]
    public final double[] y;   // shape: [nSamples]
    public final String[] featureNames;
    public final String labelName;

    public DataSet(double[][] X, double[] y, String[] featureNames, String labelName) {
        this.X = X;
        this.y = y;
        this.featureNames = featureNames;
        this.labelName = labelName;
    }

    public int samples() { return X.length; }
    public int features() { return X.length == 0 ? 0 : X[0].length; }

    public Split trainTestSplit(double testFraction, long seed) {
        int n = samples();
        int testSize = Math.max(1, (int)Math.round(n * testFraction));
        int trainSize = n - testSize;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        double[][] Xtr = new double[trainSize][];
        double[] ytr = new double[trainSize];
        double[][] Xte = new double[testSize][];
        double[] yte = new double[testSize];
        for (int i = 0; i < trainSize; i++) { Xtr[i] = X[idx[i]]; ytr[i] = y[idx[i]]; }
        for (int i = 0; i < testSize; i++) { Xte[i] = X[idx[trainSize + i]]; yte[i] = y[idx[trainSize + i]]; }
        return new Split(
            new DataSet(Xtr, ytr, featureNames, labelName),
            new DataSet(Xte, yte, featureNames, labelName)
        );
    }

    public static class Split {
        public final DataSet train;
        public final DataSet test;
        public Split(DataSet train, DataSet test) { this.train = train; this.test = test; }
    }
}
