package com.smartstudent.cli;

import com.smartstudent.data.*;
import com.smartstudent.ml.*;

public class App {
    public static void main(String[] args) {
        if (args.length == 0 || args[0].equals("help")) {
            printHelp();
            return;
        }
        String cmd = args[0];
        String task = "classification"; // or regression
        String data = "data/students.csv";
        String label = "passed";
        int epochs = 1000;
        double lr = -1.0; // default by task
        double testSplit = 0.2;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--task": task = args[++i]; break;
                case "--data": data = args[++i]; break;
                case "--label": label = args[++i]; break;
                case "--epochs": epochs = Integer.parseInt(args[++i]); break;
                case "--lr": lr = Double.parseDouble(args[++i]); break;
                case "--testSplit": testSplit = Double.parseDouble(args[++i]); break;
                default: break;
            }
        }
        if (lr <= 0) lr = task.equals("regression") ? 0.01 : 0.05;

        try {
            if (cmd.equals("train")) {
                runTrain(task, data, label, epochs, lr, testSplit);
            } else {
                System.out.println("Unknown command: " + cmd);
                printHelp();
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runTrain(String task, String csv, String label, int epochs, double lr, double testSplit) throws Exception {
    DataSet ds = DataLoader.loadNumericFeatures(csv, label);
    DataSet.Split split0 = ds.trainTestSplit(testSplit, 42);
    StandardScaler scaler = new StandardScaler();
    scaler.fit(split0.train.X);
    double[][] Ztr = scaler.transform(split0.train.X);
    double[][] Zte = scaler.transform(split0.test.X);

    DataSet train = new DataSet(Ztr, split0.train.y, ds.featureNames, ds.labelName);
    DataSet test = new DataSet(Zte, split0.test.y, ds.featureNames, ds.labelName);

        if (task.equals("regression")) {
            LinearRegressionGD model = new LinearRegressionGD();
            model.fit(train.X, train.y, epochs, lr);
            double[] preds = model.predict(test.X);
            double mse = Metrics.mse(test.y, preds);
            double r2 = Metrics.r2(test.y, preds);
            System.out.println("Task: regression");
            System.out.println("Test MSE: " + String.format("%.4f", mse));
            System.out.println("Test R2:  " + String.format("%.4f", r2));
        } else {
            LogisticRegressionGD model = new LogisticRegressionGD();
            model.fit(train.X, train.y, epochs, lr);
            double[] prob = model.predictProba(test.X);
            double acc = Metrics.accuracy(test.y, prob, 0.5);
            double f1 = Metrics.f1(test.y, prob, 0.5);
            System.out.println("Task: classification");
            System.out.println("Test Accuracy: " + String.format("%.4f", acc));
            System.out.println("Test F1:       " + String.format("%.4f", f1));
        }
    }

    private static void printHelp() {
        System.out.println("Smart Student Performance Predictor");
        System.out.println("Usage: run.sh train [--task classification|regression] --data <csv> --label <col> [--epochs N] [--lr LR] [--testSplit F]");
    }
}
