package com.smartstudent.data;

import java.io.IOException;
import java.util.*;

public class DataLoader {
    public static DataSet loadNumericFeatures(String csvPath, String labelColumn) throws IOException {
        CSVReader.CSVData csv = CSVReader.read(csvPath);
        String[] header = csv.header;
        int n = csv.rows.size();
        int labelIdx = -1;
        for (int j = 0; j < header.length; j++) {
            if (header[j].equals(labelColumn)) { labelIdx = j; break; }
        }
        if (labelIdx < 0) throw new IllegalArgumentException("Label column not found: " + labelColumn);

        List<Integer> featureIdx = new ArrayList<>();
        List<String> featureNames = new ArrayList<>();
        for (int j = 0; j < header.length; j++) {
            if (j == labelIdx) continue;
            if (isNumericColumn(csv.rows, j)) {
                featureIdx.add(j);
                featureNames.add(header[j]);
            }
        }
        if (featureIdx.isEmpty()) throw new IllegalArgumentException("No numeric features were found");

        double[][] X = new double[n][featureIdx.size()];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            String[] row = csv.rows.get(i);
            for (int k = 0; k < featureIdx.size(); k++) {
                X[i][k] = Double.parseDouble(row[featureIdx.get(k)]);
            }
            y[i] = Double.parseDouble(row[labelIdx]);
        }
        return new DataSet(X, y, featureNames.toArray(new String[0]), labelColumn);
    }

    private static boolean isNumericColumn(List<String[]> rows, int col) {
        int checks = Math.min(10, rows.size());
        for (int i = 0; i < checks; i++) {
            String v = rows.get(i)[col];
            try { Double.parseDouble(v); }
            catch (NumberFormatException e) { return false; }
        }
        return true;
    }
}
