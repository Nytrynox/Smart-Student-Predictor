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

        List<double[]> validX = new ArrayList<>();
        List<Double> validY = new ArrayList<>();
        
        for (int i = 0; i < n; i++) {
            String[] row = csv.rows.get(i);
            try {
                // Check if label is valid
                if (row[labelIdx].trim().isEmpty()) continue;
                double labelVal = Double.parseDouble(row[labelIdx]);
                
                // Parse all features
                double[] features = new double[featureIdx.size()];
                boolean validRow = true;
                for (int k = 0; k < featureIdx.size(); k++) {
                    String val = row[featureIdx.get(k)];
                    if (val.trim().isEmpty()) {
                        validRow = false;
                        break;
                    }
                    features[k] = Double.parseDouble(val);
                }
                
                if (validRow) {
                    validX.add(features);
                    validY.add(labelVal);
                }
            } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                // Skip invalid rows
                continue;
            }
        }
        
        if (validX.isEmpty()) {
            throw new IllegalArgumentException("No valid data rows found");
        }
        
        double[][] X = new double[validX.size()][featureIdx.size()];
        double[] y = new double[validY.size()];
        for (int i = 0; i < validX.size(); i++) {
            X[i] = validX.get(i);
            y[i] = validY.get(i);
        }
        return new DataSet(X, y, featureNames.toArray(new String[0]), labelColumn);
    }

    private static boolean isNumericColumn(List<String[]> rows, int col) {
        int checks = Math.min(10, rows.size());
        int validCount = 0;
        for (int i = 0; i < checks; i++) {
            String v = rows.get(i)[col];
            if (v.trim().isEmpty()) continue;
            try { 
                Double.parseDouble(v);
                validCount++;
            }
            catch (NumberFormatException e) { return false; }
        }
        return validCount > 0;
    }
}
