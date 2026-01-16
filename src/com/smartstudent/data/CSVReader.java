package com.smartstudent.data;

import java.io.*;
import java.util.*;

public class CSVReader {
    public static class CSVData {
        public final String[] header;
        public final List<String[]> rows;
        public CSVData(String[] header, List<String[]> rows) {
            this.header = header; this.rows = rows;
        }
    }

    public static CSVData read(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String headerLine = br.readLine();
            if (headerLine == null) throw new IOException("Empty CSV");
            String[] header = parseLine(headerLine);
            List<String[]> rows = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                rows.add(parseLine(line));
            }
            return new CSVData(header, rows);
        }
    }

    private static String[] parseLine(String line) {
        // Simple CSV split; assumes no quoted commas for demo
        String[] parts = line.split(",");
        // Trim whitespace from each part
        for (int i = 0; i < parts.length; i++) {
            parts[i] = parts[i].trim();
        }
        return parts;
    }
}
