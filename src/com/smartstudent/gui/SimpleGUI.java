package com.smartstudent.gui;

import com.smartstudent.data.*;
import com.smartstudent.ml.*;
import com.smartstudent.db.*;

import javax.swing.*;
import java.awt.*;
import java.io.File;

public class SimpleGUI extends JFrame {
    private JTextField dataFileField;
    private JButton trainButton;
    private JButton loadButton;
    private JTextArea resultArea;
    private MongoDBHandler dbHandler;

    public SimpleGUI() {
        setTitle("Student Performance Predictor");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(700, 500);
        setLocationRelativeTo(null);
        
        dbHandler = new MongoDBHandler();
        initUI();
        setVisible(true);
    }

    private void initUI() {
        setLayout(new BorderLayout(10, 10));
        
        // Top Panel - Title
        JPanel topPanel = new JPanel();
        JLabel titleLabel = new JLabel("📊 Student Performance Predictor");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 20));
        topPanel.add(titleLabel);
        add(topPanel, BorderLayout.NORTH);
        
        // Center Panel - File Selection
        JPanel centerPanel = new JPanel(new GridLayout(3, 1, 10, 10));
        centerPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));
        
        // File selection
        JPanel filePanel = new JPanel(new FlowLayout());
        filePanel.add(new JLabel("Data File:"));
        dataFileField = new JTextField("data/student_performance_numeric.csv", 30);
        filePanel.add(dataFileField);
        
        loadButton = new JButton("Browse");
        loadButton.addActionListener(e -> browseFile());
        filePanel.add(loadButton);
        
        centerPanel.add(filePanel);
        
        // Train button
        trainButton = new JButton("🎯 Train Model");
        trainButton.setFont(new Font("Arial", Font.BOLD, 16));
        trainButton.addActionListener(e -> trainModel());
        centerPanel.add(trainButton);
        
        add(centerPanel, BorderLayout.CENTER);
        
        // Results Panel
        JPanel resultPanel = new JPanel(new BorderLayout());
        resultPanel.setBorder(BorderFactory.createTitledBorder("Results"));
        
        resultArea = new JTextArea(15, 50);
        resultArea.setEditable(false);
        resultArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        JScrollPane scrollPane = new JScrollPane(resultArea);
        resultPanel.add(scrollPane, BorderLayout.CENTER);
        
        add(resultPanel, BorderLayout.SOUTH);
    }

    private void browseFile() {
        JFileChooser fileChooser = new JFileChooser("data");
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            dataFileField.setText(fileChooser.getSelectedFile().getPath());
        }
    }

    private void trainModel() {
        trainButton.setEnabled(false);
        resultArea.setText("Training model...\n\n");
        
        new Thread(() -> {
            try {
                String csvPath = dataFileField.getText();
                
                // Load and train
                DataSet ds = DataLoader.loadNumericFeatures(csvPath, "FinalGrade");
                DataSet.Split split = ds.trainTestSplit(0.2, 42);
                
                StandardScaler scaler = new StandardScaler();
                scaler.fit(split.train.X);
                double[][] trainX = scaler.transform(split.train.X);
                double[][] testX = scaler.transform(split.test.X);
                
                DataSet train = new DataSet(trainX, split.train.y, ds.featureNames, ds.labelName);
                DataSet test = new DataSet(testX, split.test.y, ds.featureNames, ds.labelName);
                
                // Train regression model
                LinearRegressionGD model = new LinearRegressionGD();
                model.fit(train.X, train.y, 1000, 0.01);
                
                double[] predictions = model.predict(test.X);
                double mse = Metrics.mse(test.y, predictions);
                double r2 = Metrics.r2(test.y, predictions);
                
                // Save to MongoDB
                dbHandler.savePrediction("regression", mse, r2, 1000, 0.01);
                
                // Display results
                StringBuilder result = new StringBuilder();
                result.append("✅ Training Complete!\n\n");
                result.append("Dataset: ").append(new File(csvPath).getName()).append("\n");
                result.append("Training samples: ").append(train.X.length).append("\n");
                result.append("Test samples: ").append(test.X.length).append("\n");
                result.append("Features: ").append(train.X[0].length).append("\n\n");
                result.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                result.append("📈 Results:\n");
                result.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
                result.append(String.format("MSE:  %.4f\n", mse));
                result.append(String.format("R²:   %.4f\n", r2));
                result.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
                result.append("💾 Saved to MongoDB!\n");
                
                SwingUtilities.invokeLater(() -> {
                    resultArea.setText(result.toString());
                    trainButton.setEnabled(true);
                });
                
            } catch (Exception e) {
                SwingUtilities.invokeLater(() -> {
                    resultArea.setText("❌ Error: " + e.getMessage());
                    trainButton.setEnabled(true);
                });
            }
        }).start();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception ignored) {}
            new SimpleGUI();
        });
    }
}
