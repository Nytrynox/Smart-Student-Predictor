package com.smartstudent.gui;

import com.smartstudent.data.*;
import com.smartstudent.ml.*;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.io.File;

public class PredictorGUI extends JFrame {
    private JTextField dataFileField;
    private JTextField labelField;
    private JTextField epochsField;
    private JTextField learningRateField;
    private JTextField testSplitField;
    private JComboBox<String> taskComboBox;
    private JTextArea outputArea;
    private JButton trainButton;
    private JButton browseButton;
    private JProgressBar progressBar;

    public PredictorGUI() {
        setTitle("Smart Student Analysis Predictor");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLocationRelativeTo(null);
        
        initComponents();
        setVisible(true);
    }

    private void initComponents() {
        // Main panel with padding
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(15, 15, 15, 15));
        
        // Title
        JLabel titleLabel = new JLabel("Smart Student Performance Predictor", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 24));
        titleLabel.setBorder(new EmptyBorder(0, 0, 15, 0));
        mainPanel.add(titleLabel, BorderLayout.NORTH);
        
        // Input panel
        JPanel inputPanel = createInputPanel();
        mainPanel.add(inputPanel, BorderLayout.CENTER);
        
        // Output panel
        JPanel outputPanel = createOutputPanel();
        mainPanel.add(outputPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
    }

    private JPanel createInputPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(Color.GRAY), 
            "Training Configuration",
            TitledBorder.LEFT,
            TitledBorder.TOP,
            new Font("Arial", Font.BOLD, 14)
        ));
        
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        
        int row = 0;
        
        // Data file
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Data File:"), gbc);
        
        gbc.gridx = 1; gbc.weightx = 1.0;
        dataFileField = new JTextField("data/students.csv", 30);
        panel.add(dataFileField, gbc);
        
        gbc.gridx = 2; gbc.weightx = 0;
        browseButton = new JButton("Browse");
        browseButton.addActionListener(e -> browseFile());
        panel.add(browseButton, gbc);
        
        row++;
        
        // Task type
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Task:"), gbc);
        
        gbc.gridx = 1; gbc.gridwidth = 2;
        taskComboBox = new JComboBox<>(new String[]{"classification", "regression"});
        panel.add(taskComboBox, gbc);
        
        row++;
        gbc.gridwidth = 1;
        
        // Label column
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Label Column:"), gbc);
        
        gbc.gridx = 1; gbc.gridwidth = 2;
        labelField = new JTextField("passed", 30);
        panel.add(labelField, gbc);
        
        row++;
        gbc.gridwidth = 1;
        
        // Epochs
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Epochs:"), gbc);
        
        gbc.gridx = 1; gbc.gridwidth = 2;
        epochsField = new JTextField("1000", 30);
        panel.add(epochsField, gbc);
        
        row++;
        gbc.gridwidth = 1;
        
        // Learning rate
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Learning Rate:"), gbc);
        
        gbc.gridx = 1; gbc.gridwidth = 2;
        learningRateField = new JTextField("0.05", 30);
        JPanel lrPanel = new JPanel(new FlowLayout(FlowLayout.LEFT, 0, 0));
        lrPanel.add(learningRateField);
        JLabel lrNote = new JLabel(" (leave as 0.05 for classification, 0.01 for regression)");
        lrNote.setFont(new Font("Arial", Font.ITALIC, 10));
        lrNote.setForeground(Color.GRAY);
        lrPanel.add(lrNote);
        panel.add(lrPanel, gbc);
        
        row++;
        gbc.gridwidth = 1;
        
        // Test split
        gbc.gridx = 0; gbc.gridy = row;
        panel.add(new JLabel("Test Split:"), gbc);
        
        gbc.gridx = 1; gbc.gridwidth = 2;
        testSplitField = new JTextField("0.2", 30);
        panel.add(testSplitField, gbc);
        
        row++;
        
        // Train button
        gbc.gridx = 0; gbc.gridy = row;
        gbc.gridwidth = 3;
        gbc.anchor = GridBagConstraints.CENTER;
        trainButton = new JButton("Train Model");
        trainButton.setFont(new Font("Arial", Font.BOLD, 16));
        trainButton.setPreferredSize(new Dimension(200, 40));
        trainButton.addActionListener(e -> trainModel());
        panel.add(trainButton, gbc);
        
        row++;
        
        // Progress bar
        gbc.gridy = row;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        progressBar = new JProgressBar();
        progressBar.setStringPainted(true);
        progressBar.setVisible(false);
        panel.add(progressBar, gbc);
        
        return panel;
    }

    private JPanel createOutputPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createLineBorder(Color.GRAY), 
            "Training Results",
            TitledBorder.LEFT,
            TitledBorder.TOP,
            new Font("Arial", Font.BOLD, 14)
        ));
        
        outputArea = new JTextArea(8, 60);
        outputArea.setEditable(false);
        outputArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        outputArea.setBorder(new EmptyBorder(5, 5, 5, 5));
        
        JScrollPane scrollPane = new JScrollPane(outputArea);
        panel.add(scrollPane, BorderLayout.CENTER);
        
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(e -> outputArea.setText(""));
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.add(clearButton);
        panel.add(buttonPanel, BorderLayout.SOUTH);
        
        return panel;
    }

    private void browseFile() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File("data"));
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv");
            }
            public String getDescription() {
                return "CSV Files (*.csv)";
            }
        });
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            dataFileField.setText(fileChooser.getSelectedFile().getPath());
        }
    }

    private void trainModel() {
        // Validate inputs
        if (dataFileField.getText().trim().isEmpty()) {
            showError("Please select a data file.");
            return;
        }
        
        // Disable train button during training
        trainButton.setEnabled(false);
        browseButton.setEnabled(false);
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        outputArea.setText("Training in progress...\n");
        
        // Run training in background thread
        SwingWorker<String, Void> worker = new SwingWorker<>() {
            @Override
            protected String doInBackground() throws Exception {
                String task = (String) taskComboBox.getSelectedItem();
                String csv = dataFileField.getText();
                String label = labelField.getText();
                int epochs = Integer.parseInt(epochsField.getText());
                double lr = Double.parseDouble(learningRateField.getText());
                double testSplit = Double.parseDouble(testSplitField.getText());
                
                return runTraining(task, csv, label, epochs, lr, testSplit);
            }
            
            @Override
            protected void done() {
                try {
                    String result = get();
                    outputArea.setText(result);
                } catch (Exception e) {
                    outputArea.setText("Error during training:\n" + e.getMessage());
                    e.printStackTrace();
                }
                
                trainButton.setEnabled(true);
                browseButton.setEnabled(true);
                progressBar.setVisible(false);
            }
        };
        
        worker.execute();
    }

    private String runTraining(String task, String csv, String label, int epochs, double lr, double testSplit) throws Exception {
        StringBuilder output = new StringBuilder();
        
        DataSet ds = DataLoader.loadNumericFeatures(csv, label);
        DataSet.Split split0 = ds.trainTestSplit(testSplit, 42);
        StandardScaler scaler = new StandardScaler();
        scaler.fit(split0.train.X);
        double[][] Ztr = scaler.transform(split0.train.X);
        double[][] Zte = scaler.transform(split0.test.X);

        DataSet train = new DataSet(Ztr, split0.train.y, ds.featureNames, ds.labelName);
        DataSet test = new DataSet(Zte, split0.test.y, ds.featureNames, ds.labelName);

        output.append("Dataset loaded successfully\n");
        output.append("Training samples: ").append(train.X.length).append("\n");
        output.append("Test samples: ").append(test.X.length).append("\n");
        output.append("Features: ").append(train.X[0].length).append("\n\n");
        output.append("Training model...\n\n");

        if (task.equals("regression")) {
            LinearRegressionGD model = new LinearRegressionGD();
            model.fit(train.X, train.y, epochs, lr);
            double[] preds = model.predict(test.X);
            double mse = Metrics.mse(test.y, preds);
            double r2 = Metrics.r2(test.y, preds);
            output.append("Task: Regression\n");
            output.append("═══════════════════════════════\n");
            output.append("Test MSE: ").append("%.4f".formatted(mse)).append("\n");
            output.append("Test R²:  ").append("%.4f".formatted(r2)).append("\n");
        } else {
            LogisticRegressionGD model = new LogisticRegressionGD();
            model.fit(train.X, train.y, epochs, lr);
            double[] prob = model.predictProba(test.X);
            double acc = Metrics.accuracy(test.y, prob, 0.5);
            double f1 = Metrics.f1(test.y, prob, 0.5);
            output.append("Task: Classification\n");
            output.append("═══════════════════════════════\n");
            output.append("Test Accuracy: ").append("%.4f".formatted(acc)).append("\n");
            output.append("Test F1 Score: ").append("%.4f".formatted(f1)).append("\n");
        }
        
        output.append("\nTraining completed successfully!");
        
        return output.toString();
    }

    private void showError(String message) {
        JOptionPane.showMessageDialog(this, message, "Error", JOptionPane.ERROR_MESSAGE);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception e) {
                e.printStackTrace();
            }
            new PredictorGUI();
        });
    }
}
