package com.smartstudent.db;

import com.mongodb.client.*;
import org.bson.Document;
import java.util.*;

public class MongoDBHandler {
    private MongoClient mongoClient;
    private MongoDatabase database;
    private MongoCollection<Document> collection;
    
    public MongoDBHandler() {
        try {
            mongoClient = MongoClients.create("mongodb://localhost:27017");
            database = mongoClient.getDatabase("student_db");
            collection = database.getCollection("predictions");
        } catch (Exception e) {
            System.out.println("MongoDB not connected. Will work in offline mode.");
        }
    }
    
    public void savePrediction(String task, double accuracy, double score2, int epochs, double lr) {
        if (collection == null) return;
        
        try {
            Document doc = new Document("task", task)
                .append("timestamp", new Date())
                .append("epochs", epochs)
                .append("learning_rate", lr);
            
            if (task.equals("classification")) {
                doc.append("accuracy", accuracy)
                   .append("f1_score", score2);
            } else {
                doc.append("mse", accuracy)
                   .append("r2", score2);
            }
            
            collection.insertOne(doc);
        } catch (Exception e) {
            System.out.println("Could not save to MongoDB: " + e.getMessage());
        }
    }
    
    public List<String> getHistory() {
        List<String> history = new ArrayList<>();
        if (collection == null) return history;
        
        try {
            FindIterable<Document> docs = collection.find().sort(new Document("timestamp", -1)).limit(10);
            for (Document doc : docs) {
                String entry = String.format("%s - %s: %.4f",
                    doc.getDate("timestamp"),
                    doc.getString("task"),
                    doc.get("accuracy") != null ? doc.getDouble("accuracy") : doc.getDouble("mse")
                );
                history.add(entry);
            }
        } catch (Exception e) {
            System.out.println("Could not read history: " + e.getMessage());
        }
        
        return history;
    }
    
    public void close() {
        if (mongoClient != null) {
            mongoClient.close();
        }
    }
}
