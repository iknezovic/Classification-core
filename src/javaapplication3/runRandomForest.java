/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package javaapplication3;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.RegressionResultAnalyzer;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.Classifier;


/**
 *
 * @author ivan
 */
public class runRandomForest extends Configured{
    

    public static void main(String[] args) throws InterruptedException, IOException, ClassNotFoundException {
        
        String outputFile = "data/lule24";
        String inputFile = "data/DataFraud1MTest.csv";
        String modelFile = "data/forest.seq";
        String infoFile = "data/DataFraud1M.info";
        
       
        Path dataPath = new Path(inputFile); // test data path
        Path datasetPath = new Path(infoFile);
        Path modelPath = new Path(modelFile); // path where the forest is stored
        Path outputPath = new Path(outputFile); // path to predictions file, if null do not output the predictions

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        /*
        p = Runtime.getRuntime().exec("bash /home/ivan/hadoop-1.2.1/bin/start-all.sh");
        p.waitFor();*/
      
        if (outputPath == null) {
            throw new IllegalArgumentException("You must specify the ouputPath when using the mapreduce implementation");
        }

        Classifier classifier = new Classifier(modelPath, dataPath, datasetPath, outputPath, conf);

        classifier.run();

    
        double[][] results = classifier.getResults();
      
        if(results != null) {
          
            Dataset dataset = Dataset.load(conf, datasetPath);
            Data data = DataLoader.loadData(dataset, fs, dataPath);

            Instance inst;

            for(int i = 0; i < data.size(); i++)
            { 
                inst = data.get(i);
                
                //System.out.println("Prediction:"+inst.get(7)+" Real value:"+results[i][1]);
                System.out.println(inst.get(0)+" "+inst.get(1)+" "+inst.get(2)+" "+inst.get(3)+" "+inst.get(4)+" "+inst.get(5)+" "+inst.get(6)+" "+inst.get(7)+" ");
            }

            ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");

            for (double[] res : results) {
              analyzer.addInstance(dataset.getLabelString(res[0]),
                new ClassifierResult(dataset.getLabelString(res[1]), 1.0));
                System.out.println("Prvi shit:"+res[0]+" Drugi Shit"+res[1]);
                }
            
            System.out.println(analyzer.toString());

        }
    
      
    }
    
    /*private static int getSize(Instance inst){
        int i = 0;
        
        while(inst.get(i) != null)
        {
                
        }    
    } */
    
   
}
  

