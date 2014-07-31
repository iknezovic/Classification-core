/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package javaapplication3;

import java.io.IOException;
import java.util.Arrays;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.RegressionResultAnalyzer;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.df.data.Dataset;
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
        String predictionFile = "data/prediction";
        String infoFile = "data/DataFraud1M.info";
        
       
        Path dataPath = new Path(inputFile); // test data path

        Path datasetPath = new Path(infoFile);

        Path modelPath = new Path(modelFile); // path where the forest is stored

      
        Path outputPath = new Path(outputFile); // path to predictions file, if null do not output the predictions

        Configuration conf = new Configuration();
        
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
        
            if (dataset.isNumerical(dataset.getLabelId())) {

              RegressionResultAnalyzer regressionAnalyzer = new RegressionResultAnalyzer();
              regressionAnalyzer.setInstances(results);
                System.out.println(regressionAnalyzer.toString());
              //log.info("{}", regressionAnalyzer);
            } else {

              ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
              
              for (double[] res : results) {
                analyzer.addInstance(dataset.getLabelString(res[0]),
                  new ClassifierResult(dataset.getLabelString(res[1]), 1.0));
              }
                System.out.println(analyzer.toString());
              //log.info("{}", analyzer);
            }
      }
    
        
        
        
    
    }
  
}
