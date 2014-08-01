/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package javaapplication3;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.RegressionResultAnalyzer;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.common.RandomUtils;

/**
 *
 * @author ivan
 */
public class RunRandomForestSeq {
    
    public static void main(String[] args) throws IOException {
        
        String outputFile = "data/out";
        String inputFile = "data/DataFraud1MTest.csv";
        String modelFile = "data/forest.seq";
        String infoFile = "data/DataFraud1M.info";
        
        Path dataPath = new Path(inputFile); // test data path
        Path datasetPath = new Path(infoFile);
        Path modelPath = new Path(modelFile); // path where the forest is stored
        Path outputPath = new Path(outputFile); // path to predictions file, if null do not output the predictions
        
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        
        FileSystem outFS = FileSystem.get(conf);
      
        
        //log.info("Loading the forest...");
        System.out.println("Loading the forest");
        DecisionForest forest = DecisionForest.load(conf, modelPath);

        if (forest == null) 
            System.err.println("No decision forest found!");
            //log.error("No Decision Forest found!");
                
         // load the dataset
        Dataset dataset = Dataset.load(conf, datasetPath);
        DataConverter converter = new DataConverter(dataset);
         
        //log.info("Sequential classification...");
        System.out.println("Sequential classification");
        long time = System.currentTimeMillis();

        Random rng = RandomUtils.getRandom();
        
        List<double[]> resList = Lists.newArrayList();
        if (fs.getFileStatus(dataPath).isDir()) {
         //the input is a directory of files
         testDirectory(outputPath, converter, forest, dataset, resList, rng,fs,dataPath,outFS);
         } else {
        // the input is one single file
         testFile(dataPath, outputPath, converter, forest, dataset, resList, rng, outFS ,fs);
        }
        
        time = System.currentTimeMillis() - time;
        //log.info("Classification Time: {}", DFUtils.elapsedTime(time));
        System.out.println("Classification time: "+DFUtils.elapsedTime(time));
    
        if (dataset.isNumerical(dataset.getLabelId())) {
            
            RegressionResultAnalyzer regressionAnalyzer = new RegressionResultAnalyzer();
            double[][] results = new double[resList.size()][2];
            regressionAnalyzer.setInstances(resList.toArray(results));
            //log.info("{}", regressionAnalyzer);
            System.out.println(regressionAnalyzer.toString());
            
        } else {
            ResultAnalyzer analyzer = new ResultAnalyzer(Arrays.asList(dataset.labels()), "unknown");
            for (double[] r : resList) {
              analyzer.addInstance(dataset.getLabelString(r[0]),
                new ClassifierResult(dataset.getLabelString(r[1]), 1.0));
            }
            //log.info("{}", analyzer);
            System.out.println(analyzer.toString());
        }   
    
  }
    
    private static void testDirectory(Path outPath,
                             DataConverter converter,
                             DecisionForest forest,
                             Dataset dataset,
                             Collection<double[]> results,
                             Random rng,
                             FileSystem dataFS, 
                             Path dataPath,
                             FileSystem outFS) throws IOException {
    Path[] infiles = DFUtils.listOutputFiles(dataFS, dataPath);

    for (Path path : infiles) {
      //log.info("Classifying : {}", path);
        System.out.println("Classifying "+path);
      Path outfile = outPath != null ? new Path(outPath, path.getName()).suffix(".out") : null;
      testFile(path, outfile, converter, forest, dataset, results, rng,outFS,dataFS);
    }
  }
    
  private static void testFile(Path inPath,
                        Path outPath,
                        DataConverter converter,
                        DecisionForest forest,
                        Dataset dataset,
                        Collection<double[]> results,
                        Random rng,
                        FileSystem outFS,
                        FileSystem dataFS) throws IOException {
    // create the predictions file
    FSDataOutputStream ofile = null;

    if (outPath != null) {
      ofile = outFS.create(outPath);
    }

    FSDataInputStream input = dataFS.open(inPath);
    try {
      Scanner scanner = new Scanner(input, "UTF-8");

      while (scanner.hasNextLine()) {
        String line = scanner.nextLine();
        if (line.isEmpty()) {
          continue; // skip empty lines
        }

        Instance instance = converter.convert(line);
        double prediction = forest.classify(dataset, rng, instance);

        if (ofile != null) {
          ofile.writeChars(Double.toString(prediction)); // write the prediction
          ofile.writeChar('\n');
        }
        
        results.add(new double[] {dataset.getLabel(instance), prediction});
      }

      scanner.close();
    } finally {
      Closeables.close(input, true);
    }
  }  
    
}
