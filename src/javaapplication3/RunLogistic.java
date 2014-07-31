/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package javaapplication3;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import org.apache.mahout.classifier.evaluation.Auc;
import org.apache.mahout.classifier.sgd.CsvRecordFactory;
import org.apache.mahout.classifier.sgd.LogisticModelParameters;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;



public class RunLogistic {
    
    private static final String inputFile="data/DataFraud100kTest.csv";
    private static final String modelFile = "data/model";
    
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
        Auc collector = new Auc();
        
        LogisticModelParameters lmp = LogisticModelParameters.loadFrom(new File(modelFile));

        CsvRecordFactory csv = lmp.getCsvRecordFactory();
        OnlineLogisticRegression lr = lmp.createRegression();
        
        BufferedReader in = open(inputFile);
        
        
        String line = in.readLine();
        csv.firstLine(line);
        line = in.readLine();
        int correct = 0;
        int wrong = 0;
        while(line != null){
            Vector v = new SequentialAccessSparseVector(lmp.getNumFeatures());
            int target = csv.processLine(line, v);
            
            System.out.println(line);
            String [] split = line.split(",");
            
            
            double score = lr.classifyFull(v).maxValueIndex();
            if(score == target)
               correct++;
            else
               wrong++;
                       
            System.out.println("Target is: "+target+" Score: "+score);
            line = in.readLine();
            collector.add(target, score);
            
        }
        double posto = ((double)wrong/(double)(correct+wrong))*100;
        System.out.println("Total: "+(correct+wrong)+" Correct: "+correct+" Wrong: "+wrong+" Wrong pct: "+posto +"%");
        //PrintWriter output = null;
        Matrix m = collector.confusion();
        //output.printf(Locale.ENGLISH, "confusion: [[%.1f, %.1f], [%.1f, %.1f]]%n",m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
        System.out.println("Confusion:"+m.get(0, 0)+" "+m.get(1, 0)+"\n \t   "+m.get(0, 1)+" "+m.get(1, 1)+" ");
//        m = collector.entropy();
        //output.printf(Locale.ENGLISH, "entropy: [[%.1f, %.1f], [%.1f, %.1f]]%n",m.get(0, 0), m.get(1, 0), m.get(0, 1), m.get(1, 1));
        
    }
    
    static BufferedReader open(String inputFile) throws IOException {
    InputStream in;
    try {
      in = Resources.getResource(inputFile).openStream();
    } catch (IllegalArgumentException e) {
      in = new FileInputStream(new File(inputFile));
    }
    return new BufferedReader(new InputStreamReader(in, Charsets.UTF_8));
  }
    
    
}
    

