




import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.mllib.tree.configuration.Algo;
import org.apache.spark.mllib.tree.impurity.Gini;
import org.apache.spark.mllib.tree.impurity.Impurity;




public class CtrPredict {
  public static void main(String[] args) throws IOException, URISyntaxException {
    // $Initialize Spark, http://spark.apache.org/docs/latest/programming-guide.html$
     SparkConf sparkConf = new SparkConf().setAppName("CtrPredict").setMaster("local[*]");
	// Configure some of the common properties

    // Create a JavaSparkContext object
    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // Read the file from HDFS and split the data into Training data and Testing Data
    double[] weights = new double[2];
    weights[0] = 0.75;
    weights[1] = 0.25;
    long seed = 10L;
    
    JavaRDD<String>[] ctrJavaRDD = sc.textFile("hdfs://localhost:8020/user/ctr/Small.csv", 8).randomSplit(weights, seed);   //run on cloudera
    //JavaRDD<String>[] ctrJavaRDD = sc.textFile("D:\\FAU-Class\\CAP6807-Computational Advertising & Real-Time Data Analytics\\Project2\\Small.csv", 8).randomSplit(weights, seed);  //run on local laptop
    
    // Training Data
    JavaRDD<String> train_raw_javaRDD = ctrJavaRDD[0];
    // Testing Data
    JavaRDD<String> test_raw_javaRDD = ctrJavaRDD[1];
    
    System.out.println("Train records : " + train_raw_javaRDD.count());
    System.out.println("Test records : " + test_raw_javaRDD.count());
    
    train_raw_javaRDD.cache();
    test_raw_javaRDD.cache();
    
    
    // Count the distinct 
    JavaRDD<String> train_categoryRDD = train_raw_javaRDD.flatMap(new FlatMapFunction<String, String>() {
		public Iterable<String> call(String s) {
			String[] tokens=s.split(",",-1);  //make array index

			String[] catfeatures = Arrays.copyOfRange(tokens, 5, 14);

			return Arrays.asList(catfeatures);
			
		}
	});
    Map<String, Long> OHEMap =  train_categoryRDD.distinct().zipWithIndex().collectAsMap();
    
    final Broadcast<Map<String, Long>> OHEDict = sc.broadcast(OHEMap);
    
    JavaRDD<LabeledPoint> ctrFeatures = train_raw_javaRDD.map(new Function<String, LabeledPoint>() {
        public LabeledPoint call(String s) {
        	String[] tokens = s.split(",",-1);     //make array index
        	double label = Double.valueOf(tokens[1]);
        	int[] OHEIndex = new int[9] ;
        	int t = 0;
        	double[] OHEValue= new double[9];
        	for(int i = 5; i<14; i++){
        		OHEIndex[t] = (int) OHEDict.getValue().get(tokens[i]).intValue();
        		OHEValue[t] = 1.0;
        		t++;
        	}
        		
            return new LabeledPoint(label, Vectors.sparse(OHEDict.getValue().size(), OHEIndex, OHEValue));
          }
        });
    
    final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
    .setNumClasses(2)
    .run(ctrFeatures.rdd());
    
    JavaRDD<LabeledPoint> ctrTest = test_raw_javaRDD.map(new Function<String, LabeledPoint>() {
        public LabeledPoint call(String s) {
        	String[] tokens = s.split(",",-1);      //make array index
        	double label = Double.valueOf(tokens[1]);
        	int[] OHEIndex = new int[9] ;
        	int t = 0;
        	double[] OHEValue= new double[9];
        	for(int i = 5; i<14; i++){
        		if(OHEDict.getValue().containsKey(tokens[i])){
        		OHEIndex[t] = (int) OHEDict.getValue().get(tokens[i]).intValue();
        		OHEValue[t] = 1.0;
        		t++;
        		}
        	}
        		
            return new LabeledPoint(label, Vectors.sparse(OHEDict.getValue().size(), OHEIndex, OHEValue));
          }
        });
    
    JavaRDD<Tuple2<Object, Object>> predictionAndLabels = ctrTest.map(
    		  new Function<LabeledPoint, Tuple2<Object, Object>>() {
    		    public Tuple2<Object, Object> call(LabeledPoint p) {
    		      Double prediction = model.predict(p.features());
    		      return new Tuple2<Object, Object>(prediction, p.label());
    		    }
    		  }
    		);
    
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    
    System.out.println("Precision = " + metrics.precision());
    System.out.println("confusionMatrix = " + metrics.confusionMatrix());
    System.out.println(OHEMap.size());
    
    //results are written into HDFS
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(new URI( "hdfs://localhost:8020" ), conf );
    Path outFile = new Path("hdfs://localhost:8020/user/ctr/results.txt");
  
    FSDataOutputStream out; 
    if(fs.exists(outFile))
         fs.delete(outFile,true); //fs.create(outFile, true);
    out =fs.create(outFile);
    out.write(("Precision = " + metrics.precision()+"\n").getBytes());
    out.write(("confusionMatrix = " + metrics.confusionMatrix()).getBytes()); 
  }
}
