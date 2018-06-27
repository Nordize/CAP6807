import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
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


public class ctrEstimate {

	public static void main(String[] args) throws IOException, URISyntaxException {
	// $Initialize Spark, http://spark.apache.org/docs/latest/programming-guide.html$
	SparkConf sparkConf = new SparkConf().setAppName("ctrEstimate").setMaster("local");
	// Configure some of the common properties
	JavaSparkContext sc = new JavaSparkContext(sparkConf);
	// Read the file from HDFS
	//JavaRDD<String> CtrJavaRDD = sc.textFile("hdfs://localhost:8020/user/ctr/Large.csv"); //run on cloudera
	JavaRDD<String> CtrJavaRDD = sc.textFile("D:\\FAU-Class\\CAP6807-Computational Advertising & Real-Time Data Analytics\\Project2\\Large.csv");  //run on local laptop
	
	//PairRdd
	JavaPairRDD<String, Double> pairRdd = CtrJavaRDD.mapToPair(new PairFunction<String, String, Double>() 
	{
		@Override
		public Tuple2<String, Double> call(String str)  
		{
		
			String[] token = str.split(",", -1);  //make array index
	
			return new Tuple2<String, Double>(token[5], Double.parseDouble(token[1]));  //[5] site_id groupby [1] click
			//return new Tuple2<String, Double>(token[14], Double.parseDouble(token[1]));  //[14] device_type groupby [1] click
		}
	});
	
	JavaPairRDD<String, Iterable<Double>> groupRdd = pairRdd.groupByKey(); // use groupbykey()
	
	for (Tuple2<String, Iterable<Double>> tuple : groupRdd.collect()) 
	{
		Iterator<Double> it = tuple._2.iterator();
		double sum = 0.0;  //initial
		int count = 0;   //initial
		
		while (it.hasNext()) {
			sum += it.next();
			count++;
		}
	
	if( sum >7000)
	{
		//System.out.println("Device_type, click =1 "+tuple._1 + " - " + sum);  // sum no.of click =1
		System.out.println("Site_id, click =1 "+tuple._1 + " - " + sum);
		System.out.println("Calculat CTR = "+tuple._1 + " - " + sum / count); // calculate ctr = sum no.click/total imp. count
		System.out.println("Calculat CTR% = "+tuple._1 + " - " + (sum / count)*100); //make %
	}
	
		/* //results are written into HDFS
	    Configuration conf = new Configuration();
	    FileSystem fs = FileSystem.get(new URI( "hdfs://localhost:8020" ), conf );
	    Path outFile = new Path("hdfs://localhost:8020/user/ctr/results.txt");
	  
	    FSDataOutputStream out; 
	    if(fs.exists(outFile))
	         fs.delete(outFile,true); //fs.create(outFile, true);
	    out =fs.create(outFile);
	    out.write(("Site_id, click =1 "+tuple._1 + " - " + sum +"\n").getBytes());
	    out.write(("Calculat CTR = "+tuple._1 + " - " + sum / count + "\n").getBytes()); 	
		out.write(("Calculat CTR% = "+tuple._1 + " - " + (sum / count)*100).getBytes());  */
		
	}
	}
}
