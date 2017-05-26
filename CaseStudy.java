import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Random;


public class CaseStudy {
	String location_filename;
	String dissim_filename;
	
	String working_dir;
	
	String[] S1 = {"euclidean", "sphere"};
	String[] S2 = {"euclidean", "geodesic"};
	String[] S3 = {"linear", "poly", "log", "exp"};
	
	int[] N = {100, 300, 500, 1000};
	int[] D = {2, 3};
	
	CaseStudy(String dir) {
		working_dir = dir;
	}
	void generateFiles() {
		new File(working_dir).mkdirs();
		Random rand = new Random();
		for (int i1=0; i1 < S1.length; i1++) {
			// create directorry
			String surface_case = S1[i1];
			new File(working_dir + "/" + surface_case).mkdirs();	
			for (int n1=0; n1 < N.length; n1++) {
				int n = N[n1];
				for (int d1=0; d1 < D.length; d1++) {
					int d = D[d1];
					double[][] locations = new double[n][d];
					
					location_filename = working_dir + "/" + surface_case + "/D"+d + "N"+n;		
					switch (i1) {
					case 0:	
						// euclidean surface
						locations = new double[n][d];
						for (int i=0; i<n; i++)
							for (int j=0; j<d; j++)
								locations[i][j] = rand.nextDouble();
						break;
					case 1:
						// spherical surface
						SphereMDS mds = new SphereMDS(d, 1);
						locations = mds.randomPoint(n);
						break;
					}
					MyMatrix.saveAs(locations, location_filename);
					for (int i2=0; i2 < S2.length; i2++) {
						String distance_case = S2[i2];
						double[][] dist = new double[n][n];
						switch (i2) {
						case 0:	
							// euclidean distance
							dist = euclideanDissim(locations);
							break;
						case 1:
							// geodesic distance (applied only for non-euclidean surface)
							if (i1 == 0) {
								// for euclidean surface, geodesic distance = euclidean distance
								dist = euclideanDissim(locations);
							}
							else 
								dist = geodesicDissim(locations);
							break;
						}
						
						for (int i3=0; i3 < S3.length; i3++) {
							String dissim_case = distance_case + "_" + S3[i3];
							dissim_filename = location_filename + "_" + dissim_case;
							double[][] F = generateDissim(S3[i3], dist);
							MyMatrix.saveAs(F, dissim_filename);
						
						}
					}
				}
			}
			
		}
		
	}
	
	double[][] euclideanDissim(double[][] locations) {
		// compute euclidean distances between points
		int n = locations.length;
		double[][] dist = new double[n][n];
		
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double[] x = locations[i];
				double[] y = locations[j];
	
				dist[i][j] = Misc.euclideanDist(x, y);
				dist[j][i] = dist[i][j];
			}
		}
		return dist;
	}
		
	
	double[][] geodesicDissim(double[][] locations) {
		// compute geodesic distances between points on a unit sphere
		int n = locations.length;
		int d = locations[0].length;
		double[][] dist = new double[n][n];
		
		for (int i=0; i<n; i++) {
			for (int j=0; j<i; j++) {
				double[] x = locations[i];
				double[] y = locations[j];
				SphereMDS mds = new SphereMDS(d, 1);	
				dist[i][j] = SphereMDS.geodesic(x, y, 1);
				dist[j][i] = dist[i][j];
			}
		}
		return dist;
	}
	
	
	private double[][] generateDissim(String type, double[][] dist) {
		// generate dissimilarity matrix as a function of ground-truth distance
		// normalize it such that maximum possible value is  1
		int n = dist.length;
		double[][] F = new double[n][n];
		for (int i=0; i< n; i++) {
			for (int j=0; j < i; j++) {
				double e = dist[i][j];
				if (type.compareTo("linear")==0) {
					// linear
					F[i][j] = e;
				}
				if (type.compareTo("poly")==0) {
					// linear
					F[i][j] = e*e;
				}
				if (type.compareTo("log")==0) {
					// linear
					F[i][j] = Math.log(1+e);  //to make F[i][j]>=0 and F[i][j]=0 if e=0
				}
				if (type.compareTo("exp")==0) {
					// linear
					F[i][j] = Math.exp(e)-1; // to make F[i][j]=0 if e=0
				}
				F[j][i] = F[i][j];
			}
		}
		// normalize it such that maximum possible value is  1
		return MyMatrix.normalize_01(F);
	}
	
	void setCaseStudy(int n, int d, String surface, String distance, String dissim) {
		location_filename = working_dir + "/" + surface + "/" + "/D"+d + "N"+n;
		dissim_filename = location_filename + "_" + distance + "_" + dissim;
	}
	
	
	void generate_W_Files() {
		Random rand = new Random();
		for (int n : N) {
			for (int p=10; p<=100; p+=10) {
				int[][] W = new int[n][n];
				for (int i=0; i<n; i++) 
					for(int j=0; j<i; j++) {
						W[i][j] = (rand.nextDouble() < (double)p/100.0)? 1: 0;
						W[j][i] = W[i][j];
					}
				
				String filename = working_dir + "/" + "W_N" + n + "p" + p;
				MyMatrix.saveAs(W, filename);
			}
		}
	}
	
	
	void generate_Location_Train_Files() {
		Random rand = new Random();
		for (int n : N) {
			for (int p=10; p<=100; p+=10) {
				String filename = working_dir + "/" + "L_N" + n + "p" + p;
				double location_prob=  (double) p / 100.00;
				try {
					PrintWriter pw = new PrintWriter(filename);
					boolean first=true;
					for (int i=0; i<n; i++) 
						if (rand.nextDouble() < location_prob) {
							if (first) {
								pw.print(i);
								first = false;
							}
							else pw.print(","+i);
						}
					pw.close();
				}
				catch (IOException e) {
					System.err.println("generate_Location_Train_Files(): file error" + filename);
					System.exit(-1);
				}
			}
		}
	}
	
	public static void main(String[] args) {
		
		CaseStudy cs = new CaseStudy("input_data1");
		cs.generateFiles();
		cs.generate_W_Files();
		cs.generate_Location_Train_Files();
	}

}
