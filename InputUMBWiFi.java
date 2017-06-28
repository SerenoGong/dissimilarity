import java.io.*;
import java.util.Random;

public class InputUMBWiFi {
	public  int data_dim;
	public  int loc_dim;
	public double cell_resolution = 0; // area divided into pixel cells; this is side length of a cell
	
	public  int num_points;
	public  double[][] fingerprint; // feature vector
	public	double[][] location; // value (which is location in our project)
	public double[][] location_bound;
	
	double[][] dissim_matrix; // dissimilarity matrix
	
	
	InputUMBWiFi(String fingerprint_filename) {
		loc_dim = 2;
		location_bound = new double[loc_dim][2];

		// e.g., fileFingerprints = "umbwheatley.wifi.scaled.txt" (// umb.wifi format)	
		// read fingerprint and location information
		
		int i;
		int num_features = 0;		
		double avg_num_APs_per_location = 0.0;
		
		try{
			FileInputStream fstream = new FileInputStream(fingerprint_filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			
			String strLine;
			int loc = 0;
			while((strLine = br.readLine()) != null){
				if(strLine.contains("area")){
					String[] s = strLine.split(" ");

					location_bound[0][0] =  Double.parseDouble(s[1]);
					location_bound[0][1] = Double.parseDouble(s[3]);
					
					location_bound[1][0] = Double.parseDouble(s[5]);
					location_bound[1][1] = Double.parseDouble(s[7]);
					continue;
				}
				
				if(strLine.contains("data_dim")){
					data_dim = Integer.parseInt(strLine.split("=")[1]);
					continue;
				}
				
				if(strLine.contains("num_points")){
					num_points = Integer.parseInt(strLine.split("=")[1]);
					
					fingerprint = new double[num_points][data_dim];
					location = new double[num_points][loc_dim];
					continue;
				}
				
				
				if(strLine.contains("###")){
					//get x, y coordinate pixels
					String[] l = strLine.split("###")[1].split(",");
					assert(loc_dim == location.length): "location dimension mismatch";
					for (i = 0; i < loc_dim; i++) {
						location[loc][i] = Double.parseDouble(l[i]);
						
					}
					//get the value of each feature
					strLine = br.readLine();
					String[] str = strLine.split(";");
					
					avg_num_APs_per_location += (double) str.length;
					
					for(i=0; i<str.length;i++) {
						int index = Integer.parseInt(str[i].split(":")[0]);
						double value = Double.parseDouble(str[i].split(":")[1]);
						fingerprint[loc][index] = value;						
						num_features = Math.max(num_features, index+1);
					}
					
					loc++;
					assert(loc <= num_points): "num. samples not match";
				}
			}
			
			System.out.println("avg_num_APs_per_location = " + avg_num_APs_per_location / (double) num_points);
			
			assert(num_features==data_dim): "data_dim not match";
			in.close();
			
		} 
		catch(IOException ioException){
			System.err.println("InputUMBWiFi(): "  + ioException.getMessage());
			System.exit(-1);
		}
		
		
	}

	private void saveLocationFile(String location_filename) {
		// generate dissimilarity matrix as a function of fingerprint vector
		// normalize it such that maximum possible value is  1
		MyMatrix.saveAs(location, location_filename);
	}
	
	private void saveDissimFile(String type, String dissim_filename) {
		// generate dissimilarity matrix as a function of fingerprint vector
		// normalize it such that maximum possible value is  1
		
		int n = num_points;
		double e = Double.NaN;
		
		double[][] F = new double[n][n];
		for (int i=0; i< n; i++) {
			for (int j=0; j < i; j++) {
				if (type.compareTo("euclidean")==0) {
					// linear
					e = Misc.euclideanDist(fingerprint[i], fingerprint[j]);
				}

				F[i][j] = e;
				F[j][i] = F[i][j];
			}
		}
		
		// normalize it such that maximum possible value is  1
		MyMatrix.saveAs(MyMatrix.normalize_01(F), dissim_filename);
	}
	

	void generate_W_Files(String output_dir) {
		int n= num_points;
		for (int p=20; p<=100; p+=20) {
			int[][] W = null;
			while(true) {
				// will break if W forms a connected graph
				// else, continue trying to find a good W
				W = MyMatrix.random_01Matrix(n, (double) p/100.0);
				if (Misc.isConnectedGraph(W)) break;
			}
			MyMatrix.saveAs(W, output_dir + "/W" + p + "N" + n);
		}
	}
	void generate_L_Files(String output_dir) {
		// generate a list of nodes serving as location-training nodes; i.e., those with location known
		Random rand = new Random();
		int n= num_points;
		
		for (int p=20; p<=200; p+=20) {
			String filename = output_dir + "/L" + p+"N" + n;
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
				System.err.println("generate_L_Files(): " + e.getMessage());
				System.exit(-1);
			}
		}
	
	}
	
	
	public static void main(String[] args) {
		String working_dir = "input_wifidata";
		String dataset_name = "umbcs"; 
		String rssi_filename = working_dir + "/" + dataset_name + ".wifi.scaled.txt";
		InputUMBWiFi input = new InputUMBWiFi(rssi_filename);	
		//input.saveLocationFile(rssi_filename +"_location");
		input.generate_W_Files(working_dir);
		//input.generate_L_Files(working_dir);
		//String dissim_type = "euclidean";
		//input.saveDissimFile(dissim_type, rssi_filename +"_dissim_"+dissim_type);
	}
}
