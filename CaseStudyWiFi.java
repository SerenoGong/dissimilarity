import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;


public class CaseStudyWiFi {
	String working_dir;
	
	static String[] SURFACE = {"umbcs", "umbccul", "umbwheatley"};
	static String[] DISTANCE = {"euclidean"};
	static String[] DISSIM = {"linear"};
	
	static int[] NUMPOINTS = {208};
	static int[] DIMENSION = {2};
	
	CaseStudyWiFi(String dir) {
		working_dir = dir;
	}
	void setCaseStudy(int case_study) {
		String rssi_file = "";
		String dist_file = "";	
		String validpoint_file = null;
		
		switch (case_study) {
			case 1:
				rssi_file = "umbccul.wifi.scaled.txt";
				dist_file = "umbccul.trajectory.txt"; // file storing sequences;
				validpoint_file = "umbccul.validpoints.csv";
				break;
			case 2:
				rssi_file = "umbcs.wifi.scaled.txt";
				dist_file = "umbcs.trajectory.txt"; // file storing sequences;
				validpoint_file = "umbcs.validpoints.csv";
				break;
			case 3:
				rssi_file = "umbwheatley.wifi.scaled.txt";
				dist_file = "umbwheatley.trajectory.txt"; // file storing sequences;
				validpoint_file = "umbwheatley.validpoints.csv";
				break;
			case 7:
				rssi_file = "trento.wifi.txt";
				dist_file = "NonTrajectories.257_257.txt"; // file storing sequences;
				break;
			default:
				System.out.println("wrong case study");
				System.exit(-1);
		}
		
		if (validpoint_file != null) init("input_files/" + data_set, "input_files/" + validpoint_file);	
		else init("input_files/" + data_set, null);
		
		readSequenceFile("input_files/" + sub_case);
		//saveFingerprintMatrix("input_files/" + data_set+"_"+sub_case + ".matrix");
		
		// create a directory for the main case 
		working_dir = working_dir  +  data_set + "/";
		new File(working_dir).mkdirs();		
		// create a sub-directory for the sub-case 
		working_dir = working_dir + sub_case + "/";
		new File(working_dir).mkdirs();
	}
	


	
	static String filename_location(int n, int d, String surface) {
		return surface + "/D"+d + "N"+n;
	}
	static String filename_dissim(int n, int d, String surface, String distance, String dissim) {
		return filename_location(n,d,surface) + "_" + distance + "_" + dissim;
	}
	static String filename_W(int n, int p) {
		return "W"+p +"N" + n;
	}
	static String filename_L(int n, int p) {
		return "L"+p +"N" + n;
	}
	
	void generateAllFiles() {
		generateFiles();
		
	}
	
	public static void main(String[] args) {
		
		CaseStudy cs = new CaseStudy("input_data");
		cs.generateAllFiles();
	}

}
