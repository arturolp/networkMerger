package main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import tools.FileManager;
import latentTAN.mergeModels;

/**
 *
 * @author Arturo Lopez Pineda <arl68@pitt.edu>
 * Date: June 29, 2015
 */
public class argumentHandler {

	public static ArrayList<String> models = new ArrayList<String>();
	public static Boolean em = false;
	public static String train = "";
	public static String target = "class";
	public static String type = "independent";
	public static String outputFile = "";
	//public static String lib = "";

	public static void showInfo(){
		System.out.println("Expected commands format: -networks model1.xdsl model2.xdsl [-type independent] [model3.xdsl ...] [-train train.arff] [-target class] [-outputFile merged.xdsl]");
		System.out.println("-networks file1 file2.\t At least 2 XDSL or NET files that will be merged. Must have a common target variable");
		System.out.println("-type [independent|threeway|cascade]\tThe type of merging that is seeked. Default is independent (naïve)");
		System.out.println("-train train.arff.\t The training data file that will be used to estimate the parameters of the latent variables with EM. Default: no estimation.");
		System.out.println("-target class.\tThe name of the target variable that is in common between all models. Default: class");
		System.out.println("-outputFile merged.xdsl\tThe name of the output file. Default: merged.xdsl in the location of the first model");
		//System.out.println("-lib libjsmile.jnilib\tThe SMILE library. It should be specific to your Operating System. You can download it from: https://dslpitt.org/genie/");
	}

	/**
	 * @param args the command line arguments
	 */
	public static void main(String[] args) {


		if((args.length < 3)){
			System.out.println("Insuficient arguments.");
			showInfo();
			System.exit(1);
		}


		//Capture all arguments and check for consistency in their numbers
		for(int i = 0; i < args.length; i++){
			if(args[i].equalsIgnoreCase("-networks")){
				if(i+1 < args.length){
					int j= i+1;
					while(j < args.length){
						if(args[j].startsWith("-")){
							break;
						}
						else{
							models.add(args[j]);
						}
						j++;
					}
					if(models.size() < 2){
						System.out.println("-- Please provide annother argument to (only one network given) -networks");
						showInfo();
						System.exit(1);
					}
				}
				else{
					System.out.println("-- No arguments provided for -networks");
					showInfo();
					System.exit(1);
				}
			}
			else if(args[i].equalsIgnoreCase("-type")){
				if(i+1 < args.length){
					if(type.equalsIgnoreCase("independent") || type.equalsIgnoreCase("threeway") || type.equalsIgnoreCase("cascade")){
						type = args[i+1];
					}
					else{
						System.out.println("-- The argument provided for -type is not a valid option: choose between: [independent | threeway | cascade]");
						showInfo();
						System.exit(1);
					}
				}
				else{
					System.out.println("-- No argumnet provided for -type");
					showInfo();
					System.exit(1);
				}
			}
			else if(args[i].equalsIgnoreCase("-train")){
				if(i+1 < args.length){
					train = args[i+1];
					em = true;
				}
				else{
					System.out.println("-- No argument provided for -train");
					showInfo();
					System.exit(1);
				}
			}
			else if(args[i].equalsIgnoreCase("-target")){
				if(i+1 < args.length){
					target = args[i+1];
				}
				else{
					System.out.println("-- No argument provided for -target");
					showInfo();
					System.exit(1);
				}
			}
			else if(args[i].equalsIgnoreCase("-outputFile")){
				if(i+1 < args.length){
					outputFile = args[i+1];
				}
				else{
					System.out.println("-- No argument provided for -outputFile");
					showInfo();
					System.exit(1);
				}
			}
			/*else if(args[i].equalsIgnoreCase("-lib")){
				if(i+1 < args.length){
					lib = args[i+1];
				}
				else{
					System.out.println("-- No argument provided for -lib");
					showInfo();
					System.exit(1);
				}
			}*/
		}

		//System.out.println("outputFile: "+outputFile);
		if(outputFile.equals("")){
			FileManager fm = new FileManager();

			String extension =  fm.getExtension(models.get(0));
			String path = fm.stripPath(models.get(0));
			outputFile = path+"/merged-"+type+extension;
		}
		//System.out.println("out: "+outputFile);


		//Load SMILE library
		/* This code didn't work. Instead, I copied the jnilib file to:
		 * /System/Library/Java/Extensions
		 * 
		 
		try {
			File jnilib = new File(lib);
			System.out.println("path: "+System.getProperty("java.library.path"));
			System.out.println("Lib Path: "+jnilib.getAbsolutePath());
			System.load(jnilib.getAbsolutePath());
		} catch (UnsatisfiedLinkError e) {
			System.err.println("Native code library failed to load.\n" + e);
			System.exit(1);
		}*/


		//Runner
		mergeModels merge = new mergeModels();
		merge.runner(models.toArray(new String[models.size()]), outputFile, type, train, target);

		//Finalize
		System.out.println("outputFile: "+outputFile);
		if(em == false){
			System.out.println("EM was not used");
		}
		System.out.println("----\n");

	}
}
