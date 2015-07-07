package tools;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Vector;

/**
 *
 * @author Arturo Lopez Pineda
 */
public class FileManager {

	public String read(String file) {

		StringBuffer contents = new StringBuffer();

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null; //not declared within while loop

			while ((line = input.readLine()) != null) {
				contents.append(line);
				contents.append(System.getProperty("line.separator"));
			}
		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {
			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return contents.toString();
	}

	public Vector<Vector<Integer>> readCSV(String file, String regex) {

		Vector<Vector<Integer>> contents = new Vector<Vector<Integer>>();

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null;

			//try to obtain the number of columns
			line = input.readLine();
			while (line.trim().isEmpty()) {
				line = input.readLine();
			}
			int columns = Integer.parseInt(line);

			//try to obtain the names of variables
			line = input.readLine();
			while (line.trim().isEmpty()) {
				line = input.readLine();
			}
			//String[] names = line.split(regex, columns);

			//try to get the values
			line = input.readLine();
			while (line != null) {
				//try to reduce the white spaces
				if (line.trim().isEmpty()) {
					line = input.readLine();
				} else {
					String[] row = line.split(regex, columns); //will try to get the regex ',' as many times as possible

					Vector<Integer> values = new Vector<Integer>();

					for (int i = 0; i < columns; i++) {
						if (row[i].trim().isEmpty()) {
							values.add(-1);
						} else {
							values.add(Integer.parseInt(row[i].trim()));
						}
					}

					contents.add(values);
					//System.out.println("contents.size= "+contents.size());
					line = input.readLine();
				}
			}



		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {

			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}


		return contents;
	}

	public Hashtable<String, String> readHashTable(String file) {

		Hashtable<String, String> hashTableFromFile = new Hashtable<String, String>();

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null;

			//try to obtain the number of columns
			line = input.readLine();
			while (line != null) {
				while (line.trim().isEmpty()) {
					line = input.readLine();
				}
				String content[] = line.split(",");
				if(content.length == 0){
					content = line.split("\t");
				}
				
				hashTableFromFile.put(content[0].trim(), content[1].trim());
				
				line = input.readLine();
			}
			return hashTableFromFile;






		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {

			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}


		return hashTableFromFile;
	}

	public String[] readCSVNames(String file, String regex) {

		String[] contents = new String[0];

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null;

			//try to obtain the number of columns
			line = input.readLine();
			while (line.trim().isEmpty()) {
				line = input.readLine();
			}
			int columns = Integer.parseInt(line);

			//try to obtain the names of variables
			line = input.readLine();
			while (line.trim().isEmpty()) {
				line = input.readLine();
			}
			String[] names = line.split(regex, columns);
			return names;






		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {

			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}


		return contents;
	}



	public Vector<Vector<String>> readCSV_2(String file) {

		Vector<Vector<String>> contents = new Vector<Vector<String>>();

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null;

			//try to get the values
			line = input.readLine();
			while (line != null) {
				//to reduce the within white lines
				if (line.trim().isEmpty()) {
					line = input.readLine();
				} else {
					String[] row = line.split(","); //will try to get the regex ',' as many times as possible

					Vector<String> values = new Vector<String>();

					for (int i = 0; i < row.length; i++) {
						values.add(row[i].trim());
					}

					contents.add(values);
					//System.out.println("contents.size= "+contents.size());
					line = input.readLine();
				}
			}



		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {

			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}

		contents.remove(0);


		return contents;
	}




	public ArrayList<String[]> readCSV(String file) {

		ArrayList<String[]> contents = new ArrayList<String[]>();

		BufferedReader input = null;
		try {
			input = new BufferedReader(new FileReader(file));
			String line = null;

			//try to get the values
			line = input.readLine();
			while (line != null) {
				//to reduce the within white lines
				if (line.trim().isEmpty()) {
					line = input.readLine();
				} else {
					String[] row = line.split(","); //will try to get the regex ',' as many times as possible

					contents.add(row);
					//System.out.println("contents.size= "+contents.size());
					line = input.readLine();
				}
			}

		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {

			ex.printStackTrace();
		} finally {
			try {
				if (input != null) {
					input.close();
				}
			} catch (IOException ex) {
				ex.printStackTrace();
			}
		}
		return contents;
	}


	public void write(String file, String stream) {
		// Stream to write file
		FileOutputStream fout;

		try {
			// Open an output stream
			fout = new FileOutputStream(file);

			// Print a line of text
			new PrintStream(fout).println(stream);

			// Close our output stream
			fout.close();
		} // Catches any error conditions
		catch (IOException e) {
			System.err.println("Unable to write to file '" + file + "'");
			System.exit(-1);
		}
	}

	public String setExtension(String fileName, String newVersion, String fileRoute, String ext) {
		String file = "";

		int index = 0;
		while(fileName.charAt(index) != '.') {
			if (fileName.charAt(index) != '.'){
				file = file + fileName.charAt(index);
			}
			index++;
		}
		file = fileRoute + file + newVersion + "."+ ext;

		return file;


	}
	

	public void appends(String file, String stream) {
		// Stream to write file
		FileOutputStream fout;

		try {
			// Open an output stream
			fout = new FileOutputStream(file, true);

			// Print a line of text
			new PrintStream(fout).append(stream);

			// Close our output stream
			fout.close();
		} // Catches any error conditions
		catch (IOException e) {
			System.err.println("Unable to write to file '" + file + "'");
			System.exit(-1);
		}
	}


	public void append(String fileName, String data) {
		try{

			File file =new File(fileName);

			//if file doesnt exists, then create it
			if(!file.exists()){
				file.createNewFile();
			}

			//true = append file
			FileWriter fileWritter = new FileWriter(fileName,true);
			BufferedWriter bufferWritter = new BufferedWriter(fileWritter);
			bufferWritter.append(data);
			bufferWritter.close();

		}catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public String stripName(String fileName) {
		String strip = "";
		File file = new File(fileName);

		strip = stripExtension(file.getName());
		return strip;
	}
	
	public String getExtension(String fileName) {
		String extension = "";
		File file = new File(fileName);
		String name = file.getName();
		
		int indexSeparator = name.lastIndexOf(".");
		extension = name.substring(indexSeparator, name.length());
		return extension;
	}
	
	
	public String stripExtension (String str) {
        // Handle null case specially.
        if (str == null) return null;
        
        // Get position of last '.'.
        int pos = str.lastIndexOf(".");
        
        // If there wasn't any '.' just return the string as is.
        if (pos == -1) return str;

        // Otherwise return the string, up to the dot.
        return str.substring(0, pos);
    }

	
	public String stripPath(String filename){
		String filePath = "";
		File file = new File(filename);
		String absolutePath = file.getAbsolutePath();
		filePath = absolutePath.
		    substring(0,absolutePath.lastIndexOf(File.separator));
		
		return filePath;
	}

	public String replaceExtension(String filename, String string) {
		String path = stripPath(filename);
		String name = stripName(filename);
		String newFilename = "";
		if(path.equals("")){
			newFilename = name+string;
		}
		else{
			newFilename = path+"/"+name+string;
		}
		
		return newFilename;
	}

	//end of program
}
