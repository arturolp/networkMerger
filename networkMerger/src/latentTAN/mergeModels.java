package latentTAN;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;
import smile.Network;
import smile.SMILEException;
import smile.learning.DataMatch;
import smile.learning.DataSet;
import smile.learning.EM;
import tools.FileManager;

/**
 *
 * @author arturolp
 * Date: March 12, 2012
 */
public class mergeModels {


	public Network merged = new Network();
	public Instances train;


	public mergeModels(){

	}//constructor




	public void writeOutput(String outputFile) throws IOException {
		FileManager fm = new FileManager();
		String extension = fm.getExtension(outputFile);
		if(extension.equals(".net")){
			System.out.println("Hugin");
			//change to Hugin file
			FileWriter output = new FileWriter(outputFile);
			output.write(getNetFileString());
			output.close();
		}
		else{
			FileWriter output = new FileWriter(outputFile);
			output.write(merged.writeString());
			output.close();
		}
	}



	public String getNetFileString() {
		String netfile = "";
		//ADD header
		netfile += "net \n{\n\tnode_size = (76 36);\n}\n\n";

		//Update nodes
		for(int i = 0; i < merged.getNodeCount(); i++){
			String iname = merged.getNodeName(i);

			netfile += "node "+iname+"\n{\n\tlabel = \""+iname+"\";\n";

			netfile += "\tposition = (74 278);\n";
			netfile += "\tstates = (";
			String[] states = merged.getOutcomeIds(i);
			for(int j = 0; j <states.length-1 ; j++){
				netfile += "\""+toReadableState(states[j])+"\" ";
			}
			netfile += "\""+ toReadableState(states[states.length-1])+"\");\n";
			netfile += "}\n\n";

		}

		//Update potentials
		for(int i = 0; i < merged.getNodeCount(); i++){
			String iname = merged.getNodeName(i);

			netfile += "potential (";
			netfile += iname + " | ";
			String[] iparents = merged.getParentIds(i);
			for(int j = 0; j < iparents.length-1; j++){
				netfile += iparents[j]+" ";
			}
			if(iparents.length>0){
				netfile += iparents[iparents.length-1]+")\n{\n";
			}
			else{
				netfile += ")\n{\n";
			}

			netfile += "\tdata = (";

			double[] potentials = merged.getNodeDefinition(i);

			int window = merged.getOutcomeCount(iname);
			for(int j = 0; j < potentials.length; j=j+window){
				String subtable = "";
				for(int k = 0; k < window; k++){
					subtable += potentials[j+k];
					if(k<window-1){
						subtable += " ";
					}
				}
				netfile += "(" + subtable + ")";
				if((j+window)<potentials.length){
					netfile += "\n\t";
				}
			}

			netfile += ");\n}\n\n";

		}


		return netfile;
	}



	public String toReadableState(String string) {
		//System.out.println(string);
		String readable = "(";

		String[] split = string.split("_");

		if(split.length > 1 && split[0].equals("x") ){
			//System.out.print("size: "+split.length+", ");
			boolean bracket = false;
			for(int i = 1; i < split.length; i++){
				if(split[i].equals("")){
					readable += "-";
				}
				else if(split[i].equals("inf")){
					readable += "inf";
					if(!split[i-1].equals("")){
						readable += "]";
						bracket = true;
					}
					else{
						readable += ", ";
					}
				}
				else if(split[i].length() == 1){
					readable += split[i]+"."+split[i+1];
					if((i+2)<split.length){
						readable += ", ";
					}
					i++;
				}

				//System.out.print(split[i]+", ");
			}
			if(bracket == false){
				readable += ")";
			}
			//System.out.println();
		}
		else{
			readable = string;
		}

		//System.out.println(readable);
		return readable;
	}




	public DataSet readDataset(String filename, String targetLabel) {

		DataSet data = new DataSet();
		DataSet dataFiltered = new DataSet();

		//Read weka dataset
		try {
			ArffLoader arff = new ArffLoader();
			arff.setFile(new File(filename)); 
			train = arff.getDataSet();
			int classIndex = train.numAttributes() - 1;
			train.setClassIndex(classIndex);

		} catch (IOException ex) {
			System.out.println(ex);
		}

		//Save file as CSV
		CSVSaver saver = new CSVSaver();
		FileManager fm = new FileManager();
		String filenamecsv = fm.replaceExtension(filename, ".csv");
		try {
			saver.setInstances(train);
			saver.setFile(new File(filenamecsv));
			saver.writeBatch();
		} catch (IOException e) {
			System.out.println(e);
		} 



		//Read CSV with Smile
		data.readFile(filenamecsv, "?", -1, -1.0f, true);
		System.out.println("DataSet data contains " + data.getRecordCount() + " records, over " + data.getVariableCount() + " variables");



		//Update the variable outcomes
		for(int variableIndex = 0; variableIndex < data.getVariableCount(); variableIndex++){
			String variableName = data.getVariableId(variableIndex);

			if(hasNode(variableName)){
				String[] nodeOutcomes = merged.getOutcomeIds(variableName);
				dataFiltered.addIntVariable(variableName, -1);
				dataFiltered.setStateNames(dataFiltered.getVariableCount()-1, nodeOutcomes);
			}
		}

		//Update values in dataset with the new ordering
		for(int recordIdx = 0 ; recordIdx < data.getRecordCount(); recordIdx++) {
			dataFiltered.addEmptyRecord();
			for(int varIdx = 0; varIdx < data.getVariableCount(); varIdx++){
				String variableName = data.getVariableId(varIdx);

				if(hasNode(variableName)){

					int varIdxF = dataFiltered.findVariable(variableName);
					int oldInt = data.getInt(varIdx, recordIdx);
					int newInt = -1;
					//System.out.print(variableName+"("+varIdxF+") = ");
					if(oldInt != -1){
						newInt = getNewOrder(oldInt, data.getStateNames(varIdx), dataFiltered.getStateNames(varIdxF));
					}
					dataFiltered.setInt(varIdxF, recordIdx, newInt);
					//System.out.print(data.getInt(varIdx, recordIdx)+","+dataFiltered.getInt(varIdxF, recordIdx)+",  ");
				}

			}
			//System.out.println();
		}


		//Delete CSV temporary
		File file = new File(filenamecsv);
		file.delete();

		
		//dataFiltered.writeFile(filenamecsv);


		return dataFiltered;
	}




	public boolean hasNode(String variableName) {

		boolean hasNode = false;

		String[] nodes = merged.getAllNodeIds();
		for(int i = 0; i < nodes.length; i++){
			if(nodes[i].equals(variableName)){
				hasNode = true;
				break;
			}
		}

		return hasNode;
	}




	public String getGenieConsistentName(String oldState) {

		String newState = "";
		newState += "x_"+ oldState;
		newState = newState.replace(".", "_");
		newState = newState.replace("-", "_");
		newState = newState.replace("(", "");
		newState = newState.replace(")", "");
		newState = newState.replace("[", "");
		newState = newState.replace("]", "");
		newState = newState.replace("\\", "");
		newState = newState.replace("'", "");
		newState = newState + "_";

		//System.out.println("old: "+oldState+", new: "+newState);

		return newState;
	}




	public int getNewOrder(int oldIndex, String[] stateNamesOld,
			String[] stateNamesNew) {
		int newIndex = -1;

		String oldName = stateNamesOld[oldIndex];

		for(int i = 0; i < stateNamesNew.length; i++){
			if(stateNamesNew[i].equals(oldName)){
				newIndex = i;
			}
		}

		if(newIndex == -1){
			oldName = getGenieConsistentName(oldName);

			for(int i = 0; i < stateNamesNew.length; i++){
				if(stateNamesNew[i].equals(oldName)){
					newIndex = i;
				}
			}
		}

		return newIndex;
	}




	public String[][] reorderStates(String[] stateNames) {

		String[][] orderedNames = new String[2][stateNames.length];
		List<Double> lowerBounds = new ArrayList<Double>();

		for(int i = 0; i < stateNames.length; i++){
			String lower = getLower(stateNames[i]);
			lowerBounds.add(Double.parseDouble(lower));
			//System.out.println(lower);
		}

		Collections.sort(lowerBounds);

		int index = 0;
		for(int i = 0; i < stateNames.length; i++){
			String lower = getLower(stateNames[i]);
			int oldIndex = 0;
			for(int j = 0; j < lowerBounds.size(); j++){
				//System.out.println("lower: "+lower+", lowBoundsJ: "+lowerBounds.get(j)+", "+lowerBounds.get(j).equals(lower));
				if(lowerBounds.get(j).toString().equals(lower)){
					oldIndex = j;
					break;
				}
			}
			//System.out.println("index: "+ index + ", oldIndex: "+oldIndex);
			orderedNames[0][index] = stateNames[oldIndex];
			orderedNames[1][index] = ""+oldIndex;
			index++;
		}



		return orderedNames;
	}





	public String getLower(String bin) {
		String lower = bin;
		lower = lower.replace("x_", "");
		lower = lower.replace("0_", "0.");
		String[] split = lower.split("_");

		if(split[0].equals("")){
			if(split[1].equals("inf")){
				lower = "-Infinity";
			}
			else{
				lower = "-"+split[1];
			}
		}
		else{
			lower = split[0];
		}
		return lower;
	}




	public void mergeCascade(String[] targetNames, String targetLabel) {

		addTargetNodeIfNotExists(targetNames, targetLabel);

		//add arcs between the first latent variable and the new class node
		String[] iParents = merged.getParentIds(targetNames[0]);
		if(iParents.length == 0){
			merged.addArc(targetLabel, targetNames[0]);
		}

		//add arcs between the rest latent variables in order
		for(int i = 1; i < targetNames.length; i++){
			merged.addArc(targetNames[i-1], targetNames[i]);
		}

	}

	public void mergeIndependent(String[] targetNames, String targetLabel) {

		addTargetNodeIfNotExists(targetNames, targetLabel);


		//add arcs between the latent variables and the new class node
		for(int i = 0; i < targetNames.length; i++){
			merged.addArc(targetLabel, targetNames[i]);
		}

	}

	public void addTargetNodeIfNotExists(String[] targetNames, String targetLabel) {
		String[] nodeHandle = merged.getAllNodeIds();
		boolean hasTarget = false;
		for(int i = 0; i < nodeHandle.length; i++){
			if(nodeHandle[i].equals(targetLabel)){
				hasTarget = true;
			}
		}

		if(hasTarget == false){

			merged.addNode(Network.NodeType.Cpt, targetLabel);

			String outcomeHandle[] = merged.getOutcomeIds(targetNames[0]);

			//add outcomes to class node from first network
			for(int outcomeIndex = 0; outcomeIndex < outcomeHandle.length; outcomeIndex++){
				merged.addOutcome(targetLabel, outcomeHandle[outcomeIndex]);
			}

			//Remove default outcomes (a node cannot have empty outcomes)
			merged.deleteOutcome(targetLabel, "State0");
			merged.deleteOutcome(targetLabel, "State1");
			
			
			// Add parameters to class from first network
			double[] nodeDefinition = merged.getNodeDefinition(targetNames[0]);
			merged.setNodeDefinition(targetLabel, nodeDefinition);
		}

		
	}

	public void addNet(Network network) {
		// Add all nodes
		String[] nodeHandle = network.getAllNodeIds();

		for(int i = 0; i < nodeHandle.length; i++){
			merged.addNode(Network.NodeType.Cpt, nodeHandle[i]);

			String outcomeHandle[] = network.getOutcomeIds(nodeHandle[i]);

			//add outcomes
			//System.out.print(nodeHandle[i]+"\t-->\t");
			for(int outcomeIndex = 0; outcomeIndex < outcomeHandle.length; outcomeIndex++){
				merged.addOutcome(nodeHandle[i], outcomeHandle[outcomeIndex]);
				//System.out.print(outcomeHandle[outcomeIndex]+", ");
			}
			//System.out.println();

			//Remove default outcomes (a node cannot have empty outcomes)
			merged.deleteOutcome(nodeHandle[i], "State0");
			merged.deleteOutcome(nodeHandle[i], "State1");

		}

		// Add all arcs
		for(int j = 0; j < nodeHandle.length; j++){
			String[] parents = network.getParentIds(nodeHandle[j]);
			if(parents.length>0){
				for(int k = 0; k < parents.length; k++){
					//System.out.println("parent: "+parents[k]+", child: "+nodes[j]);
					merged.addArc(parents[k], nodeHandle[j]);
				}
			}
		}

		// Add all parameters for nodes
		for(int i = 0; i < nodeHandle.length; i++){
			double[] nodeDefinition = network.getNodeDefinition(nodeHandle[i]);
			merged.setNodeDefinition(nodeHandle[i], nodeDefinition);
		}


	}

	public void emLearning(DataSet data) {
		try {

			// Set up and config EM
			EM em = new EM();

			merged.clearAllEvidence();
			em.setRandomizeParameters(false);
			em.setUniformizeParameters(false);
			em.setRelevance(true);
			

			/*for(int v = 0; v < merged.getNodeCount(); v++){
				//System.out.println(merged.getNodeId(v)+" :" + merged.getOutcomeCount(v));
				String[] ids = merged.getOutcomeIds(0);
				String[] states = data.getStateNames(0);
				for(int i = 0; i < ids.length; i++){
					//System.out.print("id: "+ids[i]);
					//System.out.println(", state: "+states[i]);
				}
				//System.out.println("-----");
			}
			 */

			//DataMatch[] autoMatched = data.matchNetwork(merged);
			DataMatch[] matched = new DataMatch[data.getVariableCount()];

			for(int i = 0; i < matched.length; i++){
				matched[i] = new DataMatch();
				matched[i].column = i;
				String variableName = data.getVariableId(i);
				int nodeIdx = getNodeIndex(variableName);
				matched[i].node = nodeIdx; 
				//System.out.println("column: "+data.getVariableId(i)+"("+matched[i].column+"), node: "+merged.getNodeId(nodeIdx)+"("+matched[i].node+")");
				//System.out.println("column: "+autoMatched[i].column+", node: "+autoMatched[i].node);

				String[] stateNames = data.getStateNames(i);
				String[] outcomeNames = merged.getOutcomeIds(nodeIdx);
				if(stateNames.length != outcomeNames.length){
					//System.out.println(variableName+"!!!");
				}
				for(int j = 0; j < stateNames.length; j++){
					if(!stateNames[j].equals(outcomeNames[j])){
						//System.out.println("   "+stateNames[j]+"   "+outcomeNames[j]);
					}
				}
			}
			
			
			

			em.learn(data, merged, matched);

			System.out.println("EM log-likelihood: " + em.getLastScore());

			/*System.out.println("\nNode Definition GENEXP: ");
			double[] nodeDefinitionGene = merged.getNodeDefinition("genexp1");
			for(int i = 0; i < nodeDefinitionGene.length; i++){
				System.out.print(nodeDefinitionGene[i] + "   ");
			}
			System.out.println("\nNode Definition METHY: ");
			double[] nodeDefinitionMethy = merged.getNodeDefinition("methy1");
			for(int i = 0; i < nodeDefinitionMethy.length; i++){
				System.out.print(nodeDefinitionMethy[i] + "   ");
			}
			System.out.println("\nNode Definition CLASS: ");
			double[] nodeDefinitionClass = merged.getNodeDefinition("class");
			for(int i = 0; i < nodeDefinitionClass.length; i++){
				System.out.print(nodeDefinitionClass[i] + "   ");
			}
			System.out.println("\n");
			*/

			//merged.updateBeliefs();
			
			



		} catch (SMILEException sme) {
			sme.printStackTrace();
		}
	}

	public int getNodeIndex(String variableName) {
		int index = 0;

		for(int i = 0; i < merged.getNodeCount(); i++){

			if(variableName.equals(merged.getNodeId(i))){
				//System.out.println(variableName + "==?" + merged.getNodeId(i) + ", "+i);
				index = i;
				break;
			}
		}

		return index;
	}




	public void runner(String[] models, String outputFile, String type, String trainFile, String targetLabel) {

		Network[] nets;
		//Check if the network models are readable
		nets = new Network[models.length];
		for(int k = 0; k < models.length; k++){
			try{
				nets[k] = new Network();
				File absolute = new File(models[k]);
				nets[k].readFile(absolute.getAbsolutePath());
			} catch (Exception ex) {
				System.out.println(ex);
				System.exit(1);
			}
		}
		
		System.out.println("Merging the following files:");

		String[] targetNames = new String[models.length];
		
		
		for(int i = 0; i < nets.length; i++){
			System.out.println("Net "+(i+1)+": "+models[i]);

			//Get network names
			FileManager fm = new FileManager();
			String networkName = fm.stripName(models[i]);
			targetNames[i] = networkName.replace("-fold", "");

			//Rename the network class
			nets[i].setNodeId(targetLabel, targetNames[i]);

			//add all nodes and arcs
			addNet(nets[i]);
		}


		//Merge the different nodes
		System.out.println("Type: "+type);
		if(type.equalsIgnoreCase("independent")){
			mergeIndependent(targetNames, targetLabel);
		}
		if(type.equalsIgnoreCase("cascade")){
			mergeCascade(targetNames, targetLabel);
		}
		if(type.equalsIgnoreCase("threeway")){
			mergeIndependent(targetNames, targetLabel);
			mergeCascade(targetNames, targetLabel);
		}

		//EM learning

		if(!trainFile.equals("")){

			System.out.println("\nExecute EM to learn parameters for merged network from data");
			System.out.println("train data: "+trainFile);

			DataSet data = readDataset(trainFile, targetLabel);

			emLearning(data);
			

		}

		try {
			writeOutput(outputFile);
			System.out.println("[done]\n");
		} catch (IOException e) {
			System.out.println(e);
		}


	}


}
