 import java.io.*;

import java.util.*;


public class Eval {

	private static String QUERY_FILE = "query.text";
	private static String QRELS = "qrels.text";
	private static final String DOC_ID = ".I";
	private static final String TITLE = ".T";
	private static final String ABSTRACT = ".W";
	private static final String PUBLICATION_DATE = ".B";
	private static final String AUTHOR = ".A";
	private static final String N =".N";
	private static final String X=".X";
	private static final String K=".K";
	private static final String SPLIT = "\\s+";
	private static String YES = "y";
	private static String NO = "n";
	private static Map<Integer, Document> documents;
	private static Map<String,DocumentFrequency> documentFrequencies;
	private static boolean stopwordsOn = true;
	private static boolean stemmerOn = true;
	private static Set<String> stopwords;
	private static int topK = 100;
	private static final String EVAL_FILE = "eval.txt";
	private static double w1=0;
	private static double w2=0;
	private static final int LIMIT =50;
	private static final double ALPHA = 0.85;
	private static final String PAGERANKS = "pagerank.txt";

	public static void setTFIDFWeights() {
		for (Integer docID : documents.keySet()) {
			Document doc = documents.get(docID);
			for (String term : doc.getTerms()) {
				double tf = doc.getFrequency(term);		
				tf = 1 + Math.log10(tf);
				DocumentFrequency docFreq = documentFrequencies.get(term);
				int df = 0;
				if (docFreq != null) {
					df = docFreq.getDocumentFrequency();
				}
				double idf = Math.log10(documents.size() / df);
				if (Double.isNaN(idf)) {
					idf = 0.0;
				}
				doc.setWeight(term, tf * idf);
			}
		}
	}

	public static Document queryVector(String query) {
		Document queryDoc = new Document();	
		queryDoc.setAbstract(query);
		DocumentParser queryParser = new DocumentParser(stemmerOn);
		queryParser.setDocument(queryDoc);
		for (String term : queryParser.findTerms()) {
			if (!stopwordsOn || !stopwords.contains(term)) {
				int tf = queryParser.calculateFrequency(term);
				queryDoc.setFrequency(term, tf);
				double weight = (1 + Math.log10(tf));
				queryDoc.setWeight(term, weight);
			}
		}
		return queryDoc;
	}

	public static double sim(Document doc, Document queryDoc) {
		double product = 0;
		double docMagnitude = 0;
		double queryMagnitude = 0;
		Set<String> terms = new HashSet<String>();
		terms.addAll(doc.getTerms());
		terms.addAll(queryDoc.getTerms());
		for(String term : terms) {			
			double docWeight = doc.getWeight(term);
			double queryWeight = queryDoc.getWeight(term);
			product += (docWeight*queryWeight);
			docMagnitude += Math.pow(docWeight, 2);
			queryMagnitude += Math.pow(queryWeight,2);
		}
		docMagnitude = Math.sqrt(docMagnitude);
		queryMagnitude = Math.sqrt(queryMagnitude);
		if (docMagnitude > 0 && queryMagnitude > 0) {
			return product / (docMagnitude * queryMagnitude);
		}
		else {
			return 0;
		}
	}

	public static Map<Integer, Set<Integer>> parseQRels(String fileName) throws IOException,FileNotFoundException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(fileName)));

		Map<Integer, Set<Integer>> qrelsList = new TreeMap<Integer, Set<Integer>>();
		for (String line = reader.readLine(); line != null; line = reader.readLine()) {
			if ((line = line.trim()).isEmpty()) {
				continue;
			}
			String[] tags = line.split("\\s+");
			int queryNum = Integer.parseInt(tags[0]);
			int docs = Integer.parseInt(tags[1]);

			Set<Integer> relevantDocs = qrelsList.get(queryNum);
			if (relevantDocs == null) {
				relevantDocs = new HashSet<Integer>();
			}
			relevantDocs.add(docs);
			qrelsList.put(queryNum, relevantDocs);
		}
		reader.close();
		return qrelsList;
	}
	
	public static  int parseID(String line) {
		String[] split = line.split(SPLIT);
		String id = split[1];
		return Integer.parseInt(id);
	}
	public  static void calculatePageRank(){
		Set<Integer> documentSet= documents.keySet();
		int n = documentSet.size();
		double pMatrix[][] = new double[n][n];//probability matrix
		double xMatrix [] = new double [n];// p-rank results matrix
		for (Integer id : documentSet ) {
			xMatrix[id-1]=0.0;//initializing the array
			Document doc = documents.get(id);
			Set<Integer> citationSet = doc.getCitation();
			double probability = 0.0;
			/* f a row of Ahas no 1’s, then 
			divide each element by NFor all other rows, do the following: 
			Divide each 1 in Aby the number of 1s in its row Multiply 
			the resulting matrix by 1-alpha; 
			add alpha/Nto every entry of the resulting matrix, to obtain P*/
			if (citationSet != null || citationSet.isEmpty()) {
				probability = (1.0/citationSet.size()) * (1.0-ALPHA) + ALPHA/n;
			}
			else {
				probability = (1-ALPHA)/n + ALPHA/n;
			}
			for (int i = 0; i < n; i++) {
				if (citationSet != null || citationSet.isEmpty()) {
					if (citationSet.contains(i+1)) {
						Document doc2 = documents.get(i+1);
						if (doc.equals(doc2)) {
							//self reference 
							pMatrix[id-1][i] = probability;
						}
						else {
							//making sure the doc is referencing an older document
							int d1 = doc.getID();
							int d2 = doc2.getID();
							
								if (doc2.getID()>(doc.getID())) {
									pMatrix[id-1][i] = probability;
								}
								pMatrix[id-1][i] = probability;
							}
					}
					else {
						pMatrix[id-1][i] = ALPHA/n;
					}
				}
				else {
					pMatrix[id-1][i] = probability;
				}
			}
		}
		xMatrix[0]=1.0; //assuming starting point is at the first document
		for (int i=1;i<=LIMIT ;i++ ) {
			double sum=0;
			double xP[]= new double [xMatrix.length];
			for (int q=0;q<n ;q++ ) {
				for (int r=0;r<n ;r++ ) {
					sum+=xMatrix[r]*pMatrix[r][q];			
				}
				xP[q]=sum;
				sum=0;//reset and calculate the next one				
			}
			xMatrix=xP;
		}
		
		
		File pageRankFile = new File(PAGERANKS);
		try {
			FileWriter writer = new FileWriter(pageRankFile);
			for (Integer docID : documents.keySet()) {
				
				Document doc = documents.get(docID);
				doc.setPageRank(xMatrix[docID-1]);
				
				writer.write("Document ID: " + docID + "\n");
				writer.write("\tPageRank Score: " + doc.getPageRank() + "\n");
			}
			writer.close();
			
		} 
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	public static double score(Document doc, Document query) {
		double similarity = w1 * sim(doc,query);
		double pageRankScore = w2 * (doc.getPageRank());
		return similarity + (pageRankScore * 100);
	}

	public static void main(String[] args) throws Exception {		
		Scanner in = new Scanner(System.in);
		System.out.println("Would you like to use stop words? y/n");
		String input = in.nextLine();

		while (!(input.equalsIgnoreCase(YES) || input.equalsIgnoreCase(NO))) {
			System.out.println("Invalid entry");
			input = in.nextLine();
		}
		if (input.equalsIgnoreCase(YES)) {
			stopwordsOn = true;
		}
		else if (input.equalsIgnoreCase(NO)) {
			stopwordsOn = false;
		}

		System.out.println("Would you like to use stemming? y/n");
		input = in.nextLine();
		while (!(input.equalsIgnoreCase(YES) || input.equalsIgnoreCase(NO))) {
			System.out.println("Invalid entry");
			input = in.nextLine();
		}		
		if (input.equalsIgnoreCase(YES)) {
			stemmerOn = true;
		}
		else if (input.equalsIgnoreCase(NO)) {
			stemmerOn = false;
		}

		System.out.println("Creating tf-idf vectors and calculating similarity scores, please wait...");

		//parsing query.text using same method as the FileParser
		BufferedReader reader = new BufferedReader(new FileReader(new File(QUERY_FILE)));
		LinkedList<Query> queries = new LinkedList<Query>();
		int currentID = 0;
			String line = reader.readLine();
				while (line != null) {
				Query q = new Query();
				if (line.startsWith(DOC_ID)) {
					currentID = parseID(line);
					
					q.setID(currentID);
					line = reader.readLine();

				}
					
				else if (line.startsWith(ABSTRACT)) {
					line = reader.readLine();
					StringBuilder builder = new StringBuilder();
					while (line != null && !(line.startsWith(DOC_ID) || line.startsWith(TITLE) || line.startsWith(ABSTRACT) ||
							line.startsWith(PUBLICATION_DATE) || line.startsWith(AUTHOR)||line.startsWith(N)||line.startsWith(X)||line.startsWith(K) )) {
						builder.append(line + "\n");
						line = reader.readLine();
					}
					q.setQuery(builder.toString());
					queries.add(q);
				}	
				
				
				
				else {
					line = reader.readLine();
				}


			}
		reader.close();

		Map<Integer,Set<Integer>> relevantDocs = parseQRels(QRELS);

		long startTime = System.currentTimeMillis();

		Invert invertedIndex = new Invert(stopwordsOn,stemmerOn);		
		invertedIndex.invert();

		stopwords = invertedIndex.getStopWords();
		documents = invertedIndex.getDocuments();
		documentFrequencies = invertedIndex.getDocumentFrequencies();

		setTFIDFWeights();
		calculatePageRank();
		

		long totalTime = System.currentTimeMillis() - startTime;

		System.out.println("Total time to create tf-idf vectors and calculate cosine scores: " + totalTime/1000 + "seconds");
		System.out.println("Please enter values for w1 and w2, where w1 + w2 = 1");
		System.out.print("w1: ");
		w1 = Double.parseDouble(in.nextLine());
		System.out.print("w2: ");
		w2 = Double.parseDouble(in.nextLine());
		while (w1 + w2 != 1) {
			System.out.println("You have entered invalid w1 and w2 values, please re-enter: ");
			System.out.print("w1: ");
			w1 = Double.parseDouble(in.nextLine());
			System.out.print("w2: ");
			w2 = Double.parseDouble(in.nextLine());
		}
		

		System.out.print("\nCreating MAP and R-Precision Values\n");
		Set<Double> mapSet = new HashSet<Double>();
		Set<Double> rPrecisionSet = new HashSet<Double>();

		File evalFile = new File(EVAL_FILE);
		FileWriter writer = new FileWriter(evalFile);
		
		long queryStart = System.currentTimeMillis();
		double averageRPrecision = 0;
		double averageMAPValue = 0;
		for (Integer queryNum : relevantDocs.keySet()) {
			Set<RelevanceScore> relevanceScores = new TreeSet<RelevanceScore>(Collections.reverseOrder());
			String currentQuery = queries.get(queryNum-1).getQuery();
			for (Integer docID : documents.keySet()) {
				RelevanceScore rs = new RelevanceScore(docID, score(documents.get(docID),queryVector(currentQuery)));	
				if (rs.getScore() > 0) {
					relevanceScores.add(rs);
				}
			}
			Set<Integer> relDocs = relevantDocs.get(queryNum);
			int r = relDocs.size();
			double j = 1;
			double relevantDocNum = 0;
			double mapValue = 0;
			if (!relevanceScores.isEmpty()) {
				for (RelevanceScore rs : relevanceScores) {
					if (j > topK) {
						break;
					}
					int docID = rs.getID();
					if (relDocs.contains(docID)) {
						if (j <= r) {
							relevantDocNum++;
						}
						mapValue += relevantDocNum / j;
					}
					j ++;
				}
				double rPrecision = relevantDocNum / r;
				rPrecisionSet.add(rPrecision);
				averageRPrecision += rPrecision;
				System.out.println("\nQuery " + queryNum + "\n\tDocuments we got: " + relevantDocNum +" / Actual Documents that exist: " + r);
				System.out.println("\tR-Precision = " + rPrecision);
				double totalMAP = mapValue / r;
				mapSet.add(totalMAP);
				averageMAPValue += totalMAP;
				System.out.println("\tMAP Value = " + totalMAP);
				writer.write("\nQuery " + queryNum + "\n\tRelevant documents retrieved: " + relevantDocNum +
						" / Total relevant documents in query: " + r + "\n");
				writer.write("\tR-Precision = " + rPrecision + "\n");
				writer.write("\tMAP Value = " + totalMAP + "\n");
			}
			else {
				System.out.println("No match for query " + queryNum + ": " + currentQuery);
			}
		}
		in.close();
		System.out.println("Average R-Precision over " + queries.size() + " queries = " + (averageRPrecision/rPrecisionSet.size()));
		System.out.println("Average MAP Value over "+ queries.size() + " queries = " + (averageMAPValue/mapSet.size()));
		long queryEnd = System.currentTimeMillis() - queryStart;
		
		System.out.println("\nTime required to create MAP and R-Precision values: " + (queryEnd)/1000 + " seconds");
		writer.write("Average R-Precision over " + queries.size() + " queries = " + (averageRPrecision/rPrecisionSet.size()) + "\n");
		writer.write("Average MAP Value over "+ queries.size() + " queries = " + (averageMAPValue/mapSet.size()) + "\n");
		writer.write("\nTime required to create MAP and R-Precision values: " + (queryEnd)/1000 + " seconds\n");
		writer.close();
	}
}
