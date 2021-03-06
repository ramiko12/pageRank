
import java.util.*;


public class Document {

	private int docID;
	private String title;
	private String content;
	private String author;
	private String publicationDate;
	private Set<Integer> citations;
	private Map<String, Integer> termFrequencies;
	private Map<String, Double> tfidfWeights;
	private double pageRank;

	

	public Document()
	{

	}

	public Document(int docID) {
		this.docID = docID;
	}

	public int getID() {
		return docID;
	}

	public void setID(int docID) {
		this.docID = docID;
	}

	public String getTitle() {
		return title;
	}

	public void setTitle(String title) {
		this.title = title;
	}

	public String getAbstract() {
		if (content != null) {
			return content;
		}
		else {
			return "";
		}
	}

	public void setAbstract(String abs) {
		content = abs;
	}
	public void setDate(String date) {
		this.publicationDate=date;
	}

	public void setAuthor(String author) {
		this.author = author;
	}

	public String getAuthor() {
		if (author != null) {
			return author;
		}
		else {
			return "";
		}
	}
	
	public Integer getFrequency(String term) {
		if (termFrequencies != null && termFrequencies.containsKey(term)) {
			return termFrequencies.get(term);
		}
		else {
			return 0;
		}
	}
	
	public void setFrequency(String term, Integer frequency) {
		if (termFrequencies == null) {
			termFrequencies = new HashMap<String,Integer>();
		}
		termFrequencies.put(term, frequency);
	}
	
	public Set<String> getTerms() {
		if (termFrequencies != null) {
		return termFrequencies.keySet();
		}
		else {
			return new HashSet<String>();
		}
	}
	public Double getWeight(String term) {
		if (tfidfWeights != null && tfidfWeights.containsKey(term)) {
			return tfidfWeights.get(term);
		}
		else {
			return 0.0;
		}
	}
	
	public void setWeight(String term, Double weight) {
		if (tfidfWeights == null) {
			tfidfWeights = new HashMap<String,Double>();
		}
		tfidfWeights.put(term, weight);
	}
	
	public void setCitation(Set<Integer> aCitation)
	{
		this.citations=aCitation;
	}

	public void addCitation(int citation)
	{
		citations.add(citation);
	}

	public Set<Integer> getCitation()
	{

		return citations;
	}

	public double getPageRank(){
		return pageRank;
	}

	public void setPageRank(double rank)
	{
		this.pageRank=rank;
	}
	
	
}
