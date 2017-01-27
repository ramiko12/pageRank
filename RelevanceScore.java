
public class RelevanceScore implements Comparable<RelevanceScore> {
	
	private int docID;
	private Double score;
	
	public RelevanceScore(int docID, double score) {
		this.docID = docID;
		this.score = score;
	}
	
	public int getID(){
		return docID;
	}
	
	public Double getScore() {
		return score;
	}
	
	public int compareTo(RelevanceScore arg0) {
		return score.compareTo(arg0.getScore());
	}
}
