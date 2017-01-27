
public class Query
{

	private String query;
	private int queryId;
	
	//Map<Integer, Query> querysres = new TreeMap<Integer, Query>();

	public Query()
	{
		
	}

	public void setID(int id)
	{
		this.queryId=id;
	}
	public void setQuery(String query) {
		this.query = query;
	}
	public String getQuery(){
		return this.query;
	}


	


}