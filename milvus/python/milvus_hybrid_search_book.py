from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("jeopardy_db")      # Get an existing collection.
collection.load()

data=[embed(text)]
search_param = {
  #"data": [[0.1, 0.2]],
  "anns_field": "embedding",
  "param": {"metric_type": "L2", "params": {"glucose": 10}, "offset": 0},
  "limit": 10,
  "expr": "word_count <= 11000",
  "output_fields":['Question']
}
res = collection.search(**search_param)

print(res)


# def search(text):
#   # Search parameters for the index
#   search_params = {
#     "metric_type": "L2"
#   }
#
#   results = j_milvus_collection.search(
#     data=[embed(text)],  # Embeded search value
#     anns_field="embedding",  # Search across embeddings
#     param=search_params,
#     limit=5,  # Limit to five results per search
#     output_fields=['Question']  # Include title field in result
#   )
#
#   ret = []
#   for hit in results[0]:
#     row = []
#     row.extend([hit.id, hit.score, hit.entity.get('Question')])  # Get the id, distance, and title for the results
#     ret.append(row)
#   return ret
#
#
# search_terms = ['glucose']
#
# for x in search_terms:
#   print('Search term:', x)
#   for result in search(x):
#     print(result)
#   print()
