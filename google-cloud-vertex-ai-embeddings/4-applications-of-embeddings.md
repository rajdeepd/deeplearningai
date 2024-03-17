---
layout: default
title: 4. Application of Embeddings
nav_order: 5
description: ".."
has_children: false
parent:  Google Cloud Vertex AI Embeddings
---

## Lesson 4: Applications of Embeddings

.



#### Project environment setup

After gaining some understanding of embeddings, let's explore their practical uses. As in previous tutorials, we'll begin by setting up our credentials and authenticating. Additionally, we'll choose the region where our service will operate. Next, we'll import the Vertex AI Python SDK to initialize our working environment. With these preliminary steps completed, we're positioned to start importing our dataset

- Load credentials and relevant Python Libraries as shown below


```python
from utils import authenticate
credentials, PROJECT_ID = authenticate()
```


```python
REGION = 'us-central1'
```


```python
import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
```      

#### Load Stack Overflow questions and answers from BigQuery

For this demonstration, we'll utilize the Stack Overflow question and answer dataset, accessible through BigQuery, Google's fully managed, serverless data warehouse. Our next step involves importing the BigQuery Python client, followed by creating a function capable of executing SQL queries within BigQuery. BigQuery is Google Cloud's serverless data warehouse. This function will accept an SQL query as input and return the query results in a pandas dataframe, facilitating various applications within our notebook. While the inner workings of this BigQuery function may not be immediately clear, its purpose is to fetch our dataset for this tutorial without needing to delve into technical specifics.

  


```python
from google.cloud import bigquery
import pandas as pd
```


```python
def run_bq_query(sql):

    # Create BQ client
    bq_client = bigquery.Client(project = PROJECT_ID, 
                                credentials = credentials)

    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run=True, 
                                         use_query_cache=False)
    bq_client.query(sql, job_config=job_config)

    # If dry run succeeds without errors, proceed to run query
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql, 
                                    job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df
```


To manage the dataset's large size and avoid memory issues, we'll focus on a smaller subset of Stack Overflow posts. This will involve creating a list of specific programming language tags to query. Our approach starts with an empty dataframe, to which we'll add data by executing SQL queries for each language tag in our list. The SQL query, designed to fetch the first 500 posts associated with each specified language tag from Stack Overflow, simplifies our data collection process without the need for extensive SQL knowledge.

```python
# define list of programming language tags we want to query

language_list = ["python", "html", "r", "css"]
```


```python
so_df = pd.DataFrame()

for language in language_list:
    
    print(f"generating {language} dataframe")
    
    query = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE 
        q.accepted_answer_id IS NOT NULL AND 
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT 
        500
    """

    
    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df], 
                      ignore_index = True) 
```


Following the data collection, the next step is to merge all the gathered results into a unified dataframe for use within this notebook. 

```python
so_df
```

After executing the data fetching cell, which may take a moment, data for each of the four specified programming languages is retrieved. 





Should there be any issues with executing the BigQuery code, or if there's a preference to avoid using BigQuery altogether, an alternative step is provided. This step involves loading the data from a CSV file, offering a straightforward solution for those encountering errors or preferring not to use BigQuery.

Upon successfully compiling our dataset, we proceed to inspect its composition. The resulting dataframe contains 2,000 Stack Overflow posts, evenly distributed across the queried programming languages, totaling 500 posts per language. It's structured into three columns: the first combines the title and question of each Stack Overflow post; the second column contains the community's accepted answer for each post; and the third specifies the programming language category.

#### Generate text embeddings

With our dataset prepared, we're set to move forward with embedding and further applications. This involves loading the text embedding model used in previous tutorials, specifically the text embedding gecko model. Given the substantial volume of data, additional steps are necessary for efficient processing. To this end, we'll introduce helper functions designed to batch the data before sending it to the embedding API. The first of these functions, named generate batches, segments our dataset into manageable batches of five. This batching is essential, as per the API documentation, which states the API's limitation to processing no more than five text entries at a time.

``python
from vertexai.language_models import TextEmbeddingModel
```


```python
model = TextEmbeddingModel.from_pretrained(
    "textembedding-gecko@001")
```


```python
import time
import numpy as np
```


```python
# Generator function to yield batches of sentences

def generate_batches(sentences, batch_size = 5):
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]
```


```python
so_questions = so_df[0:200].input_text.tolist() 
batches = generate_batches(sentences = so_questions)
```


```python
batch = next(batches)
len(batch)
```
You will get 5 batches

```
5
```


#### Get embeddings on a batch of data


To process our data for text embeddings, we'll divide it into batches of five due to the API's limitation of handling only five text instances per request. We'll demonstrate this process by applying the generateBatches function to the first 200 rows of our dataset, illustrating how it organizes the data into manageable groups of five.




```python
def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]
```


```python
batch_embeddings = encode_texts_to_embeddings(batch)
```


```python
f"{len(batch_embeddings)} embeddings of size \
{len(batch_embeddings)}

```

Upon using generateBatches on a portion of our dataset, it effectively creates batches sized perfectly for our embedding process. This method proves essential for efficiently embedding the entire dataset. Next, we introduce the encodeTextToEmbeddings function, which serves as a convenient wrapper for the getEmbeddings function previously utilized. By testing this function with a batch of five sentences, we observe that it returns five embeddings, each with a dimensionality of 768. This dimension count, 768, is characteristic of the text embedding model we're employing, indicating each text input is transformed into a vector of 768 dimensions.




#### Code for getting data on an entire data set

Furthermore, it's crucial to consider the rate limits imposed by most Google Cloud services, which restrict the number of requests per minute. To navigate these constraints, we've prepared a specialized function, encodeTextToEmbeddingBatched, designed to both batch the data and adhere to rate limits. While we won't process all 2,000 dataset rows in this session to stay within online classroom guidelines, this function is available for your use in personal projects, ensuring efficient data management and compliance with service limitations.

- Most API services have rate limits, so we've provided a helper function (in utils.py) that you could use to wait in-between API calls.
- If the code was not designed to wait in-between API calls, you may not receive embeddings for all batches of text.
- This particular service can handle 20 calls per minute.  In calls per second, that's 20 calls divided by 60 seconds, or `20/60`.

```Python
from utils import encode_text_to_embedding_batched

so_questions = so_df.input_text.tolist()
question_embeddings = encode_text_to_embedding_batched(
                            sentences=so_questions,
                            api_calls_per_second = 20/60, 
                            batch_size = 5)
```




The encodeTextToEmbeddingsBatched function is designed to streamline the process of embedding textual data by handling both the division of data into manageable batches and adhering to rate limits effectively. After organizing your data into batches of five to comply with the request limit per batch, this function proceeds to generate embeddings for each batch while also managing the frequency of requests to stay within the prescribed rate limits of the service.



#### Load the data from file

In order to handle limits of this classroom environment, we're not going to run this code to embed all of the data. But you can adapt this code for your own projects and datasets.

Next, we'll proceed to integrate these pre-generated embeddings. To ensure these embeddings accurately correspond to the Stack Overflow questions we've gathered, we'll reload the questions from a new CSV file, maintaining consistency with our BigQuery data.

Upon reloading, we use a pickle file containing all the necessary embeddings for further processing. Upon examining this embeddings vector, we observe that our dataset now comprises 2,000 Stack Overflow posts, each represented by a 768-dimensional vector, ready for application in various analytical tasks.


- We'll load the stack overflow questions, answers, and category labels (Python, HTML, R, CSS) from a .csv file.
- We'll load the embeddings of the questions (which we've precomputed with batched calls to `model.get_embeddings()`), from a pickle file.


```python
so_df = pd.read_csv('so_database_app.csv')
so_df.head()
```


```python
import pickle
```


```python
with open('question_embeddings_app.pkl', 'rb') as file:
    question_embeddings = pickle.load(file)
```


```python
print("Shape: " + str(question_embeddings.shape))
print(question_embeddings)
```

#### Cluster the embeddings of the Stack Overflow questions

One such application we'll explore is data clustering. We'll employ the K-Means clustering algorithm to group these posts based on their embeddings. To facilitate this, we import the K-Means and PCA (Principal Component Analysis) functions from Scikit-Learn, the latter aiding in the two-dimensional visualization of our clusters for ease of interpretation. For this demonstration, we'll focus on visualizing only the first 1,000 entries from our dataset, specifically those tagged with Python or HTML.

We'll initiate our clustering effort by determining the number of clusters, setting it to two, and then applying the K-Means model to the selected portion of our dataset. Following the model's application, we'll extract cluster labels to identify which cluster each post belongs to. Given the high dimensionality of our data, we resort to PCA for reducing it to two dimensions, making it visually interpretable.

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```


```python
clustering_dataset = question_embeddings[:1000]
```


```python
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0, 
                n_init = 'auto').fit(clustering_dataset)
```


```python
kmeans_labels = kmeans.labels_
```


```python
PCA_model = PCA(n_components=2)
PCA_model.fit(clustering_dataset)
new_values = PCA_model.transform(clustering_dataset)
```


```python
import matplotlib.pyplot as plt
import mplcursors
%matplotlib ipympl
```


```python
from utils import clusters_2D
clusters_2D(x_values = new_values[:,0], y_values = new_values[:,1], 
            labels = so_df[:1000], kmeans_labels = kmeans_labels)
```

- Clustering is able to identify two distinct clusters of HTML or Python related questions, without being given the category labels (HTML or Python).


After applying PCA to condense our dataset for two-dimensional visualization, we employ matplotlib to graph and examine the clusters formed. The visualization reveals two distinct clusters: one comprising HTML-tagged questions, marked by red circles, and another of Python-tagged questions, efficiently categorized by the clustering model based solely on the embeddings, without prior knowledge of the labels. These labels are subsequently added for ease of visualization, illustrating the embeddings' capacity to segregate data into clear, thematic groups.

## Anomaly / Outlier detection

Further exploring the utility of embeddings, we venture into anomaly or outlier detection. Utilizing the isolation forest algorithm from scikit-learn, known for its ability to identify anomalies in an unsupervised manner, we introduce a non-programming related query to our dataset—a question about baking. This distinct query, once embedded and added to our dataset, increases our dataset size to 2,001 entries, now including a question that diverges significantly from the programming context of the existing dataset.

Before applying the isolation forest model, we augment our dataframe with this baking query, despite it lacking an accepted answer, and label it as 'baking' for visualization purposes. Following the model's application, it classifies this baking question as an outlier, underlining its dissimilarity from the programming-focused questions. Intriguingly, some programming-related questions, specifically those about the R language, are also flagged as outliers, suggesting a closer examination of these entries might reveal why they're perceived as anomalies within the dataset.

- We can add an anomalous piece of text and check if the outlier (anomaly) detection algorithm (Isolation Forest) can identify it as an outlier (anomaly), based on its embedding.


```python
from sklearn.ensemble import IsolationForest
```


```python
input_text = """I am making cookies but don't 
                remember the correct ingredient proportions. 
                I have been unable to find 
                anything on the web."""
```


```python
emb = model.get_embeddings([input_text])[0].values
```


```python
embeddings_l = question_embeddings.tolist()
embeddings_l.append(emb)
```


```python
embeddings_array = np.array(embeddings_l)
```


```python
print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)
```


```python
# Add the outlier text to the end of the stack overflow dataframe
so_df = pd.read_csv('so_database_app.csv')
new_row = pd.Series([input_text, None, "baking"], 
                    index=so_df.columns)
so_df.loc[len(so_df)+1] = new_row
so_df.tail()
```


#### Use Isolation Forest to identify potential outliers

- `IsolationForest` classifier will predict `-1` for potential outliers, and `1` for non-outliers.
- You can inspect the rows that were predicted to be potential outliers and verify that the question about baking is predicted to be an outlier.


```python
clf = IsolationForest(contamination=0.005, 
                      random_state = 2) 
```


```python
preds = clf.fit_predict(embeddings_array)

print(f"{len(preds)} predictions. Set of possible values: {set(preds)}")
```


```python
so_df.loc[preds == -1]
```
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-10_at_3.08.34 PM.png" width="90%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-10_at_5.33.45 PM.png" width="90%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-10_at_5.33.50 PM.png" width="90%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-10_at_5.34.01 PM.png" width="90%"/>

#### Remove the outlier about baking


```python
so_df = so_df.drop(so_df.index[-1])
```


```python
so_df
```


Having identified certain posts as outliers, potentially due to mislabeling or other reasons unrelated to the R programming language, we proceed by removing the baking question from our dataset. This action restores our Stack Overflow dataset to its original state, focusing solely on programming queries and maintaining 2,000 entries.

## Classification

The final segment of this lesson explores leveraging embedding vectors as input features for supervised learning models. Embeddings transform textual input into structured vectors, suitable for various supervised learning classifiers. For the purposes of this tutorial, we choose to utilize a random forest classifier from scikit-learn, though this choice can be substituted with any preferred classifier available within the library.

We consider various possibilities for framing a classification challenge with this dataset, such as predicting mentions of specific technologies like pandas or estimating the popularity of posts based on upvote counts. However, our focus will be on classifying the programming language category of each post. To accomplish this, we import additional utilities from scikit-learn, including the accuracy score metric and the train-test split function, to prepare our dataset for the classification task. We then organize our embeddings into an array 
�
X for our feature set and extract the programming language categories as our labels, represented by 
�
Y.

The next step involves randomizing the dataset and dividing it into training and testing subsets to validate the effectiveness of our classification model accurately. This process ensures a robust evaluation of the model's ability to generalize and predict the category of new, unseen Stack Overflow posts based on their embeddings.


- Train a random forest model to classify the category of a Stack Overflow question (as either Python, R, HTML or CSS).


```python
from sklearn.ensemble import RandomForestClassifier
```

Utilizing the train test split function from scikit-learn, we organize our dataset into a training set and a testing set, with the testing set comprising 20% of the original dataset. This preparation sets the stage for applying a random forest classifier to our data. We choose to configure our classifier with 200 estimators, although this parameter can be adjusted according to preference. With the classifier ready, we proceed to train it using our embeddings as input features and their corresponding programming languages as labels.

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```

```python
# re-load the dataset from file
so_df = pd.read_csv('so_database_app.csv')
X = question_embeddings
X.shape
```
`X.shape` will be `(2000, 768)`

```python
y = so_df['category'].values
y.shape
```
Shape of y is `(2000,)` which you can check in the output console

```python
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 2)
```


```python
clf = RandomForestClassifier(n_estimators=200)
```


```python
clf.fit(X_train, y_train)
```

After training, we deploy the model to predict the categories of unseen test data, based solely on the embeddings of Stack Overflow posts. The performance of our model is evaluated using the accuracy score metric, yielding a respectable 70% accuracy. This demonstrates the effectiveness of embeddings in various applications, such as clustering for similarity analysis, classification to predict categories, and outlier detection to identify anomalous data points.


#### You can check the predictions on a few questions from the test set


```python
y_pred = clf.predict(X_test)
```


```python
accuracy = accuracy_score(y_test, y_pred) # compute accuracy
print("Accuracy:", accuracy)
```
Accuracy of the model came to `0.6875`.

#### Try out the classifier on some questions


```python
# choose a number between 0 and 1999
i = 2
label = so_df.loc[i,'category']
question = so_df.loc[i,'input_text']

# get the embedding of this question and predict its category
question_embedding = model.get_embeddings([question])[0].values
pred = clf.predict([question_embedding])

print(f"For question {i}, the prediction is `{pred[0]}`")
print(f"The actual label is `{label}`")
print("The question text is:")
print("-"*50)
print(question)
```
Output listing for the random question


```
For question 2, the prediction is `python`
The actual label is `python`
The question text is:
--------------------------------------------------
How do we test a specific method written in a list of files for functional testing in python<p>The project has so many modules. There are functional test cases being written for almost every api written like for GET requests, POST requests and PUT requests. To test an individual file we use the syntact pytest tests/file_name.py
but I want to test a specific method in that file. Is there any way to test it like that??</p>
```


In our next session, we'll shift focus from embeddings to explore the realm of text generation, providing a broader perspective on the applications of machine learning in processing and generating textual content. Join us as we continue our exploration into machine learning techniques.