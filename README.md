### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 22-04-2025
### AIM: 
To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import string
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')

###### Sample documents stored in a dictionary
    documents = 
    {
        "doc1": "The cat sat on the mat",
        "doc2": "The dog sat on the log",
        "doc3": "The cat lay on the rug"
    }

###### Preprocessing function to tokenize and remove stopwords/punctuation
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
        return " ".join(tokens)

###### Preprocess documents and store them in a dictionary
    preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

###### Compute Term frequency
    tf_vectorizer=CountVectorizer()
    tf_matrix = tf_vectorizer.fit_transform(preprocessed_docs.values()).toarray()

###### Construct TF-IDF matrix manually
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values()).toarray()

###### Convert into Pandas DataFrame for better readability
    terms = tfidf_vectorizer.get_feature_names_out()
    df_tf = pd.DataFrame(tf_matrix, index=documents.keys(), columns=tf_vectorizer.get_feature_names_out())
    df_tfidf = pd.DataFrame(tfidf_matrix, index=documents.keys(), columns=terms)

###### Calculate cosine similarity between query and documents
    def compute_cosine_similarity(query, tfidf_matrix, tfidf_vectorizer):
        query_processed = preprocess_text(query)
        query_vector = tfidf_vectorizer.transform([query_processed]).toarray()[0]

        cosine_scores = []
        for i, doc_vector in enumerate(tfidf_matrix):
            dot_product = np.dot(query_vector, doc_vector)
            magnitude_query = np.linalg.norm(query_vector)
            magnitude_doc = np.linalg.norm(doc_vector)
            cosine_similarity = dot_product / (magnitude_query * magnitude_doc) if magnitude_query and magnitude_doc else 0
            cosine_scores.append((list(documents.keys())[i], documents[list(documents.keys())[i]], cosine_similarity))

        return sorted(cosine_scores, key=lambda x: x[2], reverse=True)

###### Get input from user
    query = input("Enter your query: ")

###### Perform search
    search_results = compute_cosine_similarity(query, tfidf_matrix, tfidf_vectorizer)

###### Display term frequency matrix
    print("\nTerm Frequency Matrix:")
    print(df_tf)

###### Display TF-IDF matrix
    print("\nTF-IDF Matrix:")
    print(df_tfidf)

###### Display search results
    print("\nQuery:", query)
    for i, result in enumerate(search_results, start=1):
        print(f"\nRank: {i}")
        print("Document ID:", result[0])
        print("Document:", result[1])
        print("Similarity Score:", result[2])
        print("----------------------")

###### Get the highest rank cosine score
    highest_rank_score = max(result[2] for result in search_results)
    print("The highest rank cosine score is:", highest_rank_score)

### Output:

![query](/q.png)
<br>

![output](/image.png)
<br>

### Result:
Thus, Information Retrival is successfully implemented using vector space model in python.