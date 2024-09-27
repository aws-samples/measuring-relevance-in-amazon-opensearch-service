import math
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from matplotlib import pyplot as plt
import os

def download_s3_file(bucket_name, s3_file_key):
    """
    Downloads a file from an S3 bucket to the current directory using the same file name as in the S3 key.

    :param bucket_name: str, the name of the S3 bucket
    :param s3_file_key: str, the key (path) of the file in the S3 bucket
    """
    # Initialize an S3 client
    s3 = boto3.client('s3')

    # Extract the file name from the S3 key
    file_name = os.path.basename(s3_file_key)

    # Define the local file path (current directory + file name)
    local_file_path = os.path.join(os.getcwd(), file_name)

    try:
        # Download the file from the bucket
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File {s3_file_key} downloaded successfully to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")


def get_auth(region):
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region)

    return auth


def get_auth_serverless(region):
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, 'aoss')

    return auth


def create_client(host, username=None, password=None, port=443, verify_certs=True, serverless=False,
                  region='us-east-1'):
    if username and password is not None:
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=verify_certs,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection
        )
    else:
        if serverless:
            auth = get_auth_serverless(region)
        else:
            auth = get_auth(region)
        client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=verify_certs,
            connection_class=RequestsHttpConnection
        )
        print(client)


def create_amazon_opensearch_clients(serverless=False, domain_name='', username=None, password=None,
                                     region='us-east-1'):
    # Serverless
    if serverless:
        boto3_client_os = boto3.client('opensearchserverless')
        domain_host = boto3_client_os.batch_get_collection(names=['{}'.format(domain_name)])['collectionDetails'][0][
            'collectionEndpoint']
        domain_host = domain_host.replace("https://", "")
        print(domain_host)
        os_client = create_client(host="{}".format(domain_host), port=443, serverless=True, region=region)


    else:
        # Managed
        boto3_client_os = boto3.client('opensearch')
        domain_host = boto3_client_os.describe_domain(DomainName='{}'.format(domain_name))['DomainStatus']['Endpoint']
        os_client = create_client(host="{}".format(domain_host), port=443, serverless=False, username=username,
                                  password=password, region=region)

    return boto3_client_os, os_client


def list_of_docs(k, queries, os_client):
    """
    Generate a ranked list of k top documents in descending order of BM25 similarity.
    The output is a list of (query_id, document_id) pairs for each query in `queries`.

    If k=0 is given, retrieve 500 matching documents for each query.

    Arguments:
        k {int} -- Number of top documents to retrieve (0 for all documents).
        queries {list} -- A list of query dictionaries with "text" as the key.
        os_client {OpenSearchClient} -- An instance of OpenSearch or Elasticsearch client.

    Returns:
        list -- List of dictionaries where each contains a query_id and a list of document hits.
    """

    query_id_hits = []
    id_counter = 0

    for query in queries:
        id_counter += 1
        text = query['text']

        # Set the query size based on whether k is 0 (retrieve all)
        size = k if k > 0 else 500  # Set a reasonable upper limit for all documents

        # Construct search query
        body = {
            "size": size,
            "query": {
                "match": {
                    "text": text
                }
            }
        }

        # Send the query request to OpenSearch
        try:
            query_response = os_client.http.get(
                url='/cranfield/_search?filter_path=hits.hits._id',
                body=body
            )
        except Exception as e:
            print(f"Error querying OpenSearch for query_id {id_counter}: {e}")
            continue

        # Extract hits from response and store them with their respective query ID
        hits_list = [hit['_id'] for hit in query_response.get('hits', {}).get('hits', [])]

        query_id_hits.append({"query_id": id_counter, "hits": hits_list})

    return query_id_hits


def intersection(lst1, lst2):
    """To count number of common items between 2 lists

    Arguments:
        lst1 {list} -- list 1
        lst2 {list} -- list 2

    Returns:
        integer -- number of common items between list 1 & list 2
    """
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)


def calculate_recall(k, list_of_documents, relevant_docs):
    """Calculate recall at K for each query in the list_of_documents.

    Arguments:
        k {integer} -- number of top documents to be retrieved
        list_of_documents {list} -- list of dictionaries containing query_id and hits
        relevant_docs {dict} -- dictionary containing relevant documents for each query_id

    Returns:
        list -- list of recall values for each query
    """
    recall = []

    for query_id_hits in list_of_documents:
        docs_retrieved = query_id_hits["hits"][:k]  # Limit to top-k documents
        query_id = query_id_hits["query_id"]

        # Get the relevant documents for the query
        docs_relevant = relevant_docs.get(str(query_id), [])

        # Number of relevant documents retrieved
        a = len(set(docs_retrieved).intersection(docs_relevant))

        # Total number of relevant documents
        b = len(docs_relevant)

        # Handle case when there are no relevant documents
        r = a / b if b > 0 else 0
        recall.append(r)

    return recall


def calculate_precision(k, list_of_documents, relevant_docs):
    """Calculate precision at K for each query in the list_of_documents.

    Arguments:
        k {integer} -- number of top documents to be retrieved
        list_of_documents {list} -- list of dictionaries containing query_id and hits
        relevant_docs {dict} -- dictionary containing relevant documents for each query_id

    Returns:
        list -- list of precision values for each query
    """
    precision = []

    for query_id_hits in list_of_documents:
        docs_retrieved = query_id_hits["hits"][:k]  # Limit to top-k documents
        query_id = query_id_hits["query_id"]

        # Get the relevant documents for the query
        docs_relevant = relevant_docs.get(str(query_id), [])

        # Number of relevant documents retrieved
        a = len(set(docs_retrieved).intersection(docs_relevant))

        # Total number of documents retrieved (which is k)
        b = len(docs_retrieved)  # This should be min(k, len(hits))

        # Handle case when no documents are retrieved
        r = a / b if b > 0 else 0
        precision.append(r)

    return precision


def plot_recall_vs_k(max_k, list_of_documents, relevant_docs):
    """Plot recall vs K for the given list of documents and relevant docs."""
    k_values = list(range(1, max_k + 1))
    avg_recall_at_k = []

    # Calculate recall for each value of K
    for k in k_values:
        recall_values = calculate_recall(k, list_of_documents, relevant_docs)
        avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0
        avg_recall_at_k.append(avg_recall)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_recall_at_k, marker='o', linestyle='-', color='b', label='Recall')
    plt.title('Recall vs. K')
    plt.xlabel('K (Number of top elements)')
    plt.ylabel('Average Recall')
    plt.grid(True)
    plt.xticks(k_values)  # Ensure x-ticks align with K values
    plt.legend()
    plt.show()


def plot_precision_vs_k(max_k, list_of_documents, relevant_docs):
    """Plot precision vs K for the given list of documents and relevant docs."""
    k_values = list(range(1, max_k + 1))
    avg_precision_at_k = []

    # Calculate precision for each value of K
    for k in k_values:
        precision_values = calculate_precision(k, list_of_documents, relevant_docs)
        avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0
        avg_precision_at_k.append(avg_precision)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_precision_at_k, marker='o', linestyle='-', color='g', label='Precision')
    plt.title('Precision vs. K')
    plt.xlabel('K (Number of top elements)')
    plt.ylabel('Average Precision')
    plt.grid(True)
    plt.xticks(k_values)  # Ensure x-ticks align with K values
    plt.legend()
    plt.show()


def calculate_f1(precision, recall):
    """Calculate F1 score for each query."""
    f1 = []
    for p, r in zip(precision, recall):
        if p + r > 0:
            f1_score = 2 * (p * r) / (p + r)
        else:
            f1_score = 0
        f1.append(f1_score)
    return f1


def plot_f1_vs_k(max_k, list_of_documents, relevant_docs):
    """Plot F1 vs K for the given list of documents and relevant docs."""
    k_values = list(range(1, max_k + 1))
    avg_f1_at_k = []

    # Calculate F1 for each value of K
    for k in k_values:
        precision_values = calculate_precision(k, list_of_documents, relevant_docs)
        recall_values = calculate_recall(k, list_of_documents, relevant_docs)
        f1_values = calculate_f1(precision_values, recall_values)
        avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
        avg_f1_at_k.append(avg_f1)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_f1_at_k, marker='o', linestyle='-', color='r', label='F1 Score')
    plt.title('F1 Score vs. K')
    plt.xlabel('K (Number of top elements)')
    plt.ylabel('Average F1 Score')
    plt.grid(True)
    plt.xticks(k_values)  # Ensure x-ticks align with K values
    plt.legend()
    plt.show()


def plot_precision_recall_f1_vs_k(max_k, list_of_documents, relevant_docs):
    """Plot Precision, Recall, and F1 vs. K for the given list of documents and relevant docs."""
    k_values = list(range(1, max_k + 1))
    avg_precision_at_k = []
    avg_recall_at_k = []
    avg_f1_at_k = []

    # Calculate Precision, Recall, and F1 for each value of K
    for k in k_values:
        precision_values = calculate_precision(k, list_of_documents, relevant_docs)
        recall_values = calculate_recall(k, list_of_documents, relevant_docs)
        f1_values = calculate_f1(precision_values, recall_values)

        avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0
        avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0
        avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

        avg_precision_at_k.append(avg_precision)
        avg_recall_at_k.append(avg_recall)
        avg_f1_at_k.append(avg_f1)

    # Plot the results
    plt.figure(figsize=(10, 6))

    # Plot Precision
    plt.plot(k_values, avg_precision_at_k, marker='o', linestyle='-', color='b', label='Precision')

    # Plot Recall
    plt.plot(k_values, avg_recall_at_k, marker='s', linestyle='--', color='g', label='Recall')

    # Plot F1 Score
    plt.plot(k_values, avg_f1_at_k, marker='^', linestyle='-.', color='r', label='F1 Score')

    # Set plot labels and title
    plt.title('Precision, Recall, and F1 Score vs. K')
    plt.xlabel('K (Number of top elements)')
    plt.ylabel('Average Score')
    plt.grid(True)
    plt.xticks(k_values)  # Ensure x-ticks align with K values
    plt.legend()
    plt.show()


def calculate_precision_at_k(k, docs_retrieved, relevant_docs):
    """Calculate precision at a given cutoff k."""
    relevant_retrieved = set(docs_retrieved[:k]).intersection(relevant_docs)
    return len(relevant_retrieved) / k if k > 0 else 0


def calculate_average_precision(k, docs_retrieved, relevant_docs):
    """Calculate average precision (AP) for a single query up to the cutoff k."""
    ap = 0
    num_relevant = 0

    # Calculate precision at each relevant retrieved document
    for i in range(1, k + 1):
        if docs_retrieved[i - 1] in relevant_docs:
            num_relevant += 1
            ap += calculate_precision_at_k(i, docs_retrieved, relevant_docs)

    return ap / num_relevant if num_relevant > 0 else 0


def calculate_map_at_k(k, list_of_documents, relevant_docs):
    """Calculate MAP@K for all queries in the list_of_documents."""
    ap_scores = []

    for query_id_hits in list_of_documents:
        docs_retrieved = query_id_hits["hits"][:k]
        query_id = query_id_hits["query_id"]

        # Get the relevant documents for this query
        docs_relevant = relevant_docs.get(str(query_id), [])

        # Calculate average precision for this query
        ap = calculate_average_precision(k, docs_retrieved, docs_relevant)
        ap_scores.append(ap)

    # Return the mean of the average precision scores across all queries
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0


def plot_map_vs_k(max_k, list_of_documents, relevant_docs):
    """Plot MAP@K for different values of K."""
    k_values = list(range(1, max_k + 1))
    map_at_k = []

    # Calculate MAP for each value of K
    for k in k_values:
        map_score = calculate_map_at_k(k, list_of_documents, relevant_docs)
        map_at_k.append(map_score)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, map_at_k, marker='o', linestyle='-', color='purple', label='MAP@K')
    plt.title('MAP@K vs. K')
    plt.xlabel('K (Number of top elements)')
    plt.ylabel('Mean Average Precision')
    plt.grid(True)
    plt.xticks(k_values)
    plt.legend()
    plt.show()


def calculate_dcg(relevances, k=None):
    """Calculate the DCG for the top K elements from a list of relevances."""
    if k is None:  # If no value for K is provided, use the entire list
        k = len(relevances)

    dcg = 0
    for i, relevance in enumerate(relevances[:k]):
        # DCG formula: relevance / log2(position + 1)
        dcg += relevance / math.log2(i + 2)  # i+2 because index starts at 0
    return dcg


def get_hits_relevances(list_of_documents, query_id, qrels):
    """Get the list of relevances for hits in a specific query_id based on qrel_query.
       Assign relevance of 0 if a hit's relevance is not found or is -1 in qrel_query."""

    qrel_query = list(filter(lambda qrel: qrel['query_id'] == str(query_id), qrels))

    hits = next((doc['hits'] for doc in list_of_documents if doc['query_id'] == query_id), [])
    relevances = {entry['doc_id']: entry['relevance'] for entry in qrel_query if entry['query_id'] == str(query_id)}
    # Assign relevance of 0 if the document is not found or if relevance is -1
    hit_relevances = [relevances.get(hit_id, 0) if relevances.get(hit_id, 0) != -1 else 0 for hit_id in hits]
    return hit_relevances


def plot_ndcg_dcg_idcg(list_of_documents, query_id, qrels, max_k):
    """
    Plots the DCG, IDCG, and NDCG for the top K elements.

    Args:
        list_of_documents (list): The list of document hits with query IDs.
        query_id : The query ID to evaluate.
        qrels : The list of relevance judgments for the queries.
        max_k (int): The maximum number of top elements to consider.
    """

    qrel_query = list(filter(lambda qrel: qrel['query_id'] == str(query_id), qrels))

    # Get relevances
    hits_relevances = get_hits_relevances(list_of_documents, query_id, qrel_query)
    ideal_relevances = sorted(hits_relevances, reverse=True)

    # Calculate DCG, IDCG, and NDCG for K values from 1 to max_k
    k_values = list(range(1, min(max_k, len(hits_relevances)) + 1))
    dcg_values = [calculate_dcg(hits_relevances, k) for k in k_values]
    idcg_values = [calculate_dcg(ideal_relevances, k) for k in k_values]
    ndcg_values = [(dcg / idcg) if idcg > 0 else 0 for dcg, idcg in zip(dcg_values, idcg_values)]

    # Plot DCG and IDCG
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, dcg_values, marker='o', linestyle='-', color='b', label='DCG')
    plt.plot(k_values, idcg_values, marker='x', linestyle='--', color='g', label='IDCG')

    plt.title(f"DCG vs. IDCG for query_id = {query_id}")
    plt.xlabel("K (Number of top elements)")
    plt.ylabel("DCG / IDCG")
    plt.xticks(k_values)  # Set x-ticks for each K value
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot NDCG in a separate graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, ndcg_values, marker='s', linestyle='-', color='r', label='NDCG')

    plt.title(f"NDCG vs. K for query_id = {query_id}")
    plt.xlabel("K (Number of top elements)")
    plt.ylabel("NDCG")
    plt.xticks(k_values)  # Set x-ticks for each K value
    plt.grid(True)
    plt.legend()
    plt.show()
