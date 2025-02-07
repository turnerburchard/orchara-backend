from cluster import Clusterer
from embed import Embedder
from pickle_helpers import load_from_pkl
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import to_tree

# Data loading and model initialization
FILENAME = '../final_project/Data/50k/data_50k.pkl'
LINKAGE_FILE = '../final_project/Data/50k/280/agg50k'
all_papers = load_from_pkl(FILENAME)
data = [paper.abstract_vector for paper in all_papers]
linkage_matrix = load_from_pkl(LINKAGE_FILE).linkage_matrix

clusterer = Clusterer(data)
transform = clusterer.reduce_dimensions(273)
embedder = Embedder()

def find_target_cluster(linkage_matrix, query_vector, target_size, data, transform_vector, method='euclidean'):
    """
    Traverse the hierarchical cluster tree to locate a cluster of approximate target size.
    """
    def compute_centroid(cluster_node):
        leaf_indices = cluster_node.pre_order()
        return np.mean(data[leaf_indices], axis=0)

    tree = to_tree(linkage_matrix, rd=False)
    current_node = tree
    query_vector = transform_vector(query_vector)

    while True:
        left, right = current_node.left, current_node.right
        if len(current_node.pre_order()) <= target_size:
            break

        left_distance = cdist([query_vector], [compute_centroid(left)], metric=method)[0][0]
        right_distance = cdist([query_vector], [compute_centroid(right)], metric=method)[0][0]
        current_node = left if left_distance < right_distance else right

    return current_node.pre_order()

def search_api(query: str, cluster_size: int):
    """
    Embed the query, locate the corresponding cluster, and return a list of paper metadata.
    """
    query_vector = embedder.embed_text(query)
    indices = find_target_cluster(linkage_matrix, query_vector, cluster_size, clusterer.data, transform)
    results = []
    for i in indices:
        paper = all_papers[i]
        paper_data = {'title': paper.title}
        if hasattr(paper, 'abstract'):
            paper_data['abstract'] = paper.abstract
        results.append(paper_data)
    return results

if __name__ == '__main__':
    # Standalone test
    test_query = "Searching research texts with hierarchical clustering"
    test_cluster_size = 5
    papers = search_api(test_query, test_cluster_size)
    for paper in papers:
        print(f"Title: {paper['title']}")
        print(paper['url'])
