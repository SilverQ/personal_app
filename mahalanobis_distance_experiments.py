import os
import gc
import math  # Needed to check for NaN
import json
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from scipy.sparse import csr_matrix, dok_matrix
# from streamlit import cache
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# File to save results
RESULTS_FILE = "hyperparameter_tuning_results.json"

# Initialize lists of hyperparameters to be tuned
pca_text_components = [50, 100, 200]
pca_cpc_components = [10, 30, 50]
threshold_values = [90, 95, 97, 99]
vectorizer_types = ['tfidf', 'count']
scaling_options = [True, False]
ngram_ranges = [(1, 1), (1, 2), (1, 3)]
max_features_list = [10000, 20000, 30000]

# Load stopwords from stopwords-ko.json
with open('stopwords-ko.json', 'r', encoding='utf-8') as file:
    korean_stopwords = json.load(file)


def read_column_names(fn):
    with open(fn, encoding='utf-8') as json_f:
        json_line = json_f.readline()
    temp = json.loads(json_line)
    return temp.keys()


def get_maingr(cpc_unsplitted, main_only=1):
    # main_only=0 : 전체 CPC 메인그룹 반환, main_only=1 : 메인트렁크만 반환

    def normalize_cpc(cpc_code):
        # Assuming main group is the first 4 characters, adjust as necessary
        return cpc_code.split('/')[0].strip()

    # Return empty list if the input is not a valid string
    if not isinstance(cpc_unsplitted, str) or pd.isna(cpc_unsplitted):
        return []

    # Split the CPC codes by '|', normalize to the main group, and remove duplicates
    processed_maingr_list = set()

    cpc_codes = cpc_unsplitted.split('|')
    maingr_list = [normalize_cpc(cpc) for cpc in cpc_codes if
                   isinstance(cpc, str) and cpc.strip()]  # Ensure it's a non-empty, valid string

    if main_only == 1:
        maingr_list = [item for item in maingr_list if len(item) < 8 and item[0] != 'Y']

    processed_maingr_list.update(maingr_list)  # Remove duplicates
    processed_maingr_list = list(processed_maingr_list)

    return processed_maingr_list


def get_hier(cpc):
    tmp = set()
    if isinstance(cpc, list) == False:
        cpc = [cpc]
    for item in cpc:
        if isinstance(item, str):
            tmp.update([item[:4], item[:3], item[0]])
    #     tmp = list(set(tmp))
    tmp.update(cpc)
    tmp = list(tmp)
    return tmp


def make_cpc_vector(train_data, test_data, pca_cpc):
    # Combine the 'cpc_h' columns from both train and test data
    combined_cpc_h = pd.concat([train_data, test_data], axis=0)
    # Convert any NaN or float entries to empty lists
    combined_cpc_h = [x if isinstance(x, list) else [] for x in combined_cpc_h]

    # MultiLabelBinarizer for CPC
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(combined_cpc_h)
    # print('num_cpcs: ', len(mlb.classes_))
    train_cpc_vectors = mlb.transform(train_data)
    test_cpc_vectors = mlb.transform(test_data)

    # Combine sparse matrices if needed
    train_cpc_vectors = csr_matrix(train_cpc_vectors)
    test_cpc_vectors = csr_matrix(test_cpc_vectors)

    # Perform PCA for CPC vectors
    # pca2 = PCA(n_components=pca_cpc)
    pca2 = TruncatedSVD(n_components=pca_cpc)  # TruncatedSVD instead of PCA
    train_cpc_vectors_r = pca2.fit_transform(train_cpc_vectors)
    test_cpc_vectors_r = pca2.transform(test_cpc_vectors)
    return  train_cpc_vectors_r, test_cpc_vectors_r


def make_text_vector(train_data, test_data, vectorizer_type, max_features, ngram_range, pca_text):
    # Apply vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=korean_stopwords, min_df=2,
                                     ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(max_features=max_features, stop_words=korean_stopwords, min_df=2,
                                     ngram_range=ngram_range)

    # Transform the text data
    train_vectors = vectorizer.fit_transform(train_data).astype(np.float32)
    test_vectors = vectorizer.transform(test_data).astype(np.float32)

    try:
        # Perform PCA for text vectors
        pca1 = PCA(n_components=pca_text)
        train_vectors_r = pca1.fit_transform(train_vectors)
        test_vectors_r = pca1.transform(test_vectors)
    except Exception as e:
        # Perform TruncatedSVD for text vectors (instead of PCA)
        svd1 = TruncatedSVD(n_components=pca_text)
        train_vectors_r = svd1.fit_transform(train_vectors)
        test_vectors_r = svd1.transform(test_vectors)
    return train_vectors_r, test_vectors_r

uploaded_file = r'data\ksic01.json'
# uploaded_file = 'ksic02.json'
if uploaded_file.endswith(".json"):
    col_name = read_column_names(uploaded_file)

    temp = []
    error = []
    # uploaded_file.seek(0)  # Reset file pointer after reading column names
    with open(uploaded_file, encoding='utf-8') as json_f:
        for i, line in enumerate(json_f):
            try:
                temp.append(json.loads(line.replace('\\\\"', '\\"')))
            #             temp.append(json.loads(line.decode('utf-8').replace('\\\\"', '\\"')))
            except Exception as e:
                error.append([e, line])
        train_data = pd.DataFrame(data=temp, columns=col_name)
else:
    # Load Excel data
    train_data = pd.read_excel(uploaded_file)

train_data['text'] = train_data['title'] + ', ' + train_data['ab'] + ', ' + train_data['cl']
train_data.dropna(subset='text', inplace=True)
train_data.dropna(subset='cpc', inplace=True)
train_data['maingr'] = train_data['cpc'].apply(get_maingr)
train_data['cpc_h'] = train_data['maingr'].apply(get_hier)
train_data.dropna(subset='cpc_h', inplace=True)
train_data['Mahalanobis_Distance'] = np.nan

test_file = 'ksic_test.xlsx'
test_data = pd.read_excel(test_file)
test_data['text'] = test_data['title'] + ', ' + test_data['ab'] + ', ' + test_data['cl']
test_data.dropna(subset='text', inplace=True)
test_data.dropna(subset='cpc', inplace=True)
test_data['maingr'] = test_data['cpc'].apply(get_maingr)
test_data['cpc_h'] = test_data['maingr'].apply(get_hier)
test_data.dropna(subset='cpc_h', inplace=True)
test_data['Mahalanobis_Distance'] = np.nan


# Helper function to convert numpy and pandas data types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()  # Convert pandas DataFrame to dictionary
    elif isinstance(obj, pd.Series):
        return obj.tolist()  # Convert pandas Series to list
    else:
        return obj


saved_results = {}


# Function to run the process for a combination of hyperparameters
def run_batch_process(pca_text, pca_cpc, threshold, vectorizer_type, scaling, ngram_range, max_features):
    # Create a unique key for this parameter combination
    param_key = f"{pca_text}_{pca_cpc}_{threshold}_{vectorizer_type}_{scaling}_{ngram_range}_{max_features}"
    print('processing: ', param_key)
    # Check if results already exist for this combination
    if param_key in saved_results:
        print(f"Loading results from saved data for parameters: {param_key}")
        return saved_results[param_key]

    # # Apply vectorizer
    # if vectorizer_type == 'tfidf':
    #     vectorizer = TfidfVectorizer(max_features=max_features, stop_words=korean_stopwords, min_df=2,
    #                                  ngram_range=ngram_range)
    # else:
    #     vectorizer = CountVectorizer(max_features=max_features, stop_words=korean_stopwords, min_df=2,
    #                                  ngram_range=ngram_range)
    #
    # # Transform the text data
    # train_vectors = vectorizer.fit_transform(train_data['text']).astype(np.float32)
    # test_vectors = vectorizer.transform(test_data['text']).astype(np.float32)
    #
    # try:
    #     # Perform PCA for text vectors
    #     pca1 = PCA(n_components=pca_text)
    #     train_vectors_r = pca1.fit_transform(train_vectors)
    #     test_vectors_r = pca1.transform(test_vectors)
    # except Exception as e:
    #     # Perform TruncatedSVD for text vectors (instead of PCA)
    #     svd1 = TruncatedSVD(n_components=pca_text)
    #     train_vectors_r = svd1.fit_transform(train_vectors)
    #     test_vectors_r = svd1.transform(test_vectors)

    train_cpc_vectors_r, test_cpc_vectors_r = make_cpc_vector(train_data['cpc_h'], test_data['cpc_h'], pca_cpc)
    train_vectors_r, test_vectors_r = make_text_vector(train_data['text'], test_data['text'],
                                                       vectorizer_type, max_features, ngram_range, pca_text)
    #
    # # Combine the 'cpc_h' columns from both train and test data
    # combined_cpc_h = pd.concat([train_data['cpc_h'], test_data['cpc_h']], axis=0)
    # # Convert any NaN or float entries to empty lists
    # combined_cpc_h = [x if isinstance(x, list) else [] for x in combined_cpc_h]
    #
    # # MultiLabelBinarizer for CPC
    # mlb = MultiLabelBinarizer(sparse_output=True)
    # mlb.fit(combined_cpc_h)
    # print('num_cpcs: ', len(mlb.classes_))
    # train_cpc_vectors = mlb.transform(train_data['cpc_h'])
    # test_cpc_vectors = mlb.transform(test_data['cpc_h'])
    #
    # # Combine sparse matrices if needed
    # train_cpc_vectors = csr_matrix(train_cpc_vectors)
    # test_cpc_vectors = csr_matrix(test_cpc_vectors)
    #
    # # Perform PCA for CPC vectors
    # # pca2 = PCA(n_components=pca_cpc)
    # pca2 = TruncatedSVD(n_components=pca_cpc)  # TruncatedSVD instead of PCA
    # train_cpc_vectors_r = pca2.fit_transform(train_cpc_vectors)
    # test_cpc_vectors_r = pca2.transform(test_cpc_vectors)

    # Combine CPC and text vectors
    train_vectors_c = np.hstack([train_vectors_r, train_cpc_vectors_r])
    test_vectors_c = np.hstack([test_vectors_r, test_cpc_vectors_r])

    # Apply scaling if specified
    if scaling:
        scaler = StandardScaler()
        train_vectors_c = scaler.fit_transform(train_vectors_c)
        test_vectors_c = scaler.transform(test_vectors_c)

    selected_ksic = '28114'
    # Run the evaluation function

    # Compute Mahalanobis distances for the test set
    cov = EmpiricalCovariance().fit(test_vectors_c)
    test_mahalanobis_distances = cov.mahalanobis(test_vectors_c)

    # Select the top 20 furthest points
    test_data['Mahalanobis_Distance'] = test_mahalanobis_distances
    top_20_outliers = test_data.nlargest(20, 'Mahalanobis_Distance')

    # Calculate the performance based on the 'label' column (0 = outlier, 1 = normal)
    num_correct_outliers = (top_20_outliers['label'] == 0).sum()
    performance = (num_correct_outliers / 20) * 100  # Percentage of correct outliers

    # Log and save the performance and hyperparameters
    result = {
        'pca_text': pca_text,
        'pca_cpc': pca_cpc,
        'threshold': threshold,
        'vectorizer_type': vectorizer_type,
        'scaling': scaling,
        'ngram_range': ngram_range,
        'max_features': max_features,
        'num_correct_outliers': num_correct_outliers,
        'performance_percentage': performance
    }

    #     saved_results[param_key] = result

    #     # Ensure all values are serializable before saving to JSON
    #     serializable_results = {k: convert_to_serializable(v) for k, v in saved_results.items()}

    #     # Save results to file
    #     with open(RESULTS_FILE, 'w') as f:
    #         json.dump(saved_results, f)

    return result


# Get all combinations of hyperparameters
param_combinations = itertools.product(pca_text_components, pca_cpc_components, threshold_values, vectorizer_types,
                                       scaling_options, ngram_ranges, max_features_list)

# hyperparameter_configs = [
#     # High dimensional text and CPC, tight threshold
#     {'pca_text': 150, 'pca_cpc': 50, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 1), 'max_features': 50000},
#     # Low dimensional text and CPC, loose threshold
#     {'pca_text': 50, 'pca_cpc': 10, 'threshold': 99, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (1, 3), 'max_features': 10000},
#     # Mid-range text dimensions, wide n-gram range
#     {'pca_text': 100, 'pca_cpc': 30, 'threshold': 95, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 30000},
#     # High threshold, scaling off
#     {'pca_text': 120, 'pca_cpc': 40, 'threshold': 98, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (2, 3), 'max_features': 40000},
#     # Low threshold, narrow n-gram range
#     {'pca_text': 80, 'pca_cpc': 25, 'threshold': 92, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 20000},
#     # Larger PCA and max features, tighter threshold
#     {'pca_text': 200, 'pca_cpc': 60, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 60000},
#     # Testing CountVectorizer with smaller text PCA and scaling on
#     {'pca_text': 70, 'pca_cpc': 20, 'threshold': 96, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 25000},
#     # Mid-range configurations
#     {'pca_text': 100, 'pca_cpc': 35, 'threshold': 95, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (2, 3), 'max_features': 30000},
#     # Testing lower n-gram range with high threshold
#     {'pca_text': 120, 'pca_cpc': 40, 'threshold': 99, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (1, 1), 'max_features': 20000},
#     # Maximum n-gram range with low features and no scaling
#     {'pca_text': 60, 'pca_cpc': 15, 'threshold': 97, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (1, 3), 'max_features': 15000},
#     # High dimensions, wide n-gram range, and scaling off
#     {'pca_text': 180, 'pca_cpc': 55, 'threshold': 93, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (2, 3), 'max_features': 50000},
#     # Higher pca_cpc than pca_text, fixed threshold of 90, experimenting with various n-grams and max_features
#     {'pca_text': 50, 'pca_cpc': 70, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 25000},
#     {'pca_text': 30, 'pca_cpc': 60, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (2, 4), 'max_features': 15000},
#     {'pca_text': 80, 'pca_cpc': 100, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (2, 3), 'max_features': 35000},
#     {'pca_text': 70, 'pca_cpc': 90, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (3, 4), 'max_features': 40000},
#     {'pca_text': 40, 'pca_cpc': 60, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (2, 3), 'max_features': 20000},
#
#     # Equal pca_text and pca_cpc, fixed threshold of 90, with various vectorizers and scaling options
#     {'pca_text': 100, 'pca_cpc': 100, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (1, 1), 'max_features': 30000},
#     {'pca_text': 50, 'pca_cpc': 50, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 15000},
#     {'pca_text': 120, 'pca_cpc': 120, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (2, 4), 'max_features': 40000},
#     {'pca_text': 60, 'pca_cpc': 60, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 20000},
#     {'pca_text': 70, 'pca_cpc': 70, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 25000},
#
#     # Lower pca_text than pca_cpc, testing different n-gram ranges and scaling
#     {'pca_text': 40, 'pca_cpc': 80, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 20000},
#     {'pca_text': 20, 'pca_cpc': 40, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (2, 3), 'max_features': 15000},
#     {'pca_text': 90, 'pca_cpc': 120, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (3, 4), 'max_features': 35000},
#     {'pca_text': 60, 'pca_cpc': 100, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 30000},
#     {'pca_text': 50, 'pca_cpc': 70, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (2, 4), 'max_features': 25000},
#
#     # High pca_text and pca_cpc, fixed threshold of 90, testing high max_features
#     {'pca_text': 150, 'pca_cpc': 150, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 60000},
#     {'pca_text': 120, 'pca_cpc': 120, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (1, 1), 'max_features': 50000},
#     {'pca_text': 200, 'pca_cpc': 200, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (2, 4), 'max_features': 75000},
#     {'pca_text': 140, 'pca_cpc': 140, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (3, 3), 'max_features': 60000},
#     {'pca_text': 160, 'pca_cpc': 160, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 55000},
#
#     # Lower pca_cpc and pca_text values, experimenting with lower max_features
#     {'pca_text': 20, 'pca_cpc': 30, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (1, 2), 'max_features': 10000},
#     {'pca_text': 40, 'pca_cpc': 60, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (1, 1), 'max_features': 12000},
#     {'pca_text': 60, 'pca_cpc': 90, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (2, 3), 'max_features': 15000},
#     {'pca_text': 50, 'pca_cpc': 80, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (3, 4), 'max_features': 18000},
#     {'pca_text': 30, 'pca_cpc': 50, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (1, 2), 'max_features': 10000},
#
#     # Mixed scaling, various n-gram ranges
#     {'pca_text': 90, 'pca_cpc': 110, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': False,
#      'ngram_range': (2, 3), 'max_features': 40000},
#     {'pca_text': 130, 'pca_cpc': 130, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': True,
#      'ngram_range': (3, 4), 'max_features': 45000},
#     {'pca_text': 40, 'pca_cpc': 50, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 30000},
#     {'pca_text': 60, 'pca_cpc': 80, 'threshold': 90, 'vectorizer_type': 'count', 'scaling': False,
#      'ngram_range': (2, 4), 'max_features': 20000},
#     {'pca_text': 120, 'pca_cpc': 130, 'threshold': 90, 'vectorizer_type': 'tfidf', 'scaling': True,
#      'ngram_range': (1, 3), 'max_features': 35000},
# ]


# Define the possible values for each hyperparameter
pca_text_values = [50, 60, 70, 80, 100, 120, 150, 200]
pca_cpc_values = [15, 20, 25, 30, 35, 40, 50, 55, 60]
threshold_values = [90]  # Fixed threshold
vectorizer_types = ['tfidf', 'count']
scaling_options = [True, False]
ngram_ranges = [(1, 1), (1, 2), (1, 3), (2, 3)]
max_features_values = [10000, 15000, 20000]

# Generate all combinations of the parameters
hyperparameter_configs = [
    {'pca_text': pca_text, 'pca_cpc': pca_cpc, 'threshold': threshold,
     'vectorizer_type': vectorizer_type, 'scaling': scaling,
     'ngram_range': ngram_range, 'max_features': max_features}
    for pca_text, pca_cpc, threshold, vectorizer_type, scaling, ngram_range, max_features in product(
        pca_text_values, pca_cpc_values, threshold_values,
        vectorizer_types, scaling_options, ngram_ranges, max_features_values)
]

# Initialize a list to store results
results_list = []

# Run the batch process with each hyperparameter configuration
for i, config in enumerate(hyperparameter_configs):
    result = run_batch_process(**config)
    print(
        f"{i+1}. Tested config {config}\n -> Performance: {result['performance_percentage']}%, {result['num_correct_outliers']}")

    # Format the result as a dictionary
    result_dict = {
        'config': config,
        'performance_percentage': result['performance_percentage']
    }

    # Append the result to the list
    results_list.append(result_dict)

    # Delete large variables to free up memory
    del result
    del config

    # Run garbage collection to free up memory
    gc.collect()

# Save the results list to a JSON file
output_file = 'experiment_results.json'
with open(output_file, 'w') as file:
    json.dump(results_list, file, indent=4)

print(f"Results saved to {output_file}")
