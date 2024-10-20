import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

global log_content
best_config = {
            "pca_text": 60,
            "pca_cpc": 100,
            "threshold": 90,
            "vectorizer_type": "count",
            "scaling": True,
            "ngram_range": [1, 3],
            "max_features": 30000
        }

# Load stopwords from stopwords-ko.json
with open('data/stopwords-ko.json', 'r', encoding='utf-8') as file:
    korean_stopwords = json.load(file)


# Function to read column names from a JSON file
def read_column_names(file):
    # Read the first line from the uploaded JSON file-like object
    return json.loads(file.readline().decode('utf-8')).keys()


def get_maingr(cpc_unsplitted, main_only=1):
    # main_only=0 : 전체 CPC 메인그룹 반환, main_only=1 : 메인트렁크만 반환

    def normalize_cpc(cpc_code):
        # Assuming main group is the first 4 characters, adjust as necessary
        return cpc_code.split('/')[0].strip()

    # Split the CPC codes by '|', normalize to the main group, and remove duplicates
    processed_maingr_list = set()
    #     cpc_list = cpc_unsplitted.split('|')

    cpc_codes = cpc_unsplitted.split('|')
    maingr_list = [normalize_cpc(cpc) for cpc in cpc_codes]
    if main_only == 1:
        maingr_list = [item for item in maingr_list if len(item) < 8 and item[0] != 'Y']
    processed_maingr_list.update(maingr_list)  # Remove duplicates
    processed_maingr_list = list(processed_maingr_list)

    return processed_maingr_list


# def clean_numeric_data(df):
#     # Select only numeric columns from the dataframe
#     numeric_df = df.select_dtypes(include=[np.number])
#     return numeric_df
#
#
def get_hier(cpc):
    tmp = set()
    for item in cpc:
        if isinstance(item, str):
            tmp.update([item[:4], item[:3], item[0]])
    #     tmp = list(set(tmp))
    tmp.update(cpc)
    tmp = list(tmp)
    return tmp


def run():
    # Initialize the session state for logging and results
    # if 'log_content' not in st.session_state:
    #     st.session_state['log_content'] = ""
    # if 'filtered_data' not in st.session_state:
    #     st.session_state['filtered_data'] = None
    # if 'outlier_data' not in st.session_state:
    #     st.session_state['outlier_data'] = None
    # if 'combined_vectors' not in st.session_state:
    #     st.session_state['combined_vectors'] = None

    # Function to process KSIC input and evaluate data
    # def eval_data(ksic, threshold, scaler=True):
    #     tmp_index = data['ksic'] == ksic
    #     cluster1_vectors = combined_vectors[tmp_index]
    #
    #     # Log the number of records at this stage
    #     # update_log(f"Total records with selected KSIC '{ksic}': {len(cluster1_vectors)}")
    #
    #     # Check for low variance or collinearity issues
    #     pca = PCA(n_components=130)  # Example: Adjust components to capture meaningful variance
    #     reduced_vectors = pca.fit_transform(cluster1_vectors)
    #     # update_log(f"Variance explained by PCA: {pca.explained_variance_ratio_.sum():.2f}")
    #     # update_log(f"Number of records after PCA: {len(reduced_vectors)}")
    #
    #     # Scale data before Mahalanobis distance calculation
    #     if scaler:
    #         scaler = StandardScaler()
    #         reduced_vectors = scaler.fit_transform(reduced_vectors)
    #
    #     # Check variance to ensure scaling is correct
    #     variances = np.var(reduced_vectors, axis=0)
    #     # update_log(f"Data Variances:, sample: {variances[:3]}")
    #
    #     cov = EmpiricalCovariance().fit(reduced_vectors)
    #     mahalanobis_distances = cov.mahalanobis(reduced_vectors)
    #     # update_log(f"Computed Mahalanobis distances, sample: {mahalanobis_distances[:5]}")
    #     threshold_value = np.percentile(mahalanobis_distances, threshold)
    #     # update_log(f"Distance threshold for outliers: {threshold_value}")
    #
    #     # Add Mahalanobis distances to the data
    #     data.loc[tmp_index, 'Mahalanobis_Distance'] = mahalanobis_distances
    #
    #     # Separate data into filtered and outlier sets
    #     filtered_data = data.loc[tmp_index].copy()
    #     data1 = filtered_data[filtered_data['Mahalanobis_Distance'] <= threshold_value]
    #     data2 = filtered_data[filtered_data['Mahalanobis_Distance'] > threshold_value]
    #
    #     # Log the number of filtered and outlier records
    #     # update_log(f"Number of filtered records: {len(data1)}")
    #     # update_log(f"Number of outlier records: {len(data2)}")
    #
    #     return data1, data2

    # In your existing pipeline, clean the data before applying PCA
    def visualize_outliers(filtered_data, outlier_data, combined_vectors):
        # Clean the combined_vectors to ensure only numeric data
        # cleaned_combined_vectors = clean_numeric_data(pd.DataFrame(combined_vectors))
        # Ensure you have enough samples and features for PCA
        num_samples, num_features = combined_vectors.shape
        if num_samples < 2 or num_features < 2:
            st.error("Not enough data to run PCA for visualization.")
            return

        # Apply PCA to reduce the dimensionality to 2D for visualization
        pca = PCA(n_components=2)
        with tab4:
            # st.write(combined_vectors.shape())
            reduced_vectors = pca.fit_transform(combined_vectors)

            # Separate the data for plotting
            filtered_indices = filtered_data.index.to_list()
            outlier_indices = outlier_data.index.to_list()

            plt.figure(figsize=(10, 6))

            # Plot filtered data points
            plt.scatter(reduced_vectors[filtered_indices, 0], reduced_vectors[filtered_indices, 1],
                        color='blue', label='Filtered Data')

            # Plot outliers with a different color
            plt.scatter(reduced_vectors[outlier_indices, 0], reduced_vectors[outlier_indices, 1],
                        color='red', label='Outliers')

            # Add labels and legend
            plt.title("2D Visualization of Patent Data with Outliers")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            plt.legend()
            plt.grid(True)

            # Show plot
            st.pyplot(plt)

    def update_log(log_content, new_entry):
        # # global log_content
        log_content += new_entry + "\n"
        # log_area.text_area("Log Output", log_content, height=150)

    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        processed_data = output.getvalue()
        return processed_data

    # Streamlit UI
    st.title("Outlier Detection using Mahalanobis Distance")
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data", "Run", "Results", "Download", "Visualization", "Log"])

    with tab5:
        # Logging
        log_area = st.empty()
        # Function to update the log with new entries, keeping the previous logs
        log_content = ""

    with tab0:
        # Step 1: Upload file (either Excel or JSON)
        #     col01, col02 = st.columns([10, 8])
        # with col01:
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader("Upload an Excel or JSON file", type=["xlsx", "xls", "json"])

        if uploaded_file:
            if uploaded_file.name.endswith(".json"):
                fname = uploaded_file.name
                col_name = read_column_names(uploaded_file)

                temp = []
                error = []
                uploaded_file.seek(0)  # Reset file pointer after reading column names
                for i, line in enumerate(uploaded_file):
                    try:
                        temp.append(json.loads(line.decode('utf-8').replace('\\\\"', '\\"')))
                    except Exception as e:
                        error.append([e, line])
                data = pd.DataFrame(data=temp, columns=col_name)
            else:
                # Load Excel data
                data = pd.read_excel(uploaded_file)

        #     with col02:
            # Step 2: Display the first 3 rows of the uploaded data as a preview
            st.subheader(f"Preview of Data(결측치 제외 {len(data)}건)")

            # Preprocess the data
            data['text'] = data['title'] + ', ' + data['ab'] + ', ' + data['cl']
            data.dropna(subset=['text', 'cpc'], inplace=True)
            data['maingr'] = data['cpc'].apply(get_maingr)
            data['cpc_h'] = data['maingr'].apply(get_hier)
            data.dropna(subset=['text', 'cpc_h'], inplace=True)
            data['Mahalanobis_Distance'] = np.nan

            st.write(data[['title', 'ksic', 'an', 'cpc']].head(10))  # Show first 3 rows

            # Check if there are any documents left after preprocessing
            if data['text'].str.strip().eq('').all():
                st.error("All text documents are empty after preprocessing.")
            else:
                # korean_stopwords = ['그리고', '그러나', '하지만', '또한', '그리고', '때문에', '그러므로', '그런데', '따라서',
                #                     '입니다', '있는', '이런', '있는', '이렇게', '저렇게', '그렇게']
                # Adjust TfidfVectorizer to handle stop words and ignore terms that are too frequent
                vectorizer = TfidfVectorizer(
                    max_features=30000, stop_words=korean_stopwords, min_df=2, ngram_range=(1, 3))
                # vectorizer = CountVectorizer(max_features=30000, stop_words=korean_stopwords, min_df=2, ngram_range=(1, 2))

                # Fit and transform the text data
                text_vectors = vectorizer.fit_transform(data['text']).toarray()
                # Perform PCA for text vectors
                pca1 = PCA(n_components=best_config['pca_text'])
                text_vectors_r = pca1.fit_transform(text_vectors)

        #         scaler = StandardScaler()
        #         text_vectors_scaled = scaler.fit_transform(text_vectors)

            # MultiLabelBinarizer for CPC
            mlb = MultiLabelBinarizer()
            cpc_vectors = mlb.fit_transform(data['cpc_h'])
            # Perform PCA for CPC vectors
            pca2 = PCA(n_components=best_config['pca_cpc'])
            cpc_vectors_r = pca2.fit_transform(cpc_vectors)

            # Combine CPC and text vectors
            vectors_c = np.hstack([text_vectors_r, cpc_vectors_r])

            # Apply scaling if specified
            if best_config['scaling']:
                scaler = StandardScaler()
                vectors_c = scaler.fit_transform(vectors_c)

    with tab1:
        if uploaded_file:
            # Step 4: Layout with columns to display n_components, KSIC code, and threshold at the same height
            col1, col2, col3, col4 = st.columns([4, 3, 3, 3])

            if uploaded_file:
                with col1:
                    ksic_list = data['ksic'].drop_duplicates()
                    selected_ksic = st.selectbox("Select KSIC Code", ksic_list, key='ksic_code')

                with col2:
                    n_components_text = st.number_input("Text Vector dim(PCA)",
                                                   min_value=2, max_value=min(text_vectors.shape[1], 1000), value=60, key='n_components_txt')

                with col3:
                    n_components_cpc = st.number_input("CPC Vector dim(PCA)",
                                                   min_value=2, max_value=min(cpc_vectors.shape[1], 1000), value=100, key='n_components_cpc')

                with col4:
                    # threshold = st.slider("Distance Threshold %", 80, 99, 90, key='threshold_slider')
                    threshold = st.slider("Chi-squared confidence level (%)", 90, 99, 95, key='threshold_slider')

                    # # Step 3: Ask for the dimension size to reduce the vectors
                    # n_components = st.number_input("Enter the number of dimensions for PCA", min_value=2, max_value=min(combined_vectors.shape[1], 1000), value=50)

                if st.button(f"Run Outlier Detection", key="run_detection"):
                    tmp_index = data['ksic'] == selected_ksic

                    # Compute Mahalanobis distances for the test set
                    cov = EmpiricalCovariance().fit(vectors_c[tmp_index])
                    mahalanobis_distances = cov.mahalanobis(vectors_c[tmp_index])

                    # Select the top 20 furthest points
                    data.loc[tmp_index, 'Mahalanobis_Distance'] = mahalanobis_distances
                    top_20_outliers = data[tmp_index].nlargest(20, 'Mahalanobis_Distance')

                    # # cutoff 계산
                    # threshold_value = np.percentile(mahalanobis_distances, threshold)
                    # Chi-squared 분포 기반 cutoff 계산
                    n_features = vectors_c.shape[1]
                    threshold_value = chi2.ppf(threshold / 100, df=n_features)

                    # Separate data into filtered and outlier sets
                    filtered_data = data.loc[tmp_index].copy()
                    data1 = filtered_data[filtered_data['Mahalanobis_Distance'] <= threshold_value]
                    data2 = filtered_data[filtered_data['Mahalanobis_Distance'] > threshold_value]

                    # # eval_data(selected_ksic, threshold)
                    # filtered_data, outlier_data = eval_data(selected_ksic, threshold)
                    #
                    # # Combine CPC and text vectors
                    # train_vectors_c = np.hstack([train_vectors_r, train_cpc_vectors_r])
                    #
                    # combined_result = pd.concat([filtered_data, outlier_data])

                    # Store the results in session state to persist
                    st.session_state['filtered_data'] = data1
                    st.session_state['outlier_data'] = data2
                    # st.session_state['combined_vectors'] = combined_result

                    with tab2:
                        st.subheader(f"Filtered Data({len(data1)})건")
                        st.write(data1[['Mahalanobis_Distance', 'ksic', 'maingr', 'title', 'ab', 'cl', 'an', 'cpc']].head(10))

                        st.subheader(f"Outliers({len(data2)})건")
                        st.write(data2[['Mahalanobis_Distance', 'ksic', 'maingr', 'title', 'ab', 'cl', 'an', 'cpc']].head(10))

            # Reset log and session state when KSIC changes
            if selected_ksic != st.session_state.get("last_ksic"):
                log_content = ""  # Reset log
                st.session_state["last_ksic"] = selected_ksic  # Update the KSIC
                if 'filtered_data' in st.session_state:
                    del st.session_state['filtered_data']
                if 'outlier_data' in st.session_state:
                    del st.session_state['outlier_data']
                if 'combined_vectors' in st.session_state:
                    del st.session_state['combined_vectors']

    # with tab3:
    #     # Check if results are available
    #     if st.session_state.get('filtered_data') is not None and st.session_state.get('outlier_data') is not None:
    #         # Excel download button
    #         st.download_button(
    #             label=f"Download Excel({len(st.session_state['filtered_data']) + len(st.session_state['outlier_data'])}건)",
    #             data=convert_df_to_excel(st.session_state['combined_vectors']),
    #             file_name=f'{selected_ksic}_MahalanobisDistance.xlsx',
    #             mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    #         )
    #
    # with tab4:
    #     # Visualization button
    #     if st.button("Run Visualizer", key="run_visualizer"):
    #         visualize_outliers(st.session_state['filtered_data'], st.session_state['outlier_data'], st.session_state['combined_vectors'])
