import streamlit as st
import pandas as pd
import itertools
import re
import networkx as nx
import numpy as np
import warnings
from io import BytesIO

# openpyxl 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --- 함수 정의 ---

@st.cache_data
def create_yearly_edge_dfs(df):
    """
    입력 DataFrame에서 연도별 협력 관계(엣지) DataFrame 딕셔너리를 생성합니다.
    """
    # '제N특허권자명' 형태의 컬럼을 동적으로 찾기
    patent_holder_pattern = re.compile(r'제(\d+)특허권자명')
    dynamic_cols_to_combine = []
    for col in df.columns:
        match = patent_holder_pattern.match(col)
        if match:
            dynamic_cols_to_combine.append((int(match.group(1)), col))
    
    dynamic_cols_to_combine.sort()
    cols_to_combine = [col_name for _, col_name in dynamic_cols_to_combine]

    if not cols_to_combine:
        st.error("입력 파일에 '제N특허권자명' 형태의 컬럼이 없습니다.")
        return None

    yearly_edge_dfs = {}
    try:
        if '등록년도' not in df.columns or '협력유형4' not in df.columns:
            raise KeyError("'등록년도' 또는 '협력유형4' 컬럼을 찾을 수 없습니다.")

        years_to_process = sorted(df['등록년도'].dropna().unique().tolist())
        
        for year in years_to_process:
            df_filtered_by_year = df[df['등록년도'] == year]
            df_final_filtered = df_filtered_by_year[df_filtered_by_year['협력유형4'] == '법인 간'].copy()
            
            if df_final_filtered.empty:
                continue

            all_combinations_for_this_year = []
            for _, row in df_final_filtered.iterrows():
                current_row_values_raw = [str(row[col]) for col in cols_to_combine if pd.notna(row[col])]
                current_row_unique_values = sorted(list(set(current_row_values_raw)))
                
                if len(current_row_unique_values) >= 2:
                    row_combinations = list(itertools.combinations(current_row_unique_values, 2))
                    all_combinations_for_this_year.extend(row_combinations)
            
            if not all_combinations_for_this_year:
                continue

            yearly_edge_dfs[str(int(year))] = pd.DataFrame(all_combinations_for_this_year, columns=['source', 'target'])
        
        return yearly_edge_dfs
    except KeyError as e:
        st.error(f"입력 파일에 필수 컬럼({e})이 없습니다. ('등록년도', '협력유형4', '제N특허권자명')")
        return None

@st.cache_data
def analyze_networks(_edge_dfs):
    """
    엣지 DataFrame 딕셔너리를 기반으로 네트워크 지표를 계산하고,
    전체 지표 DataFrame과 중심성 DataFrame 딕셔너리를 반환합니다.
    """
    overall_metrics = []
    all_sheets_centrality_results = {}

    for year, df_sheet_data in _edge_dfs.items():
        G = nx.from_pandas_edgelist(df_sheet_data, 'source', 'target')

        if G.number_of_nodes() == 0:
            continue

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        average_degree = sum(d for _, d in G.degree()) / num_nodes if num_nodes > 0 else 0
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        avg_path_length, num_nodes_giant_component, proportion_giant_component = np.nan, 0, 0.0
        if connected_components:
            largest_component_nodes = max(connected_components, key=len)
            subgraph_gc = G.subgraph(largest_component_nodes)
            if subgraph_gc.number_of_nodes() > 1:
                 avg_path_length = nx.average_shortest_path_length(subgraph_gc)
            num_nodes_giant_component = len(largest_component_nodes)
            if num_nodes > 0:
                proportion_giant_component = (num_nodes_giant_component / num_nodes) * 100
        max_eigenvalue, alpha_reciprocal = np.nan, np.nan
        if num_nodes > 1:
            try:
                adj_matrix = nx.to_numpy_array(G)
                eigenvalues = np.linalg.eigvals(adj_matrix)
                max_eigenvalue = np.max(eigenvalues.real)
                if max_eigenvalue > 1e-9:
                    alpha_reciprocal = 1.0 / max_eigenvalue
            except Exception: pass
        overall_metrics.append({
            'Year': year, 'Nodes': num_nodes, 'Edges': num_edges, 'Density': density,
            'Avg Degree': average_degree, 'Avg Path Length (GC)': avg_path_length,
            'Components': num_components, 'Giant Component Nodes': num_nodes_giant_component,
            'Giant Component Ratio (%)': proportion_giant_component, 'Max Eigenvalue': max_eigenvalue,
            'Alpha (1/Max Eigenvalue)': alpha_reciprocal
        })
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-8)
        except nx.PowerIterationFailedConvergence:
            eigenvector_centrality = {n: np.nan for n in G.nodes()}
            st.warning(f"'{year}'년 고유벡터 중심성 계산에 실패했습니다 (수렴 실패).")
        katz_alpha = alpha_reciprocal * 0.9 if pd.notna(alpha_reciprocal) and alpha_reciprocal > 0 else 0.01
        try:
            katz_centrality = nx.katz_centrality(G, alpha=katz_alpha, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            katz_centrality = {n: np.nan for n in G.nodes()}
            st.warning(f"'{year}'년 가츠 중심성 계산에 실패했습니다 (수렴 실패).")
        df_centrality = pd.DataFrame({
            '기관명': list(G.nodes()),
            '연결중심성': [degree_centrality.get(n, 0) for n in G.nodes()],
            '근접중심성': [closeness_centrality.get(n, 0) for n in G.nodes()],
            '매개중심성': [betweenness_centrality.get(n, 0) for n in G.nodes()],
            '고유벡터중심성': [eigenvector_centrality.get(n, 0) for n in G.nodes()],
            '가츠중심성': [katz_centrality.get(n, 0) for n in G.nodes()],
        }).sort_values(by='연결중심성', ascending=False)
        all_sheets_centrality_results[year] = df_centrality

    df_overall = pd.DataFrame(overall_metrics).set_index('Year')
    return df_overall, all_sheets_centrality_results

@st.cache_data
def to_excel(_df_dict):
    """
    DataFrame 딕셔너리를 엑셀 파일 형식의 바이트 스트림으로 변환합니다.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in _df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    processed_data = output.getvalue()
    return processed_data

def run():
    """
    SNA 서브앱을 실행하는 메인 함수
    """
    st.set_page_config(layout="wide")
    st.title("🔬 사회연결망 분석 (Social Network Analysis)")
    st.markdown("""
    엑셀 파일을 업로드하여 연도별 협력 네트워크를 분석합니다.
    - **필수 컬럼**: `등록년도`, `협력유형4`, `제1특허권자명`, `제2특허권자명`, ...
    - **분석 조건**: `협력유형4`가 '법인 간'인 데이터만 사용합니다.
    """)

    # --- 1. 데이터 입력 UI ---
    st.divider()
    st.header("1. 데이터 입력")
    uploaded_file = st.file_uploader("분석할 엑셀 파일을 업로드하세요.", type=['xlsx'])

    if uploaded_file is not None:
        try:
            df_input = pd.read_excel(uploaded_file)
            st.success(f"✅ '{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")
            with st.expander("업로드한 데이터 미리보기 (상위 5개 행)"):
                st.dataframe(df_input.head())

            # --- 2. 동작 실행 버튼 ---
            st.divider()
            st.header("2. 분석 실행")
            if st.button("🚀 분석 실행하기"):
                
                # --- 3. 결과 표시 영역 ---
                st.divider()
                st.header("3. 분석 결과")

                # 단계 1: 엣지 리스트 생성
                with st.spinner("1단계: 연도별 협력 관계(엣지)를 생성 중입니다..."):
                    edge_dfs = create_yearly_edge_dfs(df_input)
                
                if not edge_dfs:
                    st.error("엣지 리스트 생성에 실패하여 분석을 중단합니다. 파일의 컬럼 구성을 확인해주세요.")
                    st.stop()
                st.success("✅ 1단계: 엣지 리스트 생성 완료!")

                # 단계 2: 네트워크 분석
                with st.spinner("2단계: 네트워크 지표 및 중심성을 계산 중입니다..."):
                    df_metrics, df_centralities = analyze_networks(edge_dfs)

                if df_metrics is None or df_centralities is None:
                    st.error("네트워크 분석에 실패하여 중단합니다.")
                    st.stop()
                st.success("✅ 2단계: 네트워크 분석 완료!")

                # 결과 표시
                st.subheader("📊 네트워크 거시 지표")
                st.markdown("연도별 네트워크의 구조적 특성을 나타내는 지표입니다.")
                st.dataframe(df_metrics.style.format(precision=4))

                st.subheader("🔭 노드 중심성 분석")
                st.markdown("각 연도별 네트워크에서 어떤 기관이 중심적인 역할을 하는지 나타냅니다.")
                
                # 중심성 결과 다운로드 버튼
                excel_data = to_excel(df_centralities)
                st.download_button(
                    label="📥 중심성 분석 결과 다운로드 (Excel)",
                    data=excel_data,
                    file_name="중심성_분석_결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                years = list(df_centralities.keys())
                if not years:
                    st.warning("표시할 중심성 데이터가 없습니다.")
                else:
                    sorted_years = sorted(years, key=int)
                    tabs = st.tabs(sorted_years)
                    for i, year in enumerate(sorted_years):
                        with tabs[i]:
                            st.dataframe(df_centralities[year])
        
        except Exception as e:
            st.error(f"파일을 읽거나 분석하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run()
