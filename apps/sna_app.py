import streamlit as st
import pandas as pd
import itertools
import re
import networkx as nx
import numpy as np
import warnings
from io import BytesIO
from pyvis.network import Network
import plotly.graph_objects as go

# openpyxl 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --------------------------
# 캐시 함수 (위젯 호출 금지)
# --------------------------
@st.cache_data
def create_yearly_edge_dfs(df: pd.DataFrame):
    """
    입력 DataFrame에서 연도별 협력 관계(엣지) DataFrame 딕셔너리를 생성합니다.
    필수 컬럼: 등록년도, 협력유형4, 제N특허권자명(N>=1)
    """
    patent_holder_pattern = re.compile(r'제(\d+)특허권자명')
    dynamic_cols_to_combine = []
    for col in df.columns:
        m = patent_holder_pattern.match(col)
        if m:
            dynamic_cols_to_combine.append((int(m.group(1)), col))
    dynamic_cols_to_combine.sort()
    cols_to_combine = [c for _, c in dynamic_cols_to_combine]

    if not cols_to_combine:
        raise ValueError("입력 파일에 '제N특허권자명' 형태의 컬럼이 없습니다.")

    if '등록년도' not in df.columns or '협력유형4' not in df.columns:
        raise KeyError("'등록년도' 또는 '협력유형4' 컬럼을 찾을 수 없습니다.")

    yearly_edge_dfs = {}
    years_to_process = sorted(df['등록년도'].dropna().unique().tolist())

    for year in years_to_process:
        df_y = df[df['등록년도'] == year]
        # 문자열 메서드는 반드시 .str 체인으로 처리해야 함
        df_final = df_y[df_y['협력유형4'].astype(str).str.strip().str.replace(" ", "", regex=False) == '법인간'].copy()
        if df_final.empty:
            continue

        combos = []
        for _, row in df_final.iterrows():
            vals = [str(row[c]) for c in cols_to_combine if pd.notna(row[c])]
            uniq = sorted(set(vals))
            if len(uniq) >= 2:
                combos.extend(itertools.combinations(uniq, 2))

        if combos:
            yearly_edge_dfs[str(int(year))] = pd.DataFrame(combos, columns=['source', 'target'])

    return yearly_edge_dfs


@st.cache_data
def analyze_networks(_edge_dfs: dict):
    """
    엣지 DF 딕셔너리를 기반으로 네트워크 지표 및 중심성 계산.
    """
    overall_metrics = []
    all_sheets_centrality_results = {}
    warn_msgs = []

    for year, df_edges in _edge_dfs.items():
        G = nx.from_pandas_edgelist(df_edges, 'source', 'target')
        if G.number_of_nodes() == 0:
            continue

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        avg_deg = sum(d for _, d in G.degree()) / n_nodes if n_nodes else 0

        comps = list(nx.connected_components(G))
        n_comp = len(comps)
        avg_path_len, gc_nodes, gc_ratio = np.nan, 0, 0.0
        if comps:
            largest = max(comps, key=len)
            sub_gc = G.subgraph(largest)
            if sub_gc.number_of_nodes() > 1:
                try:
                    avg_path_len = nx.average_shortest_path_length(sub_gc)
                except Exception as e:
                    warn_msgs.append(f"[{year}] 평균 최단경로 계산 실패: {e}")
            gc_nodes = len(largest)
            if n_nodes > 0:
                gc_ratio = (gc_nodes / n_nodes) * 100

        max_eig, alpha_recip = np.nan, np.nan
        if n_nodes > 1:
            try:
                adj = nx.to_numpy_array(G)
                eig = np.linalg.eigvals(adj)
                max_eig = np.max(eig.real)
                if max_eig > 1e-9:
                    alpha_recip = 1.0 / max_eig
            except Exception as e:
                warn_msgs.append(f"[{year}] 고유값 계산 실패: {e}")

        overall_metrics.append({
            'Year': year, 'Nodes': n_nodes, 'Edges': n_edges, 'Density': density,
            'Avg Degree': avg_deg, 'Avg Path Length (GC)': avg_path_len,
            'Components': n_comp, 'Giant Component Nodes': gc_nodes,
            'Giant Component Ratio (%)': gc_ratio, 'Max Eigenvalue': max_eig,
            'Alpha (1/Max Eigenvalue)': alpha_recip
        })

        deg_c = nx.degree_centrality(G)
        clo_c = nx.closeness_centrality(G)
        bet_c = nx.betweenness_centrality(G)
        try:
            eig_c = nx.eigenvector_centrality(G, max_iter=1000, tol=1.0e-8)
        except nx.PowerIterationFailedConvergence:
            eig_c = {n: np.nan for n in G.nodes()}
            warn_msgs.append(f"[{year}] 고유벡터 중심성 계산에 실패했습니다 (수렴 실패).")

        katz_alpha = alpha_recip * 0.9 if pd.notna(alpha_recip) and alpha_recip > 0 else 0.01
        try:
            katz_c = nx.katz_centrality(G, alpha=katz_alpha, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            katz_c = {n: np.nan for n in G.nodes()}
            warn_msgs.append(f"[{year}] 가츠 중심성 계산에 실패했습니다 (수렴 실패).")

        df_c = pd.DataFrame({
            '기관명': list(G.nodes()),
            '연결중심성': [deg_c.get(n, 0) for n in G.nodes()],
            '근접중심성': [clo_c.get(n, 0) for n in G.nodes()],
            '매개중심성': [bet_c.get(n, 0) for n in G.nodes()],
            '고유벡터중심성': [eig_c.get(n, 0) for n in G.nodes()],
            '가츠중심성': [katz_c.get(n, 0) for n in G.nodes()],
        }).sort_values(by='연결중심성', ascending=False)

        all_sheets_centrality_results[year] = df_c

    df_overall = pd.DataFrame(overall_metrics).set_index('Year')
    return df_overall, all_sheets_centrality_results, warn_msgs


@st.cache_data
def to_excel(_df_dict: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in _df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


# --------------------------
# 시각화 유틸
# --------------------------
def _aggregate_weighted_edges(df_edges: pd.DataFrame, min_weight: int) -> pd.DataFrame:
    if df_edges.empty:
        return df_edges.assign(weight=1)
    g = df_edges.groupby(['source', 'target']).size().reset_index(name='weight')
    g = g[g['weight'] >= int(max(1, min_weight))]
    return g


def _make_graph_with_weights(df_edges_w: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in df_edges_w.iterrows():
        G.add_edge(r['source'], r['target'], weight=float(r['weight']))
    return G


def _choose_layout(G: nx.Graph, layout_name: str, seed: int = 42) -> dict:
    name = (layout_name or "spring").lower()
    if name == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if name == "circular":
        return nx.circular_layout(G)
    if name == "shell":
        return nx.shell_layout(G)
    if name == "spectral":
        return nx.spectral_layout(G)
    if name == "random":
        return nx.random_layout(G, seed=seed)
    return nx.spring_layout(G, seed=seed)


def _normalize(values: np.ndarray, out_min: float, out_max: float, use_log: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if use_log:
        arr = np.log1p(np.maximum(arr, 0))
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if np.isclose(vmin, vmax):
        return np.full_like(arr, (out_min + out_max) / 2.0)
    return out_min + (arr - vmin) * (out_max - out_min) / (vmax - vmin)


def _metric_series(df_centrality: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "degree":
        metric = "연결중심성"
    col = metric if metric in df_centrality.columns else "연결중심성"
    return df_centrality.set_index("기관명")[col].fillna(0.0)


def visualize_network_plotly_advanced(
    df_edges: pd.DataFrame,
    df_centrality: pd.DataFrame,
    year: str,
    *,
    layout_name: str = "spring",
    node_size_metric: str = "연결중심성",
    node_color_metric: str = "연결중심성",
    node_size_range: tuple = (8, 32),
    node_size_log: bool = False,
    colorscale: str = "YlGnBu",
    reverse_colorscale: bool = True,
    show_labels: str = "Top-N",  # "Off" / "Top-N" / "All"
    label_top_n: int = 20,
    label_font_size: int = 11,
    label_position: str = "top center",
    min_edge_weight: int = 1,
    min_degree: int = 0,
    keep_top_n_nodes_by_metric: int = 0,
    edge_width_range: tuple = (0.5, 6.0),
    edge_opacity: float = 0.35,
    edge_bins: int = 4
) -> go.Figure:
    df_w = _aggregate_weighted_edges(df_edges, min_edge_weight)
    if df_w.empty:
        return go.Figure(layout=go.Layout(title=dict(text=f"{year} 빈 그래프", font=dict(size=16)), height=600))

    G = _make_graph_with_weights(df_w)

    if min_degree > 0:
        to_keep = [n for n, d in G.degree() if d >= min_degree]
        G = G.subgraph(to_keep).copy()

    if G.number_of_nodes() == 0:
        return go.Figure(layout=go.Layout(title=dict(text=f"{year} (필터 결과 노드 없음)", font=dict(size=16)), height=600))

    if keep_top_n_nodes_by_metric and keep_top_n_nodes_by_metric > 0:
        s_metric = _metric_series(df_centrality, node_size_metric)
        top_nodes = set(s_metric.sort_values(ascending=False).head(keep_top_n_nodes_by_metric).index)
        G = G.subgraph([n for n in G.nodes() if n in top_nodes]).copy()
        if G.number_of_nodes() == 0:
            return go.Figure(layout=go.Layout(title=dict(text=f"{year} (Top-N 필터 후 노드 없음)", font=dict(size=16)), height=600))

    pos = _choose_layout(G, layout_name)

    s_size = _metric_series(df_centrality, node_size_metric).reindex(G.nodes(), fill_value=0.0)
    s_color = _metric_series(df_centrality, node_color_metric).reindex(G.nodes(), fill_value=0.0)

    node_sizes = _normalize(s_size.values, node_size_range[0], node_size_range[1], use_log=node_size_log)
    node_colors = s_color.values

    w = np.array([G.edges[e]['weight'] for e in G.edges()], dtype=float)
    if w.size == 0:
        w = np.array([1.0])
    widths = _normalize(w, edge_width_range[0], edge_width_range[1], use_log=False)

    if edge_bins < 1:
        edge_bins = 1
    bins = np.linspace(widths.min(), widths.max(), edge_bins + 1)
    bin_idx = np.digitize(widths, bins, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, edge_bins - 1)

    edge_traces = []
    for b in range(edge_bins):
        xs, ys = [], []
        this_w = None
        for (edge, wid, idx) in zip(G.edges(), widths, bin_idx):
            if idx != b:
                continue
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            this_w = wid
        if xs:
            edge_traces.append(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(width=float(this_w), color='rgba(100,100,100,1.0)'),
                hoverinfo='none',
                opacity=float(edge_opacity),
                showlegend=False
            ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    hover_text = [
        (f"기관명: {n}<br>"
         f"연결중심성: {float(_metric_series(df_centrality,'연결중심성').get(n,0)):.4f}<br>"
         f"근접중심성: {float(_metric_series(df_centrality,'근접중심성').get(n,0)):.4f}<br>"
         f"매개중심성: {float(_metric_series(df_centrality,'매개중심성').get(n,0)):.4f}<br>"
         f"고유벡터중심성: {float(_metric_series(df_centrality,'고유벡터중심성').get(n,0)):.4f}<br>"
         f"가츠중심성: {float(_metric_series(df_centrality,'가츠중심성').get(n,0)):.4f}")
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition=label_position,
        textfont=dict(size=int(label_font_size)),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            showscale=True,
            colorscale=colorscale,
            reversescale=bool(reverse_colorscale),
            colorbar=dict(thickness=16, title=dict(text=f'{node_color_metric}', side='right')),
            line=dict(width=0.5, color='#333')
        ),
        text=None,   # 아래서 채움
        texttemplate='%{text}',
        hovertext=hover_text,
        showlegend=False
    )

    labels = [""] * len(G.nodes())
    if show_labels == "All":
        labels = list(G.nodes())
    elif show_labels == "Top-N":
        s_for_label = _metric_series(df_centrality, node_size_metric).reindex(G.nodes(), fill_value=0.0)
        topN = min(int(label_top_n), len(s_for_label))
        top_nodes = set(s_for_label.sort_values(ascending=False).head(topN).index)
        labels = [n if n in top_nodes else "" for n in G.nodes()]
    # Off면 빈 라벨 유지
    node_trace.text = labels

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        title=dict(text=f"<br>{year}년 협력 네트워크 (Plotly 고급)", font=dict(size=18)),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=48),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=820
    )
    return fig


def visualize_network(G: nx.Graph, year: str):
    net = Network(height='800px', width='100%', notebook=False, cdn_resources='in_line', heading=f'{year}년 협력 네트워크')
    for node in G.nodes():
        degree = G.degree(node)
        G.nodes[node]['size'] = degree * 3 + 10
        G.nodes[node]['title'] = f"기관명: {node}<br>연결수: {degree}"
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    return net.generate_html(notebook=False)


# --------------------------
# 메인 실행 (8:2 레이아웃 적용)
# --------------------------
def run():
    st.title("🔬 사회연결망 분석 (Social Network Analysis)")
    st.markdown("""
    엑셀 파일을 업로드하여 연도별 협력 네트워크를 분석합니다.
    - **필수 컬럼**: `등록년도`, `협력유형4`, `제1특허권자명`, `제2특허권자명`, ...
    - **분석 조건**: `협력유형4`가 '법인 간'인 데이터만 사용합니다.
    """)

    if 'sna_results' not in st.session_state:
        st.session_state.sna_results = None

    # 1) 데이터 입력 & 실행 버튼 — 8:2 컬럼
    st.divider()
    st.header("1. 데이터 입력 & 실행")
    col_input, col_run = st.columns([8, 2], gap="large")

    with col_input:
        uploaded_file = st.file_uploader(
            "분석할 엑셀 파일을 업로드하세요.",
            type=['xlsx'],
            on_change=lambda: st.session_state.update(sna_results=None),
            key="sna_uploader"
        )
        if uploaded_file is not None:
            try:
                df_input = pd.read_excel(uploaded_file)
                st.success(f"✅ '{uploaded_file.name}' 파일이 업로드되었습니다.")
                with st.expander("업로드한 데이터 미리보기 (상위 10개 행)"):
                    st.dataframe(df_input.head(10))
            except Exception as e:
                st.error(f"파일 읽기 오류: {e}")
                df_input = None
        else:
            df_input = None

    with col_run:
        st.markdown("### 실행")
        st.caption("파일 업로드 후 실행하세요.")
        go_analyze = st.button("🚀 분석 실행하기", use_container_width=True, disabled=(df_input is None))

    # 2) 분석 실행
    if go_analyze and df_input is not None:
        try:
            with st.spinner("1단계: 연도별 협력 관계(엣지)를 생성 중입니다..."):
                edge_dfs = create_yearly_edge_dfs(df_input)
        except Exception as e:
            st.error(f"엣지 리스트 생성 실패: {e}")
            st.stop()

        if not edge_dfs:
            st.error("엣지 리스트 생성에 실패했습니다. 파일의 컬럼 구성을 확인해주세요.")
            st.stop()
        st.success("✅ 1단계 완료: 엣지 리스트 생성")

        with st.spinner("2단계: 네트워크 지표 및 중심성 계산 중입니다..."):
            df_metrics, df_centralities, warn_msgs = analyze_networks(edge_dfs)

        st.success("✅ 2단계 완료: 네트워크 분석")
        for msg in warn_msgs:
            st.warning(msg)

        st.session_state.sna_results = {
            "edge_dfs": edge_dfs,
            "df_metrics": df_metrics,
            "df_centralities": df_centralities
        }

    # 3) 분석 결과 & 4) 시각화
    if st.session_state.sna_results:
        res = st.session_state.sna_results
        edge_dfs = res["edge_dfs"]
        df_metrics = res["df_metrics"]
        df_centralities = res["df_centralities"]
        years = list(df_centralities.keys())

        st.divider()
        st.header("2. 분석 결과")
        st.subheader("📊 네트워크 거시 지표")
        st.dataframe(df_metrics.round(4))

        st.subheader("🔭 노드 중심성 분석")
        excel_data = to_excel(df_centralities)
        st.download_button(
            label="📥 중심성 분석 결과 다운로드 (Excel)",
            data=excel_data,
            file_name="중심성_분석_결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if years:
            sorted_years = sorted(years, key=int)
            tabs = st.tabs(sorted_years)
            for i, y in enumerate(sorted_years):
                with tabs[i]:
                    st.dataframe(df_centralities[y])
        else:
            st.warning("표시할 중심성 데이터가 없습니다.")

        # ---- 시각화 ----
        st.divider()
        st.header("3. 네트워크 시각화")
        if years:
            sorted_years = sorted(years, key=int)
            selected_year = st.selectbox("시각화할 연도를 선택하세요.", sorted_years)

            left, right = st.columns([1, 1])
            with left:
                vis_library = st.radio(
                    "시각화 라이브러리를 선택하세요:",
                    ('pyvis (빠른 인터랙션)', 'Plotly (고급 설정)'),
                    horizontal=True
                )

            adv_opts = {}
            if vis_library == 'Plotly (고급 설정)':
                with st.expander("⚙️ 고급 시각화 옵션", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        layout_name = st.selectbox("레이아웃 알고리즘",
                                                   ["spring", "kamada_kawai", "circular", "shell", "spectral", "random"])
                        node_size_metric = st.selectbox("노드 사이즈 기준",
                                                        ["연결중심성", "근접중심성", "매개중심성", "고유벡터중심성", "가츠중심성", "degree"])
                        node_color_metric = st.selectbox("노드 색상 기준",
                                                         ["연결중심성", "근접중심성", "매개중심성", "고유벡터중심성", "가츠중심성", "degree"])
                    with c2:
                        size_min, size_max = st.slider("노드 사이즈 범위", 4, 80, value=(8, 32))
                        size_log = st.checkbox("노드 사이즈 로그 스케일", value=False)
                        color_scale = st.selectbox("Color Scale",
                                                   ["YlGnBu", "Viridis", "Plasma", "Cividis", "Turbo", "Magma",
                                                    "Inferno", "RdBu", "Portland"])
                        reverse_cs = st.checkbox("색상 반전", value=True)
                    with c3:
                        label_mode = st.selectbox("라벨 표시", ["Top-N", "All", "Off"], index=0)
                        topn = st.slider("라벨 Top-N", 1, 200, 20)
                        label_font_size = st.slider("라벨 폰트 크기", 8, 24, 11)
                        label_pos = st.selectbox("라벨 위치",
                                                 ["top center", "middle center", "bottom center", "top left", "top right"])

                    st.markdown("---")
                    c4, c5, c6 = st.columns(3)
                    with c4:
                        min_edge_weight = st.number_input("최소 에지 가중치(동일쌍 빈도)", min_value=1, value=1, step=1, format="%d")
                        min_degree = st.number_input("최소 노드 차수", min_value=0, value=0, step=1, format="%d")
                    with c5:
                        keep_top_n_nodes = st.number_input("선택 기준 Top-N 노드만 유지(0=해제)", min_value=0, value=0, step=1, format="%d")
                        edge_opacity = st.slider("에지 투명도", 0.05, 1.0, 0.35)
                    with c6:
                        wmin, wmax = st.slider("에지 두께 범위", 0.2, 12.0, value=(0.5, 6.0))
                        edge_bins = st.slider("에지 두께 구간(bin)", 1, 8, 4)

                adv_opts = dict(
                    layout_name=layout_name,
                    node_size_metric=node_size_metric,
                    node_color_metric=node_color_metric,
                    node_size_range=(int(size_min), int(size_max)),
                    node_size_log=bool(size_log),
                    colorscale=color_scale,
                    reverse_colorscale=bool(reverse_cs),
                    show_labels=label_mode,
                    label_top_n=int(topn),
                    label_font_size=int(label_font_size),
                    label_position=label_pos,
                    min_edge_weight=int(min_edge_weight),
                    min_degree=int(min_degree),
                    keep_top_n_nodes_by_metric=int(keep_top_n_nodes),
                    edge_width_range=(float(wmin), float(wmax)),
                    edge_opacity=float(edge_opacity),
                    edge_bins=int(edge_bins),
                )

            if st.button(f"🕸️ {selected_year}년 네트워크 시각화"):
                year_edges_df = edge_dfs.get(selected_year)
                if year_edges_df is None or year_edges_df.empty:
                    st.warning(f"{selected_year}년도에 대한 네트워크 데이터가 없습니다.")
                else:
                    if vis_library == 'pyvis (빠른 인터랙션)':
                        with st.spinner(f"{selected_year}년 pyvis 네트워크 생성 중…"):
                            G = nx.from_pandas_edgelist(year_edges_df, 'source', 'target')
                            html_source = visualize_network(G, selected_year)
                            st.subheader(f"{selected_year}년 협력 네트워크 (pyvis)")
                            st.components.v1.html(html_source, height=800, scrolling=True)
                    else:
                        with st.spinner(f"{selected_year}년 Plotly 네트워크 생성 중…"):
                            fig = visualize_network_plotly_advanced(
                                df_edges=year_edges_df,
                                df_centrality=df_centralities[selected_year],
                                year=selected_year,
                                **adv_opts
                            )
                            st.subheader(f"{selected_year}년 협력 네트워크 (Plotly)")
                            st.plotly_chart(fig, use_container_width=True)

                            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
                            st.download_button(
                                label="📥 Plotly 그래프 HTML 다운로드",
                                data=html_str.encode("utf-8"),
                                file_name=f"network_{selected_year}.html",
                                mime="text/html"
                            )
        else:
            st.warning("시각화할 네트워크가 없습니다.")


if __name__ == "__main__":
    run()
