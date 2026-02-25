def render_svg(map_dict, param_dict):

    import pandas as pd
    import numpy as np
    import networkx as nx
    import cairo
    import math
    from typing import List, Tuple
    import io


    def map2nodes(transcript_map):
        # """
        # Convert a transcript event map into a node map with
        # start/end frames and transition states.
        # """
        event_map = transcript_map.copy()

        # ---------- HELPERS ----------

        def frame_from_pos(pos):
            return (pos - 1) % 3 + 1

        # ---------- NODE HANDLERS ----------

        def handle_init(df):
            df['start_frame'] = 0
            df['end_frame'] = frame_from_pos(df['pos'])
            df['end'] = df['pos']
            return df

        def handle_stop(df):
            df['start_frame'] = frame_from_pos(df['pos'])
            df['end_frame'] = 0
            df['end'] = df['pos']
            return df

        def handle_shift(df):
            df['shift'] = (
                df['type']
                .str.extract(r'shift([+-]?\d+)')
                .astype(int)
            )
            df['start_frame'] = frame_from_pos(df['pos'])
            df['end'] = df['pos'] + df['shift']
            df['end_frame'] = frame_from_pos(df['end'])
            return df

        def handle_ires(df):
            df['start_frame'] = -1
            df['end_frame'] = frame_from_pos(df['pos'])
            df['end'] = df['pos']
            return df

        def handle_5cap(df):
            
            df['pos'] = -1
            df['start_frame'] = -1
            df['end_frame'] = 0
            df['end'] = 1
            
            return df

        # ---------- HANDLER REGISTRY ----------

        HANDLERS = {
            'init': handle_init,
            'stop': handle_stop,
            'ires': handle_ires,
            '5cap': handle_5cap,
        }

        # ---------- APPLY HANDLERS ----------

        for node_type, handler in HANDLERS.items():
            mask = event_map['type'] == node_type
            if mask.any():
                event_map.loc[mask, ['start_frame', 'end_frame', 'end']] = (
                    handler(event_map.loc[mask].copy())
                    [['start_frame', 'end_frame', 'end']]
                    .values
                )

        # Shift nodes handled separately due to regex-based type
        shift_mask = event_map['type'].str.contains('shift', na=False)
        if shift_mask.any():
            event_map.loc[shift_mask, ['start_frame', 'end_frame', 'end']] = (
                handle_shift(event_map.loc[shift_mask].copy())
                [['start_frame', 'end_frame', 'end']]
                .values
            )

        

        # ---------- HANDLE END NODE ----------

        end_pos = param_dict['transcript_length'] + 1 

        max_idx = event_map.index.max()

        terminal_nodes = pd.DataFrame(
            [
                [end_pos, 'end-scan0', 0, 1, 0, -1, end_pos],
                [end_pos, 'end-tran1', 0, 1, 1, -1, end_pos],
                [end_pos, 'end-tran2', 0, 1, 2, -1, end_pos],
                [end_pos, 'end-tran3', 0, 1, 3, -1, end_pos],
            ],
            columns=event_map.columns,
            index=range(max_idx + 1, max_idx + 5),
        )

        event_map = pd.concat([event_map, terminal_nodes])

        # ---------- FINAL CLEANUP ----------

        event_map[['pos', 'end', 'start_frame', 'end_frame']] = (
            event_map[['pos', 'end', 'start_frame', 'end_frame']].astype(int)
        )

        # Probability bookkeeping
        event_map['cont_prob'] = 1 - (event_map['prob'] + event_map['drop_prob'])


        event_map['pos'][event_map['type'] == '5cap'] = -1
        event_map['drop_prob'][event_map['type'] == '5cap'] = 0
        event_map['drop_prob'][event_map['type'] == 'ires'] = 0

        event_map = (
            event_map
            .sort_values('pos')
            .reset_index(drop=True)
            [['pos', 'end', 'type', 'start_frame', 'end_frame',
            'prob', 'drop_prob', 'cont_prob']]
        )

        return event_map


    def build_eventon_map(nodemap: pd.DataFrame) -> pd.DataFrame:
        """
        Split init / stop / shift nodes into begin and finish event-nodes
        and normalize event metadata in a single pass.
        """

        nodemap = nodemap.copy()

        # ---------- SELECT SPLITTABLE NODES ----------
        split_mask = (
            nodemap['type'].isin(['init', 'stop', '5cap', 'ires']) |
            nodemap['type'].str.contains('shift', na=False)
        )

        split_nodes = nodemap[split_mask].copy()
        keep_nodes = nodemap[~split_mask].copy()

        # ---------- BEGIN EVENTS ----------
        begin = split_nodes.assign(
            pos=split_nodes['pos'],
            start_frame=split_nodes['start_frame'],
            end_frame=split_nodes['end_frame'],
            phase='begin',
            prob=split_nodes['prob'],
            drop_prob=split_nodes['drop_prob'],
            cont_prob=split_nodes['cont_prob'],
        )

        # ---------- FINISH EVENTS ----------
        finish = split_nodes.assign(
            pos=split_nodes['end'],
            start_frame=split_nodes['end_frame'],
            end_frame=split_nodes['end_frame'],
            phase='finish',
            prob=split_nodes['prob'].where(split_nodes['prob'] == 0, 1),
            drop_prob=0,
            cont_prob=1,
        ).dropna(subset=['pos'])

        # ---------- EVENT CLASS ----------
        for df in (begin, finish, keep_nodes):
            df['event_class'] = df['type']
            df['event_class'] = df['event_class'].str.replace(
                r'/(begin|finish)$', '', regex=True
            )
            df['event_class'] = df['event_class'].str.replace(
                r'shift[+-]?\d+', 'shift', regex=True
            )
            df.loc[df['event_class'].str.startswith('end-', na=False),
                'event_class'] = 'end'

        # ---------- SHIFT AMOUNT ----------
        for df in (begin, finish):
            df['shift_amount'] = (
                df['type']
                .str.extract(r'shift([+-]?\d+)')[0]
                .astype(float)
                .fillna(0)
                .astype(int)
            )

        keep_nodes['shift_amount'] = 0
        keep_nodes['phase'] = None

        # ---------- COMBINE ----------
        out = pd.concat(
            [
                keep_nodes.drop(columns=['end'], errors='ignore'),
                begin.drop(columns=['end'], errors='ignore'),
                finish.drop(columns=['end'], errors='ignore'),
            ],
            ignore_index=True,
        )

        # ---------- NODE IDENTIFIER ----------
        out.loc[(out['event_class'] == 'ires') &
                (out['phase'] == 'begin'), 'pos'] = -1
        out['node'] = list(zip(out['pos'], out['start_frame']))

        return out



    def advance_position(pos, event_class, phase, shift_amount, cont):
        """
        Apply position change caused by shift events.
        """
        if event_class == 'shift' and phase == 'begin' and not cont:
            return pos + shift_amount
        return pos


    def find_next_node(node_map, pos, frame, node):
        """
        Find the next node matching position and frame.
        """
        candidates = node_map[
            (node_map['pos'] >= pos) &
            (node_map['start_frame'] == frame) &
            (node_map['node'] != node) 
            ]

        if candidates.empty:
            return (None,None)

        return candidates.loc[candidates['pos'].idxmin(), 'node']


    # cont == True / False → (event_class, phase) → state
    STATE_TRANSITIONS = {
        True: {
            ('init',  'begin'):  lambda f: 'scan',
            ('init',  'finish'): lambda f: f'tran{f}',

            ('shift', 'begin'):  lambda f: f'tran{f}',
            ('shift', 'finish'): lambda f: f'tran{f}',

            ('stop',  'begin'):  lambda f: f'tran{f}',
            ('stop',  'finish'): lambda f: 'scan',

            ('ires',  'begin'):  lambda f: f'rcyc{f}',
            ('ires',  'finish'): lambda f: f'tran{f}',

            ('5cap',  'begin'):  lambda f: f'rcyc{f}',
            ('5cap',  'finish'): lambda f: 'scan',
        },

        False: {
            ('init',  'begin'):  lambda f: 'init',
            ('init',  'finish'): lambda f: f'tran{f}',

            ('shift', 'begin'):  lambda f: 'shift',
            ('shift', 'finish'): lambda f: f'tran{f}',

            ('stop',  'begin'):  lambda f: 'rein',
            ('stop',  'finish'): lambda f: 'scan',

            ('ires',  'begin'):  lambda f: f'load',
            ('ires',  'finish'): lambda f: f'tran{f}',

            ('5cap',  'begin'):  lambda f: f'load',
            ('5cap',  'finish'): lambda f: 'scan',
        },
        }


    def resolve_state(event_class, phase, frame, cont):
        """
        Resolve ribosome state after a transition.
        """
        table = STATE_TRANSITIONS[cont]

        # Try exact match
        key = (event_class, phase)
        if key in table:
            return table[key](int(frame))

        # Fallback for phase-less events
        key = (event_class, None)
        if key in table:
            return table[key](int(frame))

        # Default safety fallback
        return 'scan'


    def next_node(node, pos, frame, prob, node_map: pd.DataFrame, event_row, cont: bool):
        """
        Determine the next node and ribosome state.
        """
        if prob == 0:
            return pd.Series([None, None])

        # ---- advance position ----
        pos = advance_position(
            pos,
            event_row['event_class'],
            event_row['phase'],
            event_row['shift_amount'],
            cont
        )

        # ---- find next node ----
        next_node_id = find_next_node(
            node_map,
            pos,
            frame,
            node
        )

        if next_node_id == (None, None):
            return pd.Series([None, None])

        # ---- resolve state ----
        state = resolve_state(
            event_row['event_class'],
            event_row['phase'],
            frame,
            cont,
        )

        return pd.Series([next_node_id, state])



    def prune_finish(graph: pd.DataFrame) -> pd.DataFrame:
        df_fixed = graph.copy()
        df_fixed["change"] = df_fixed["change"].astype(object)
        df_fixed["cont"] = df_fixed["cont"].astype(object)

        # All changes from begin phase
        begin_changes = set(df_fixed.loc[df_fixed['phase'] == 'begin', 'change'])

        # Finish rows whose node does NOT match any begin change
        invalid_finish_mask = (
            (df_fixed['phase'] == 'finish') &
            (~df_fixed['node'].isin(begin_changes))
        )
        invalid_finish = df_fixed[invalid_finish_mask]

        # Rewire parent → child
        for _, row in invalid_finish.iterrows():
            finish_node = row['node']
            child_node = row['cont'] if row['cont'] not in [None, (-1,-1)] else row['change']
            if child_node in [None, (-1,-1)]:
                continue

            # Rewire references in change column
            for i in df_fixed.index:
                if df_fixed.at[i, 'change'] == finish_node:
                    df_fixed.at[i, 'change'] = child_node

            # Rewire references in cont column
            for i in df_fixed.index:
                if df_fixed.at[i, 'cont'] == finish_node:
                    df_fixed.at[i, 'cont'] = child_node

        # Drop invalid finish nodes
        df_fixed = df_fixed.loc[~invalid_finish_mask].reset_index(drop=True)
        return df_fixed


    def prune_parentless(graph: pd.DataFrame) -> pd.DataFrame:
        """
        Remove nodes that are not referenced by any change, cont, or drop edge.
        """
        out = graph.copy()

        # Flatten all referenced nodes into a set
        referenced_nodes = set()
        for col in ['change', 'cont', 'drop']:
            referenced_nodes.update(out[col].dropna().values)

        # Keep only nodes that are referenced
        out = out[out['node'].isin(referenced_nodes)].reset_index(drop=True)

        return out


    def ribograph(node_map: pd.DataFrame) -> pd.DataFrame:
        """
        Construct a ribosome transition graph from a node map.
        """

        graph = node_map.copy()

        # ---------- CHANGE EDGES (state-changing transitions) ----------
        graph[['change', 'change_type']] = graph.apply(
            lambda row: next_node(
                pos=row['pos'],
                node=row['node'],
                frame=row['end_frame'],
                prob=row['prob'],
                node_map=node_map,
                event_row=row,
                cont=False
            ),
            axis=1
        )

        # ---------- CONTINUATION EDGES ----------
        graph[['cont', 'cont_type']] = graph.apply(
            lambda row: next_node(
                pos=row['pos'],
                node=row['node'],
                frame=row['start_frame'],
                prob=row['cont_prob'],
                node_map=node_map,
                event_row=row,
                cont=True
            ),
            axis=1
        )

        # ---------- DROP EDGES ----------
        graph['drop'] = graph['drop_prob'].apply(
            lambda p: (-1, -1) if p > 0 else None
        )
        graph['drop_type'] = 'drop'

        # ---------- BULK NODE ----------
        bulk_row = {
            'pos': -1,
            'type': 'bulk',
            'start_frame': -1,
            'end_frame': -1,
            'prob': 1,
            'drop_prob': 0,
            'cont_prob': 0,
            'node': (-1, -1),
            'event_class': 'bulk',
            'phase': None,
            'shift_amount': 0,
            'change': None,
            'change_type': None,
            'cont': None,
            'cont_type': None,
            'drop': None,
            'drop_type': None,
        }

        graph = pd.concat([graph, pd.DataFrame([bulk_row])], ignore_index=True)

        # ---------- PRUNE UNREACHABLE NODES ----------
        referenced_nodes = set(
            graph[['change', 'cont', 'drop']]
            .values
            .flatten()
        )

        graph = graph[
            graph['node'].isin(referenced_nodes)
        ]

        graph = prune_finish(graph=graph)
        
        graph = prune_parentless(graph=graph)
        return graph.reset_index(drop=True)



    def table2edgelist(graph: pd.DataFrame) -> pd.DataFrame:
        """
        Convert graph table into an edge list.
        """

        EDGE_SPECS = [
            ('change', 'prob', 'change_type'),
            ('cont',   'cont_prob', 'cont_type'),
            ('drop',   'drop_prob', 'drop_type'),
        ]

        edges = []

        for target_col, weight_col, state_col in EDGE_SPECS:
            df = graph[['node', target_col, weight_col, state_col]].dropna().copy()
            df.columns = ['source', 'target', 'weight', 'state']
            edges.append(df)

        return (
            pd.concat(edges, ignore_index=True)
            .drop_duplicates()
        )


    def get_ribopaths(graph: nx.DiGraph, node_map: pd.DataFrame) -> List[List[Tuple]]:
        """
        Find all ribosome paths from load/ires nodes to the bulk node (-1,-1).

        Args:
            graph: networkx.DiGraph representing ribosome transitions.
            node_map: DataFrame containing node info (must have 'type' and 'node' columns).

        Returns:
            List of paths, each path is a list of edge tuples (source, target).
        """
        all_paths = []

        # source nodes: 'load' or 'ires'
        source_nodes = graph.successors((-1,-1))


        target_node = (-1, -1)  # bulk node
        for source in source_nodes:
            paths = nx.all_simple_edge_paths(graph, source=source, target=target_node)
            for path in paths:
                path.insert(0, ((-1,-1), path[0][0]))
                all_paths.extend([path])

        return all_paths



    def calc_flux(graph: nx.DiGraph, edgelist: pd.DataFrame, edge_paths: List[List[Tuple]]) -> pd.DataFrame:
        """
        Calculate flux for each edge based on ribosome paths.

        Args:
            graph: nx.DiGraph with edge attribute 'weight'.
            edgelist: DataFrame with columns ['node', 'target', 'weight', 'state'].
            edge_paths: List of paths, each a list of edge tuples (source, target).

        Returns:
            DataFrame with columns ['node','target','weight','state','flux'].
        """

        out = edgelist.copy()
        out['edge_tuple'] = list(zip(out['source'], out['target']))

        # Initialize flux accumulator
        flux_accum = pd.Series(0, index=out.index)
        for path in edge_paths:
            # Calculate path flux (product of edge weights along path)
            path_flux = graph[path[0][0]][path[0][1]]['weight']
            for edge in path[1:]:
                path_flux *= graph[edge[0]][edge[1]]['weight']

            # Map this path flux to the edges in the edgelist
            mask = out['edge_tuple'].isin(path)
            flux_accum[mask] += path_flux

        out['flux'] = flux_accum
        return out[['source', 'target', 'weight', 'state', 'flux']]


    def build_flux_graph(
        prob_graph: nx.DiGraph,
        edgelist: pd.DataFrame,
        ribopaths: List[List[Tuple]],
    ) -> Tuple[pd.DataFrame, nx.DiGraph]:
        """
        Calculate ribosome flux along edges and generate a weighted DiGraph.

        Args:
            prob_graph: Original probabilistic DiGraph.
            edgelist: DataFrame with edges ['node','target','weight','state'].
            ribopaths: List of paths (list of edge tuples) through the graph.
            csv_path: Optional path to save the fluxed edge table as CSV.

        Returns:
            flux_edges: DataFrame with additional 'flux' column.
            flux_graph: NetworkX DiGraph with edge attributes ['state','weight','flux'].
        """
        # Compute flux along all paths
        flux_edges = calc_flux(prob_graph, edgelist, ribopaths).copy()

        # Standardize column names
        flux_edges.columns = ['source', 'target', 'weight', 'state', 'flux']
        
        # Build DiGraph with edge attributes
        flux_graph = nx.from_pandas_edgelist(
            flux_edges,
            source='source',
            target='target',
            edge_attr=['state', 'weight', 'flux'],
            create_using=nx.DiGraph
        )


        return flux_edges, flux_graph


    def get_sister_edge(edge: pd.Series, edgelist: pd.DataFrame) -> pd.DataFrame:
        """Return sister edges sharing the same source but different target."""
        source = edge['source']
        target = edge['target']
        return edgelist[(edgelist['source'] == source) & (edgelist['target'] != target)]


    def get_parent_edges(edge: pd.Series, edgelist: pd.DataFrame) -> pd.DataFrame:
        """Return edges whose target is the source of the current edge."""
        source = edge['source']
        return edgelist[edgelist['target'] == source]



    color_map = {
        "scan": "grey",
        "tran1" : 'lightblue',
        "tran2" : "blue",
        "tran3" : 'darkblue',
        "init" : "green",
        "drop" : "purple",
        'rein': 'orange',
        'shift': 'red',
        'load': 'purple'
    }

    COLOURS = {
        "grey":   (0.6, 0.6, 0.6),
        "green":  (0.2, 0.7, 0.3),
        "blue":   (0.1, 0.1, 0.9),
        "purple": (0.6, 0.3, 0.7),
        'orange': (1, 0.6, 0 ),
        'darkblue': (0,0,0.4),
        'lightblue': (0.6,0.6,0.9),
        'red': (0.9,0.1,0.1)
    }



    def add_vert_category(edgelist: pd.DataFrame, log_reduction: float = 1.5, height_scale: float = 2) -> pd.DataFrame:
        """
        Add vertical direction and log-scaled x positions to the edge list.

        Args:
            edgelist: DataFrame with 'source' and 'target' as stringified tuples.
            log_reduction: Base for log scaling gaps in x positions.

        Returns:
            DataFrame with updated 'source_x', 'source_y', 'target_x', 'target_y', 'vert'.
        """

        out = edgelist.copy()

        # # --- Parse source/target coordinates ---
        # out['source'] = out['source'].apply(ast.literal_eval)
        # out['target'] = out['target'].apply(ast.literal_eval)

        # --- Split into x/y columns ---
        for point in ['source', 'target']:
            out[f'{point}_x'] = out[point].apply(lambda v: int(v[0]))
            out[f'{point}_y'] = out[point].apply(lambda v: int(v[1]))


        # --- Adjust negative target_y values ---
        mask_neg = out['source_y'] < 0
        out.loc[mask_neg, 'source_y'] = out.loc[mask_neg, 'target_y']

        # --- Adjust negative target_y values ---
        mask_neg = out['source_x'] < 0
        out.loc[mask_neg, 'source_x'] = out.loc[mask_neg, 'target_x'] - 1

        # --- Adjust negative target_y values ---
        mask_neg = out['target_x'] < 0
        out.loc[mask_neg, 'target_x'] = out.loc[mask_neg, 'source_x']

        if log_reduction != 1: #SCALE GRAPH BASED ON LOG VALUES OF DISTANCES
            # --- Compute unique sorted x positions ---
            x_pos = sorted(set(list(out['source_x']) + [out['target_x'].iloc[-1]]))

            # --- Compute gaps and log-scale ---
            gaps = [x_pos[i+1] - x_pos[i] for i in range(len(x_pos)-1)]
            log_gaps = [math.log(g, log_reduction) + 1 for g in gaps]

            # --- Map original x to log-scaled x ---
            log_x_pos = [x_pos[0]]
            for gap in log_gaps:
                log_x_pos.append(log_x_pos[-1] + gap)


            log_map = dict(zip(x_pos, log_x_pos))

            
            out['source_x'] = out['source_x'].map(log_map)
            out['target_x'] = out['target_x'].map(log_map)



        # --- Compute vertical direction (+1/-1), 0 if same ---
        delta_y = out['target_y'] - out['source_y']
        out['vert'] = (delta_y / delta_y.abs()).fillna(0).astype(int)

        negatives = out[out['target'] == (-1,-1)]
        for idx, edge in negatives.iterrows():
            sibling = get_sister_edge(edge, edgelist=out)
            if sibling.empty:
                out.loc[idx, 'target_y'] = out.loc[idx, 'source_y'] + 1
            else:
                if sibling['vert'].values[0] == 1:
                    out.loc[idx, 'target_y'] = out.loc[idx, 'source_y'] - 1
                    continue
                out.loc[idx, 'target_y'] = out.loc[idx, 'source_y'] + 1
            
            # --- Compute vertical direction (+1/-1), 0 if same ---
        delta_y = out['target_y'] - out['source_y']
        out['vert'] = (delta_y / delta_y.abs()).fillna(0).astype(int)
        
        # --- Scale y coordinates ---
        out['source_y'] *= height_scale
        out['target_y'] *= height_scale

        out = out.sort_values(by='source_x', axis=0)

        return out


    def assign_vertical_positions(edgelist: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 'bot' and 'top' positions for horizontal edges (vert == 0),
        based on parent edges, sister edges, and flux.
        """

        out = edgelist.copy()
        out['bot'] = np.nan
        out['top'] = np.nan

        horizontal_edges = out[out['vert'] == 0]

        for idx, edge in horizontal_edges.iterrows():
            if edge['state'] == 'load':
                out.at[idx, 'bot'] = edge['source_y']
                out.at[idx, 'top'] = out.at[idx, 'bot'] + out.at[idx, 'flux']
                continue
            parent_edges = get_parent_edges(edge, out)
            num_parents = len(parent_edges)

            if num_parents == 0:
                out.at[idx, 'bot'] = edge['source_y']

            elif num_parents == 1:
                parent_vert = parent_edges['vert'].values[0]

                if parent_vert != 0:
                    out.at[idx, 'bot'] = edge['source_y']
                else:
                    sister = get_sister_edge(edge, out)
                    if len(sister) == 0:
                        out.at[idx, 'bot'] = parent_edges['bot'].values[0]
                        out.at[idx, 'top'] = out.at[idx, 'bot'] + out.at[idx, 'flux']
                        continue
                    if len(sister) == 1:
                        sister_vert = sister['vert'].values[0]
                        if sister_vert == 1:
                            out.at[idx, 'bot'] = parent_edges['bot'].values[0]
                        elif sister_vert == -1:
                            out.at[idx, 'bot'] = parent_edges['bot'].values[0] + \
                                                    parent_edges['flux'].values[0] - \
                                                    edge['flux']
                    else:
                        out.at[idx, 'bot'] = parent_edges['bot'].values[0] + \
                            parent_edges['flux'].values[0] - \
                            edge['flux'] - sister.loc[sister['vert']==1]['flux']


            elif num_parents == 2:
                nonzero_vert = parent_edges[parent_edges['vert'] != 0]['vert'].values[0]
                zero_vert_parent = parent_edges[parent_edges['vert'] == 0]

                if nonzero_vert == 1:

                    out.at[idx, 'bot'] = zero_vert_parent['bot'].values[0] + \
                                            zero_vert_parent['flux'].values[0] - \
                                            edge['flux']
                else:
                    out.at[idx, 'bot'] = zero_vert_parent['bot'].values[0]

            # Compute top based on flux
            out.at[idx, 'top'] = out.at[idx, 'bot'] + out.at[idx, 'flux']

        return out


    def get_child_edges(edge: pd.Series, edgelist: pd.DataFrame) -> pd.DataFrame:
        """Return edges whose source is the current edge's target."""
        return edgelist[edgelist['source'] == edge['target']]


    def assign_vertical_edges(edgelist: pd.DataFrame, drop_length, height_scale) -> pd.DataFrame:
        """
        Assign bot/top/left/right positions for vertical edges (vert != 0).
        """

        out = edgelist.copy()
        out[['top_left', 'bot_left']] = np.nan

        vertical_edges = out[out['vert'] != 0]

        for idx, edge in vertical_edges.iterrows():
            half_flux = 0.5 * edge['flux']
            parents = get_parent_edges(edge, out)
            children = get_child_edges(edge, out)

            source_left  = edge['source_x'] - half_flux
            target_left  = edge['target_x'] - half_flux

            # # ---- UPWARD EDGE ----
            if edge['vert'] == 1:
                if parents.empty or children.empty:
                    continue

                parent = parents.iloc[0]
                child = children.iloc[0]
                if edge['state'] == 'drop':
                    out.at[idx, 'bot'] = parent['top']
                    out.at[idx, 'top'] = out.at[idx, 'bot'] + drop_length * height_scale
                else:
                    out.at[idx, 'bot'] = parent['bot'] + parent['flux']
                    out.at[idx, 'top'] = child['bot']
                #corners of the parallelagram
                out.at[idx, 'bot_left']  =  source_left 
                out.at[idx, 'top_left']  =  target_left 

            # ---- DOWNWARD EDGE ----
            elif edge['vert'] == -1:
                if parents.empty or children.empty:
                    continue

                parent = parents.iloc[0]
                child = children.iloc[0]
                if edge['state'] == 'drop':
                    out.at[idx, 'top'] = parent['bot']
                    out.at[idx, 'bot'] = out.at[idx, 'top'] - drop_length * height_scale
                else:
                    out.at[idx, 'top'] = parent['bot']
                    out.at[idx, 'bot'] = child['bot'] + child['flux']
                
                out.at[idx, 'top_left']  =  source_left 
                out.at[idx, 'bot_left']  =  target_left 


        return out





    def adjust_vertical(edgelist: pd.DataFrame):
        out = edgelist.copy()

        equal_mask = out['top_left'].eq(out['bot_left'])
        vert_mask = out['vert'] != 0

        # candidate alignment value
        out["left_value"] = np.select(
            [out["vert"] == 1, out["vert"] == -1],
            [out["bot_left"], out["top_left"]],
            default=np.nan
        )

        # per-source minimum
        out.loc[vert_mask, 'left_value'] = (
            out.loc[vert_mask]
            .groupby('source')['left_value']
            .transform('min')
        )

        # case 1: originally unequal → adjust only one side
        out['top_left'] = np.select(
            [(out['vert'] == -1) & (~equal_mask)],
            [out['left_value']],
            default=out['top_left']
        )

        out['bot_left'] = np.select(
            [(out['vert'] == 1) & (~equal_mask)],
            [out['left_value']],
            default=out['bot_left']
        )

        # case 2: originally equal → move both together
        both_mask = equal_mask & vert_mask
        out.loc[both_mask, 'top_left'] = out.loc[both_mask, 'left_value']
        out.loc[both_mask, 'bot_left'] = out.loc[both_mask, 'left_value']

        # recompute right values
        out['top_right'] = out['top_left'] + out['flux']
        out['bot_right'] = out['bot_left'] + out['flux']

        return out.drop(columns='left_value')



    import numpy as np
    import pandas as pd

    def assign_horizontal_edges(edgelist: pd.DataFrame, color_map: dict) -> pd.DataFrame:
        """
        Assign left/right positions for horizontal edges (vert == 0).
        """

        out = edgelist.copy()
        horizontal = out[out['vert'] == 0]

        for idx, edge in horizontal.iterrows():
            parents = get_parent_edges(edge, out)
            children = get_child_edges(edge, out)

            # ------------------
            # LEFT coordinate
            # ------------------
            if parents.empty:
                continue
                left = edge['source_x']
                out.loc[idx, ['top_left', 'bot_left']] = left
                out.loc[idx, ['top_right', 'bot_right']] = right
                
            elif len(parents) == 1:

                left = parents.iloc[0]['bot_right']

            else:
                # Multiple parents: choose vertical parent
                vert_parent = parents[parents['vert'] != 0]
                if not vert_parent.empty:
                    if vert_parent['vert'].values[0] == 1:
                        left = vert_parent.iloc[0]['top_right']
                    elif vert_parent['vert'].values[0] == -1:
                        left = vert_parent.iloc[0]['bot_right']
                else:
                    left = parents.iloc[0]['bot_right']

            # ------------------
            # RIGHT coordinate
            # ------------------
            if not children.empty and (children['vert'] != 0).any():
                vert_child = children[children['vert'] != 0]
                if vert_child.iloc[0]['vert'] == 1:
                    right = vert_child.iloc[0]['bot_left']
                else:
                    right = vert_child.iloc[0]['top_left']

            else:
                # Look at child's vertical parent
                if not children.empty:
                    child = children.iloc[0]
                    child_parents = get_parent_edges(child, out)
                    vert_child_parent = child_parents[child_parents['vert'] != 0]

                    if not vert_child_parent.empty:
                        if vert_child_parent.iloc[0]['vert'] == 1:
                            right = vert_child_parent.iloc[0]['top_right']
                        else:
                            right = vert_child_parent.iloc[0]['bot_right']
                    else:
                        right = edge['target_x']
                else:
                    right = np.nan

            if edge['state'] == 'load':
                left = edge['source_x']
            out.loc[idx, ['top_left', 'bot_left']] = left
            out.loc[idx, ['top_right', 'bot_right']] = right

        # ------------------
        # Assign colors
        # ------------------
        out['colour'] = out['state'].map(color_map)


        return out



    def rectangles_to_circles(rectangles: pd.DataFrame) -> pd.DataFrame:
        """
        Convert rectangle edges into circles for plotting.
        Each rectangle can generate multiple circles based on corners and child edges.
        """
        circles = []

        for _, rect in rectangles.iterrows():
            # Dictionary of corners
            corners = {
                'top_left': (rect['top_left'], rect['top']),
                'top_right': (rect['top_right'], rect['top']),
                'bot_left': (rect['bot_left'], rect['bot']),
                'bot_right': (rect['bot_right'], rect['bot'])
            }

            # Base circle: pick a primary corner depending on vert
            if rect['vert'] == 1:
                primary_corner = 'bot_left'
                primary_quarter = 1
                secondary_corner = 'top_right'
                secondary_quarter = 3
            elif rect['vert'] == -1:
                primary_corner = 'top_left'
                primary_quarter = 4
                secondary_corner = 'bot_right'
                secondary_quarter = 2
            else:
                continue

            # Main circle
            circles.append({
                'centre_x': corners[primary_corner][0],
                'centre_y': corners[primary_corner][1],
                'radius': rect['flux'],
                'quarter': primary_quarter,
                'colour': rect['colour']
            })


            if rect['state'] == 'drop':
                continue
            # If rectangle has children, add secondary circle
            children = get_child_edges(rect, rectangles)
            if len(children) > 0:
                child_colour = children['colour'].values[0]

                circles.append({
                    'centre_x': corners[secondary_corner][0],
                    'centre_y': corners[secondary_corner][1],
                    'radius': rect['flux'],
                    'quarter': secondary_quarter,
                    'colour': child_colour
                })

        return pd.DataFrame(circles).dropna()



    def compute_triangle_from_line(source, target, width, height, vert, base_at_end=False ):
        """
        Compute triangle vertices pointing from source -> target.
        Tip is guaranteed to lie ON the source-target line.
        """
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        length = np.hypot(dx, dy)

        if length == 0:
            return source[0], source[1], source[0], source[1], source[0], source[1]

        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        # Base position
        if base_at_end:
            base_x = target[0]
            base_y = target[1]
            tip_x, tip_y = base_x, base_y + height * vert
        else:
            mid_x = (source[0] + target[0]) / 2
            mid_y = (source[1] + target[1]) / 2
            base_x = mid_x - ux * height / 2
            base_y = mid_y - uy * height / 2
            tip_x = mid_x + ux * height / 2
            tip_y = mid_y + uy * height / 2

        half_w = width / 2

        left_x = base_x + px * half_w
        left_y = base_y + py * half_w
        right_x = base_x - px * half_w
        right_y = base_y - py * half_w

        return tip_x, tip_y, left_x, left_y, right_x, right_y



    def rectangles_to_triangles(rects: pd.DataFrame) -> pd.DataFrame:
        triangles = []

        for _, r in rects.iterrows():



            # Special case: load (flat arrows)
            if r.state == 'load':
                for y_tip, y_base, x_left, x_right in [
                    (r.top + r.flux / 2, r.top, r.top_left, r.top_right),
                    (r.bot - r.flux / 2, r.bot, r.bot_left, r.bot_right)
                ]:
                    triangles.append({
                        'tip_x': x_left,
                        'tip_y': y_tip,
                        'left_x': x_left,
                        'left_y': y_base,
                        'right_x': x_right,
                        'right_y': y_base,
                        'colour': r.colour
                    })

                
                continue

            if r.vert == 0:
                continue

            # Triangle size
            width = max(r.flux * 2, 0.4)
            height = width * 0.8

            # Rectangle centerline
            top_center = (
                (r.top_left + r.top_right) / 2,
                r.top
            )
            bottom_center = (
                (r.bot_left + r.bot_right) / 2,
                r.bot
            )

            # Direction from vert
            if r.vert < 0:
                source, target = top_center, bottom_center
            else:
                source, target = bottom_center, top_center

            base_at_end = (r.state == 'drop')
            if not base_at_end:
                continue

            tip_x, tip_y, left_x, left_y, right_x, right_y = compute_triangle_from_line(
                source,
                target,
                width,
                height,
                r.vert,
                base_at_end=base_at_end,
            )

            triangles.append({
                'tip_x': tip_x,
                'tip_y': tip_y,
                'left_x': left_x,
                'left_y': left_y,
                'right_x': right_x,
                'right_y': right_y,
                'colour': r.colour
            })

        return pd.DataFrame(triangles)



    def calc_bounds(rects, circles, triangles, px_per_unit = 100):
        x_min = min(rects['top_left'].min(), circles['centre_x'].min(), triangles[['tip_x', 'left_x', 'right_x']].min().min())
        x_max = max(rects['top_right'].max(), circles['centre_x'].max(), triangles[['tip_x', 'left_x', 'right_x']].max().max())
        y_min = min(rects['bot'].min(), circles['centre_y'].min(), triangles[['tip_y', 'left_y', 'right_y']].min().min())
        y_max = max(rects['top'].max(), circles['centre_y'].max(), triangles[['tip_y', 'left_y', 'right_y']].max().max())
        


        WIDTH = int((x_max - x_min) * px_per_unit)
        HEIGHT = int((y_max - y_min) * px_per_unit)

        # Scale factors
        x_scale = WIDTH / (x_max - x_min)
        y_scale = HEIGHT / (y_max - y_min)
        scale = min(x_scale, y_scale)  # uniform scale to preserve aspect ratio

        # Margins to center the drawing
        x_margin = (WIDTH - scale * (x_max - x_min)) / 2
        y_margin = (HEIGHT - scale * (y_max - y_min)) / 2

        scale_dict = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'WIDTH': WIDTH,
            'HEIGHT': HEIGHT,
            'scale': scale,
            'x_margin': x_margin,
            'y_margin': y_margin

        }

        return scale_dict

    # -----------------------
    # Coordinate transforms
    # -----------------------
    def sx(x, scale_dict):
        """Transform data x-coordinate to pixel x-coordinate."""
        return scale_dict['x_margin'] + (x - scale_dict['x_min']) * scale_dict['scale']

    def sy(y, scale_dict):
        """Transform data y-coordinate to pixel y-coordinate (y-axis inverted)."""
        return scale_dict['HEIGHT'] - (scale_dict['y_margin'] + (y - scale_dict['y_min']) * scale_dict['scale'])



    def draw_rect(ctx, rects, scale_dict):
        for _, r in rects.iterrows():
            # Rectangle
            x0, y0 = sx(r.top_left, scale_dict),  sy(r.top, scale_dict)
            x1, y1 = sx(r.top_right, scale_dict), sy(r.top, scale_dict)
            x2, y2 = sx(r.bot_right, scale_dict), sy(r.bot, scale_dict)
            x3, y3 = sx(r.bot_left, scale_dict),  sy(r.bot, scale_dict)

            ctx.move_to(x0, y0)
            ctx.line_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.line_to(x3, y3)
            ctx.close_path()

            ctx.set_source_rgba(*COLOURS[r.colour], 1)
            ctx.fill()


    def draw_triangles(ctx, triangles, scale_dict):
        for _, t in triangles.iterrows():
            ctx.new_path()
            ctx.move_to(sx(t.tip_x, scale_dict), sy(t.tip_y, scale_dict))
            ctx.line_to(sx(t.left_x, scale_dict), sy(t.left_y, scale_dict))
            ctx.line_to(sx(t.right_x, scale_dict), sy(t.right_y, scale_dict))
            ctx.close_path()
            ctx.set_source_rgba(*COLOURS[t.colour], 1)
            ctx.fill()


    def draw_arcs(ctx, circles, scale_dict):
        QUARTER_ANGLES = {
            1: (0, math.pi/2),
            2: (math.pi/2, math.pi),
            3: (math.pi, 3*math.pi/2),
            4: (3*math.pi/2, 2*math.pi),
        }

        for _, a in circles.iterrows():
            cx = sx(a.centre_x, scale_dict)
            cy = sy(a.centre_y, scale_dict)
            r  = sx(a.centre_x + a.radius, scale_dict) - sx(a.centre_x, scale_dict)

            start, end = QUARTER_ANGLES[a.quarter]

            ctx.move_to(cx, cy)
            ctx.arc(cx, cy, r, start, end)
            ctx.close_path()

            ctx.set_source_rgba(*COLOURS[a.colour], 1)
            ctx.fill()
            # ctx.set_source_rgb(0, 0, 0)
            # ctx.stroke()


    transcript_map = pd.DataFrame(map_dict)
    
    nodemap = map2nodes(transcript_map=transcript_map)
    

    eventon_map = build_eventon_map(nodemap)
    
    graph = ribograph(node_map=eventon_map)
    
    edgelist = table2edgelist(graph=graph)
    

    prob_graph = nx.from_pandas_edgelist(edgelist, edge_attr=['weight', 'state'], create_using=nx.DiGraph)
    
    paths = get_ribopaths(prob_graph, node_map=graph)
    
    flux_edges, flux_graph = build_flux_graph(prob_graph, edgelist, paths)
    

    vertified = add_vert_category(edgelist=flux_edges, log_reduction=param_dict['log_reduction'], height_scale=param_dict['height_scale'])
    

    vert_pos = assign_vertical_positions(vertified)
    
    vert_all = assign_vertical_edges(vert_pos, drop_length=param_dict['drop_length'], height_scale=param_dict['height_scale'])
    adjusted = adjust_vertical(vert_all)

    rects = assign_horizontal_edges(adjusted, color_map=color_map)

    circles = rectangles_to_circles(rectangles=rects)

    triangles = rectangles_to_triangles(rects=rects)

    scale_dict = calc_bounds(rects=rects, triangles=triangles, circles=circles)

    buffer = io.BytesIO()

    surface = cairo.SVGSurface(buffer, scale_dict['WIDTH'], scale_dict['HEIGHT'])
    ctx = cairo.Context(surface)

    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    draw_rect(ctx=ctx, rects=rects, scale_dict=scale_dict)
    draw_arcs(ctx=ctx, circles=circles, scale_dict=scale_dict)
    draw_triangles(ctx=ctx, triangles=triangles, scale_dict=scale_dict)

    surface.finish()

    svg_bytes = buffer.getvalue()
    svg_text = svg_bytes.decode("utf-8")

    return svg_text