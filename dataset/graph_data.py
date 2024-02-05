import itertools

import numpy as np
import paddle
import paddle.distributed as dist
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from dataset.data_utils import haversine

try:
    import hssinfo
except:
    pass


def convert_origin_graph_files(
    graph_path,
    save_graph_path,
    node_nums,
    src_name="src",
    dst_name="to",
    distance_name="cost",
):
    # Read the graph data from the CSV file
    dataframe = pd.read_csv(graph_path)

    # Extract the source, destination, and distance values from the dataframe
    src = dataframe[src_name].values.tolist()
    dst = dataframe[dst_name].values.tolist()
    distance = dataframe[distance_name].values.tolist()

    # Create an empty array to represent the graph
    array = np.zeros([node_nums, node_nums])

    # Populate the array with the distances between nodes
    for i in range(len(src)):
        array[src[i]][dst[i]] = distance[i]
        array[dst[i]][src[i]] = distance[i]

    # Create lists to store the source indices, destination indices, and edge weights
    edge_src_idx, edge_dst_idx, edge_weights = [], [], []

    # Iterate over all possible pairs of nodes
    for i, j in itertools.product(range(node_nums), range(node_nums)):
        # If there is a distance between the nodes and i <= j
        if array[i][j] > 0 and i <= j:
            # Add the source index, destination index, and edge weight to the respective lists
            edge_src_idx.append(i)
            edge_dst_idx.append(j)
            edge_weights.append(array[i][j])

    # Normalize the edge weights to a range of [0.1, 1]
    edge_weights = [-w for w in edge_weights]
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    print(graph_path, max_weight, min_weight)
    if max_weight > min_weight:
        edge_weights = [
            (w - min_weight) * 0.9 / (max_weight - min_weight) + 0.1
            for w in edge_weights
        ]
    else:
        edge_weights = [1 for w in edge_weights]

    # Create a new dataframe with the source indices, destination indices, and normalized edge weights
    dataframe = pd.DataFrame(
        {
            "src": edge_src_idx,
            "dst": edge_dst_idx,
            "weight": edge_weights,
        }
    )

    # Save the new graph data to a CSV file
    dataframe.to_csv(save_graph_path, index=False)


class SpatialGraph:
    def __init__(self, args, build=True):
        """_summary_

        Args:
            args (_type_): args
            node_df (pd.DataFrame): the node dataframe is a csv file
                                        with the following format:
            ID,dist,lat,lon
            308511,3,38.761062,-120.569835
            308512,3,38.761182,-120.569866
            311831,3,38.409253,-121.48412
            311832,3,38.409782,-121.48468
        """
        self.args = args
        self.node_nums = self.args.num_nodes

        if build:
            self.node_df = pd.read_csv(args.node_path)
            self.edge_src_idx = []
            self.edge_dst_idx = []
            self.edge_weights = []
            self.build_graph()
            self.save_graph()
        else:
            self.load_graph()

        self.build_laplacian_matrix()
        self.chebyshev_matrix = self.build_chebyshev_polynomials(k=3)

    def build_graph(self):
        """build graph according to the node dataframe"""
        self.node_nums = len(self.node_df)
        lon = self.node_df["lon"].values
        lat = self.node_df["lat"].values

        node_distances = {}
        for i in range(self.node_nums):
            # print(i, flush=True)
            row_i = self.node_df.iloc[i]
            # haversine method to calculate the distance
            distance = haversine(row_i["lon"], row_i["lat"], lon, lat)
            node_distances[i] = distance

        for id, distance in node_distances.items():
            if len(distance) == 0:
                continue
            topk_indices = np.argpartition(distance, self.args.node_top_k)
            topk_indices = topk_indices[: self.args.node_top_k]

            # build undirected graph. from node with small num to node with big num
            for k in topk_indices:
                if id <= k and distance[k] < self.args.node_max_dis:
                    self.edge_src_idx.append(id)
                    self.edge_dst_idx.append(k)
                    self.edge_weights.append(distance[k])

        # Normalize the weights
        self.edge_weights = [-w for w in self.edge_weights]
        max_weight = max(self.edge_weights)
        min_weight = min(self.edge_weights)
        self.edge_weights = [
            (w - min_weight) * 0.9 / (max_weight - min_weight) + 0.1
            for w in self.edge_weights
        ]

    def build_laplacian_matrix(self):
        # self.matrix_sparse = sparse.csc_matrix(
        #     (self.edge_weights, (self.edge_src_idx, self.edge_dst_idx)),
        #     shape=(self.node_nums, self.node_nums),
        # )
        self.matrix_sparse = sparse.csc_matrix(
            (
                np.ones_like(np.array(self.edge_weights)),
                (self.edge_src_idx, self.edge_dst_idx),
            ),
            shape=(self.node_nums, self.node_nums),
        )
        self.matrix_sparse = self.matrix_sparse + self.matrix_sparse.T  # 转换为无向图结构
        id = sparse.identity(self.node_nums, format="csc")
        self.matrix_sparse = self.matrix_sparse - id  # 相加过程中自权重翻倍，减去自权重

        # A_{sym} = I - D^{-0.5} * A * D^{-0.5}
        row_sum = self.matrix_sparse.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.0
        row_sum_inv_sqrt[np.isnan(row_sum_inv_sqrt)] = 0.0
        deg_inv_sqrt = sparse.diags(row_sum_inv_sqrt, format="csc")
        sym_norm_adj = deg_inv_sqrt.dot(self.matrix_sparse).dot(deg_inv_sqrt)

        self.laplacian_matrix = id - sym_norm_adj

    def build_chebyshev_polynomials(self, k):
        """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
        if getattr(self, "laplacian_matrix", None) is None:
            self.build_laplacian_matrix()

        degree_diags_matrix = sparse.identity(self.node_nums, format="csc")
        max_eigen_val = linalg.norm(self.laplacian_matrix, 2)

        # Calculate Chebyshev polynomials
        scaled_L = (2.0 / max_eigen_val) * self.laplacian_matrix - degree_diags_matrix
        T_k = [degree_diags_matrix, scaled_L]

        for i in range(2, k + 1):
            T_k.append(2 * scaled_L * T_k[i - 1] - T_k[i - 2])

        return T_k

    def get_cluster_group(self, node_nums, edge_src_idx, edge_dst_idx, edge_weights):
        res = paddle.zeros([node_nums], dtype=paddle.int32)
        if paddle.distributed.get_rank() == 0:
            res = hssinfo.cluster(
                paddle.arange(node_nums),
                paddle.to_tensor(edge_src_idx, dtype=paddle.int32),
                paddle.to_tensor(edge_dst_idx, dtype=paddle.int32),
                paddle.to_tensor(edge_weights, dtype=paddle.float32),
            )
            expected_place = paddle.framework._current_expected_place()
            res = res._copy_to(expected_place, False)
        if paddle.distributed.get_world_size() > 1:
            dist.barrier()
            dist.broadcast(res, src=0)
        res = res.numpy()
        paddle.device.cuda.empty_cache()
        return res

    def build_group_graph(self, n):
        assert n >= 2, "group levels should be larger than or equal to 2"
        # 分组划分映射，二部图
        self.group_mapping_node_nums = {}
        self.group_mapping_edge_src_idx = {i: [] for i in range(n)}
        self.group_mapping_edge_dst_idx = {i: [] for i in range(n)}

        self.group_connect_edge_src_idx = {}
        self.group_connect_edge_dst_idx = {}
        self.group_connect_edge_weights = {}

        res = self.get_cluster_group(
            self.node_nums, self.edge_src_idx, self.edge_dst_idx, self.edge_weights
        )

        self.group_mapping_edge_src_idx[1] = [idx for idx in range(self.node_nums)]
        self.group_mapping_edge_dst_idx[1] = list(res)
        self.group_mapping_node_nums[1] = (self.node_nums, max(res) + 1)

        conn_src, conn_dst, conn_weights = self._build_group_edge(
            res,
            edge_src_idx=self.edge_src_idx,
            edge_dst_idx=self.edge_dst_idx,
            edge_weights=self.edge_weights,
        )
        self.group_connect_edge_src_idx[1] = conn_src
        self.group_connect_edge_dst_idx[1] = conn_dst
        self.group_connect_edge_weights[1] = conn_weights

        for i in range(2, n):
            last_group_nums = self.group_mapping_node_nums[i - 1][1]
            res = self.get_cluster_group(
                last_group_nums, conn_src, conn_dst, conn_weights
            )
            if max(res) + 1 == last_group_nums:
                # without re-grouping
                break

            self.group_mapping_edge_src_idx[i] = [idx for idx in range(last_group_nums)]
            self.group_mapping_edge_dst_idx[i] = list(res)
            self.group_mapping_node_nums[i] = (last_group_nums, max(res) + 1)

            conn_src, conn_dst, conn_weights = self._build_group_edge(
                res,
                edge_src_idx=conn_src,
                edge_dst_idx=conn_dst,
                edge_weights=conn_weights,
            )
            self.group_connect_edge_src_idx[i] = conn_src
            self.group_connect_edge_dst_idx[i] = conn_dst
            self.group_connect_edge_weights[i] = conn_weights

    def _build_group_edge(
        self, res: np.ndarray, edge_src_idx, edge_dst_idx, edge_weights
    ):
        res_dict = {}  # id => group
        weights_dict = {}
        edges_dict = []
        for i in range(len(res)):
            res_dict[i] = res[i]

        for edge_idx in range(len(edge_src_idx)):
            src_idx, dst_idx = edge_src_idx[edge_idx], edge_dst_idx[edge_idx]
            src_group, dst_group = res_dict[src_idx], res_dict[dst_idx]
            if src_group != dst_group:
                key = (src_group, dst_group)
                if key not in edges_dict:
                    edges_dict.append(key)
                    weights_dict[key] = edge_weights[edge_idx]
                else:
                    weights_dict[key] += edge_weights[edge_idx]

        connect_edge_src_idx = []
        connect_edge_dst_idx = []
        connect_edge_weights = []
        for key in weights_dict:
            src_group, dst_group = key
            connect_edge_src_idx.append(src_group)
            connect_edge_dst_idx.append(dst_group)
            connect_edge_weights.append(weights_dict[key])
        return connect_edge_src_idx, connect_edge_dst_idx, connect_edge_weights

    def save_graph(self, path=None):
        dataframe = pd.DataFrame(
            {
                "src": self.edge_src_idx,
                "dst": self.edge_dst_idx,
                "weight": self.edge_weights,
            }
        )
        if path is not None:
            dataframe.to_csv(path, index=False)
        else:
            dataframe.to_csv(self.args.adj_path, index=False)

    def save_group_graph(self, path=None):
        dataframe = pd.DataFrame(
            {
                "src": self.group_connect_edge_src_idx[1],
                "dst": self.group_connect_edge_dst_idx[1],
                "weight": self.group_connect_edge_weights[1],
            }
        )
        if path is not None:
            dataframe.to_csv(path, index=False)
        else:
            dataframe.to_csv(self.args.group_path, index=False)

    def save_group_mapping(self, path=None):
        dataframe = pd.DataFrame(
            {
                "src": self.group_mapping_edge_src_idx[1],
                "dst": self.group_mapping_edge_dst_idx[1],
            }
        )
        if path is not None:
            dataframe.to_csv(path, index=False)
        else:
            dataframe.to_csv(self.args.mapping_path, index=False)

    def load_graph(self, path=None):
        if path is not None:
            dataframe = pd.read_csv(path)
        else:
            dataframe = pd.read_csv(self.args.adj_path)
        self.edge_src_idx = dataframe["src"].values.tolist()
        self.edge_dst_idx = dataframe["dst"].values.tolist()
        self.edge_weights = dataframe["weight"].values.tolist()


if __name__ == "__main__":
    convert_origin_graph_files(
        graph_path="data/PEMS03/PEMS03_adj.csv",
        save_graph_path="data/PEMS03/PEMS03_adj_weights.csv",
        node_nums=358,
        src_name="from",
        dst_name="to",
        distance_name="distance",
    )
    convert_origin_graph_files(
        graph_path="data/PEMS04/PEMS04.csv",
        save_graph_path="data/PEMS04/PEMS04_adj_weights.csv",
        node_nums=307,
        src_name="from",
        dst_name="to",
        distance_name="cost",
    )
    convert_origin_graph_files(
        graph_path="data/PEMS07/PEMS07.csv",
        save_graph_path="data/PEMS07/PEMS07_adj_weights.csv",
        node_nums=883,
        src_name="from",
        dst_name="to",
        distance_name="cost",
    )
    convert_origin_graph_files(
        graph_path="data/PEMS08/PEMS08.csv",
        save_graph_path="data/PEMS08/PEMS08_adj_weights.csv",
        node_nums=170,
        src_name="from",
        dst_name="to",
        distance_name="cost",
    )
    convert_origin_graph_files(
        graph_path="data/HZME_INFLOW/HZME_INFLOW.csv",
        save_graph_path="data/HZME_INFLOW/HZME_INFLOW_adj_weights.csv",
        node_nums=80,
        src_name="from",
        dst_name="to",
        distance_name="distance",
    )
    convert_origin_graph_files(
        graph_path="data/HZME_OUTFLOW/HZME_OUTFLOW.csv",
        save_graph_path="data/HZME_OUTFLOW/HZME_OUTFLOW_adj_weights.csv",
        node_nums=80,
        src_name="from",
        dst_name="to",
        distance_name="distance",
    )
