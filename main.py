import paddle
from hssinfo import cluster

from args import args
from models.graphconv import GraphST

if __name__ == "__main__":
    graph = GraphST(args=args, build=False)
    res = cluster(
        paddle.arange(graph.node_nums),
        paddle.to_tensor(graph.edge_dst_idx, dtype=paddle.int32),
        paddle.to_tensor(graph.edge_src_idx, dtype=paddle.int32),
        paddle.to_tensor(graph.edge_weights, dtype=paddle.float32),
    )
    print(res.numpy())
