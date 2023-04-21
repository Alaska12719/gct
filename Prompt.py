import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.utils import sort_edge_index, add_self_loops, to_undirected
from torch_geometric.data import Batch
from utils import seed_everything, seed
from torch_geometric.loader.cluster import ClusterData
class Prompt(torch.nn.Module):
    def __init__(self, token_dim, token_num, prune_thre=0.9, isolate_tokens=False, inner_prune=None):
        super(Prompt, self).__init__()
        self.prune_thre = prune_thre
        if inner_prune is None:
            self.inner_prune = prune_thre
        else:
            self.inner_prune = inner_prune
        self.isolate_tokens = isolate_tokens
        self.token_x = torch.nn.Parameter(torch.empty(token_num, token_dim))
        self.initial_prompt()

    def initial_prompt(self, init_mode='kaiming_uniform'):
        if init_mode == 'metis':  # metis_num = token_num
            self.initial_prompt_with_metis()
        elif init_mode == 'node_labels':  # label_num = token_num
            self.initial_prompt_with_node_labels()
        elif init_mode == 'orthogonal':
            torch.nn.init.orthogonal_(self.token_x, gain=torch.nn.init.calculate_gain('leaky_relu'))
        elif init_mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.token_x, gain=torch.nn.init.calculate_gain('tanh'))
        elif init_mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.token_x, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        elif init_mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.token_x, nonlinearity='leaky_relu')
        elif init_mode == 'uniform':
            torch.nn.init.uniform_(self.token_x, 0.99, 1.01)
        else:
            raise KeyError("init_mode {} is not defined!".format(init_mode))

    def initial_prompt_with_metis(self, data=None, save_dir=None):
        if data is None:
            raise KeyError("you are calling initial_prompt_with_metis with empty data")
        metis = ClusterData(data=data, num_parts=self.token_x.shape[0], save_dir=save_dir)
        b = Batch.from_data_list(list(metis))
        x = global_mean_pool(b.x, b.batch)
        self.token_x.data = x

    def initial_prompt_with_node_labels(self, data=None):
        x = global_mean_pool(data.x, batch=data.y)
        self.token_x.data = x

    def forward(self, graph_batch: Batch):
        num_tokens = self.token_x.shape[0] #node num of the prompt graph 
        node_x = graph_batch.x  #input graph
        num_nodes = node_x.shape[0] #number of nodes in the graph
        num_graphs = graph_batch.num_graphs #number of graphs in the batch
        node_batch = graph_batch.batch #the list of nodes in each graph
        token_x_repeat = self.token_x.repeat(num_graphs, 1) #repeat the token_x tensor for each graph
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)])
        batch_one = torch.cat([node_batch, token_batch], dim=0)
        token_batch = torch.LongTensor([j for j in range(num_graphs) for i in range(num_tokens)]) + num_graphs
        batch_two = torch.cat([node_batch, token_batch], dim=0)
        edge_index = graph_batch.edge_index

        token_dot = torch.mm(self.token_x, torch.transpose(node_x, 0, 1))  # (T,d) (d, N)--> (T,N)
        token_sim = torch.sigmoid(token_dot)  # 0-1
        cross_adj = torch.where(token_sim < self.prune_thre, 0, token_sim)
        tokenID2nodeID = cross_adj.nonzero().t().contiguous()
        batch_value = node_batch[tokenID2nodeID[1]]
        new_token_id_in_cross_edge = tokenID2nodeID[0] + num_nodes + num_tokens * batch_value
        tokenID2nodeID[0] = new_token_id_in_cross_edge
        cross_edge_index = tokenID2nodeID
        if self.isolate_tokens:
            new_edge_index = torch.cat([edge_index, cross_edge_index], dim=1)
        else:
            token_dot = torch.mm(self.token_x, torch.transpose(self.token_x, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            tokenID2tokenID = inner_adj.nonzero().t().contiguous()
            inner_edge_index = torch.cat([tokenID2tokenID + num_nodes + num_tokens * i for i in range(num_graphs)],
                                         dim=1)
            new_edge_index = torch.cat([edge_index, cross_edge_index, inner_edge_index], dim=1)

        new_edge_index, _ = add_self_loops(new_edge_index)
        new_edge_index = to_undirected(new_edge_index)
        edge_index_xp = sort_edge_index(new_edge_index)

        return (torch.cat([node_x, token_x_repeat], dim=0),
                edge_index_xp,
                batch_one,
                batch_two)