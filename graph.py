import torch
import models
from layer import Layer, LayerGroup
import options as opt

def get_graph_vgg(vgg):
    D = dict()
    n = 0
    V = []
    E = [[]]

    def record_hook(module, input, output):
        key = id(module)
        if key not in D:
            D[key] = len(V)
            V.append(Layer(module, input[0].shape, output.shape))

    hooks = []
    for module in vgg.modules():
        if isinstance(module, Layer.supported_base):
            hooks.append(module.register_forward_hook(record_hook))
    input = torch.rand(1, 3, 32, 32, device=opt.device)
    output = vgg(input)
    for hook in hooks:
        hook.remove()

    n = len(V)
    E = [([False] * n) for i in range(n)]
    for i in range(n - 1):
        E[i][i + 1] = True
    return n, V, E

def get_graph_resnet(resnet):
    D = dict()
    n = 0
    V = []
    E = [[]]

    def record_hook(module, input, output):
        key = id(module)
        if key not in D:
            D[key] = len(V)
            V.append(Layer(module, input[0].shape, output.shape))

    def add_edge(src, dst):
        i = D[id(src)]
        j = D[id(dst)]
        E[i][j] = True

    def add_chain(ls):
        for i in range(len(ls) - 1):
            add_edge(ls[i], ls[i + 1])

    hooks = []
    for module in resnet.modules():
        if isinstance(module, Layer.supported_base):
            hooks.append(module.register_forward_hook(record_hook))
    input = torch.rand(1, 3, 32, 32, device=opt.device)
    output = resnet(input)
    for hook in hooks:
        hook.remove()

    n = len(V)
    E = [([False] * n) for i in range(n)]

    chain = [resnet.conv1, resnet.bn1, resnet.relu1]
    add_chain(chain)

    src = [resnet.relu1]
    for module in resnet.modules():
        if isinstance(module, models.BasicBlockM):
            chain = [module.conv1, module.bn1, module.relu1,
                     module.conv2, module.bn2, module.relu2]
            add_chain(chain)
            dst = [module.conv1]
            src_ = [module.relu2]

            if module.downsample is not None:
                chain = list(module.downsample.children())
                add_chain(chain)
                dst.append(chain[0])
                add_edge(chain[-1], module.relu2)
            else:
                dst.append(module.relu2)

            for s in src:
                for d in dst:
                    add_edge(s, d)
            src = src_
            dst = []

    chain = [resnet.avgpool, resnet.flatten, resnet.fc]
    for s in src:
        add_edge(s, chain[0])
    add_chain(chain)

    return n, V, E

def get_graph_shufflenet(shufflenet):
    D = dict()
    n = 0
    V = []
    E = [[]]

    def record_hook(module, input, output):
        key = id(module)
        if key not in D:
            D[key] = len(V)
            in_shape = input[0][0].shape if isinstance(input[0], list) else input[0].shape
            out_shape = output.shape
            V.append(Layer(module, in_shape, out_shape))

    def add_edge(src, dst):
        i = D[id(src)]
        j = D[id(dst)]
        E[i][j] = True

    def add_chain(ls):
        for i in range(len(ls) - 1):
            add_edge(ls[i], ls[i + 1])

    hooks = []
    for module in shufflenet.modules():
        if isinstance(module, Layer.supported_base):
            hooks.append(module.register_forward_hook(record_hook))
    input = torch.rand(1, 3, 32, 32, device=opt.device)
    output = shufflenet(input)
    for hook in hooks:
        hook.remove()

    n = len(V)
    E = [([False] * n) for i in range(n)]

    chain = [shufflenet.conv1, shufflenet.bn1, shufflenet.relu1]
    add_chain(chain)

    src = [shufflenet.relu1]
    for module in shufflenet.modules():
        if isinstance(module, models.BottleneckM):
            chain = [module.conv1, module.bn1, module.relu1, module.shuffle,
                     module.conv2, module.bn2,
                     module.conv3, module.bn3]
            add_chain(chain)
            dst = [module.conv1]
            src_ = [module.relu3]

            if module.stride == 2:
                dst.append(module.conv4)
                add_edge(module.conv4, module.avgpool)
                add_edge(module.avgpool, module.concat)
                add_edge(module.bn3, module.concat)
                add_edge(module.concat, module.relu3)
            else:
                add_edge(module.bn3, module.relu3)
                dst.append(module.relu3)

            for s in src:
                for d in dst:
                    add_edge(s, d)
            src = src_
            dst = []

    chain = [shufflenet.avgpool, shufflenet.flatten, shufflenet.fc]
    for s in src:
        add_edge(s, chain[0])
    add_chain(chain)

    return n, V, E

def get_groups(V):
    if opt.co_graph_gen == 'get_graph_shufflenet':
        groups = []
        in_layers = list(range(1, 4))
        out_layers = list(range(0, 3))
        groups.append(LayerGroup(-1, in_layers, out_layers))
        in_layers = list(range(4, 11)) + list(range(13, 43))
        out_layers = list(range(3, 11)) + list(range(13, 42))
        groups.append(LayerGroup(-1, in_layers, out_layers))
        in_layers = list(range(43, 50)) + list(range(52, 118))
        out_layers = list(range(42, 50)) + list(range(52, 117))
        groups.append(LayerGroup(-1, in_layers, out_layers))
        in_layers = list(range(118, 125)) + list(range(127, 159))
        out_layers = list(range(117, 125)) + list(range(127, 158))
        groups.append(LayerGroup(-1, in_layers, out_layers))
        return groups

    else:
        n = len(V)
        vis = [([False] * 2) for i in range(n)]
        vis[0][0] = True
        vis[-1][1] = True
        groups = []
        for i in range(n):
            for j in range(2):
                if not vis[i][j]:
                    F = V[i].out_shape[1] if j else V[i].in_shape[1]
                    in_layers = []
                    out_layers = []
                    for k in range(n):
                        if not vis[k][0] and V[k].in_shape[1] == F:
                            in_layers.append(k)
                            vis[k][0] = True
                        if not vis[k][1] and V[k].out_shape[1] == F:
                            out_layers.append(k)
                            vis[k][1] = True
                    groups.append(LayerGroup(F, in_layers, out_layers))
        return groups

def get_links(E):
    n = len(E)
    in_links = [[] for i in range(n)]
    out_links = [[] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if E[i][j]:
                in_links[j].append(i)
                out_links[i].append(j)
    return in_links, out_links

def get_plot(name, n, V, E, reduced=False):
    from graphviz import Digraph
    dot = Digraph(name=name)
    for i, v in enumerate(V):
        node_name = '%d %s %s->%s' % (i, v.base_type, 
            str(list(v.in_shape)[1:]), str(list(v.out_shape)[1:]))
        colors = ['gray', 'gray', 'gray', 'gray', 'red', 'yellow', 'yellow', 'green', 'cyan', 'blue']
        if v.base_type != 'Identity' or not reduced:
            color = colors[Layer.supported_base.index(type(v.base))]
            dot.node(str(i), node_name, shape='box', color=color)
    if reduced:
        for i in range(n):
            if V[i].base_type == 'Identity':
                in_links = []
                out_links = []
                for j in range(n):
                    if E[j][i]:
                        in_links.append(j)
                        E[j][i] = False
                    if E[i][j]:
                        out_links.append(j)
                        E[i][j] = False
                for u in in_links:
                    for v in out_links:
                        E[u][v] = True
    for i in range(n):
        for j in range(n):
            if E[i][j]:
                dot.edge(str(i), str(j))
    dot.view()