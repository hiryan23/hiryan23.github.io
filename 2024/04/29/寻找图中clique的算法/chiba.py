import time

import numpy as np
from sortedcontainers import SortedList


class UndirectedGraph(object):
    def __init__(self):
        self.vertex_to_id_dict: dict = {}
        self.id_to_vertex_dict: dict = {}
        self.vertex_count: int = 0
        self.id_list: SortedList = SortedList()
        self.adj_list: list = []
        self.deg_list: list = []

    def load_from_txt(self, file: str):
        edge_list = np.loadtxt(file, dtype=int)
        for src, dst in edge_list:
            self.add_edge(src, dst)

    def add_edge(self, vertex1, vertex2):
        if vertex1 not in self.vertex_to_id_dict:
            self.add_vertex(vertex1)
        if vertex2 not in self.vertex_to_id_dict:
            self.add_vertex(vertex2)
        self.adj_list[self.vertex_to_id_dict[vertex1]].add(self.vertex_to_id_dict[vertex2])
        self.adj_list[self.vertex_to_id_dict[vertex2]].add(self.vertex_to_id_dict[vertex1])

        self.deg_list[self.vertex_to_id_dict[vertex1]] += 1
        self.deg_list[self.vertex_to_id_dict[vertex2]] += 1

    def add_vertex(self, vertex):
        v_id = self.vertex_count
        self.vertex_to_id_dict[vertex] = v_id
        self.id_to_vertex_dict[v_id] = vertex
        self.vertex_count += 1
        self.id_list.add(v_id)
        self.adj_list.append(SortedList())
        self.deg_list.append(0)

    def del_vertex(self, vertex, inplace=False):
        v_id = self.vertex_to_id_dict[vertex]
        v_adj = self.adj_list[v_id]
        if v_id in self.id_list:
            if inplace:
                for u_id in v_adj:
                    self.deg_list[u_id] -= 1
                    self.adj_list[u_id].remove(v_id)
                self.deg_list[v_id] = -1
                self.id_list.remove(v_id)
                self.adj_list[v_id] = SortedList()
            else:
                new_g = self
                for u_id in v_adj:
                    new_g.deg_list[u_id] -= 1
                    new_g.adj_list[u_id].remove(v_id)
                new_g.deg_list[v_id] = -1
                new_g.id_list.remove(v_id)
                new_g.adj_list[v_id] = SortedList()

                return new_g
        else:
            pass


class K3(object):
    """
    Print all triangles.
    """

    def __init__(self, g: UndirectedGraph):
        self.graph = g
        self.process()

    def sort_by_deg(self) -> list:
        return sorted(range(len(self.graph.deg_list)), key=lambda x: self.graph.deg_list[x], reverse=True)

    def process(self):
        sorted_vertex_list = self.sort_by_deg()[:len(self.graph.id_list)]
        for v in sorted_vertex_list[:-2]:
            v_adj = self.graph.adj_list[v]
            mark = set(v_adj)
            for u in v_adj:
                u_adj = self.graph.adj_list[u]
                for w in u_adj:
                    if w in mark:
                        print(f"({self.graph.id_to_vertex_dict[v]},"
                              f"{self.graph.id_to_vertex_dict[u]},"
                              f"{self.graph.id_to_vertex_dict[w]})")
                mark.remove(u)
            self.graph.del_vertex(self.graph.id_to_vertex_dict[v], True)


class C4(object):
    def __init__(self, g: UndirectedGraph):
        self.graph = g
        self.U = [set() for _ in self.graph.deg_list]
        self.process()

    def sort_by_deg(self) -> list:
        return sorted(range(len(self.graph.deg_list)), key=lambda x: self.graph.deg_list[x], reverse=True)

    def set_id_to_vertex(self, s: set):
        return {self.graph.id_to_vertex_dict[i] for i in s}

    def process(self):
        sorted_vertex_list = self.sort_by_deg()[:len(self.graph.id_list)]
        for v in sorted_vertex_list:
            for u in self.graph.adj_list[v]:
                for w in self.graph.adj_list[u]:
                    if w != v:
                        self.U[w] = self.U[w] | {u}
            for w in range(len(self.graph.deg_list)):
                if len(self.U[w]) >= 2:
                    print(f"({self.graph.id_to_vertex_dict[v]},{self.graph.id_to_vertex_dict[w]},"
                          f"{self.set_id_to_vertex(self.U[w])})")
            for w in range(len(self.graph.deg_list)):
                if self.U[w]:
                    self.U[w] = set()
            self.graph.del_vertex(self.graph.id_to_vertex_dict[v], True)


class COMPLETE(object):
    def __init__(self, order: int, g: UndirectedGraph):
        self.order = order
        self.graph = g
        self.C = []

        self.f = open('ans1.txt', 'w')
        self.K(self.order, self.graph)
        self.f.close()

    def K(self, k: int, gk: UndirectedGraph):
        if k == 2:
            for x in gk.id_list:
                res = [{gk.id_to_vertex_dict[x], gk.id_to_vertex_dict[y]} | set(self.C)
                       for y in gk.adj_list[x] if x < y]
                _ = {print(i, file=self.f) for i in res}

        else:
            length = len(gk.id_list)
            for _ in range(length):
                v = gk.id_list[0]
                gk_1 = graph_induced_by_vertex_neighbor(gk, gk.id_to_vertex_dict[v])
                self.C.append(gk.id_to_vertex_dict[v])
                self.K(k - 1, gk_1)
                self.C.pop()
                gk.del_vertex(gk.id_to_vertex_dict[v], True)


class NEW_COMPLETE(object):
    def __init__(self, order: int, g: UndirectedGraph):
        self.order = order
        self.graph = g
        self.C = []
        self.graph_label = [order] * len(g.id_list)
        self.adj_list = [list(g.adj_list[i]) for i in range(len(g.adj_list))]

        name_list = [self.graph.id_to_vertex_dict[v_id] for v_id in self.graph.id_list]

        self.f = open('ans2.txt', 'w')
        self.K(self.order, name_list, self.graph.deg_list)
        self.f.close()

    def K(self, k: int, u_name_list: list, k_deg_list: list):
        if k == 2:
            sub_g = graph_induced_by_vertices(self.graph, u_name_list)
            for x in sub_g.id_list:
                res = [{sub_g.id_to_vertex_dict[x], sub_g.id_to_vertex_dict[y]} | set(self.C)
                       for y in sub_g.adj_list[x] if x < y]
                _ = {print(i, file=self.f) for i in res}
        else:
            name_to_id_dict = dict(zip(u_name_list, range(len(u_name_list))))
            # id_to_name_dict = dict(zip(range(len(u_name_list)), u_name_list))

            sorted_list = sorted(u_name_list, key=lambda y: k_deg_list[name_to_id_dict[y]], reverse=True)

            for vi_name in sorted_list:
                u1_list = [self.graph.id_to_vertex_dict[v] for v in self.adj_list[self.graph.vertex_to_id_dict[vi_name]]
                           if self.graph_label[v] == k]

                for u in u1_list:
                    self.graph_label[self.graph.vertex_to_id_dict[u]] = k - 1

                k1_deg_list = []

                for u in u1_list:
                    d = 0
                    for i in self.adj_list[self.graph.vertex_to_id_dict[u]]:
                        if self.graph.id_to_vertex_dict[i] in u1_list:
                            self.adj_list[self.graph.vertex_to_id_dict[u]].remove(i)
                            self.adj_list[self.graph.vertex_to_id_dict[u]].append(i)
                            d += 1
                    k1_deg_list.append(d)

                self.C.append(vi_name)

                self.K(k-1, u1_list, k1_deg_list)

                self.C.pop()

                for u in u1_list:
                    self.graph_label[self.graph.vertex_to_id_dict[u]] = k

                self.graph_label[self.graph.vertex_to_id_dict[vi_name]] = k + 1

                for v in u1_list:
                    pos = 0
                    for i in range(len(self.adj_list[self.graph.vertex_to_id_dict[v]])):
                        if self.graph_label[self.adj_list[self.graph.vertex_to_id_dict[v]][i]] == k:
                            pos = i
                            break

                    self.adj_list[self.graph.vertex_to_id_dict[v]].remove(self.graph.vertex_to_id_dict[vi_name])
                    self.adj_list[self.graph.vertex_to_id_dict[v]].insert(pos, self.graph.vertex_to_id_dict[vi_name])


def graph_induced_by_vertex_neighbor(g: UndirectedGraph, v) -> UndirectedGraph:
    g_ind_v = UndirectedGraph()

    neighbor_vertex_list = set(g.adj_list[g.vertex_to_id_dict[v]])
    # neighbor_vertex_list.add(g.vertex_to_id_dict[v])

    for v_id in neighbor_vertex_list:
        _ = {g_ind_v.add_edge(g.id_to_vertex_dict[v_id], g.id_to_vertex_dict[u])
             for u in g.adj_list[v_id] if u in neighbor_vertex_list and v_id < u}

    return g_ind_v


def graph_induced_by_vertices(g: UndirectedGraph, v_name_list) -> UndirectedGraph:
    g_ind_v = UndirectedGraph()
    v_id_list = [g.vertex_to_id_dict[v] for v in v_name_list]

    for v_id in v_id_list:
        _ = {g_ind_v.add_edge(g.id_to_vertex_dict[v_id], g.id_to_vertex_dict[u])
             for u in g.adj_list[v_id] if u in v_id_list and v_id < u}

    return g_ind_v


if __name__ == "__main__":
    G = UndirectedGraph()
    G.load_from_txt('example.txt')
    # K3 = K3(G)
    # C4 = C4(G)

    time1 = time.time()
    # C = COMPLETE(6, G)
    C = NEW_COMPLETE(6, G)
    time2 = time.time()

    print(f"Process time: {time2-time1}s")
