## Day 8

### Dijkstra’s algorithm

- While BFS helps find the path with the fewest segments, Dijkstra’s algorithm helps find the fastest path.
- Algorithm:
    - Find the cheapest node i.e the node that you can reach in the least amount of time
    - Check whether there is a cheaper path to the neighbors of this node. If so, update the costs of the neighbors of this node.
    - Repeat until this process has been done for all nodes
    - Calculate the final path.
- You run Dijkstra’s algorithm on all nodes except the final/ target node
- In this algorithm we work with weighted graphs (all weights must be positive, if any is negative consider using Bellman-Ford algorithm)
- So, to find the shortest path in an unweighted graph, use `BFS` and in weighted graph use `Dijkstra`
- An undirected graph is a cycle
- Dijkstra works only for `directed acyclic graphs: DAG`
- Once you process a node it means there is no cheaper way to get to that node and we cannot update the cost of a node after it has been processed

### Implementation

- We require three hash tables: graph, costs, parent
- The costs and parents hash tables are updated as the algorithm progresses
- You can represent infinity in python as float(’inf’)
- We also require an array to keep track of all the nodes processed

```
# graph implementation
graph["start"] = {}
graph["start"]["a"] = 6
graph["start"]["b"] = 2

graph["a"] = {}
graph["a"]["fin"] = 1

graph["b"] = {}
graph["b"]["a"] = 3
graph["b"]["fin"] = 5

graph["fin"] = {}

# hashtable for costs
infinity = float('inf')
costs = {}
costs['a'] = 6
costs['b'] = 2
costs['fin'] = infinity

# Hashtable for parents
parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["fin"] = None

processed = []

def find_lowest_cost_node(costs):
    lowest_cost = float("inf")
    lowest_cost_node = None
    for node in costs:
        cost = costs[node]
        if cost < lowest_cost and node not in processed:
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node
 
node = lowest_cost_node(costs)
while node is not None:
    cost = costs[node]
    neighbors = graph[node]
    for n in neighbors.keys():
        new_cost = cost + neighbors[n]
        if new_cost < costs[n]:
            costs[n] = new_cost
            parents[n] = node
    processed.append(node)
    node = lowest_cost_node(costs)
            
print(graph)
print(parents)
print(costs)
```