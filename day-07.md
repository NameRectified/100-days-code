### Breadth First Search (BFS)

- BFS allows us to find the shortest distance between two things, and shortest distance can mean many things such as a checkers ai that calculates the fewest moves to victory, a spell checker that gives the fewest edits required from the misspelled word to a real word, a doctor or lawyer closest to you in your network, and etc.
- A graph is a collection of nodes and edges, where an edge connects two nodes
- BFS helps to answer two types of questions:
    - Is there a path between two nodes? (if it is present BFS will find shortest path)
    - What is shortest path to reach from one node to another?
- In BFS queue data structure is used, which is first in first out `FIFO`. It contains two operations `enqueue` to add items and `dequeue` to remove items, it is similar to push and pop and the terms are used interchangably.
- Directed graphs have arrows and the relationship is only one way, undirected graphs have no arrows and the relationship is both ways.
- Algorithm:
    - Have an hash table for the graph with each node as the `key` and all its neighbors as a list of `values`
    - create a search queue and add all the neighbors on the start node to the queue
    - while the queue is not empty, dequeue the element and if that element has not been checked already, check that element, if that element is what we are searching for return true and we are done, else add the neighbors of that element to the queue and the element to the searched list.
    - if the queue becomes empty and we have not found, it means the element is not present so return false.
- `running time` O(V+E) where V is vertices or node count and E is edge count.
- A `tree` is a special type of graph where no edges point backward
- If you have a problem like “find the shortest X”, try modeling the problem as a graph and use BFS
- `Stacks` are `LIFO`
- You need to search elements in the order they were added to the search list, so the search list must be a queue, else we will not get the shortest path.
- Once an element is checked, dont check them again as it might lead to an infiite loop.

### Exercises

6.1) 2

6.2) 2, from cab to cat to bat

6.3) A: Invalid as breakfast comes before brushing, B: Valid, C: Invalid as shower comes before waking up

6.4) Wake up > Exercise > Shower > Get Dressed > Brush > Breakfast > Pack lunch

6.5) A, C

```python
from collections import deque #double ended queue

# neighbors
graph = {}
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["thom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["thom"] = []
graph["jonny"] = []

def bfs(name):
    search_queue = deque()
    search_queue += graph[name]
    searched = []
    
    while search_queue:
        person = search_queue.popleft()
        if not person in searched:
            if not person_is_seller(person):
                search_queue += graph[person]
                searched.append(person)
            else:
                print(f"{person} is a mango seller.")
                return True
    return False

def person_is_seller(name):
    return len(name) == 4
    
bfs("you")
```