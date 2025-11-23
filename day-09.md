## Day 9
### Classroom Scheduing Problem:
- Problem: You want to hold the maximum number of classes, some of the class timings overlap.
- Greedy solution:
    - Pick the class that ends the soonest.
    - Then select the class that starts after the first class that also ends the soonest among the options
    - Repeat
- This is a greedy algorithm

### Greedy algorithms

- These algorithms are simple: at each step pick the local optimal solution and at the end you either end up with the global optimal solution or get atleast close to it
- They don’t work always
- When you just need an algorithm that solves the problem pretty well and not necessarily a perfect one, greedy algorithms are the best choice as they are simple to write and usually are pretty close to the perfect solution.
- Greedy algorithms are easy to write and fast to run, so they make good approximation algorithms

### Set Covering Problem
- Problem:  figure out the smallest set of stations you can play on to cover all 50 states?
- Greedy algorithm:
    - Pick the station that covers the most number of uncovered stations. It’s fine if the station covers some stations that have already been covered
    - Repeat this step until all the stations have been covered
- This is an example of `approximation algorithm`
    - When calculating the exact solution for a problem takes a very long time, we use an approximation algorithm.
    - Approximation algorithms are judged by
        - How fast they are
        - and how close they are to the exact solution
- Travelling Salesman problem approximate solution:
    - Choose a random start city (if no particular city is mentioned)
    - Pick the closest unvisited city
    - Repeat till all cities have been visited

```python
## Radio station problem

'''
set of all the states yet to be covered

hash map of all the stations as the keys and the cities they cover as the values

initialize the best station as none, and an empty set of all the states_covered
for each of the station find if the intersection between states to cover and the states the station covers is bigger than the current. if it is update it as the best station and update the states_covered
'''

states_needed = set(["mt", "wa", "or", "id", "nv", "ut",
"ca", "az"])

stations = {}
stations["kone"] = set(["id", "nv", "ut"])
stations["ktwo"] = set(["wa", "id", "mt"])
stations["kthree"] = set(["or", "nv", "ca"])
stations["kfour"] = set(["nv", "ut"])
stations["kfive"] = set(["ca", "az"])

final_stations = set() # will be containing the final answer

while states_needed: # looping till this set is empty
    best_station = None # local best station
    states_covered = set()
    for station, states_in_station in stations.items():
        covered = states_needed & states_in_station
        if len(covered) > len(states_covered):
            best_station = station
            states_covered = covered
    states_needed = states_needed - states_covered # set difference
    final_stations.add(best_station)

print(final_stations)
```

### Sets

- They are like lists, except that they do not contain any duplicates
- They gives us additional operations like:
    - Intersection: Present in both sets
    - Union: Present in either of the sets
    - Set difference: All elements in set a that are not present in set b

### NP Completeness

- These are problems that are hard to solve
- If you know that the problem you are solving is an NP-complete problem, you can stop trying to solve it perfectly and devise an approximate algorithm instead
- But it’s hard to know  if a problem is np-complete or not as most of the time there is a small difference between a problem that is easy to solve and an NP-complete problem.
- Eg: It easy to find the shortest path from point a to b, but to find the shortest path that connects several points becomes travelling salesman problem which is NP-complete.
- There is no easy way to tell if the problem is NP-complete or not, but these are some clues:
    - The algorithm runs quickly for few items but really slows down with more items
    - “All combinations of X” usually means an NP-complete problem
    - If you have to calculate “every possible version” of X because you can’t break it down into smaller-sub problems, it might be np-complete
    - If the problem involves a sequence( like a sequence of cities in travelling salesman problem) or it involves a set(like a set of radio stations) and they are hard to solve, maybe its np-complete
    - If you can restate the problem as a set-covering problem or TSP, it is definitely np-complete
- They have no known fast solution
- If you have an NP-complete problem, your best bet is to use an approximation algorithm.

### Exercises

8.1) Select the largest box that fits in remaining space, no

8.2) Select the place or activity that have highest points that can be visited or done in the remaining time. I dont think this gives the optimal solution, this problem is similar to knapsack problem and since greedy approach does not work for knapsack, I guess it fails for this also.

8.3) Quicksort is not a greedy algorithm: we are not selecting any local optimal solution, also quicksort always works and greedy algorithms dont work sometimes

8.4) BFS - No, we go through all nodes in a particular level before going to the next level (but the answer at the back says yes???)

8.5) Dijkstra - Yes, we choose the local optimal in each step, and that’s the reason it fails when negative edges are present

8.6) Yes, similar to TSP

8.7) Yes - We need to check by taking each person as the start point

8.8) Yes - similar to set-covering problem