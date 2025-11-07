## Day 5

### Divide and Conquer (D&C)

- We divide a problem into smaller problems that can be solved.
- There are two steps to solve a D&C problem:
    - Find the base case, which is the simplest possible case
    - Divide the problem until it becomes the base case

### Quicksort

- A divide and conquer based sorting algorithm.
- Best, average case: O(n logn) (when the left and right arrays are balanced, that is a good pivot is chosen)
- Worst case: O(n^2) (when the left and right arrays are imbalanced, that is the pivot is either the smallest or largest element
- So the runtime of this algorithm depends on the pivot chosen

### Big O notation revisited

- Although mergesort also takes O(n logn) time always and quicksort takes O(n logn) time only for its best and average case, it is faster and most widely used because:
    - The average case is the most often hit than worst case in real world scenarios
    - The constant that is generally ignored in Big O notation is bigger in the case of mergesort compared to quicksort.

### Exercises

4.1)
``` 
def sumOfArray(arr):
    if len(arr) < 2: #base case
        return arr[0]
    else:
        return arr[0] + sumOfArray(arr[1:])
    
    
arr = [1, 2, 3, 4, 5]
print(sumOfArray(arr))

```

4.2)
```
def countItems(arr):
    if not arr: #base case
        return 0
    else:
        return 1 + countItems(arr[1:])
    
    
arr = [1, 2, 3, 4, 5]
print(countItems(arr))
```

4.3)
```
def maxItem(arr):
    if len(arr) < 2: #base case
        return arr[0]
    else:
        return max(arr[0], maxItem(arr[1:]))
    
    
arr = [1, 2, 3, 4, 5]
print(maxItem(arr))
```

4.4)
```
def binarySearch(key, start, end):
    mid = (start + end)//2
    if key == arr[mid]:
        return mid
    elif key >= arr[mid]:
        return binarySearch(key, mid+1, end)
    elif key < arr[mid]:
        return binarySearch(key, start, mid-1)
    else:
        return -1

arr = [1,2, 5,6,7]
print(binarySearch(5, 0, len(arr)))
```

4.5) O(n) as each value must be read

4.6) O(n) as each value must be read

4.7) O(1) constant time as only one values needs to be read

4.8) O(n^2) as table must be created for each element in the array (n elements) and you multiply each element with n elements