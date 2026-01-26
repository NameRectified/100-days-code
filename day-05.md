## Day 5

### Divide and Conquer (D&C)

- We divide the given problem into problems that are smallest versions of the given problem and solve it
- There are two steps to solve a D&C problem:
    - Find the base case, which is the simplest possible case
    - Divide the problem until it becomes the base case

### Quicksort

- A divide and conquer based sorting algorithm.
- Best, average case: O(n logn) (when the left and right arrays are balanced, that is a good pivot is chosen)
- Worst case: O(n^2) (when the left and right arrays are imbalanced, that is the pivot is either the smallest or largest element
- So the runtime of this algorithm depends on the pivot chosen
- we choose a pivot and all numbers greater than the pivot go on the right and all numbers less than pivot go on the left.
    ```
    # -------------- # 
    # This is more efficient #
    from typing import List
    def sortArray(nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums
        else:
            pivot = random.choice(nums)
            less = [i for i in nums if i<pivot]
            equal = [i for i in nums if i==pivot]
            more = [i for i in nums if i>pivot]
            return sortArray(less) + equal + sortArray(more)
    
    
    def quicksort(arr):
        if len(arr) < 2:
            return arr
        else:
            pivot = arr[0]
            lesser = [i for i in arr[1:] if i<pivot] # we are slicing the the array because we have selected the first element as pivot
            greater = [i for i in arr[1:] if i >= pivot]
            return quicksort(lesser) + [pivot] + quicksort(greater)
	 
    print(quicksort([1, 9, 2,4]))

    
    ```


### Big O notation revisited

- Although mergesort also takes O(n logn) time always and quicksort takes O(n logn) time only for its best and average case, it is faster and most widely used because:
    - The average case is the most often hit than worst case in real world scenarios
    - The constant that is generally ignored in Big O notation is bigger in the case of mergesort compared to quicksort.

### Exercises

4.1) Write a recursive function (D&C) for finding array sum
``` 
def sumOfArray(arr):
    if len(arr) < 2: #base case
        return arr[0]
    else:
        return arr[0] + sumOfArray(arr[1:])
    
    
arr = [1, 2, 3, 4, 5]
print(sumOfArray(arr))

```

4.2) Write a recursive function to length of array
```
def countItems(arr):
    if not arr: #base case
        return 0
    else:
        return 1 + countItems(arr[1:])
    
    
arr = [1, 2, 3, 4, 5]
print(countItems(arr))
```

4.3) Write a recursive function to find the maximum element in the array
```
def maxItem(arr):
    if len(arr) < 2: #base case
        return arr[0]
    else:
        return max(arr[0], maxItem(arr[1:]))
    
    
arr = [1, 2, 3, 4, 5]
print(maxItem(arr))
```

4.4) Find the base and recursive case of binary search
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

### How long would each of these operations take in Big O notation?
4.5) Printing the value of each element in an array: O(n) as each value must be read

4.6) Doubling the value of each element in an array: O(n) as each value must be read

4.7) Doubling the value of just the first element in an array:  O(1) constant time as only one values needs to be read

4.8) Creating a multiplication table with all the elements in the array. So
if your array is [2, 3, 7, 8, 10], you first multiply every element by 2,
then multiply every element by 3, then by 7, and so on: O(n^2) as the multiplication table must be created for each element in the array (n elements) and you multiply each element with n elements
