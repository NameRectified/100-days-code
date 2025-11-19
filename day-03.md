### Linked Lists vs Arrays

- Linked lists are good for insertion. Even if there are’nt enough contiguous blocks of memory, the elements can be inerted in empty blocks that are not continuous as each element has the address of the next.
- Arrays are great for accessing elements in random order as we know the address of every element in the array, where as for a linked list to access an element we have to start from the very first element and reach our desired element.
- Lists have a constant insertion time and linear reading time, where as arrays have a constant reading time and linear insertion time.
- Lists are also better if we want to insert elemets into the middle or delete an element.
- Insertions may fail if there is not enough space in memory but deletions always work
- Arrays provide random access, where as linked lists provide sequential access
- Elements in array have to be of the same data type, but list can contain mixed data types.

### Selection sort

- We can divide selection sort into two functions. One finds the smallest element in the array and the other appends the smallest element into a new array.
- findSmallest(): Store the first element as the smallest and its index as smallest_index. Then loop through the array starting from 2 element(index 1) all the way to the end of the array, using an if condition if you find a smaller element in the array update the smallest and smallest_index variables, return the smallest_index
- selectionSort(): create a new empty array, loop through the array from the beginning, find index of smallest element by calling findSmallest(), pop that element from the array and store that element and add it to the new array. return the new array.
- As we are going through all the elements in the list O(n) and we are doing this n times, the time complexity of selection sort is O(n^2)
```
def smallest_element_idx(arr):
    smallest = arr[0]
    smallest_idx = 0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_idx = i
    return smallest_idx

def selection_sort(arr):
    new_arr = []
    for _ in range(len(arr)):
        smallest = smallest_element_idx(arr)
        new_arr.append(arr.pop(smallest))
    return new_arr

print(selection_sort([1,5,3,2]))
```

### Exercise:

2.1 ) If we have lots of inserts and few reads we should use a list.

2.2) As elements(orders) are constantly being inserted and deleted and the orders are executed sequentially(that is reading speed does not matter), lists are a better choice

2.3) As for allowing a valid user to login requires reading the credentials and since the problem mentions users login multiple times, credentials are added to the database only once, so as there are more read operations than write operations, array will be my choice

2.4) We might run out contiguous blocks of memory and have to copy the users array to a new location, as the insertion in an array is very slow and since we are using binary search, we may need to insert the element somewhere in the middle of the array, and when we insert we have to move each element by one block

2.5) it will be faster as the insertion time is constant and the reading time is also faster as although we are going through the whole list for searching a user we are going to go through only 1 of the 26 available lists
