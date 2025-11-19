## Day 6

### Hash functions

- A function that lets you map strings to numbers
- It must be:
    - Consistent i.e it should always return the same output for the same input
    - Different words must be mapped to different numbers, and mapping to different number for every different word is the best case.
- It gives the index where the item will be stored in the array
- As the hash fuction knows how big the array is, it will return only valid indexes
- A good hash function distributes the values in the array evenly and a bad one groups them together and produces lot of collisions.

### Hashtable use cases

- They are used for lookups as in average case, it takes constant time for searching. eg: Phone book, DNS
- Preventing duplicate entries: as the same input is mapped to the same output, we can easily detect duplicates
- Using as a cache: we can store and return frequently searched data, eg: caching web pages

### Collisions

- They occur when the hash function gives the same output for two different inputs.
- The simplest way to solve this is to create a linked list in that index where collision occurs
- If the linked list gets too long, or worst case if the hash table is empty in all slots but one where it is a super long linked list, it just becomes a linked list like data structure with worse time complexity
- To avoid/ reduce collisions we need a good hash function and a low load factor
- Ideally, the hash function would map all the keys evenly all over the hash slots
- Load factor = number of items in the hashtable  / total number of slots
- If the load factor starts to grow, we need to resize the array and this process is called, resizing :)
- In resizing:
    - You create a new array with a bigger size, the rule of thumb is to make the array twice the size, and the reinsert all the elements into the new hash table with the hash function.
    - A good rule of thumb is to resize when the load factor is greater than 0.7
    - Resizing is expensive, but averaged out hash tables take O(1) even with resizing

### Performance

- In the average case it takes O(1) time for insertion, search and delete for a hash table.
- In the worst case, they all become O(n).

- You can create a hash table by combining a hash function with an array

### Exercises

5.1) Yes - because the same number is being returned for the same input (although this same output is given for every input)

5.2) No - as it is random the same input may mapped to different output

5.3) No - after an element has been inserted if the same element is given to the hash function the next empty slot will be different and we end up getting different output for the same input.

5.4) Yes - the length of the word does not change

5.5) D - By process of elimination, A is not suitable for being a hash function for any kind of problem, B fails as 3 of the names have the same length, C fails as Bob and Ben start with same letter. When i say fail I mean a collision, it is a fail because as mentioned in the question there are 10 slots and an ideal hash function will not result in a collision when the number of elements is less than the slots.

5.6) B - as all of them are of different length

5.7) B or C