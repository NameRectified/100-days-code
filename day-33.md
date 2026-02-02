## Day 33

## **Concatenation of Array**

**You are given an integer array `nums` of length `n`. Create an array `ans` of length `2n` where `ans[i] == nums[i]` and `ans[i + n] == nums[i]` for `0 <= i < n` (0-indexed).**

- create a new array or size 2n
- for each num in nums present at index i, append in the new array at position i and i+len(nums)

## **Contains Duplicate**

**Given an integer array `nums`, return `true` if any value appears more than once in the array, otherwise return `false`.**

- check if the length of the nums and set(nums) is equal(return false i.e no duplicate) or not(return True i.e duplicate exists)

## **Valid Anagram**

**Given two strings `s` and `t`, return `true` if the two strings are anagrams of each other, otherwise return `false`.**

NOTE: **An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.**

- construct two dictionaries(hash map) that contains the frequency of characters in s and t, if the two resulting dictionaries are equal, then they are anagrams

## **Two Sum**

**Given an array of integers `nums` and an integer `target`, return the indices `i` and `j` such that `nums[i] + nums[j] == target` and `i != j`.**

**You may assume that *every* input has exactly one pair of indices `i` and `j` that satisfy the condition. Return the answer with the smaller index first.**

- create a hash map
- for every num present at index i in nums, check if difference: target - num[i] is present in dictionary
- if it is present return index of difference, i
- else append the difference as a value to the key nums[i]

`NOTE` If it is mentioned that the array is sorted, we can use two pointers (inward moving) approach

## **Longest Common Prefix**

**You are given an array of strings `strs`. Return the longest common prefix of all the strings.**

**If there is no longest common prefix, return an empty string `""`.**

- sort the array
- find the common prefix between the first and last words.
- as the array is sorted, the first and last words are the most different from each other, so if we find a common prefix between them, it means that prefix is present in every word in the array

## **Group Anagrams**

**Given an array of strings `strs`, group all *anagrams* together into sublists. You may return the output in any order.**

- create a hashmap to store the groups
- use a list of length 26 that will contain the frequency of the characters in the word
- convert this list into a tuple and use it as the key for grouping different words based on whether they are anagrams or not
- return the hashmaps values

## **Remove Element**

**You are given an integer array `nums` and an integer `val`. Your task is to remove all occurrences of `val` from `nums` in-place.**

**After removing all occurrences of `val`, return the number of remaining elements, say `k`, such that the first `k` elements of `nums` do not contain `val`.**

- as we have to do it inplace, we cannot create a new list
- An important point to note over here is, we need not remove this element from the array, we just have to make sure it is not present in the first k values of the array i.e shift this element to the end of the array
- we can use a two pointer approach for this
- the left pointer will point to the index where the non-val element must be kept, and the right pointer will be used to find the next non-val element
- we will use a for loop for defining the right pointer value as it needs to increase always, no condition is present to prevent it from increasing and the left index will be zero at the start
- if the number at right index is not equal to the val, swap the numbers present in left and right index and increment left by 1
- return left as at the end it will be at the positon where there last non-val element is present

## **Majority Element**

**Given an array `nums` of size `n`, return the majority element.**

**The majority element is the element that appears more than `⌊n / 2⌋` times in the array. You may assume that the majority element always exists in the array.**

- store the frequency of each number in a hashmap
- after you increment the frequency, check if the value is greater than n//2, if it is return the key of that value (while looping, you will use i, so return)
- or
- sort the array and the element present at position n//2 is the majority element

## **Design Hashset**

**Design a HashSet without using any built-in hash table libraries.**

**Implement `MyHashSet` class:**

**`void add(key)` Inserts the value `key` into the HashSet.`bool contains(key)` Returns whether the value `key` exists in the HashSet or not.`void remove(key)` Removes the value `key` in the HashSet. If `key` does not exist in the HashSet, do nothing.**

- `void add(key)` Inserts the value `key` into the HashSet.
- `bool contains(key)` Returns whether the value `key` exists in the HashSet or not.
- `void remove(key)` Removes the value `key` in the HashSet. If `key` does not exist in the HashSet, do nothing.

- create a list of false of a huge size, one larger than the largest size mentioned in the constraints
- for add: make the false into true at that index(index is the key provided as parameter)
- contains: return the boolean value present at that index
- remove: just make the true into false at that index

## **Design HashMap**

**Design a HashMap without using any built-in hash table libraries.**

**Implement the `MyHashMap` class:**

**`MyHashMap()` initializes the object with an empty map.`void put(int key, int value)` inserts a `(key, value)` pair into the HashMap. If the `key` already exists in the map, update the corresponding `value`.`int get(int key)` returns the `value` to which the specified `key` is mapped, or `-1` if this map contains no mapping for the `key`.`void remove(key)` removes the `key` and its corresponding `value` if the map contains the mapping for the `key`.**

- `MyHashMap()` initializes the object with an empty map.
- `void put(int key, int value)` inserts a `(key, value)` pair into the HashMap. If the `key` already exists in the map, update the corresponding `value`.
- `int get(int key)` returns the `value` to which the specified `key` is mapped, or `1` if this map contains no mapping for the `key`.
- `void remove(key)` removes the `key` and its corresponding `value` if the map contains the mapping for the `key`.

- similar to the above question
- create a list of -1’s with the size based on the constraint for key, value
- put: make the -1 into the value provided at that key index
- get: return the value at the key index
- remove: make the value at the key index into -1