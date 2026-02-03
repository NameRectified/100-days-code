## Day 34

## **Sort an Array**

**You are given an array of integers `nums`, sort the array in ascending order and return it.**

**You must solve the problem without using any built-in functions in `O(nlog(n))` time complexity and with the smallest space complexity possible.**

- I will be using quicksort to sort the array
- In quick sort we choose and element called the pivot, all the elements less than the pivot are added to the left of the pivot and greater elements are added to the right
- We repeat this process for the left and right sublists
- using recursion:
    - base case: length is less than 2, return the nums (array)
    - choose a random pivot
    - store all elments less than pivot in an array called less
    - all elements greater than pivot are stored in array called more
    - elements equal to pivot are stored in equal
    - return function(less) + equal + function(more)

## Sort colors

**You are given an array `nums` consisting of `n` elements where each element is an integer representing a color:**

**`0` represents red, `1` represents white, `2` represents blue**

- `0` represents red
- `1` represents white
- `2` represents blue

**Your task is to sort the array in-place such that elements of the same color are grouped together and arranged in the order: red (0), white (1), and then blue (2).**

**You must not use any built-in sorting functions to solve this problem.**

- We can use count sort to sort the array as the range of numbers is quite less
- count the frequency of each number and store it in an array called count
- then initialize start as 0
- for each of the unique numbers, 0, 1, 2 and in this case
- define end = start + count[i] where count[i] has the frequency of i
- inner loop for j in range (start, end): nums[j] = i (as we need to modify in place)
- start = end, outside inner loop but inside outer loop

## **Longest Substring Without Repeating Characters**

**Given a string `s`, find the *length of the longest substring* without duplicate characters.**

**A substring is a contiguous sequence of characters within a string.**

- we will use sliding window approach for this
- initialize left ,right to 0 and also max_len to 0
- initialize a hashmap that will keep track of the last seen index of each character
- while right is less than length of string: if check s[right] in hashmap and hash_map[s[right]] > left, what this condition is checking is that if a duplicate character is within window
    - move left index, left = hash_map[s[right]] + 1, we are moving it ahead of the previous position of the duplicate character
- max_len = max( max_len , right -left + 1 i.e current window size)
- hash_map[s[right]] = right # store index of unique characters
- right += 1
- return max_len