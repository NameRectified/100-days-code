## Day 4

### Recursion

- It is a function calling itself.
- It uses a stack called `call stack` to keep track of the function calls.
- A recursive function contains two cases:
    - base case: when the recursion must be stopped
    - recurive case: the function calling itself
- Without a base case a recursion will run indefinitely and the computer may run out of memory
- There are two types of recursion:
    - Direct
    - Indirect

- `NOTE`: When designing a recursive function, assume any recursive call to that function will behave as intended, even if the function hasn’t been fully implemented - Coding Interview Patterns. 

### Stack

- It is a `last in - first out` data structure. LIFO for short.
- Recursions use stacks. 
- When a function is called, a block is created in the stack, when another function is called, the current function execution is paused and a new block is created on top of the old block. This continues until the last function call.
- When the last call is made, the top most block is popped and the value is returned to the previous call, this continues till the first function call and then the stack becomes empty.
- Stack has two operations: `push` to add elements and `pop` to remove and read the element.