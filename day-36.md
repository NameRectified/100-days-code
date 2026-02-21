### Chat roles and system messages

- Single turn tasks: eg: text generation, classification
- Multi turn conversations: build on previous prompts and responses
- Roles:
    - System: allows us to specify a message to control the behaviour of the assistant. eg: if it is a customer bot, we can state that the assistant is polite and helpful customer service assistant
        - use it for templating the format of the response
        - use it also to include guardrails
    - User: used to instruct the assistant
        - use it to provide the context required for new input (often single turn)
    - Assistant: response to user instruction
        - can also be written by the developer to provide examples
        - for single turn tasks, no content is sent to the assistant role as the model uses only existing knowledge, the system message and user prompt to generate the response
        - use this to provide example conversations
    
    ```jsx
    client = OpenAI(api_key="<OPENAI_API_TOKEN>")
    
    # Create a request to the Chat Completions endpoint
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      max_completion_tokens=150,
      messages=[
        {"role": "system",
         "content": "You are a study planning assistant that creates plans for learning new skills."},
        {"role": "user",
         "content": "I want to learn to speak Dutch."}
      ]
    )
    
    # Extract the assistant's text response
    print(response.choices[0].message.content)
    ```
    
    ### adding assistant messages
    
    ```jsx
    client = OpenAI(api_key="<OPENAI_API_TOKEN>")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # Add a user and assistant message for in-context learning
        messages=[
            {"role": "system", "content": "You are a helpful Geography tutor that generates concise summaries for different countries."},
            {"role": "user", "content": "Give me a quick summary of Portugal."},
            {"role": "assistant", "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa."},
            {"role": "user", "content": "Give me a quick summary of Greece."}
        ]
    )
    
    print(response.choices[0].message.content)
    ```
    
    ### more examples
    
    ```jsx
    client = OpenAI(api_key="<OPENAI_API_TOKEN>")
    
    response = client.chat.completions.create(
       model="gpt-4o-mini",
       # Add in the extra examples and responses
       messages=[
           {"role": "system", "content": "You are a helpful Geography tutor that generates concise summaries for different countries."},
           {"role": "user", "content": "Give me a quick summary of Portugal."},
           {"role": "assistant", "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa."},
           {"role": "user", "content": example1},
           {"role": "assistant", "content": response1},
           {"role": "user", "content": example2},
           {"role": "assistant", "content": response2},
           {"role": "user", "content": example3},
           {"role": "assistant", "content": response3},
           {"role": "user", "content": "Give me a quick summary of Greece."}
       ]
    )
    
    print(response.choices[0].message.content)
    ```
    

### coding a conversation

```jsx
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = [
    {"role": "system", "content": "You are a helpful math tutor that speaks concisely."},
    {"role": "user", "content": "Explain what pi is."}
]

# Send the chat messages to the model
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_completion_tokens=100
)

# # Extract the assistant message from the response
assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}

# # Add assistant_dict to the messages dictionary
messages.append(assistant_dict)
print(messages)
```

creating an ai chatbot

```jsx
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

messages = [{"role": "system", "content": "You are a helpful math tutor that speaks concisely."}]
user_msgs = ["Explain what pi is.", "Summarize this in two bullet points."]

# Loop over the user questions
for q in user_msgs:
    print("User: ", q)
    
    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)
    
    # Create the API request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=100
    )
    
    # Append the assistant's message to messages
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")
```

### Mitigating misuse

- System message can include guardrails that are specific instructions on what the model can generate
- eg: place a restriction on model outputs preventing learning plans *not* related to languages, as your system is beginning to find its niche in that space. You'll design a custom message for users requesting these type of learning plans so they understand this change.

```jsx
client = OpenAI(api_key="<OPENAI_API_TOKEN>")

sys_msg = """You are a study planning assistant that creates plans for learning new skills.

If these skills are non related to languages, return the message:

'Apologies, to focus on languages, we no longer create learning plans on other topics.'
"""

# Create a request to the Chat Completions endpoint
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": sys_msg},
    {"role": "user", "content": "Help me learn to program."}
  ]
)

print(response.choices[0].message.content)
```

### Planning a trip to Paris with openai api project

```jsx

conversation = [{"role": "system", 
                 "content": "You are a travel tour assistant. Be polite and concise. Give only destination and tourism related suggestions."},
               {"role":"user", 
                "content": "What is the most famous landmark in Paris?"
               },
                {
                        "role":"assistant",
                        "content":"The most famous landmark in Paris is the Eiffel Tower."
                    }]
user_msgs = ["How far away is the Louvre from the Eiffel Tower (in miles) if you are driving?", "Where is the Arc de Triomphe?","What are the must-see artworks at the Louvre Museum?" ]

# Loop over the user questions
for q in user_msgs:
    print("User: ", q)
    
    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    conversation.append(user_dict)
    
    # Create the API request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        max_completion_tokens=100,
        temperature=0.0
    )
    
    # Append the assistant's message to messages
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    conversation.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")
```

## Python toolbox

### Iterators

- iter is used to convert an iterable to an iterator
- next is used to get the next value

```jsx
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for item in flash:
    print(item)

# Create an iterator for flash: superhero
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

```

- `range()` doesn't actually create the list; instead, it creates a range object with an iterator that produces the values until it reaches the limit
- If `range()` created the actual list, calling it with a value of  may not work, especially since a number as big as that may go over a regular computer's memory.

```jsx
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10**100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))

```

### enumerate

```jsx
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1,value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)

```

### zip

```jsx
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)
```

### **Using *(splat) and zip to 'unzip'**

```jsx
# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

```

### **Using iterators to load large files into memory**

- if the data is so large to be held in memory, one possible solution is to load it in chunks

```jsx
# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)

```

### making it into a function
```
# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)
```