## Day 35

Here is your content rewritten in clean, properly structured **Markdown format**:

---

# OpenAI Chat Completions Examples (Python)

## Create the OpenAI Client

```python
from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")
```

---

## Basic Text Generation

### Example: Accepting a Job Offer

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Write a polite reply accepting an AI Engineer job offer."
        }
    ]
)

print(response.choices[0].message.content)
```

---

## 3️⃣ Text Replacement

### Replace "car" with "plane"

```python
prompt = """
Replace car with plane and adjust phrase:
A car is a vehicle that is typically powered by an internal combustion engine or an electric motor.
It has four wheels, and is designed to carry passengers and/or cargo on roads or highways.
Cars have become a ubiquitous part of modern society, and are used for a wide variety of purposes,
such as commuting, travel, and transportation of goods. Cars are often associated with freedom,
independence, and mobility.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=100
)

print(response.choices[0].message.content)
```

---

## Text Summarization

```python
finance_text = "..."  # Your finance content here

prompt = f"""
Summarize the following text into two concise bullet points:
{finance_text}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=400
)

print(response.choices[0].message.content)
```

---

## Estimating API Cost

Before deploying AI features at scale, estimate usage cost based on token consumption and pricing.

```python
max_completion_tokens = 200  # Example value

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=max_completion_tokens
)

input_token_price = 0.15 / 1_000_000
output_token_price = 0.6 / 1_000_000

input_tokens = response.usage.prompt_tokens
output_tokens = max_completion_tokens

cost = (input_tokens * input_token_price) + (output_tokens * output_token_price)

print(f"Estimated cost: ${cost}")
```

---

## Text Generation with Temperature

The `temperature` parameter controls randomness:

- Lower → More deterministic
- Higher → More creative/random

### Example: Restaurant Slogan

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Create a slogan for an Indian restaurant"}
    ],
    max_completion_tokens=100
)

print(response.choices[0].message.content)
```

---

### Example: Product Description

```python
prompt = """
Generate a persuasive product description for SonicPro headphones with:
- Active Noise Cancellation (ANC)
- 40-hour battery life
- Foldable design
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=200,
    temperature=0.5
)

print(response.choices[0].message.content)
```

---

# Prompting Techniques

## Zero-Shot Prompting

No examples provided — just instructions.

```python
prompt = """
Classify the sentiment of the given statements using numbers 1 to 5:
1. Unbelievably good!
2. Shoes fell apart on the second use.
3. The shoes look nice, but they aren't very comfortable.
4. Can't wait to show them off!
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=100
)

print(response.choices[0].message.content)
```

---

## One-Shot Prompting

One example included.

```python
prompt = """
Classify sentiment as 1–5 (negative to positive):
1. Love these! = 5
2. Unbelievably good! =
3. Shoes fell apart on the second use. =
4. The shoes look nice, but they aren't very comfortable. =
5. Can't wait to show them off! =
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=100
)

print(response.choices[0].message.content)
```

---

## Few-Shot Prompting

Multiple examples provided for better guidance.

```python
prompt = """
Classify sentiment as 1–5 (negative to positive):
1. Comfortable but not pretty = 2
2. Love these! = 5
3. Unbelievably good! =
4. Shoes fell apart on the second use. =
5. The shoes look nice, but they aren't very comfortable. =
6. Can't wait to show them off! =
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=100
)

print(response.choices[0].message.content)
```

---