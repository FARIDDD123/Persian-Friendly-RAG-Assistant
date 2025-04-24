from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
