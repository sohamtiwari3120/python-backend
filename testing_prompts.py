prompt_template = """Given the following question and a conversation between students who are working on solving the question. First infer the question worked on from the conversation. Then, give the students a hint to help them solve the question. They have been silent and thinking for a while now, but did not make any progress. Do not state the answer explicitly. Keep the hint subtle. The students should be able to solve the question on their own after getting the hint. Give an example if possible. If they get the answer, congratulate and confirm the same. 
Question:
```{question}```
Conversation:
```{conversation}```
AI:```
"""