PleIAs/SYNTH: 
- "Do tropical cyclones typically cause more damage to steel-framed buildings than to wooden structures in coastal regions?"
- "Geographers and oceanographers use several established frameworks to further divide the world's largest ocean, the Pacific. All of the following are recognized methods for subdividing the Pacific Ocean EXCEPT:

A. Dividing it into a northern and a southern component based on ocean circulation patterns.
B. Informally segmenting it into four distinct quadrants using the International Date Line as a reference.
C. Categorizing it into different zones based on the varying depths of its seafloor.
D. Differentiating it into an eastern and a western half based on its relationship to the continents of Asia and the Americas."
- "Hey so like, how come depriving baby animals of seeing stuff makes their brain cells move around more during that critical period but when they just don't develop vision normally, the little spines are all chill?"
- "I need a theatrical piece that captures the dialectical tension between metropolitan cultural authority and journalistic integrity—specifically, could you craft a Latin dialogue set in contemporary New York City that explores the epistemological challenges faced by cultural gatekeepers? I envision this as literary realism that interrogates the institutional memory of publications like New York magazine, perhaps featuring characters drawn from the magazine's editorial lineage—Clay Felker as a mentor figure, Milton Glaser representing aesthetic vision, and contemporary voices like Olivia Nuzzi embodying current cultural criticism. The Latin should elevate the discourse beyond mere journalistic concerns to philosophical questions about cultural authenticity, the commodification of urban experience, and the tension between local specificity and national relevance. I'm particularly interested in how the dialogue form can illuminate the collaborative yet competitive dynamics within editorial communities that shape metropolitan consciousness."


nvidia/Nemotron-Math-v2:
- "Beginning with simplification of the expressions, calculate the derivatives of the following functions. $f(x)=\left(\sqrt {1-x^4}+1\right): \left(\frac{1}{\sqrt {1+x^2}}+\sqrt {1-x^2}\right)$."
- "Consider a static zero-sum game with two players and a payoff matrix \( U \) such that \( \sum_{i \neq j} u_{i,j} = u_{i,i} \) for all \( i \in \{1, \ldots, n\} \). Player 1 and Player 2 choose numbers from 1 to \( n \). If both choose the same number \( i \), Player 1 pays \( u_{i,i} \). Otherwise, if Player 1 chooses \( i \) and Player 2 chooses \( j \), Player 1 earns \( u_{i,j} \). Compute the value of the game (or the expected payoff of the Nash equilibrium under mixed strategies)."
- "For the function \( f(x) = (x^2 - 2mx - 4(m^2 + 1))(x^2 - 4x - 2m(m^2 + 1)) \), determine the value of \( k \) such that \( f(x) = 0 \) has three distinct real roots when \( m = k \)."
- "Determine the minimum value of $\frac{IA^3+IB^3+IC^3}{AB^3+BC^3+CA^3}$, where $ABC$ is any triangle, and $I$ is the incenter."

open-r1/OpenR1-Math-220k
- "1. Solve the equation: $\frac{8 x+13}{3}=\frac{19-12 x}{2}-\left(16 x-\frac{7-4 x}{6}\right)$."
- "1. Let the universal set be the set of real numbers. If $A=\{x \mid \sqrt{x-2} \leqslant 0\}, B=\left\{x \mid 10^{x^{2}-2}=10^{x}\right\}$, then $A \cap \bar{B}$ is
A. $\{2\}$
B. $\{-1\}$
C. $\{x \mid x \leqslant 2\}$
D. $\varnothing$"

PrimeIntellect/Hendrycks-Math:
- "Calculate: $(243)^{\frac35}$"
- "Find the slope of the line that is perpendicular to the line $2x + 3y = 6$."

qwedsacf/competition_math
- "Evaluate $\left\lceil-\sqrt{\frac{49}{4}}\right\rceil$."

nvidia/OpenCodeReasoning-2
**Cant get these prompts yet, waiting for access**

nvidia/Nemotron-Comptetitive-Programming-v1
**Cant get these prompts yet, waiting for access**

nvidia/OpenCodeInstruct:
- "You are given a list of integers `numbers`. Your task is to implement a function `find_mode` that determines the mode of the list, which is the element that appears most frequently. In the event of a tie, where multiple elements share the highest frequency, return the smallest of these elements. If the list is empty, the function should return `None`.

**Input:**
- A list of integers `numbers` (0 ≤ len(numbers) ≤ 10^5), where each integer is in the range [-10^9, 10^9].

**Output:**
- An integer representing the mode of the list, or `None` if the list is empty.

**Sample Input 1:**
```
[1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5]
```

**Sample Output 1:**
```
4
```

**Sample Input 2:**
```
[6, 6, 7, 7, 8]
```

**Sample Output 2:**
```
6
```

**Sample Input 3:**
```
[]
```

**Sample Output 3:**
```
None
```"
- "You are given a list of integers. Your task is to implement a function `find_mode` that determines the mode(s) of the list. The mode is the number that appears most frequently. If there are multiple numbers with the same highest frequency, return all of them in a list. The input list will contain at least one integer.

**Sample Input:**
```
[1, 2, 2, 3, 3, 4]
```

**Sample Output:**
```
[2, 3]
```"

agentica-org/DeepCoder-Preview-Dataset
- "The two versions are different problems. You may want to read both versions. You can make hacks only if both versions are solved.

You are given two positive integers $n$, $m$.

Calculate the number of ordered pairs $(a, b)$ satisfying the following conditions:

* $1\le a\le n$, $1\le b\le m$; * $b \cdot \gcd(a,b)$ is a multiple of $a+b$.

Input Format:
Each test contains multiple test cases. The first line contains the number of test cases $t$ ($1\le t\le 10^4$). The description of the test cases follows.

The first line of each test case contains two integers $n$, $m$ ($1\le n,m\le 2 \cdot 10^6$).

It is guaranteed that neither the sum of $n$ nor the sum of $m$ over all test cases exceeds $2 \cdot 10^6$.

Output Format:
For each test case, print a single integer: the number of valid pairs.

Tags:
['brute force', 'math', 'number theory']

Time Limit: 2000 ms
Memory Limit: 256 MB"
- "Monocarp is opening his own IT company. He wants to hire $n$ programmers and $m$ testers.

There are $n+m+1$ candidates, numbered from $1$ to $n+m+1$ in chronological order of their arriving time. The $i$-th candidate has programming skill $a_i$ and testing skill $b_i$ (a person's programming skill is different from their testing skill). The skill of the team is the sum of the programming skills of all candidates hired as programmers, and the sum of the testing skills of all candidates hired as testers.

When a candidate arrives to interview, Monocarp tries to assign them to the most suitable position for them (if their programming skill is higher, then he hires them as a programmer, otherwise as a tester). If all slots for that position are filled, Monocarp assigns them to the other position.

Your task is, for each candidate, calculate the skill of the team if everyone except them comes to interview. Note that it means that exactly $n+m$ candidates will arrive, so all $n+m$ positions in the company will be filled.

Input Format:
The first line contains a single integer $t$ ($1 \le t \le 10^4$) — the number of test cases.

Each test case consists of three lines:

* the first line contains two integers $n$ and $m$ ($0 \le n, m \le 2 \cdot 10^5$; $2 \le n + m + 1 \le 2 \cdot 10^5$) — the number of programmers and the number of testers Monocarp wants to hire, respectively; * the second line contains $n + m + 1$ integers $a_1, a_2, \dots, a_{n+m+1}$ ($1 \le a_i \le 10^9$), where $a_i$ is the programming skill of the $i$-th candidate; * the third line contains $n + m + 1$ integers $b_1, b_2, \dots, b_{n+m+1}$ ($1 \le b_i \le 10^9$; $b_i \ne a_i$), where $b_i$ is the testing skill of the $i$-th candidate.

Additional constraint on the input: the sum of $(n + m + 1)$ over all test cases doesn't exceed $2 \cdot 10^5$.

Output Format:
For each test case, print $n + m + 1$ integers, where the $i$-th integer should be equal to the skill of the team if everyone except the $i$-th candidate comes to interview.

Tags:
['binary search', 'dp', 'greedy', 'implementation', 'two pointers']

Time Limit: 2000 ms
Memory Limit: 256 MB"

HuggingFaceH4/ultrachat_200k:
- "De León, previewing the speech he will give today, said he will highlight his Senate Bill 535, which directs a quarter of the proceeds from the Greenhouse Gas Reduction Fund to projects that benefit disadvantaged communities.
On Thursday, de León nodded approvingly as a string of leading scientists and religious leaders gathered for hours of weedy policy discussions on the impacts of climate change, including gloomy predictions on mortality attributable to air pollution.
SEIU HEADS TO THE BAR: Employees of the State Bar of California represented by SEIU are planning a picket line today at the bar building in Los Angeles to protest the latest contract offer. What is the reason for SEIU employees planning a picket line at the State Bar of California building in Los Angeles?"
- "Can you explain the organizational structure of the Habsburg imperial court and the roles of various officials?"

OpenAssistant/oasst1:
- "Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research."

bcui19/chat-v2-anthropic-helpfulness
- "How do I teach my kids to speak French?"
- "I'm planning on incorporating a business in the station of Maine. Can you tell me what papers I have to file with the state to create a legal entity?"

NousResearch/hermes-function-calling-v1
- "[
{
"from": "system",
"value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"set_smart_light_color\", \"description\": \"Sets the color and brightness of smart lights in a specified room.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"room\": {\"type\": \"string\", \"description\": \"The room where the smart lights are located.\"}, \"brightness\": {\"type\": \"string\", \"description\": \"The desired brightness level.\", \"enum\": [\"bright\", \"dim\", \"off\"]}, \"color\": {\"type\": \"string\", \"description\": \"The desired color setting.\", \"enum\": [\"warm\", \"cool\", \"daylight\"]}}, \"required\": [\"room\", \"brightness\", \"color\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"sync_lights_with_automation_system\", \"description\": \"Syncs the smart lights with the home automation system for a specified room.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"room\": {\"type\": \"string\", \"description\": \"The room where the smart lights are located.\"}}, \"required\": [\"room\"]}}}]\n</tools>\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-dict>}\n</tool_call>\n"
},
{
"from": "human",
"value": "I am preparing my living room for a cozy movie night and I would like to set the mood with the appropriate lighting. Could you please adjust the smart lights in the living room to emit a warm glow at a dim brightness level? Additionally, I need the lights to be in sync with my home automation system to create a seamless experience. For the function call, please use the living room as the specified room, set the brightness to 'dim', and the color to 'warm'. Then, ensure that the lights are well-integrated with the automation system in the living room."
},
{
"from": "gpt",
"value": "<tool_call>\n{\"name\": \"set_smart_light_color\", \"arguments\": {\"room\": \"living room\", \"brightness\": \"dim\", \"color\": \"warm\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"sync_lights_with_automation_system\", \"arguments\": {\"room\": \"living room\"}}\n</tool_call>\n"
}
]"