# LLM generation prompts

The prompts used to generate the LLM ad explanations (GPT-4o and Gemma2). Each
query is assembled as a fixed **system prompt** plus a **scenario prompt** built
from four parts: the role (perspective) description, the scenario narrative, the
task instructions, and the feature list. The viewer and advertiser role
descriptions are identical to those shown to human participants in the study
preregistration.

Explanations were generated for all 8 scenarios × 2 perspectives (viewer,
advertiser) × 3 target lengths (short / medium / long), with 6 generations per
cell, yielding 288 explanations per model (576 in total).

## System prompt

```
You are an assistant that provides concise, context-specific explanations.

Please read the following scenario and generate 6 different explanations for why the user is seeing this ad. Each explanation should start with '### Explanation X' or '**Explanation X:**', so we can split them easily.
```

## Role (perspective) descriptions

**Viewer**

```
As a viewer of the advertisement, your role is to discern what features would be important to include in the explanation for the advertisement. The main goal of the explanation is to answer the question "Why am I seeing this ad?" for the hypothetical user of the platform, presented with the features describing them. While you may not have access to specific details about the advertisement itself, you can still draw upon your experience and interactions with Video-on-Demand platforms to identify relevant features to provide a compelling and informative explanation.
```

**Advertiser**

```
As an advertiser, your primary goal is to effectively communicate the value and appeal of the product or service being promoted to the target audience. The main goal of the explanation is to answer the question "Why am I seeing this ad?" for the hypothetical user of the platform, presented with the features describing them. While you may not have detailed information about the specific product or service advertised, you can approach the selection of features for the advertisement explanation by considering broader marketing strategies, focusing on brand awareness. Try to identify features that are generally appealing and likely to resonate with consumers across various contexts. Try to create an explanation, which will be persuasive, informative and effectively captures the attention and interest of the audience, even without detailed knowledge about the specific product or service being promoted.
```

## Scenario prompt template

The scenario prompt appends, after the role description above:

```
<scenario narrative: a third-person description of the hypothetical user>

Formulate a message to be presented to the user:
Your objective is to explain why he/she is presented with a targeted advertisement.
Speak directly to the user, from the platform point of view.
Don't mention your role in the explanation.
Include information you consider useful from the list below in your explanation.

  List of information:
<feature list: one "Feature: value" pair per line, from the scenario>

Only use important information from the list as you formulate your explanation.

Start the explanation with "You are seeing this ad, because...".

Provide 6 different explanations. Limit -- <N> characters.
```

The character limit `<N>` is the **target-length** parameter and varied across the
short / medium / long conditions (the worked example below uses 117).

## Full worked example (viewer, scenario 1)

```
As a viewer of the advertisement, your role is to discern what features would be important to include in the explanation for the advertisement. The main goal of the explanation is to answer the question "Why am I seeing this ad?" for the hypothetical user of the platform, presented with the features describing them. While you may not have access to specific details about the advertisement itself, you can still draw upon your experience and interactions with Video-on-Demand platforms to identify relevant features to provide a compelling and informative explanation.

This user is a male aged 18-24, who primarily enjoys watching movies. He is an "intellectual explorer" (his preferred genres include news, current events, science docs, thriller series, travel, fantasy, and sci-fi). He typically watches from 2 to 5 programs per week for less than 2 hours per day (when active), mostly during workdays and between 9:00 AM and 6:00 PM. He prefers watching content on his PC.

Formulate a message to be presented to the user:
Your objective is to explain why he is presented with a targeted advertisement.
Speak directly to the user, from the platform point of view.
Don't mention your role in the explanation.
Include information you consider useful from the list below in your explanation.

  List of information:
Gender: Male,
Age group: 18-24,
Preferable content type: Movies,
Preferable genres: Intellectual Explorer (news, current events, science docs, thriller series, travel, fantasy, sci-fi),
Programs watching per week: 2 to 5,
Hours spent watching content per day (when active): less than 2,
Preferable type of the day for watching content: Workday,
Preferable time of the day for watching content: 9:00 AM - 6:00 PM,
Preferable device: PC.

Only use important information from the list as you formulate your explanation.

Start the explanation with "You are seeing this ad, because...".

Provide 6 different explanations. Limit -- 117 characters.
```

## Generation settings

Both models used the **same prompts** and the **same decoding settings** — `temperature=0.7`, `max_tokens=500`, `seed=123`, 6 explanations per prompt — differing only in the model:

- **Gemma2** — served locally via [Ollama](https://ollama.com).
- **GPT-4o** — OpenAI's GPT-4o (the version current at the time of data collection, late 2024).
