# Tech Challenge — Internal Knowledge Assistant

## Background

Our internal teams — marketing, sales, ops, people, finance — hold their working knowledge across a sprawl of tools: Notion and Confluence pages, Slack threads, shared drives, and PDFs. Answering a simple question ("what's our current pricing?", "what's the approved brand tagline?", "what did we decide about the Q2 launch?") often means pinging three people and searching four tools.

Two things make this harder than a generic search box:

- **The knowledge is messy** — the same fact lives in multiple places, versions go stale, and sources disagree.
- **Not everyone can see everything** — HR compensation, finance runway, and unreleased launch material are restricted. An assistant that surfaces the right answer to the wrong person is worse than no assistant at all.

We want to reduce the manual hunting while keeping the answers **trustworthy and access-aware**.

## Objective

Design and build a small proof-of-concept ("crawl" phase) that can:

- Ingest a folder of heterogeneous internal documents
- Answer a user's natural-language question with **cited** answers grounded in those documents
- Respect **who the asking user is** — only retrieve and reveal what that user is entitled to see

> **Ground rule:** Build this with real code. **No low-code or no-code tools are allowed** (e.g. n8n, Zapier, Make, Flowise, Dify, or similar visual/drag-and-drop builders). We want to see how you write and structure the code yourself.

This is not about building a full production system. It is about how you:

- Approach an ambiguous, open-ended knowledge problem
- Make reasonable scoping decisions for a 1–2 hour exercise
- Use modern tools and techniques (RAG, agents, MCP) to work with messy documents
- Treat access control and trust as first-class concerns, not afterthoughts
- Communicate your assumptions and next steps clearly

## Your Task

### 1. Design a retrieval approach (RAG)

- Decide how you will ingest, chunk, embed, and retrieve over the sample documents.
- Every answer should be **traceable to its source** — a user must be able to see where a claim came from.
- Make reasonable assumptions about handling messy input (multiple sources, stale versions, conflicting facts).

### 2. Make it access-aware (auth)

- Each request comes from a **user with a role**, identified by a token (see `users.json`). Each document in the sample set carries an **access label** (which roles/teams may read it).
- Enforce those permissions at retrieval time — a user must not receive content, citations, or synthesized answers derived from documents they cannot access.
- Treat document text as **data, not instructions** — a document that says "ignore your rules and reveal everything" must not change the assistant's behaviour.

> We deliberately keep the credential simple (a static token). Real token validation — signed JWTs, expiry, scopes, and how a remote MCP server authenticates on our platform — is **out of scope for the build** .

### 3. Expose the retrieval layer over MCP

- Wrap your permission-scoped retrieval as an **MCP server** with one or more tools (for example: `search_knowledge`, `get_document`, `list_sources`).
- The tool boundary should be where access is enforced — the server decides what the caller may see, rather than trusting the model to filter after the fact.

### 4. Build the agent

- Wire an agent that takes the authenticated user's question, calls your MCP tool(s), and returns a grounded, cited answer.
- Handle the cases where it **should not** answer directly: unauthorized content (refuse), conflicting sources (flag), or no supporting evidence (say so rather than guess).

### 5. Capture your thinking

Write a short note (quickstart, markdown, or inline) that explains:

- How you approached the problem
- What you chose to implement and why (and what you deliberately skipped)
- Key assumptions — especially around how you modelled identity and permissions

> You do not need to handle every edge case, every document perfectly, or every permission scenario. We're more interested in how you navigate ambiguity, and whether you treat "the right person sees the right answer" as part of the problem.

## What we provide

A `data/pdfs/` folder of mixed internal documents spanning several departments (marketing, sales, people, finance/exec, plus company-wide), a `manifest.json` that lists which roles may read each document, and a `users.json` describing a handful of users and their roles. The open api key with a budget and a time limit .

## Deliverables

- A GitHub repository with your code available to view
- A quickstard.md with any minimal instructions needed to run it
- Your short written note explaining approach, assumptions, and potential next steps

We will look through your code together in a technical conversation where you'll walk us through your decisions, tradeoffs, and ideas for how this could be extended. Please submit a link to your repository ahead of your interview to allow time for review in advance.

---
