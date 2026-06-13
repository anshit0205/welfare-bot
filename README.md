# Welfare Scheme Assistant

## Overview

The Welfare Scheme Assistant is an AI-powered multilingual advisory system designed to help citizens discover, understand, and access government welfare schemes through natural language conversations. The system supports multiple Indian languages and provides personalized scheme guidance, eligibility assessment, document requirements, application assistance, and real-time welfare information through an intuitive conversational interface.

The platform combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), semantic search, memory management, caching, and real-time web search to deliver accurate, contextual, and user-centric assistance.

---

# Key Features

## Multilingual Conversational Interface

The system supports interactions in multiple Indian languages:

* English
* Hindi (हिंदी)
* Marathi (मराठी)
* Bengali (বাংলা)
* Tamil (தமிழ்)

Users can select their preferred language at the beginning of the conversation or naturally interact in their chosen language. Responses remain consistent in the selected language throughout the session.

---

## Intelligent Welfare Scheme Discovery

Users can ask questions in natural language without needing to know the exact scheme name.

Examples:

* "What is the NREGA wage in Uttarakhand?"
* "Schemes for farmers"
* "Benefits for widows"
* "Scholarships for students"

The assistant identifies user intent, understands context, and retrieves the most relevant welfare schemes from its knowledge base.

---

## Personalized Eligibility Assessment

The system conducts an interactive eligibility assessment by collecting relevant household and demographic information.

Information collected may include:

* Occupation
* Age
* State of residence
* Land ownership
* Annual household income
* BPL status
* Presence of daughters in the family
* Pregnancy or maternal status within the household

Using this information, the assistant evaluates eligibility across multiple schemes and generates personalized recommendations.

---

## Retrieval-Augmented Generation (RAG)

The platform uses a hybrid Retrieval-Augmented Generation architecture that combines:

* Dense vector retrieval (FAISS)
* Sparse keyword retrieval (BM25)
* Large Language Models

This approach grounds responses in verified welfare scheme data and minimizes hallucinations.

---

## Hybrid Search Engine

The retrieval system employs:

### FAISS Semantic Search

Dense vector embeddings are generated using multilingual sentence-transformer models, enabling semantic understanding of user queries.

### BM25 Keyword Search

Traditional lexical search captures exact keyword matches, ensuring strong retrieval performance for scheme names, document names, and official terminology.

### Weighted Reciprocal Rank Fusion (RRF)

Results from semantic and keyword retrieval are combined using weighted rank fusion, improving relevance and retrieval accuracy.

---

## Context-Aware Conversations

The assistant maintains conversational context across interactions.

Examples:

User:

> What is the NREGA wage in West Bengal?

User:

> What about Madhya Pradesh?

The system understands that the follow-up question refers to NREGA wages and provides the appropriate state-specific response without requiring the user to repeat the scheme name.

---

## Session Memory

The platform maintains a persistent user session that stores:

* Language preference
* Collected eligibility information
* Previous conversation context
* Recently discussed schemes
* Conversation summaries

This enables smoother multi-turn interactions and reduces repetitive questioning.

---

## Query-Based Caching

The platform implements exact-query caching for frequently repeated requests.

When a user submits a query that has already been processed previously, the system can return the cached response instead of performing a new retrieval and inference cycle. This reduces response latency and minimizes repeated computation.

The caching layer is particularly effective for commonly asked welfare-related questions such as wage rates, scheme benefits, document requirements, and application procedures.

## Real-Time Web Search Integration

For queries involving changing information, the assistant can perform real-time web searches to supplement knowledge-base results.

This helps answer questions involving:

* Updated wages
* New scheme announcements
* Policy changes
* Recently launched government initiatives

The system intelligently decides when external information is required.

---

## Intelligent Intent Classification

A dedicated intent classification pipeline identifies user objectives such as:

* Scheme information requests
* Eligibility checks
* Document requirements
* Application procedures
* General welfare inquiries

Intent detection enables appropriate routing and response generation.

---

## Document Checklist Generation

For eligible schemes, the assistant generates personalized document checklists based on scheme requirements.

Examples include:

* Aadhaar Card
* Income Certificate
* Bank Passbook
* BPL Card
* Land Records
* Residence Proof

This helps users prepare application documents efficiently.

---

## State-Specific Welfare Information

Many welfare schemes contain state-dependent benefits and eligibility criteria.

The assistant can retrieve and present:

* State-wise wage rates
* Regional benefits
* Localized eligibility requirements
* State-specific implementation details

---

## Safety and Guardrails

The system incorporates multiple guardrail mechanisms to ensure reliable responses.

### Response Validation

Generated responses are validated before being delivered to users.

### Hallucination Reduction

Knowledge-grounded retrieval minimizes unsupported claims and factual inaccuracies.

### Controlled Eligibility Decisions

Eligibility recommendations are based on explicit scheme rules rather than free-form model assumptions.

### Error Recovery

Fallback mechanisms ensure graceful handling of unexpected failures and incomplete information.

---

## Usage Analytics and Monitoring

The platform records operational metrics for monitoring and optimization.

Tracked information includes:

* Request latency
* Token usage
* Model utilization
* Retrieval performance
* User interaction statistics

These analytics support system evaluation and continuous improvement.

---

## Modular Architecture

The system is organized into modular components including:

* Intent Classification
* Eligibility Assessment
* Retrieval Engine
* Memory Manager
* Caching Layer
* Translation Utilities
* Web Search Integration
* Response Generation

This architecture promotes maintainability, extensibility, and scalability.

---

# Core Technologies

* FastAPI
* NVIDIA NIM APIs
* Llama Models
* FAISS
* BM25
* Sentence Transformers
* Tavily Search
* SQLite Session Storage
* Exact Caching
* Retrieval-Augmented Generation (RAG)

---

# Summary

The Welfare Scheme Assistant combines multilingual conversational AI, intelligent retrieval, eligibility reasoning, contextual memory, semantic caching, and real-time information retrieval to provide citizens with accurate, personalized, and accessible welfare scheme guidance. The system is designed to simplify welfare discovery, improve information accessibility, and assist users throughout their scheme exploration and eligibility assessment journey.
