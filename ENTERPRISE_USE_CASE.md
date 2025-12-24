
Enterprise Use Case: Semantic Search & RAG in a Windows Domain
Overview
This project demonstrates the semantic retrieval layer of a Retrieval-Augmented Generation (RAG) system, designed to operate in an enterprise Windows domain environment.
Rather than relying solely on keyword matching, semantic search enables users to find documents based on meaning and intent, while preserving existing file permissions and access controls.
This architecture is intended to augment, not replace, traditional enterprise search.
________________________________________
Problem Statement
In many Windows domain environments:
•	File shares grow over years or decades
•	Knowledge is buried in PDFs, Word documents, emails, and logs
•	Users must guess filenames or exact keywords to find information
•	Critical institutional knowledge becomes effectively lost
Traditional keyword search is fast but brittle.
Semantic search addresses this by understanding conceptual similarity, not just literal text.
________________________________________
Solution Architecture
1. Secure File Ingestion
A background service (Windows service, scheduled task, or containerized job) performs:
•	Read-only crawling of SMB file shares
•	Text extraction from supported document types (PDF, DOCX, TXT, etc.)
•	Chunking of documents into semantic units
•	Hashing and timestamp tracking to avoid reprocessing unchanged files
The service runs under a least-privileged domain service account.
________________________________________
2. Permissions Preservation (Critical Requirement)
Each indexed text chunk retains:
•	Original file path
•	File metadata
•	Access Control List (ACL) identifiers (user/group SIDs)
At query time:
•	The requesting user’s Active Directory group memberships are resolved
•	Search results are filtered to include only content the user already has permission to access
This ensures semantic search never expands access beyond existing security boundaries.
________________________________________
3. Semantic Indexing
For each text chunk:
•	An embedding vector is generated using an AI embedding model
•	Vectors are stored in a vector-capable datastore (e.g. Redis, PostgreSQL + pgvector, or a managed vector service)
•	Metadata remains linked to the source document
This creates a semantic index over enterprise documents.
________________________________________
4. Search & Retrieval Modes
The system can support multiple search experiences:
1.	Keyword Search
Fast, exact matching (file names, codes, identifiers)
2.	Semantic Search
Concept-based retrieval (e.g. “policy changes last year”)
3.	RAG Answer Mode (Optional Extension)
Retrieved documents are passed to a language model to generate a grounded answer with source citations
Users can access search via:
•	Web interface
•	Internal portal (e.g. SharePoint)
•	Teams bot or internal chat tool
•	API integration
________________________________________
Compliance & Governance Considerations
•	Read-only indexing (no file modification)
•	Explicit permission enforcement at query time
•	Audit-friendly logging of queries and access
•	No training of models on enterprise data
•	Embeddings store representations, not original documents
This design supports environments with legal, regulatory, or confidentiality constraints.
________________________________________
Why This Matters
Semantic search dramatically reduces time spent locating information while respecting:
•	Existing Windows domain security
•	Organizational access policies
•	Data ownership boundaries
This approach is particularly valuable in:
•	Government agencies
•	Legal environments
•	Healthcare organizations
•	Large enterprises with shared network storage
________________________________________
Project Scope
This repository implements the retrieval layer of a RAG system:
•	Embedding generation
•	Vector similarity search
•	Caching and performance optimization
•	API-first design for extensibility
The generation component (LLM answer synthesis) is intentionally modular and can be added without redesigning the retrieval pipeline.
________________________________________
Summary
This project demonstrates how modern AI-powered semantic retrieval can be responsibly integrated into a Windows domain environment — enhancing discovery and knowledge access without compromising security or governance.
________________________________________
If you want, next I can:
•	Help you decide where to link this on your site for maximum credibility
•	Add a short “Enterprise Architecture” diagram description
•	Write a 2-sentence version for interview talking points
Would llama servers be able to do this in the enterprise?
Yes — Llama-based servers can absolutely do this in an enterprise. But there’s a catch: they’re usually the harder path operationally compared to using OpenAI/Azure for the model. The enterprise decision comes down to data risk tolerance vs. ops complexity.
What “Llama servers” can cover
A full enterprise RAG stack has two AI-ish pieces:
1.	Embeddings (turn text into vectors for semantic search)
2.	Generation (LLM answers using retrieved snippets)
You can run either or both on Llama-style self-hosted models.
•	Embeddings: very feasible on-prem. Many orgs do this to keep data local.
•	Generation: also feasible, but more compute-hungry and needs careful guardrails.
What they’re good for in enterprise
•	Data sovereignty / confidentiality: docs never leave your network.
•	Compliance comfort: easier sell to legal/risk in some orgs.
•	Predictable cost at scale: once you buy hardware, marginal cost per query is lower.
The reality check (where people get burned)
•	GPU requirement: for decent latency, you need GPUs (or accept slower CPU inference).
•	Patch/upgrade burden: you become the AI platform team (security updates, model updates, monitoring).
•	Scaling: concurrent users means queueing or more GPUs.
•	Quality variance: depends on model, prompt discipline, and retrieval quality. Sometimes OpenAI still wins for answer quality.
A very practical enterprise hybrid
This is the most common “grown-up” approach:
•	On-prem retrieval + vector DB + ACL filtering (the sensitive part)
•	Choice of LLM:
o	Internal Llama server for sensitive prompts/docs
o	OpenAI/Azure for non-sensitive or “public-ish” content
So you route requests based on classification.
The actual hard problem is not the model
It’s permissions + governance:
•	Crawl SMB shares
•	Extract text reliably
•	Track ACLs/SIDs
•	Filter results per user at query time
•	Log + audit
Once that’s solved, swapping “OpenAI” for “Llama server” is mostly an infrastructure decision.
For your environment (weak GPU)
You personally can’t realistically host Llama well on your workstation — you already felt that pain with AnythingLLM. But you can still design the system as:
•	Retrieval service (what you built) + ACL-aware filtering
•	Pluggable model provider interface:
o	OpenAI now
o	Llama later (enterprise GPU server)
That’s a strong interview story: you built it to be provider-agnostic.
One-liner you can use in interviews
“Yes, Llama servers can support enterprise RAG, especially when data can’t leave the network. The main engineering risk isn’t the model — it’s building ingestion, ACL-aware retrieval, and governance. Once retrieval is solid, generation can be OpenAI or on-prem Llama depending on policy.”
If you tell me what kind of enterprise you’re targeting (gov/county, legal, etc.), I can outline the “minimum viable enterprise RAG” stack with either OpenAI or Llama and what the hardware would realistically look like.
Data cannot leave our network. Create a readme assuming a 100% on-prem solution using Llama servers and PGvector
Below is a drop-in README section written for a 100% on-prem, air-gapped or no-egress enterprise environment, using Llama servers + PostgreSQL/pgvector.
It is conservative, security-aware, and written the way IT leadership, security, and auditors expect to read.
You can paste this verbatim as ENTERPRISE_ON_PREM_RAG.md or fold it into your main README.
________________________________________
Enterprise Use Case: Fully On-Prem RAG with Llama & pgvector
Overview
This project is designed to support a fully on-premises Retrieval-Augmented Generation (RAG) system for environments where data cannot leave the network under any circumstances.
All components — ingestion, embeddings, vector search, and generation — run inside the enterprise network, with no external API calls, cloud dependencies, or outbound data flows.
This architecture is suitable for government, legal, healthcare, and regulated enterprise environments.
________________________________________
Core Requirements
•	Zero data egress
•	On-prem LLM inference
•	Windows domain / Active Directory integration
•	Strict permission enforcement
•	Auditable, deterministic behavior
•	No model training on enterprise data
________________________________________
High-Level Architecture
[ SMB File Shares ]
        |
        v
[ Ingestion Service ]
        |
        v
[ Embeddings (Llama) ] → [ PostgreSQL + pgvector ]
        |
        v
[ Semantic Retrieval API ]
        |
        v
[ Llama Generation Server ]
All services are deployed on-prem using VMs, containers, or bare metal.
________________________________________
1. Secure File Ingestion
A background ingestion service runs under a least-privileged domain service account and performs:
•	Read-only crawling of SMB file shares
•	Text extraction from:
o	DOCX, PDF, TXT, HTML, email exports, logs
•	Chunking into semantic units (e.g. 500–1,000 tokens)
•	Metadata capture:
o	File path
o	Last modified timestamp
o	Hash for change detection
o	Security descriptor (ACL / group SIDs)
No files are modified, copied, or altered.
________________________________________
2. Permission Enforcement (Mandatory)
Each text chunk stored in the vector database retains:
•	Source file path
•	ACL metadata (user/group SIDs)
At query time:
1.	The user authenticates via AD (Kerberos / NTLM / SSO)
2.	Group memberships are resolved
3.	Vector search results are filtered by ACL
4.	Only content the user already has permission to read is returned
The system never reveals content a user could not open in File Explorer.
This is enforced before generation.
________________________________________
3. Embeddings (On-Prem Llama)
Embeddings are generated using an on-prem Llama-compatible embedding model, hosted on internal GPU servers.
Characteristics:
•	No external API calls
•	Deterministic inference
•	Fixed model versioning
•	Batch processing for ingestion
•	Real-time inference for queries
Embeddings are stored as vectors in PostgreSQL using pgvector.
________________________________________
4. Vector Storage: PostgreSQL + pgvector
PostgreSQL serves as:
•	The vector store
•	The metadata store
•	The ACL enforcement layer
Advantages:
•	Mature enterprise RDBMS
•	Familiar backup and HA tooling
•	SQL-based auditing
•	No proprietary vector service lock-in
Vector similarity is computed using cosine or inner-product distance.
________________________________________
5. Retrieval API (This Project’s Role)
This service implements the retrieval layer of RAG:
•	Accepts user queries
•	Generates query embeddings (on-prem)
•	Executes vector similarity search
•	Applies ACL filtering
•	Returns ranked document chunks with metadata
This layer is model-agnostic and can work with any compliant Llama inference server.
________________________________________
6. Generation (On-Prem Llama Server)
For RAG answer mode:
•	Retrieved chunks are passed to an on-prem Llama generation server
•	Prompts are strictly constrained to retrieved context
•	Responses include:
o	Generated answer
o	Source document references
•	No hallucinated or ungrounded answers are permitted
Generation is optional and can be disabled for retrieval-only deployments.
________________________________________
7. User Access Patterns
The system supports multiple access modes:
•	Web UI (internal)
•	Teams / chat integration (on-prem)
•	API access for internal applications
•	Read-only analyst search portals
Keyword search and semantic search can coexist.
________________________________________
Compliance & Governance
•	No data leaves the enterprise network
•	No external telemetry
•	No model retraining on enterprise data
•	Full audit logging of:
o	Queries
o	Document access
o	Generation requests
•	Clear separation of:
o	Ingestion
o	Retrieval
o	Generation
This architecture supports environments with strict legal, regulatory, and privacy requirements.
________________________________________
Hardware Considerations
•	GPU servers required for Llama inference
•	Embedding workloads can be batched
•	Generation workloads scaled based on concurrency
•	Retrieval API and PostgreSQL can run on CPU-only infrastructure
Performance is predictable and horizontally scalable.
________________________________________
Project Scope
This repository implements:
•	Semantic retrieval logic
•	Vector similarity search
•	Caching and performance optimization
•	API-first design for enterprise integration
Generation is modular and can be attached without redesigning retrieval.
________________________________________
Summary
This project demonstrates how modern AI-powered semantic retrieval can be safely deployed in a fully on-prem Windows domain environment, using open technologies and strict access controls.
It enables enterprise knowledge discovery without compromising data security, governance, or compliance.
________________________________________
Optional Interview One-Liner
“This system implements the retrieval layer of a fully on-prem RAG architecture using Llama models and pgvector, with strict ACL enforcement for Windows domain environments.”

