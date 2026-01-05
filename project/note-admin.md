# Notes Admin: Complete Study & Implementation Execution Plan

> **Project Goal:** Build an AI-powered Obsidian plugin that serves as an autonomous research partner and knowledge OS, with multi-agent architecture suitable for academic publication.

> **Time Commitment:** 4 hours/day  
> **Total Duration:** ~60 days (12 weeks)  
> **Dual Purpose:** Production Obsidian Plugin + AI Agent Research Paper

---

## Table of Contents

1. [Executive Summary](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#1-executive-summary)
2. [Project Overview](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#2-project-overview)
3. [Knowledge Gap Analysis](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#3-knowledge-gap-analysis)
4. [Reference Systems Study Guide](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#4-reference-systems-study-guide)
5. [Technology Stack Decisions](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#5-technology-stack-decisions)
6. [Study Phase: Weeks 1-3](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#6-study-phase-weeks-1-3)
7. [Implementation Phase: Weeks 4-10](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#7-implementation-phase-weeks-4-10)
8. [Evaluation & Paper Phase: Weeks 11-12](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#8-evaluation--paper-phase-weeks-11-12)
9. [Architecture Reference](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#9-architecture-reference)
10. [Risk Mitigation](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#10-risk-mitigation)
11. [Resource Library](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#11-resource-library)
12. [Daily Checklist Templates](https://claude.ai/chat/390ee355-f14d-4bd9-89bc-cdde33dffa18#12-daily-checklist-templates)

---

## 1. Executive Summary

### Timeline Overview

|Phase|Weeks|Days|Focus|
|---|---|---|---|
|**Study**|1-3|1-21|TypeScript, Obsidian, Agents, RAG|
|**Implementation**|4-10|22-70|Build complete plugin|
|**Evaluation & Paper**|11-12|71-84|Experiments, paper writing|

### Key Deliverables

1. **Week 3:** Working mini-prototype (RAG + single agent)
2. **Week 6:** Core agent framework with 4 specialized agents
3. **Week 8:** Complete safety layer with diff-based review
4. **Week 10:** Release-ready Obsidian plugin
5. **Week 12:** Paper draft ready for submission

### Success Criteria

- [ ] Plugin installable from Obsidian community plugins
- [ ] All 7 core features functional
- [ ] Multi-agent coordination working
- [ ] Evaluation metrics collected
- [ ] Paper submitted to target venue

---

## 2. Project Overview

### 2.1 The Core 7 Features

|#|Feature|Agent|Description|
|---|---|---|---|
|1|**Template-Based Tree Organization**|Librarian|Monitor inbox folder, classify files based on user-defined rules in `Directory_Structure.md`, move files to appropriate locations with user confirmation|
|2|**Smart De-Duplication**|Librarian|Detect semantically similar notes using embeddings, present side-by-side comparison, offer merge/delete options|
|3|**Provenance & Citation Tracking**|Researcher|When generating summaries or merging notes, automatically inject source citations linking back to original files and line numbers|
|4|**Dialectical Synthesis**|Researcher|Detect contradictions between notes (e.g., "Paper A claims X" vs "Experiment B shows not-X"), alert user and offer to create synthesis note|
|5|**Visual Knowledge Graph 2.0**|Teacher|Extend Obsidian's graph with: red edges for conflicts, faded nodes for decayed knowledge (not reviewed in 6+ months), interactive actions|
|6|**Project Command Center**|Manager|Auto-generate dashboard notes aggregating tasks, deadlines, related files, and health metrics per project|
|7|**Conversational Init Wizard**|Manager|Chat-based setup that creates folder structure and templates through natural conversation|

### 2.2 Multi-Agent Architecture

The system uses four specialized agents coordinated by an orchestrator:

**Librarian Agent**

- Responsibility: File organization, hygiene, deduplication
- Tools: file_read, file_move, file_list, template_parse, similarity_check
- Proactive: Monitors inbox folder for new files

**Researcher Agent**

- Responsibility: Provenance tracking, conflict detection, synthesis
- Tools: rag_search, citation_inject, conflict_detect, summarize
- Proactive: Scans for contradictions between recently modified notes

**Teacher Agent**

- Responsibility: Knowledge decay tracking, active recall, quiz generation
- Tools: decay_check, quiz_generate, review_schedule
- Proactive: Daily check for notes not reviewed in threshold period

**Manager Agent**

- Responsibility: Dashboard creation, task extraction, project health
- Tools: dashboard_create, task_extract, health_metrics
- Proactive: Updates dashboards when project files change

**Orchestrator**

- Decomposes user requests into sub-tasks
- Dispatches to appropriate agent(s)
- Aggregates results and handles agent delegation
- Manages shared context between agents

### 2.3 Paper Contribution Angles

1. **Novel Multi-Agent Architecture for PKM** - First system with specialized agents for personal knowledge management
2. **Proactive vs. Reactive Paradigm** - Agents that anticipate needs rather than just respond
3. **Trust-Calibrated Autonomy** - Configurable human-in-the-loop based on task risk
4. **Long-Term State Management** - Agents maintain awareness across sessions

### 2.4 Target Publication Venues

| Venue    | Fit   | Why                            | Deadline  |
| -------- | ----- | ------------------------------ | --------- |
| **CHI**  | ⭐⭐⭐⭐⭐ | HCI + AI, knowledge work focus | September |
| **UIST** | ⭐⭐⭐⭐⭐ | Novel UI techniques            | April     |
| **IUI**  | ⭐⭐⭐⭐  | Intelligent user interfaces    | October   |
| **CSCW** | ⭐⭐⭐⭐  | Collaborative knowledge work   | April     |

---

## 3. Knowledge Gap Analysis

### 3.1 Your Current Skills

✅ **Strong Foundation:**

- LLM systems infrastructure and optimization
- Distributed systems algorithms
- Academic research and paper writing
- Python programming
- Machine learning fundamentals

### 3.2 Skills to Acquire

|Skill|Priority|Why Needed|Study Hours|
|---|---|---|---|
|**TypeScript**|Critical|Obsidian plugins are TypeScript|12|
|**Obsidian Plugin API**|Critical|Target platform|12|
|**Agent Patterns (ReAct)**|Critical|Core architecture|10|
|**Tool Calling (Claude/OpenAI)**|Critical|Agent execution|6|
|**RAG Implementation**|Critical|Knowledge retrieval|8|
|**Vector Stores (Orama)**|Critical|Local search|4|
|**DOM/CSS Basics**|Important|UI components|8|
|**Multi-Agent Coordination**|Important|Orchestrator design|6|

**Total Study Hours: ~66 hours (16-17 days at 4 hrs/day)**

### 3.3 Learning Dependencies

```
TypeScript Basics
       │
       ▼
Obsidian Plugin API ──────┐
       │                  │
       ▼                  ▼
   RAG Implementation    UI Components
       │                  │
       └────────┬─────────┘
                │
                ▼
         Agent Patterns
                │
                ▼
      Multi-Agent Coordination
```

---

## 4. Reference Systems Study Guide

### 4.1 PaperDebugger Analysis

**What it is:** A Chrome extension for Overleaf that provides multi-agent AI assistance for academic writing.

**Repository:** https://github.com/PaperDebugger/paperdebugger  
**Paper:** arXiv:2512.02589

**Architecture Overview:**

- Frontend: Chrome Extension (TypeScript) injected into Overleaf
- Backend: Go + Gin HTTP server + gRPC for internal communication
- Agent Layer: Kubernetes pods for isolated agent execution
- Tool Protocol: MCP (Model Context Protocol) for standardized tool interface
- Storage: MongoDB for state, arXiv vector store for literature

**Key Patterns to Study:**

|Pattern|How PaperDebugger Does It|How to Adapt for Notes Admin|
|---|---|---|
|**Agent Pipeline**|Critique → Enhance → Patch (sequential)|Librarian → Researcher → Teacher (parallel capable)|
|**Tool Protocol**|MCP with JSON schemas|Simplified TypeScript interfaces|
|**Safety Layer**|Diff-based patches, never auto-apply|Same approach - accumulate edits, show diff, require approval|
|**Isolation**|Each agent in K8s pod|Logical separation in TypeScript classes|
|**State Management**|MongoDB persistence|Obsidian plugin data storage|

**What to Read in Their Code:**

1. `proto/` folder - Understand their API contracts and tool definitions
2. `internal/agent/` - How they structure agent logic
3. `internal/mcp/` - MCP tool protocol implementation
4. `webapp/` - Chrome extension TypeScript patterns

**Key Takeaways for Notes Admin:**

- Diff-based patching is essential for user trust
- Tool protocols should be well-defined and typed
- Agent isolation helps with debugging and testing
- Pipeline patterns work well for sequential tasks

---

### 4.2 Obsidian-Copilot Analysis

**What it is:** The most popular AI plugin for Obsidian with RAG-based vault search and agentic capabilities.

**Repository:** https://github.com/logancyang/obsidian-copilot  
**Stars:** 5.6k+

**Architecture Overview:**

- Pure TypeScript plugin running locally in Obsidian
- RAG using Orama (pure JS vector store)
- Multiple LLM provider support (OpenAI, Anthropic, Ollama)
- Agent mode with autonomous tool calling (Plus version)

**Key Patterns to Study:**

|Pattern|How Obsidian-Copilot Does It|How to Adapt for Notes Admin|
|---|---|---|
|**Vault QA**|BM25 fallback + optional embeddings|Hybrid search from day one|
|**@ Commands**|`@vault`, `@web`, `@youtube` for context injection|Similar command system for agent dispatch|
|**Composer**|Select text → Apply AI action → Preview → Apply|Same workflow for edits|
|**Settings**|Obsidian native settings tab|Same approach|
|**Streaming**|Real-time response rendering|Same for chat interface|

**What to Read in Their Code:**

1. `src/main.ts` - Plugin lifecycle, command registration, view setup
2. `src/settings/` - How they structure and persist settings
3. `src/services/llm/` - Provider abstraction pattern
4. `src/services/vectorStore/` - Orama integration
5. `src/agents/` - Agent tool calling implementation

**Key Takeaways for Notes Admin:**

- Orama works well for local vector search in Obsidian
- Hybrid search (BM25 + vector) improves results
- Provider abstraction is essential for flexibility
- Obsidian's native UI components are sufficient

---

### 4.3 Comparative Analysis

|Aspect|PaperDebugger|Obsidian-Copilot|Notes Admin (Planned)|
|---|---|---|---|
|**Platform**|Chrome Extension|Obsidian Plugin|Obsidian Plugin|
|**Backend**|Go + Kubernetes|Local TypeScript|Local TypeScript|
|**Agents**|Multi-agent pipeline|Single agent + tools|Multi-agent + orchestrator|
|**RAG**|arXiv vector store|Orama (local)|Orama (local)|
|**Safety**|Diff-based patching|Composer preview|Diff-based patching|
|**Proactive**|No|Limited|Yes (scheduled scans)|
|**Tool Protocol**|MCP (formal)|Custom (informal)|Custom (typed)|

**What to Adopt from Each:**

From PaperDebugger:

- Multi-agent pipeline architecture
- Diff-based safety layer
- Formal tool definitions
- Audit logging

From Obsidian-Copilot:

- Orama vector store setup
- Provider abstraction
- Settings management
- UI component patterns
- Streaming responses

---

## 5. Technology Stack Decisions

### 5.1 Core Stack

|Component|Choice|Rationale|
|---|---|---|
|**Language**|TypeScript|Required for Obsidian plugins|
|**Plugin Framework**|Obsidian API|Target platform|
|**Vector Store**|Orama|Pure JS, proven in Obsidian, no native dependencies|
|**LLM Provider**|Claude API (primary)|Best tool calling, falls back to OpenAI|
|**Embeddings**|OpenAI API|text-embedding-3-small, cost-effective|
|**Diff Library**|diff (npm)|Standard, well-tested|
|**Validation**|Zod|Runtime type checking for tool params|
|**PDF Parsing**|unpdf|Better text extraction than alternatives|

### 5.2 Why These Choices

**Orama over alternatives:**

- sqlite-vec requires native bindings (complex in Obsidian)
- Pinecone/Weaviate require external servers
- Orama is pure JS, works in browser environment, sufficient for vault-scale

**Claude over OpenAI for agents:**

- Better instruction following
- More reliable tool calling
- Longer context window
- Can switch to OpenAI as fallback

**Zod for validation:**

- Runtime type checking for LLM outputs
- Integrates well with TypeScript
- Can generate JSON schemas for tool definitions

### 5.3 Development Environment

|Tool|Purpose|
|---|---|
|VS Code|IDE with TypeScript support|
|ESLint + Prettier|Code quality|
|esbuild|Fast bundling|
|Jest|Unit testing|
|Obsidian (dev vault)|Live testing|

---

## 6. Study Phase: Weeks 1-3

### Week 1: TypeScript & Obsidian Fundamentals

#### Day 1 (4 hours): TypeScript Basics I

**Topics to Cover:**

- Basic types: string, number, boolean, array, object
- Type annotations vs type inference
- Interfaces: defining object shapes
- Optional properties and readonly modifiers
- Type aliases vs interfaces (when to use each)

**Learning Approach:**

1. Read TypeScript Handbook chapters 1-3 (2 hours)
2. Complete TypeScript exercises on exercism.io (1 hour)
3. Convert a simple Python script to TypeScript (1 hour)

**Completion Checklist:**

- [ ] Can define interfaces with nested objects
- [ ] Understand when TypeScript infers types
- [ ] Can annotate function parameters and returns
- [ ] Know difference between `interface` and `type`

---

#### Day 2 (4 hours): TypeScript Basics II

**Topics to Cover:**

- Generics: generic functions, interfaces, and classes
- Generic constraints with `extends`
- Union types (`A | B`) and intersection types (`A & B`)
- Type guards and type narrowing
- Utility types: Partial, Required, Pick, Omit, Record

**Learning Approach:**

1. Read TypeScript Handbook: Generics chapter (1.5 hours)
2. Read TypeScript Handbook: Narrowing chapter (1 hour)
3. Practice: Define generic Tool interface (1.5 hours)

**Completion Checklist:**

- [ ] Can write generic functions with constraints
- [ ] Can use union types for message types
- [ ] Can write type guards for runtime checking
- [ ] Know when to use Partial vs Required

---

#### Day 3 (4 hours): Async TypeScript & Modules

**Topics to Cover:**

- Promises: creation, chaining, error handling
- async/await syntax and patterns
- Error handling with try/catch in async code
- ES6 modules: import/export syntax
- Default vs named exports
- Module resolution in TypeScript

**Learning Approach:**

1. Read javascript.info async section (1.5 hours)
2. Read TypeScript Handbook: Modules (1 hour)
3. Practice: Write async functions with error handling (1.5 hours)

**Completion Checklist:**

- [ ] Can write async functions with proper error handling
- [ ] Understand Promise.all for parallel operations
- [ ] Can structure code across multiple modules
- [ ] Know difference between default and named exports

---

#### Day 4 (4 hours): Obsidian Plugin Basics I

**Topics to Cover:**

- Plugin structure: main.ts, manifest.json, styles.css
- Plugin class and lifecycle: onload(), onunload()
- The App object and its properties
- Vault API: accessing and manipulating files
- TFile and TFolder classes

**Learning Approach:**

1. Read Obsidian Developer Docs: Getting Started (1 hour)
2. Clone and study obsidian-sample-plugin (1 hour)
3. Create "Hello World" plugin that loads successfully (2 hours)

**Setup Steps:**

1. Clone obsidian-sample-plugin
2. Run `npm install` and `npm run dev`
3. Create test vault at ~/test-vault
4. Symlink plugin to test vault's plugins folder
5. Enable plugin in Obsidian settings

**Completion Checklist:**

- [ ] Plugin loads without errors
- [ ] Understand manifest.json fields
- [ ] Can access this.app.vault
- [ ] Console.log works in Obsidian dev tools

---

#### Day 5 (4 hours): Obsidian Plugin Basics II

**Topics to Cover:**

- Reading file content with vault.read()
- Creating files with vault.create()
- Modifying files with vault.modify()
- Moving/renaming with vault.rename()
- Deleting with vault.delete()
- File events: create, modify, delete, rename

**Learning Approach:**

1. Read Obsidian API reference for Vault class (1 hour)
2. Implement file operations in test plugin (2 hours)
3. Set up event listeners and test (1 hour)

**Exercises:**

1. List all markdown files in vault
2. Read content of a specific file
3. Create a new note with content
4. Watch for file creation events

**Completion Checklist:**

- [ ] Can read any file in vault
- [ ] Can create files in specific folders
- [ ] Can modify existing files
- [ ] Can listen for file change events

---

#### Day 6 (4 hours): Obsidian UI Components

**Topics to Cover:**

- Modal class: creating popup dialogs
- Setting class: building settings UI
- PluginSettingTab: settings page
- Notice: toast notifications
- ItemView: custom sidebar views
- Commands: registering keyboard shortcuts

**Learning Approach:**

1. Read Obsidian Docs: User Interface section (1 hour)
2. Study UI code in obsidian-copilot (1 hour)
3. Build simple modal and settings tab (2 hours)

**Exercises:**

1. Create modal with input field and buttons
2. Build settings tab with text input and toggle
3. Register command with keyboard shortcut
4. Show notice on command execution

**Completion Checklist:**

- [ ] Can create modal dialogs
- [ ] Can build settings tab
- [ ] Can register commands
- [ ] Can show notifications

---

#### Day 7 (4 hours): Study Obsidian-Copilot Codebase

**Objective:** Deep dive into how a production Obsidian AI plugin is structured

**Files to Study (with time allocation):**

1. **src/main.ts** (1 hour)
    
    - How they initialize services
    - Command registration patterns
    - View registration
    - Plugin lifecycle management
2. **src/settings/** (30 minutes)
    
    - Settings interface structure
    - Default values pattern
    - Settings persistence
3. **src/services/llm/** (1 hour)
    
    - Provider abstraction pattern
    - How they handle multiple LLM providers
    - Streaming response handling
4. **src/services/vectorStore/** (1 hour)
    
    - Orama initialization
    - Index schema definition
    - Search implementation

**Notes to Take:**

- Document their folder structure rationale
- Note patterns you want to reuse
- Identify potential improvements for Notes Admin
- List questions for further research

**Completion Checklist:**

- [ ] Can explain obsidian-copilot's architecture
- [ ] Documented their provider abstraction pattern
- [ ] Understand their RAG implementation approach
- [ ] Have list of patterns to adopt

---

### Week 2: Agent Patterns & RAG

#### Day 8 (4 hours): ReAct Pattern Deep Dive

**Topics to Cover:**

- ReAct paper key concepts
- Thought-Action-Observation loop
- When to think vs when to act
- Stopping conditions
- Error handling in agent loops

**Learning Approach:**

1. Read ReAct paper (arXiv:2210.03629) - focus on sections 1-3 (2 hours)
2. Watch explanatory videos on ReAct (30 minutes)
3. Diagram the ReAct loop for Notes Admin use cases (1.5 hours)

**Key Concepts to Understand:**

- **Thought:** Agent's internal reasoning about what to do next
- **Action:** Tool invocation with specific parameters
- **Observation:** Result from tool execution
- **Iteration:** Loop continues until goal achieved or max steps

**Design Exercise:** Map the ReAct loop to a specific Notes Admin task: "Organize my inbox" → What thoughts, actions, observations would occur?

**Completion Checklist:**

- [ ] Can explain ReAct pattern to someone else
- [ ] Understand when agent should think vs act
- [ ] Can design ReAct flow for file organization task
- [ ] Know how to handle errors in agent loop

---

#### Day 9 (4 hours): LLM Tool Calling

**Topics to Cover:**

- Claude tool/function calling API
- Tool definition structure (name, description, parameters)
- Handling tool_use responses
- Multi-turn conversations with tools
- OpenAI function calling (for comparison)

**Learning Approach:**

1. Read Anthropic Tool Use documentation (1.5 hours)
2. Study tool definition examples (1 hour)
3. Design tool definitions for Notes Admin (1.5 hours)

**Tools to Design (conceptually):**

1. `file_read` - Read file content
2. `file_write` - Write/modify file
3. `file_move` - Move file to new location
4. `rag_search` - Search vault semantically
5. `get_metadata` - Get file metadata

**For Each Tool, Define:**

- Name and description (for LLM to understand)
- Input parameters with types
- Expected output format
- Error conditions

**Completion Checklist:**

- [ ] Understand Claude tool calling flow
- [ ] Can design tool definitions with proper schemas
- [ ] Know how to handle tool results
- [ ] Have draft tool designs for Notes Admin

---

#### Day 10 (4 hours): RAG Fundamentals

**Topics to Cover:**

- Document chunking strategies
- Why chunking matters for retrieval
- Embedding generation process
- Vector similarity search
- Hybrid search (BM25 + vector)

**Learning Approach:**

1. Read about chunking strategies (1 hour)
2. Study obsidian-copilot's chunking code (1 hour)
3. Learn about reciprocal rank fusion (1 hour)
4. Design chunking strategy for Notes Admin (1 hour)

**Chunking Strategies to Understand:**

|Strategy|Description|Best For|
|---|---|---|
|Fixed size|Split every N tokens|Simple documents|
|Semantic|Split by headers/sections|Markdown|
|Sliding window|Overlapping chunks|Dense text|
|Recursive|Hierarchical splitting|Complex documents|

**For Notes Admin:**

- Primary: Semantic chunking (respect Markdown headers)
- Fallback: Paragraph-based for flat documents
- Metadata: Preserve file path, line numbers, headers

**Completion Checklist:**

- [ ] Understand why chunking strategy matters
- [ ] Can explain hybrid search benefits
- [ ] Have chunking strategy designed for Markdown
- [ ] Know how to preserve provenance metadata

---

#### Day 11 (4 hours): Vector Store with Orama

**Topics to Cover:**

- Orama architecture and concepts
- Schema definition
- Indexing documents
- Vector search configuration
- Hybrid search implementation
- Persistence strategies

**Learning Approach:**

1. Read Orama documentation (1.5 hours)
2. Study obsidian-copilot's Orama usage (1 hour)
3. Design schema for Notes Admin (1.5 hours)

**Schema Design Considerations:**

- `content`: The chunk text (string, searchable)
- `embedding`: Vector representation (vector[1536] for OpenAI)
- `path`: Source file path (string, filterable)
- `lineStart`/`lineEnd`: Source location (number)
- `header`: Section header if any (string)
- `modified`: Last modification time (number, for decay)

**Persistence Strategy:**

- Store index in `.obsidian/plugins/notes-admin/index.json`
- Rebuild incrementally on file changes
- Full rebuild option for consistency

**Completion Checklist:**

- [ ] Understand Orama's data model
- [ ] Have schema designed for Notes Admin
- [ ] Know how to implement hybrid search
- [ ] Have persistence strategy planned

---

#### Day 12 (4 hours): Multi-Agent Patterns

**Topics to Cover:**

- Agent specialization patterns
- Inter-agent communication
- Orchestrator responsibilities
- Delegation protocols
- Conflict resolution between agents

**Learning Approach:**

1. Read PaperDebugger paper sections on multi-agent (1.5 hours)
2. Read CAMEL paper on multi-agent (arXiv:2303.17760) (1 hour)
3. Design Notes Admin multi-agent architecture (1.5 hours)

**Patterns to Understand:**

|Pattern|Description|Use Case|
|---|---|---|
|Pipeline|Sequential agent execution|Critique → Enhance → Apply|
|Parallel|Concurrent agent execution|Search + Analyze simultaneously|
|Hierarchical|Orchestrator dispatches to specialists|User request → appropriate agent|
|Collaborative|Agents share context and iterate|Complex synthesis tasks|

**For Notes Admin:**

- Hierarchical as primary pattern
- Orchestrator analyzes request, dispatches to agent(s)
- Agents can delegate to other agents
- Shared context through orchestrator

**Completion Checklist:**

- [ ] Understand different multi-agent patterns
- [ ] Can explain orchestrator role
- [ ] Have delegation protocol designed
- [ ] Know how agents will share context

---

#### Day 13 (4 hours): Study PaperDebugger Architecture

**Objective:** Extract architectural patterns from PaperDebugger

**Reading Plan:**

1. Paper (arXiv:2512.02589) - Sections 2-4 (2 hours)
2. GitHub repo structure analysis (1 hour)
3. Document learnings for Notes Admin (1 hour)

**Key Sections to Focus On:**

- Section 2: System Overview - architecture diagram
- Section 3: Agent Layer - how agents are structured
- Section 4: Workflows - real usage patterns

**Questions to Answer:**

1. How do they define agent roles?
2. How do agents communicate?
3. How is the diff patching implemented?
4. What safety mechanisms exist?
5. How do they handle errors?

**Architecture Elements to Document:**

- Agent definition structure
- Tool protocol (MCP)
- Edit/patch workflow
- Audit logging approach

**Completion Checklist:**

- [ ] Can draw PaperDebugger architecture diagram
- [ ] Documented their agent definition pattern
- [ ] Understand their safety mechanisms
- [ ] Have notes on what to adapt for Notes Admin

---

#### Day 14 (4 hours): Integration Exercise

**Objective:** Build mini-prototype combining Week 1-2 learnings

**Build a simple plugin that:**

1. Has a command to search vault
2. Uses Orama for vector search
3. Runs simple agent to summarize results
4. Shows results in modal

**Components to Implement:**

- Basic Orama setup with hardcoded test data
- Simple embedding generation (can mock initially)
- Single-agent that takes query and returns summary
- Modal to display results

**This exercise validates:**

- TypeScript skills are sufficient
- Obsidian plugin basics work
- RAG concepts understood
- Agent pattern can be implemented

**Completion Checklist:**

- [ ] Plugin loads and runs command
- [ ] Orama search returns results
- [ ] Agent processes results
- [ ] Modal displays output

---

### Week 3: UI & Advanced Integration

#### Day 15 (4 hours): Diff Implementation

**Topics to Cover:**

- Computing text differences
- Line-level vs character-level diffs
- Rendering diff visualization
- Side-by-side vs unified diff
- Accept/reject interaction patterns

**Learning Approach:**

1. Study `diff` npm package documentation (1 hour)
2. Design diff modal UI (1 hour)
3. Plan accept/reject workflow (1 hour)
4. Study how PaperDebugger shows diffs (1 hour)

**UI Design Decisions:**

- Side-by-side view (clearer for larger changes)
- Color coding: green for additions, red for deletions
- Per-hunk accept/reject for granular control
- "Accept All" and "Reject All" buttons
- Preview of final result

**Workflow Design:**

1. Agent proposes edits → accumulated in pending list
2. User triggers review → diff modal opens
3. User reviews each change
4. Accepted changes applied atomically
5. Audit log updated

**Completion Checklist:**

- [ ] Understand diff library capabilities
- [ ] Have UI mockup for diff modal
- [ ] Workflow designed end-to-end
- [ ] Know how to handle partial acceptance

---

#### Day 16 (4 hours): Chat Interface

**Topics to Cover:**

- Obsidian ItemView for sidebar panels
- Message rendering (user vs assistant)
- Streaming response display
- Input handling (Enter to send, Shift+Enter for newline)
- Source citation display

**Learning Approach:**

1. Study obsidian-copilot's chat implementation (1.5 hours)
2. Design chat UI components (1 hour)
3. Plan message data structure (1 hour)
4. Design streaming update mechanism (30 minutes)

**UI Components Needed:**

- Chat container (scrollable message list)
- Message bubble (different styles for user/assistant)
- Input area (textarea + send button)
- Source links (clickable to open note)
- Agent indicator (which agent is responding)
- Typing indicator (while streaming)

**Message Data Structure:**

- id: unique identifier
- role: user | assistant | system
- content: message text
- timestamp: when sent
- agent: which agent responded (for assistant)
- sources: array of SearchResult (for citations)
- edits: array of proposed edits (if any)

**Completion Checklist:**

- [ ] Understand ItemView implementation
- [ ] Have chat UI designed
- [ ] Message structure defined
- [ ] Know how to handle streaming

---

#### Day 17 (4 hours): Settings & Persistence

**Topics to Cover:**

- Obsidian settings API
- Settings interface design
- Default values pattern
- Secure API key storage
- Settings migration for updates

**Learning Approach:**

1. Study obsidian-copilot's settings (1 hour)
2. Design Notes Admin settings structure (1.5 hours)
3. Plan settings UI layout (1 hour)
4. Consider migration scenarios (30 minutes)

**Settings Categories:**

|Category|Settings|
|---|---|
|**API**|LLM provider, API key, model selection|
|**Embedding**|Provider, model, dimensions|
|**Agents**|Enable/disable each agent|
|**Safety**|Require approval, auto-backup|
|**Paths**|Inbox folder, template file, dashboard folder|
|**Schedule**|Scan intervals for proactive features|
|**Advanced**|Max tokens, temperature, debug mode|

**Security Considerations:**

- API keys stored in plugin data (not plain text in settings.json)
- Never log API keys
- Validate API keys before saving

**Completion Checklist:**

- [ ] Settings interface designed
- [ ] All settings categories defined
- [ ] Security approach planned
- [ ] Migration strategy considered

---

#### Days 18-19 (8 hours): Proactive Scheduling

**Topics to Cover:**

- Background task scheduling in Obsidian
- File system watching for changes
- Notification aggregation
- Priority queue for tasks
- Resource management (not blocking UI)

**Learning Approach:**

1. Study Obsidian's event system (2 hours)
2. Design scheduler architecture (2 hours)
3. Plan notification UX (2 hours)
4. Consider performance implications (2 hours)

**Scheduled Tasks to Design:**

| Task             | Interval  | Agent      | Trigger                      |
| ---------------- | --------- | ---------- | ---------------------------- |
| Inbox scan       | 5 min     | Librarian  | Also on file create in inbox |
| Conflict scan    | Daily     | Researcher | Also on file modify          |
| Decay check      | Daily     | Teacher    | Manual trigger available     |
| Dashboard update | On change | Manager    | File modify in project       |

**Notification Strategy:**

- Don't spam user with individual notifications
- Aggregate related notifications
- Single notification: "Notes Admin: 5 items need attention"
- Click to open notification center
- Mark as dismissed/actioned

**Performance Considerations:**

- Use requestIdleCallback for non-urgent tasks
- Debounce file change handlers
- Index updates should be incremental
- Long-running tasks show progress

**Completion Checklist:**

- [ ] Scheduler architecture designed
- [ ] All scheduled tasks defined
- [ ] Notification UX planned
- [ ] Performance strategy documented

---

#### Days 20-21 (8 hours): Integration & Testing

**Objective:** Combine all Week 3 components and test

**Integration Tasks:**

1. Wire diff modal to agent edit output
2. Connect chat view to orchestrator
3. Implement settings persistence
4. Set up basic scheduled task
5. End-to-end test: query → search → agent → diff → apply

**Testing Scenarios:**

|Scenario|Components Tested|
|---|---|
|Simple query|Chat → Agent → Response|
|Query with search|Chat → RAG → Agent → Response with sources|
|Edit proposal|Agent → Diff Modal → Accept → File modified|
|Settings change|Settings tab → Save → Reload → Verify|
|Inbox notification|File watcher → Notification → Action|

**Bug Categories to Watch:**

- Async race conditions
- Memory leaks from event listeners
- UI not updating after state change
- Settings not persisting
- Files not found errors

**Completion Checklist:**

- [ ] All components integrated
- [ ] Test scenarios pass
- [ ] No critical bugs
- [ ] Ready for implementation phase

---

## 7. Implementation Phase: Weeks 4-10

### Overview

|Phase|Week|Days|Deliverable|
|---|---|---|---|
|0: Setup|4|22-24|Project scaffold, types, evaluation framework|
|1: RAG|4-5|25-31|Complete RAG pipeline|
|2: Agent Core|5-6|32-42|Base agent + 4 specialized agents|
|3: Orchestrator|7|43-49|Multi-agent coordination|
|4: Safety|8|50-56|Diff-based review system|
|5: UI|9|57-63|Chat, dashboard, graph|
|6: Integration|10|64-70|Polish, testing, release prep|

---

### Phase 0: Project Setup (Days 22-24)

#### Day 22 (4 hours): Project Scaffold

**Tasks:**

1. Initialize npm project with TypeScript
2. Install all dependencies
3. Configure ESLint and Prettier
4. Set up esbuild for bundling
5. Create directory structure
6. Create manifest.json
7. Verify plugin loads in Obsidian

**Directory Structure to Create:**

```
notes-admin/
├── src/
│   ├── agents/base/
│   ├── agents/
│   ├── tools/
│   ├── rag/chunker/
│   ├── rag/
│   ├── safety/
│   ├── ui/chat/
│   ├── ui/dashboard/
│   ├── ui/graph/
│   ├── ui/
│   ├── scheduler/
│   ├── llm/prompts/
│   ├── llm/
│   ├── evaluation/
│   ├── main.ts
│   ├── types.ts
│   └── settings.ts
├── evaluation/
├── docs/
├── paper/
└── [config files]
```

**Deliverable:** Empty plugin that loads successfully

---

#### Day 23 (4 hours): Type Definitions

**Tasks:** Create comprehensive type definitions in `src/types.ts`:

1. **Agent Types**
    
    - AgentName (union of agent names)
    - AgentRole (name, prompt, tools, delegation)
    - AgentState (goal, thoughts, actions, observations)
    - AgentMessage (thought | action | observation | final)
2. **Tool Types**
    
    - Tool (name, description, parameters, execute)
    - ToolCall (tool name, params, timestamp)
    - ToolResult (success, data, error)
    - ToolContext (vault, settings, vectorStore)
3. **RAG Types**
    
    - Chunk (content, embedding, metadata)
    - ChunkMetadata (path, lines, header)
    - SearchResult (chunk, score, highlights)
4. **Edit Types**
    
    - Edit (type, path, content, agent, status)
    - EditBatch (multiple edits with summary)
5. **UI Types**
    
    - ChatMessage (role, content, sources, edits)
    - Notification (type, message, action)
6. **Settings Types**
    
    - NotesAdminSettings (all configuration)
7. **Evaluation Types**
    
    - EvaluationMetrics (for paper)

**Deliverable:** Complete type definitions for entire project

---

#### Day 24 (4 hours): Evaluation Framework

**Tasks:** Create evaluation infrastructure in `src/evaluation/`:

1. **metrics.ts**
    
    - MetricsCollector class
    - Start/end task tracking
    - Token usage recording
    - Tool call counting
    - Success/failure tracking
2. **logger.ts**
    
    - Trace logging for agent execution
    - Structured log format for analysis
    - Export to JSON for paper
3. **benchmark.ts**
    
    - Test scenario definitions
    - Automated task execution
    - Result comparison

**Metrics to Track:**

- Task success rate (per agent, per task type)
- Latency (end-to-end, per tool call)
- Token usage (input, output, total)
- Tool calls (count, which tools)
- Edit acceptance rate
- User satisfaction (if collected)

**Deliverable:** Metrics collection ready for experiments

---

### Phase 1: RAG Subsystem (Days 25-31)

#### Days 25-26 (8 hours): Document Chunker

**Tasks:** Create chunking system in `src/rag/chunker/`:

1. **markdown.ts**
    
    - Parse Markdown structure
    - Split by headers (h1-h6)
    - Handle code blocks (don't split mid-block)
    - Handle lists (keep together)
    - Preserve frontmatter
    - Track line numbers
2. **pdf.ts**
    
    - Use unpdf for extraction
    - Page-based chunking
    - Handle multi-column layouts
    - Extract metadata (title, author)
3. **index.ts**
    
    - Unified chunker interface
    - File type detection
    - Chunk size configuration
    - Overlap handling

**Chunking Parameters:**

- Max chunk size: 500 tokens (configurable)
- Overlap: 50 tokens
- Min chunk size: 100 tokens (don't create tiny chunks)

**Deliverable:** Chunker handles Markdown and PDF

---

#### Days 27-28 (8 hours): Embedding Service

**Tasks:** Create embedding service in `src/rag/embeddings.ts`:

1. **Provider Abstraction**
    
    - EmbeddingProvider interface
    - OpenAI implementation
    - Ollama implementation (local)
2. **Batch Processing**
    
    - Batch multiple texts
    - Respect rate limits
    - Progress callback
3. **Caching**
    
    - Cache embeddings in memory during session
    - Optional: store in frontmatter
    - Cache invalidation on content change
4. **Error Handling**
    
    - Retry with backoff
    - Fallback behavior
    - Clear error messages

**Deliverable:** Embedding service with provider switching

---

#### Days 29-30 (8 hours): Vector Store

**Tasks:** Create vector store in `src/rag/store.ts`:

1. **Orama Setup**
    
    - Schema definition
    - Index creation
    - Configuration options
2. **CRUD Operations**
    
    - Insert chunks
    - Update existing
    - Delete by file path
    - Clear all
3. **Search Operations**
    
    - Vector similarity search
    - BM25 text search
    - Hybrid search with fusion
4. **Persistence**
    
    - Save index to disk
    - Load on plugin start
    - Handle corrupted index

**Deliverable:** Persistent vector store with hybrid search

---

#### Day 31 (4 hours): Indexer & Retriever

**Tasks:**

1. **indexer.ts**
    
    - Full vault indexing
    - Incremental updates
    - File change handling
    - Progress reporting
    - Exclusion patterns
2. **retriever.ts**
    
    - High-level search API
    - Result formatting
    - Source tracking
    - Relevance filtering

**Indexing Strategy:**

- On plugin load: check if index exists, rebuild if needed
- On file create/modify: update affected chunks
- On file delete: remove chunks
- Manual full rebuild option

**Deliverable:** Working RAG pipeline end-to-end

---

### Phase 2: Agent Core (Days 32-42)

#### Days 32-34 (12 hours): Base Agent & ReAct Loop

**Tasks:** Create base agent infrastructure in `src/agents/base/`:

1. **agent.ts**
    
    - BaseAgent class
    - Role configuration
    - Tool binding
    - Execution entry point
2. **loop.ts**
    
    - ReAct loop implementation
    - Thought generation
    - Action execution
    - Observation recording
    - Termination conditions
    - Max iterations guard
3. **state.ts**
    
    - AgentState management
    - History tracking
    - Context window management
    - State serialization

**Loop Implementation Details:**

1. Initialize state with goal
2. Generate thought (what should I do?)
3. If thought indicates done → return final answer
4. Generate action (which tool, what params?)
5. Execute tool, record observation
6. Add to history, continue loop
7. Max 10 iterations, then force conclusion

**Deliverable:** Working single-agent execution

---

#### Days 35-37 (12 hours): Tool System

**Tasks:** Create tool infrastructure in `src/tools/`:

1. **registry.ts**
    
    - Tool registration
    - Tool discovery by name
    - Parameter validation (Zod)
    - Execution wrapper
2. **file-ops.ts**
    
    - file_read: Read file content
    - file_write: Write through safety layer
    - file_move: Move file
    - file_list: List folder contents
    - file_delete: Delete through safety layer
3. **rag-tools.ts**
    
    - rag_search: Semantic search
    - rag_similar: Find similar notes
    - rag_context: Get context around chunk
4. **analysis-tools.ts**
    
    - llm_analyze: Sub-LLM call
    - summarize: Generate summary
    - extract_entities: Find key entities
5. **dashboard-tools.ts**
    
    - dashboard_create: Create project dashboard
    - task_extract: Find tasks in notes
    - health_metrics: Calculate project health

**Tool Design Principles:**

- Each tool does one thing well
- Clear parameter descriptions for LLM
- Consistent error format
- All file modifications go through safety layer

**Deliverable:** Complete tool registry with 12+ tools

---

#### Days 38-42 (20 hours): Specialized Agents

**Tasks:** Create four specialized agents:

**Day 38-39: Librarian Agent (8 hours)** `src/agents/librarian.ts`

Responsibilities:

- Parse Directory_Structure.md template
- Classify incoming files
- Move files to correct locations
- Detect duplicates
- Report broken links

System Prompt Focus:

- File organization expert
- Follows user's template exactly
- Conservative (asks before moving)
- Explains reasoning

Tools Used:

- file_read, file_list, file_move
- template_parse (custom for librarian)
- similarity_check (for dedup)

---

**Day 39-40: Researcher Agent (8 hours)** `src/agents/researcher.ts`

Responsibilities:

- Track provenance for all generated content
- Inject citations [Source: file.md#L45]
- Detect contradictions between notes
- Generate synthesis notes

System Prompt Focus:

- Academic rigor
- Always cite sources
- Identify conflicts explicitly
- Balanced synthesis

Tools Used:

- rag_search, file_read
- citation_inject (custom)
- conflict_detect (custom)
- summarize

---

**Day 40-41: Teacher Agent (8 hours)** `src/agents/teacher.ts`

Responsibilities:

- Track knowledge decay (last review date)
- Identify notes needing review
- Generate quiz questions
- Schedule spaced repetition

System Prompt Focus:

- Encouraging tone
- Effective quiz questions
- Spaced repetition awareness
- Learning science principles

Tools Used:

- decay_check (custom)
- quiz_generate (custom)
- rag_search
- update_frontmatter

---

**Day 41-42: Manager Agent (8 hours)** `src/agents/manager.ts`

Responsibilities:

- Create project dashboards
- Extract tasks from notes
- Track project health
- Conversational project setup

System Prompt Focus:

- Project management expertise
- Task prioritization
- Clear dashboard formatting
- Proactive suggestions

Tools Used:

- dashboard_create, task_extract
- file_read, file_write
- health_metrics

**Deliverable:** Four working specialized agents

---

### Phase 3: Orchestrator (Days 43-49)

#### Days 43-45 (12 hours): Orchestrator Implementation

**Tasks:** Create orchestrator in `src/agents/orchestrator.ts`:

1. **Goal Decomposition**
    
    - Analyze user request
    - Break into sub-tasks
    - Determine required agents
    - Create execution plan
2. **Agent Dispatch**
    
    - Route tasks to agents
    - Handle parallel execution where possible
    - Manage sequential dependencies
    - Track task status
3. **Result Aggregation**
    
    - Combine agent outputs
    - Resolve conflicts
    - Generate unified response
    - Collect all proposed edits
4. **Delegation Handling**
    
    - Agent A can request Agent B
    - Pass context between agents
    - Prevent infinite delegation

**Orchestrator Prompts:**

- Planning prompt: Given request, which agents needed?
- Synthesis prompt: Combine these agent outputs
- Conflict prompt: These agents disagree, resolve

**Deliverable:** Working multi-agent coordination

---

#### Days 46-47 (8 hours): Proactive Scheduler

**Tasks:** Create scheduler in `src/scheduler/`:

1. **proactive.ts**
    
    - Schedule registration
    - Interval-based execution
    - File watcher integration
    - Task prioritization
2. **queue.ts**
    
    - Priority queue for tasks
    - Debouncing
    - Batch processing
    - Resource limiting

**Scheduled Tasks:**

|Task|Interval|Implementation|
|---|---|---|
|Inbox scan|5 min|Check inbox folder, notify if files present|
|Decay check|Daily|Find notes not modified in 6 months|
|Conflict scan|Hourly|Compare recently modified notes|
|Dashboard update|On change|Regenerate when project files change|

**Deliverable:** Background task scheduling working

---

#### Days 48-49 (8 hours): Agent Communication

**Tasks:**

1. **messaging.ts**
    
    - Message format for inter-agent communication
    - Context passing protocol
    - Result format standardization
2. **context.ts**
    
    - Shared context management
    - Context window optimization
    - Relevant context selection

**Communication Protocol:**

- Delegation request: { to: AgentName, task: string, context: string[] }
- Delegation response: { success: boolean, result: string, edits?: Edit[] }
- Context sharing: Summarized, not raw (save tokens)

**Deliverable:** Inter-agent messaging working

---

### Phase 4: Safety Layer (Days 50-56)

#### Days 50-52 (12 hours): Edit System

**Tasks:** Create safety infrastructure in `src/safety/`:

1. **accumulator.ts**
    
    - Collect edits from agents
    - Group by file
    - Batch related edits
    - Track edit provenance
2. **diff.ts**
    
    - Compute diffs using `diff` library
    - Handle create/modify/delete/move
    - Generate unified diff format
    - Statistics (additions, deletions)
3. **audit.ts**
    
    - Log all proposed edits
    - Log all applied edits
    - Timestamp and agent attribution
    - Export for analysis

**Edit Lifecycle:**

1. Agent proposes edit → Added to accumulator
2. Batch complete → Diff computed
3. User reviews → Accepts/rejects each
4. Accepted edits → Applied atomically
5. All edits → Logged to audit

**Deliverable:** Edit collection and diff generation

---

#### Days 53-54 (8 hours): Human Review UI

**Tasks:** Create diff modal in `src/ui/diff-modal.ts`:

1. **Modal Structure**
    
    - Header with summary stats
    - Scrollable diff view
    - Per-hunk controls
    - Action buttons
2. **Diff Rendering**
    
    - Side-by-side or unified (user preference)
    - Syntax highlighting for code
    - Line numbers
    - Color coding
3. **Interaction**
    
    - Accept individual changes
    - Reject individual changes
    - Accept all / Reject all
    - Preview final result

**UX Flow:**

1. User clicks "Review Changes" notification
2. Modal opens with all pending edits
3. User can expand/collapse each file
4. Accept/reject at file or hunk level
5. "Apply" button commits accepted changes

**Deliverable:** Diff review modal with accept/reject

---

#### Days 55-56 (8 hours): Trust Calibration

**Tasks:** Create trust system in `src/safety/trust.ts`:

1. **Autonomy Levels**
    
    - Full manual: All edits require approval
    - Smart: Low-risk auto-apply, high-risk manual
    - Full auto: Apply all (not recommended)
2. **Risk Assessment**
    
    - File delete = high risk
    - File move = medium risk
    - Content add = low risk
    - Content modify = medium risk
3. **Per-Agent Settings**
    
    - Librarian: Default medium autonomy
    - Researcher: Default low autonomy
    - Teacher: Default high autonomy (just frontmatter)
    - Manager: Default medium autonomy
4. **Undo Support**
    
    - Track applied edits
    - Store original content
    - Undo last batch option

**Deliverable:** Configurable trust system with undo

---

### Phase 5: UI (Days 57-63)

#### Days 57-59 (12 hours): Chat Interface

**Tasks:** Create chat UI in `src/ui/chat/`:

1. **view.ts**
    
    - ItemView implementation
    - Layout management
    - State management
2. **messages.ts**
    
    - Message component
    - Different styles per role
    - Source citations
    - Edit proposals
3. **Streaming**
    
    - Real-time token display
    - Typing indicator
    - Cancellation support
4. **Input**
    
    - Textarea with auto-resize
    - Send button
    - Keyboard shortcuts
    - @ command support (future)

**Chat Features:**

- Conversation history (in memory)
- Clear conversation
- Export conversation
- Agent indicator (which agent responding)
- Source links (click to open note)
- Pending edits indicator

**Deliverable:** Full chat interface with streaming

---

#### Days 60-61 (8 hours): Dashboard View

**Tasks:** Create dashboard in `src/ui/dashboard/`:

1. **view.ts**
    
    - Custom view for project dashboard
    - Project selector
    - Refresh functionality
2. **Components**
    
    - Task list with checkboxes
    - File list for project
    - Health metrics display
    - Quick actions

**Dashboard Content:**

- Project name and description
- Task list (from extracted tasks)
- Recent files in project
- Health metrics (staleness, completeness)
- Quick actions (new note, scan, etc.)

**Deliverable:** Project dashboard view

---

#### Days 62-63 (8 hours): Knowledge Graph Enhancement

**Tasks:** Create graph enhancements in `src/ui/graph/`:

1. **renderer.ts**
    
    - Extend Obsidian's graph view
    - Custom node styling
    - Custom edge styling
2. **overlays.ts**
    
    - Decay overlay (faded nodes)
    - Conflict overlay (red edges)
    - Interactive elements

**Approach (MVP):**

- Use Obsidian's existing graph
- Inject custom CSS based on node metadata
- Add CSS classes for decay status
- Add CSS classes for conflict nodes

**Future Enhancement:**

- Custom D3 graph renderer
- More interactive features
- Filtering by agent insight type

**Deliverable:** Basic knowledge graph visualization

---

### Phase 6: Integration (Days 64-70)

#### Days 64-66 (12 hours): Feature Integration

**Tasks:**

1. Wire all components together
2. End-to-end testing for each feature
3. Performance optimization
4. Bug fixes

**Integration Checklist:**

- [ ] Chat → Orchestrator → Agents → Response
- [ ] Agent edits → Safety layer → Diff modal → Apply
- [ ] Settings → All components respect settings
- [ ] Scheduler → Notifications → User actions
- [ ] RAG → All search features working
- [ ] Graph → Decay and conflict visualization

**Performance Optimization:**

- Lazy load heavy components
- Debounce frequent operations
- Cache expensive computations
- Profile and fix bottlenecks

---

#### Days 67-68 (8 hours): Polish & Documentation

**Tasks:**

1. **Code Cleanup**
    
    - Remove console.logs
    - Add JSDoc comments
    - Consistent error handling
    - Code review
2. **User Documentation**
    
    - README.md with features
    - Getting started guide
    - Settings explanation
    - Troubleshooting
3. **Developer Documentation**
    
    - Architecture overview
    - API documentation
    - Contributing guide

---

#### Days 69-70 (8 hours): Release Preparation

**Tasks:**

1. **Build Optimization**
    
    - Minimize bundle size
    - Tree shaking
    - Production build testing
2. **manifest.json**
    
    - Version number
    - Min Obsidian version
    - Description
    - Author info
3. **Beta Testing**
    
    - Install on fresh vault
    - Test all features
    - Fix critical issues
4. **Community Submission Prep**
    
    - GitHub repository public
    - License file
    - Screenshots for listing

**Deliverable:** Release-ready plugin

---

## 8. Evaluation & Paper Phase: Weeks 11-12

### Week 11: Evaluation (Days 71-77)

#### Days 71-72: Benchmark Dataset Creation

**Tasks:**

1. **Create Test Vault**
    
    - 100+ notes of varying types
    - Some contradictions planted
    - Some duplicates planted
    - Realistic folder structure
    - Project-like note clusters
    
2. **Define Test Scenarios**
    
    |Category|Scenarios|
    |---|---|
    |Organization|5 inbox sorting tasks|
    |Conflict Detection|5 contradiction finding tasks|
    |Summarization|5 summary generation tasks|
    |Dashboard|5 project dashboard tasks|
    |Mixed|5 multi-agent tasks|
    
3. **Ground Truth**
    
    - Expected outcomes for each task
    - Success criteria defined
    - Partial credit rubric

---

#### Days 73-74: Automated Evaluation

**Tasks:**

1. **Run All Test Scenarios**
    
    - Execute each task 3 times (variance)
    - Collect all metrics
    - Log agent traces
2. **Compute Metrics**
    
    - Success rate per agent
    - Success rate per task type
    - Average latency
    - Token usage statistics
    - Edit acceptance rate (if simulated)
3. **Statistical Analysis**
    
    - Mean, std dev, confidence intervals
    - Correlation analysis
    - Significance tests if comparing conditions

---

#### Days 75-76: Ablation Studies

**Experiments to Run:**

1. **Single Agent vs. Multi-Agent**
    
    - Run same tasks with orchestrator disabled
    - Route all to single "general" agent
    - Compare success rate and quality
2. **Proactive vs. Reactive Only**
    
    - Disable scheduler
    - Only respond to explicit commands
    - Measure difference in user experience
3. **With RAG vs. Without RAG**
    
    - Disable vector search
    - Use only BM25 keyword search
    - Compare retrieval quality
4. **Trust Levels**
    
    - Full manual approval
    - Smart auto-approval
    - Compare user effort and error rate

---

#### Day 77: User Study (Optional)

**If Time Permits:**

1. **Recruitment**
    
    - 3-5 participants
    - Obsidian users preferred
    - Mix of experience levels
2. **Study Protocol**
    
    - 30 min training
    - 5 tasks to complete
    - Think-aloud protocol
    - Post-task questionnaire
3. **Metrics**
    
    - Task completion time
    - Success rate
    - Satisfaction (Likert scale)
    - Qualitative feedback

---

### Week 12: Paper Writing (Days 78-84)

#### Days 78-79: Architecture Section

**Content:**

- System overview diagram
- Component descriptions
- Design rationale for decisions
- Agent specialization justification
- Tool protocol design

**Figures to Create:**

- Architecture diagram (like the one in Section 2)
- Agent interaction flow
- ReAct loop illustration
- UI screenshots

---

#### Days 80-81: Implementation Section

**Content:**

- Technology stack and why
- Key algorithms (chunking, hybrid search, ReAct)
- Integration challenges and solutions
- Performance characteristics

**Include:**

- Pseudocode for critical algorithms
- System parameters table
- Implementation statistics (LOC, etc.)

---

#### Days 82-83: Evaluation Section

**Content:**

- Experimental setup
- Benchmark description
- Metrics definitions
- Results tables and figures
- Ablation study results
- Discussion of findings
- Limitations

**Figures to Create:**

- Results tables
- Performance comparison charts
- Ablation study visualization

---

#### Day 84: Polish & Submission

**Tasks:**

- Write/refine abstract
- Complete related work section
- Proofread entire paper
- Check formatting requirements
- Prepare supplementary materials
- Submit!

---

## 9. Architecture Reference

### 9.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           User Interface                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Chat View  │  │  Diff Modal  │  │  Dashboard   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          └────────────────┬┴─────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────┐
│                    Orchestrator Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Orchestrator                            │   │
│  │  • Goal Decomposition  • Agent Dispatch  • Result Aggregation│   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────────┐
│                       Agent Layer                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │ Librarian  │  │ Researcher │  │  Teacher   │  │  Manager   │   │
│  │   Agent    │  │   Agent    │  │   Agent    │  │   Agent    │   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │
└────────┼───────────────┼───────────────┼───────────────┼────────────┘
         │               │               │               │
         └───────────────┴───────┬───────┴───────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────┐
│                          Tool Layer                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  file_read │ file_write │ file_move │ rag_search │ llm_call  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│   RAG Layer     │  │  Safety Layer   │  │   Scheduler Layer       │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────────────┐ │
│ │   Chunker   │ │  │ │ Accumulator │ │  │ │  Proactive Tasks    │ │
│ ├─────────────┤ │  │ ├─────────────┤ │  │ ├─────────────────────┤ │
│ │  Embeddings │ │  │ │    Diff     │ │  │ │   File Watcher      │ │
│ ├─────────────┤ │  │ ├─────────────┤ │  │ ├─────────────────────┤ │
│ │Vector Store │ │  │ │   Audit     │ │  │ │  Notifications      │ │
│ ├─────────────┤ │  │ ├─────────────┤ │  │ └─────────────────────┘ │
│ │  Retriever  │ │  │ │   Trust     │ │  └─────────────────────────┘
│ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │   Obsidian Vault    │
         │   (File System)     │
         └─────────────────────┘
```

### 9.2 Directory Structure

```
notes-admin/
├── src/
│   ├── main.ts                 # Plugin entry, lifecycle
│   ├── types.ts                # All type definitions
│   ├── settings.ts             # Settings interface & defaults
│   │
│   ├── agents/
│   │   ├── base/
│   │   │   ├── agent.ts       # BaseAgent class
│   │   │   ├── loop.ts        # ReAct implementation
│   │   │   └── state.ts       # State management
│   │   ├── orchestrator.ts    # Multi-agent coordinator
│   │   ├── librarian.ts       # File organization
│   │   ├── researcher.ts      # Provenance & conflicts
│   │   ├── teacher.ts         # Knowledge decay
│   │   ├── manager.ts         # Dashboards
│   │   └── messaging.ts       # Inter-agent protocol
│   │
│   ├── tools/
│   │   ├── registry.ts        # Tool registration
│   │   ├── file-ops.ts        # File operations
│   │   ├── rag-tools.ts       # Search tools
│   │   ├── analysis-tools.ts  # LLM tools
│   │   └── dashboard-tools.ts # Dashboard tools
│   │
│   ├── rag/
│   │   ├── chunker/
│   │   │   ├── markdown.ts    # MD chunking
│   │   │   └── pdf.ts         # PDF chunking
│   │   ├── embeddings.ts      # Embedding service
│   │   ├── store.ts           # Orama vector store
│   │   ├── indexer.ts         # Vault indexing
│   │   └── retriever.ts       # Hybrid search
│   │
│   ├── safety/
│   │   ├── accumulator.ts     # Edit collection
│   │   ├── diff.ts            # Diff computation
│   │   ├── audit.ts           # Logging
│   │   └── trust.ts           # Autonomy config
│   │
│   ├── ui/
│   │   ├── chat/
│   │   │   ├── view.ts        # Chat sidebar
│   │   │   └── messages.ts    # Message components
│   │   ├── dashboard/
│   │   │   └── view.ts        # Dashboard view
│   │   ├── graph/
│   │   │   └── overlays.ts    # Graph enhancements
│   │   ├── diff-modal.ts      # Review modal
│   │   └── settings-tab.ts    # Settings UI
│   │
│   ├── scheduler/
│   │   ├── proactive.ts       # Task scheduler
│   │   └── queue.ts           # Priority queue
│   │
│   ├── llm/
│   │   ├── provider.ts        # LLM abstraction
│   │   └── prompts/           # System prompts
│   │
│   └── evaluation/
│       ├── metrics.ts         # Metrics collection
│       ├── logger.ts          # Trace logging
│       └── benchmark.ts       # Test scenarios
│
├── evaluation/                 # Paper data
│   ├── datasets/
│   ├── results/
│   └── analysis/
│
├── docs/                       # Documentation
├── paper/                      # Paper drafts
└── [config files]
```

### 9.3 Data Flow

**User Query Flow:**

```
User types in Chat
       │
       ▼
Chat View captures input
       │
       ▼
Orchestrator receives query
       │
       ▼
Orchestrator analyzes & plans
       │
       ▼
Dispatches to Agent(s)
       │
       ▼
Agent runs ReAct loop
       │
       ├──▶ Uses Tools (RAG, files, LLM)
       │
       ▼
Agent returns result + edits
       │
       ▼
Orchestrator aggregates
       │
       ▼
Edits → Safety Layer → Diff Modal
       │
       ▼
Response → Chat View (streaming)
```

**Proactive Flow:**

```
Scheduler triggers task
       │
       ▼
Task executes (e.g., scan inbox)
       │
       ▼
If action needed → Create notification
       │
       ▼
User clicks notification
       │
       ▼
Opens appropriate view/modal
       │
       ▼
User takes action
```

---

## 10. Risk Mitigation

### 10.1 Technical Risks

|Risk|Likelihood|Impact|Mitigation|
|---|---|---|---|
|Orama performance issues at scale|Medium|High|Test with 10k+ notes early, have fallback to simpler search|
|LLM rate limiting|Medium|Medium|Implement retry with backoff, queue requests|
|Agent loops (infinite)|Medium|High|Hard max iterations, timeout, user cancel|
|Token context overflow|High|Medium|Summarize context, truncate old history|
|Obsidian API changes|Low|Medium|Pin Obsidian version, test on updates|

### 10.2 Schedule Risks

|Risk|Likelihood|Impact|Mitigation|
|---|---|---|---|
|Study phase takes longer|Medium|High|Can skip some advanced topics, learn while building|
|Feature creep|High|High|Strict MVP scope, defer features to v2|
|Bug-fixing delays|High|Medium|Buffer time built into Phase 6|
|Paper deadline pressure|Medium|High|Start writing during Phase 5-6, not just Week 11-12|

### 10.3 Contingency Plans

**If behind schedule:**

1. Cut knowledge graph to basic CSS-only approach
2. Reduce to 2 agents (Librarian + Researcher)
3. Skip user study, rely on automated metrics
4. Submit to later venue deadline

**If blocked on technical issue:**

1. Check obsidian-copilot for reference
2. Ask in Obsidian Discord
3. Simplify feature if needed
4. Document limitation for paper

---

## 11. Resource Library

### 11.1 Essential Reading

**Papers:**

- ReAct (arXiv:2210.03629) - Agent pattern
- PaperDebugger (arXiv:2512.02589) - Multi-agent reference
- CAMEL (arXiv:2303.17760) - Multi-agent communication
- Generative Agents (arXiv:2304.03442) - Agent memory

**Documentation:**

- TypeScript Handbook: https://www.typescriptlang.org/docs/handbook/
- Obsidian Developer Docs: https://docs.obsidian.md/
- Orama Docs: https://docs.oramasearch.com/
- Anthropic Tool Use: https://docs.anthropic.com/claude/docs/tool-use

### 11.2 Code Repositories

|Repository|What to Learn|
|---|---|
|obsidian-sample-plugin|Plugin structure basics|
|obsidian-copilot|RAG, chat UI, settings|
|paperdebugger|Multi-agent, safety layer|
|langchain-js|Agent patterns in TypeScript|

### 11.3 Tools & Services

|Tool|Purpose|URL|
|---|---|---|
|Anthropic Console|API keys, usage|console.anthropic.com|
|OpenAI Platform|Embeddings API|platform.openai.com|
|Obsidian Discord|Community help|discord.gg/obsidianmd|
|TypeScript Playground|Quick testing|typescriptlang.org/play|

---

## 12. Daily Checklist Templates

### Study Day Template

```markdown
## Day [X]: [Topic]

### Morning (2 hours)
- [ ] Read: [Resource]
- [ ] Notes: Key concepts documented

### Afternoon (2 hours)
- [ ] Practice: [Exercise]
- [ ] Build: [Small component]

### End of Day
- [ ] Completion criteria met
- [ ] Tomorrow's plan clear
- [ ] Questions noted for research
```

### Implementation Day Template

```markdown
## Day [X]: [Component]

### Tasks
- [ ] Task 1: [Description]
- [ ] Task 2: [Description]
- [ ] Task 3: [Description]

### Testing
- [ ] Unit tests pass
- [ ] Manual testing done
- [ ] No regressions

### End of Day
- [ ] Code committed
- [ ] Documentation updated
- [ ] Blockers noted
```

### Weekly Review Template

```markdown
## Week [X] Review

### Completed
- [List of completed items]

### Challenges
- [What was difficult]
- [How resolved or still pending]

### Learnings
- [Key insights]

### Next Week Focus
- [Priority items]

### Schedule Status
- [ ] On track
- [ ] Behind (by how much?)
- [ ] Ahead
```

---

## Final Notes

### Keys to Success

1. **Consistency over intensity** - 4 focused hours daily beats sporadic 10-hour days
2. **Study first, then build** - The 3-week study phase will make implementation faster
3. **Test incrementally** - Don't wait until the end to test
4. **Document as you go** - Notes now become paper sections later
5. **Use reference systems** - Don't reinvent; learn from obsidian-copilot and PaperDebugger

### When Stuck

1. Check reference implementations first
2. Simplify the problem
3. Ask in Obsidian Discord
4. Sleep on it (seriously)
5. Document the issue for paper's limitations section

### Success Metrics

- [ ] Week 3: Working mini-prototype
- [ ] Week 6: All 4 agents functional
- [ ] Week 10: Beta-ready plugin
- [ ] Week 12: Paper submitted

---

**Good luck, Wei! This is an ambitious but achievable project. The combination of production plugin and research paper is a great dual-purpose goal. Stick to the plan, and you'll have an impressive system to show for it.**