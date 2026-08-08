# Core Barter System Architecture

## Overview

The Core Barter System is a real-time audio conversation monitoring platform that enforces topic adherence during barter/negotiation sessions. It uses a microservices architecture with four FastAPI services and a React frontend.

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        FE[React Frontend<br/>Port 5173]
    end

    subgraph "API Gateway / Core Service"
        BE[Backend API<br/>Port 8000<br/>FastAPI + PostgreSQL]
    end

    subgraph "ML Services"
        AP[Audio Pipeline<br/>Port 8001<br/>Whisper STT]
        SA[Semantic Analysis<br/>Port 8002<br/>Sentence-BERT]
        WE[Warning Engine<br/>Port 8003<br/>Escalation Logic]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Port 5432)]
    end

    subgraph "Shared Package"
        SH[shared/<br/>types.py]
    end

    FE -->|HTTP/WebSocket| BE
    BE -->|REST| AP
    BE -->|REST| SA
    BE -->|REST| WE
    AP -->|Audio Data| SA
    SA -->|Classification| WE
    WE -->|Warnings| FE
    BE -->|SQL| PG
    AP -->|SQL| PG
    WE -->|SQL| PG
    SH -.->|Shared Types| BE
    SH -.->|Shared Types| AP
    SH -.->|Shared Types| SA
    SH -.->|Shared Types| WE
```

## Service Details

### 1. Frontend (React SPA - Port 5173)
- **Technology**: React 18+, Vite, MediaRecorder API
- **Purpose**: User interface for session management
- **Key Screens**:
  - Setup screen for session configuration
  - Live session screen with real-time audio capture
  - Post-session screen with verdict and trust scores

### 2. Backend API (FastAPI - Port 8000)
- **Technology**: FastAPI, SQLAlchemy, PostgreSQL
- **Purpose**: Core business logic and orchestration
- **Components**:
  - `main.py` - FastAPI application entry point
  - `routes.py` - REST API endpoints
  - `models.py` - SQLAlchemy database models
  - `schemas.py` - Pydantic validation schemas
  - `websocket.py` - Real-time communication
  - `escrow.py` - Escrow management logic
  - `safety.py` - Safety and validation utilities

### 3. Audio Pipeline (FastAPI - Port 8001)
- **Technology**: FastAPI, OpenAI Whisper, FFmpeg, PyTorch
- **Purpose**: Audio transcription service
- **Responsibilities**:
  - Receive audio chunks from frontend
  - Transcribe audio using Whisper model
  - Return text transcriptions to backend

### 4. Semantic Analysis (FastAPI - Port 8002)
- **Technology**: FastAPI, Sentence-BERT (sentence-transformers)
- **Purpose**: Semantic similarity analysis
- **Responsibilities**:
  - Compute cosine similarity between transcript and topic
  - Classify windows as `correct`, `weakly_correct`, or `incorrect`
  - Use configurable thresholds (UPPER=0.55, LOWER=0.35)

### 5. Warning Engine (FastAPI - Port 8003)
- **Technology**: FastAPI, PostgreSQL
- **Purpose**: Warning escalation and session termination
- **Responsibilities**:
  - Track consecutive off-topic windows
  - Issue escalating warnings:
    - 1 window: Silent warning
    - 2 windows: Strong warning
    - 3+ windows: Severe warning + auto-terminate
  - Log warnings to database

### 6. PostgreSQL Database (Port 5432)
- **Technology**: PostgreSQL 16
- **Tables**:
  - `users` - User accounts
  - `barter_sessions` - Session records
  - `session_contracts` - Session agreements
  - `window_results` - Per-window classifications
  - `warnings_log` - Warning history
  - `verdicts` - Session verdicts
  - `confirmations` - User confirmations
  - `wallets` - User wallets
  - `escrows` - Escrow records
  - `credit_transactions` - Transaction history

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant AudioPipeline
    participant SemanticAnalysis
    participant WarningEngine
    participant Database

    User->>Frontend: Start Session
    Frontend->>Backend: Create Session
    Backend->>Database: Save Session
    Backend->>Frontend: Session Created
    
    loop Real-time Audio Processing
        Frontend->>Frontend: Capture Audio
        Frontend->>Backend: Send Audio Chunk
        Backend->>AudioPipeline: Transcribe
        AudioPipeline-->>Backend: Transcription
        
        Backend->>SemanticAnalysis: Analyze Relevance
        SemanticAnalysis-->>Backend: Classification
        
        Backend->>WarningEngine: Check Warnings
        WarningEngine-->>Backend: Warning Level
        
        Backend->>Database: Save Window Result
        Backend->>Frontend: Window Status
    end
    
    User->>Frontend: End Session
    Frontend->>Backend: End Session
    Backend->>SemanticAnalysis: Generate Verdict
    Backend->>Database: Update Trust Scores
    Backend->>Frontend: Session Results
```

## Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: Session Created
    Created --> Active: Both Users Confirm
    Active --> Active: Windows Processed
    Active --> Pending_Verdict: Session Ends
    Pending_Verdict --> Complete: Verdict Generated
    Complete --> [*]
    
    note right of Active
      - Audio captured every ~30s window
      - Each window classified as correct/weakly_correct/incorrect
      - Warnings escalate after 2+ consecutive off-topic
    end note
```

## Escrow System Flow

```mermaid
graph LR
    A[User Wallet] -->|Lock| B[Escrow]
    B -->|Full Release| C[User Wallet<br/>QA >= 0.8]
    B -->|Partial Release| D[User Wallet<br/>QA >= 0.5]
    B -->|Penalty| E[System<br/>QA < 0.5]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#fff8e1
    style E fill:#ffebee
```

## Technology Stack Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | React 18+, Vite | User interface |
| Backend | FastAPI, Python | Core API |
| Database | PostgreSQL | Data persistence |
| Audio STT | OpenAI Whisper | Speech-to-text |
| Semantic Analysis | Sentence-BERT | Topic relevance |
| ML Runtime | PyTorch | Model inference |
| Audio Processing | FFmpeg | Audio format handling |
| Real-time | WebSocket | Live updates |
| Container | Docker | Service deployment |

## Inter-Service Communication

- **Frontend → Backend**: HTTP REST + WebSocket
- **Backend → ML Services**: REST API calls
- **All Services → Database**: SQL via SQLAlchemy
- **Shared Types**: Python package import from `packages/shared/`