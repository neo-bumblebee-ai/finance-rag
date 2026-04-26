# Finance RAG — Stakeholder Presentation Outline
**"From Prototype to Production: A Secure, Reliable AI System for SEC Filing Analysis"**

Target audience: Product leads, compliance officers, engineering managers, investors
Format: 12–15 min video with screen demos + architecture diagrams
Tone: Confident, business-focused, no jargon overload

---

## Slide 1 — Hook (0:00–0:45)
**Title: "What if your analysts could ask any 10-K a direct question and get a cited answer in 2 seconds?"**

- Open with a live demo: type _"What are NVIDIA's stated risk factors around AI chip competition?"_ → show structured JSON response with citations, confidence score, and decision recommendation
- Don't explain the tech yet — let the output speak first
- Tagline: _"Grounded answers. Zero hallucinations. Audit-ready citations."_

---

## Slide 2 — The Problem We're Solving (0:45–2:00)
**"Reading 10-Ks is a $500M/year analyst problem"**

- A typical 10-K is 80–150 pages. Fortune 500 companies file 4–8 per year across subsidiaries
- Analysts spend 30–40% of research time on document extraction, not insight generation
- Existing tools: keyword search misses context; LLM chatbots hallucinate numbers without citations
- Compliance risk: an answer without a source is legally unusable in a fiduciary context

**Business impact framing:**
> "One analyst covering 20 companies × 3 filings/year × 2 hours per filing = 120 hours/year on extraction alone. We eliminate that."

---

## Slide 3 — What We Built (2:00–3:30)
**System architecture — one diagram, plain language**

Walk through the pipeline visually (use the Mermaid diagram from the README):
1. **SEC EDGAR ingestion** — pulls 10-Ks and 10-Qs directly from the SEC, no manual uploads
2. **Hybrid search** — BM25 catches tickers and exact financial terms; vector search captures semantic meaning; fused together for best recall
3. **Cohere reranking** — cross-encoder selects the 5 most relevant chunks from 40 candidates
4. **GPT-4o structured output** — generates answer + per-claim citations + confidence score + decision recommendation

Key message: _"Every answer has a page number. If it's not in the filing, we don't say it."_

---

## Slide 4 — Production Upgrade: Security (3:30–5:00)
**"Enterprise-grade access control built in from day one"**

**Role-Based Access Control (JWT RBAC):**
- Three roles: Admin, Analyst, Viewer
- Analysts are scoped to their coverage universe (e.g., Alice sees only AAPL, MSFT, NVDA)
- JWT tokens expire every 60 minutes — no permanent credentials
- Demo: show the `POST /auth/token` call, then a 403 when Bob (Viewer, AAPL-only) tries to query NVDA
- All API keys stored in **AWS Secrets Manager** — never in code, never in environment variables in the container

**LangChain Guardrails:**
- Two-layer content filter runs before every query:
  - **Scope filter** — blocks off-topic questions (e.g., "write me a poem") before they reach GPT-4o
  - **Safety filter** — detects toxicity, competitor promotion, market manipulation language
- Cost: ~$0.00008 per check (GPT-4o-mini) — effectively free at scale
- Demo: show a blocked out-of-scope query returning a clean 400 with a user-friendly message

---

## Slide 5 — Production Upgrade: Reliability & Eval (5:00–7:00)
**"Every code change is automatically tested against 25 real financial questions"**

**RAGAS Evaluation Gate:**
| Metric | What it measures | Threshold |
|---|---|---|
| Faithfulness | Is every claim in the answer supported by the retrieved context? | ≥ 0.85 |
| Answer Relevancy | Does the answer actually address the question? | ≥ 0.80 |
| Context Precision | Are the retrieved chunks actually useful? | ≥ 0.78 |
| Context Recall | Does retrieval capture everything needed to answer? | ≥ 0.75 |

- Pull Request cannot merge if any metric drops below threshold
- Demo: show the GitHub Actions CI badge and the failing PR check when faithfulness drops

**LangSmith Trend Tracking:**
- Every eval run uploads scores to LangSmith — you can see faithfulness trend over 3 months
- Screenshot: LangSmith eval dashboard showing 4 metrics across 8 eval runs

---

## Slide 6 — Production Upgrade: Observability (7:00–8:30)
**"We know exactly what every query cost, how long it took, and whether the answer was confident"**

**Dual observability stack:**

| Tool | What it captures |
|---|---|
| Langfuse | Per-request: cost, latency, confidence score, user_id, chunks used — real-time |
| LangSmith | Aggregate trends: daily spend by user/model, p95 latency, low-confidence query alerts |
| CloudWatch | Infrastructure: ECS CPU/memory, ALB 5xx rate, healthy task count |

**Cost visibility demo:**
- Run `python deploy/monitoring/langsmith_monitor.py --days 7`
- Show output: $X total spend, breakdown by user, p95 latency = 2.8s, 3 low-confidence queries flagged

**Alert example:**
> "If p95 latency exceeds 5 seconds for 2 consecutive 5-minute windows, CloudWatch fires an alarm. The on-call engineer gets a PagerDuty page before users report a slowdown."

---

## Slide 7 — Production Deployment: AWS Architecture (8:30–10:00)
**"Deployed on AWS — scales automatically, zero downtime, no shared secrets"**

Architecture walkthrough (diagram):
```
Internet → ALB (HTTPS, TLS termination)
         → ECS Fargate (private subnet, auto-scales 2–10 tasks)
         → Secrets Manager (API keys injected at runtime)
         → CloudWatch (logs + alarms)
         → ECR (container registry, image scanning on push)
```

Key points:
- **HTTPS enforced** — HTTP redirected to HTTPS at the ALB; no plaintext traffic
- **Zero secrets in code** — all keys pulled from Secrets Manager by the task execution role at start
- **Auto-scaling** — CPU > 70% → adds tasks in 60s; scales in after 5 min cooldown
- **Deployment circuit breaker** — failed deploys automatically roll back; no manual intervention
- **2 AZ deployment** — service remains available if an entire availability zone goes down

CI/CD flow:
> `git push` → GitHub Actions runs lint + tests → RAGAS eval gate → build Docker image → push to ECR → CloudFormation deploys → ECS rolling update → ALB health checks confirm → done

---

## Slide 8 — Business Value Summary (10:00–11:30)
**"What this means for your team"**

| Before | After |
|---|---|
| Analyst reads 100-page 10-K manually | Answer + citation in 2.1s (p50) |
| No audit trail on AI answers | Every claim has filing + page number |
| LLM answers vary by user prompt | Structured output — same schema every time |
| No access control on sensitive docs | Role-scoped: analyst sees only their coverage |
| Hallucination risk on financial numbers | Citation enforcement — claim without source = blocked |
| Unknown LLM costs | Per-request cost tracked, daily budget alerts available |
| "Hope it works" reliability | RAGAS CI gate blocks regressions automatically |

**ROI framing:**
- If this system saves 2 hours/week per analyst across a 10-person team → 1,040 hours/year reclaimed
- At $150/hr fully-loaded analyst cost → $156,000/year in capacity freed
- Infrastructure cost at current scale: ~$400–600/month on AWS Fargate

---

## Slide 9 — What's Coming Next (11:30–12:30)
**Series roadmap — this is Part 4 of 6**

| Month | Project | Status |
|---|---|---|
| Jan | Databricks AI Engineering Challenge | Complete |
| Feb | Traditional RAG Pipeline | Complete |
| Mar | Agentic RAG with LangGraph | Complete |
| **Apr** | **Finance RAG — Production Edition (this)** | **Complete** |
| May | LLMOps Evaluation Platform | Coming |
| Jun | Enterprise AI Platform (Capstone) | Coming |

May preview:
> "Next month we're building the evaluation platform that sits above all of these systems — a unified dashboard for tracking model quality, cost trends, and regression detection across multiple RAG pipelines simultaneously."

---

## Slide 10 — Call to Action (12:30–13:00)
**Close strong**

- GitHub: `github.com/neo-bumblebee-ai/finance-rag` — live repo, all code open source
- Demo: `POST /auth/token` → `POST /ask` — 2 API calls to get a cited answer on any 10-K
- Ask for: feedback, collaboration, follow (building in public, monthly cadence)

Closing line:
> _"Production AI isn't about the model — it's about the system around the model. Security, evaluation, observability, deployment. That's what makes it enterprise-ready."_

---

## Recording Tips

- **Show, don't tell**: Every claim should have a screen recording or live demo
- Keep the Swagger UI (`/docs`) open — it's the best way to demo the API live
- Use two windows: terminal for `langsmith_monitor.py` output, browser for Swagger
- Record at 1920×1080, zoom in on code/JSON so it's readable at 720p
- Add captions — institutional audiences often watch without audio
- Total target: 12–13 minutes. Cut anything over 14 — stakeholders will drop off.
