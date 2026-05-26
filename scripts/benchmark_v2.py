"""
Benchmark v2 — 50 different legal questions, sequential execution.
Run: python scripts/benchmark_v2.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import time
import json

API_URL = "http://127.0.0.1:8003"

QUESTIONS = [
    # === SIMPLE QUESTIONS (→ Haiku) — short, direct, single-fact ===

    # Simple factual (10)
    "Who is the licensor?",
    "What is the effective date?",
    "What is the governing law?",
    "What is the term length?",
    "Is there an arbitration clause?",
    "What is the notice period?",
    "Who are the parties?",
    "What is the territory covered?",
    "Is this agreement exclusive?",
    "What is the payment frequency?",

    # Simple yes/no (5)
    "Does this contract auto-renew?",
    "Is there a non-compete clause?",
    "Are there royalty fees?",
    "Is there a force majeure clause?",
    "Is there a severability clause?",

    # === COMPLEX QUESTIONS (→ Sonnet) — multi-hop, comparison, analysis, reasoning ===

    # Multi-hop reasoning (10)
    "Compare the termination rights of both parties and explain which party has more favorable exit conditions.",
    "Analyze the relationship between the payment terms and the termination provisions — what happens to unpaid fees if the contract is terminated early?",
    "How do the confidentiality obligations interact with the intellectual property rights after the agreement expires?",
    "What are the cumulative financial obligations of the licensee considering royalties, minimum payments, and any penalties described across all sections?",
    "Explain how the indemnification provisions, limitation of liability, and insurance requirements work together to allocate risk between the parties.",
    "If a force majeure event occurs that lasts longer than 90 days, what are the combined effects on termination rights, payment obligations, and performance deadlines?",
    "Compare and contrast the obligations of the franchisor versus the franchisee in terms of training, marketing, and quality control requirements.",
    "What happens to intellectual property rights, confidential information, and non-compete restrictions if the agreement is terminated due to a material breach by the licensor?",
    "Analyze the dispute resolution mechanism — what steps must be taken before litigation, and how does the governing law choice affect enforcement?",
    "How do the renewal terms, price adjustment mechanisms, and performance metrics interact to determine whether the agreement continues beyond the initial term?",

    # Comparative analysis (8)
    "What are the differences between the termination for convenience and termination for cause provisions?",
    "Compare the warranty obligations versus the disclaimer of warranties — are there contradictions?",
    "How do the exclusivity provisions limit the licensor's ability to compete versus the licensee's non-compete obligations?",
    "Contrast the confidentiality obligations during the agreement versus the survival period after termination.",
    "What is the relationship between the minimum revenue commitment and the termination rights if minimums are not met?",
    "Compare the remedies available for IP infringement versus remedies for payment default.",
    "How do the assignment restrictions differ for each party, and what implications does this have for mergers or acquisitions?",
    "Analyze whether the limitation of liability cap is consistent with the indemnification obligations — could indemnification exceed the liability cap?",

    # Synthesis and reasoning (10)
    "Summarize all the financial obligations of the licensee across the entire agreement, including one-time fees, recurring payments, and conditional payments.",
    "What are the three most significant risks for the licensee in this agreement, considering termination, IP ownership, and financial obligations together?",
    "If both parties want to exit this agreement amicably, what is the most efficient path considering notice periods, transition obligations, and post-termination restrictions?",
    "Evaluate whether the non-compete and non-solicitation restrictions would likely be enforceable given their scope, duration, and geographic limitations.",
    "What regulatory compliance failures could trigger termination, and how do the indemnification provisions protect each party in that scenario?",
    "Construct a timeline of all key obligations and deadlines from contract execution through the first renewal period.",
    "What are the potential conflicts between the exclusivity grant and the licensor's retained rights?",
    "How would a change of control event affect the rights, obligations, and termination options available to each party?",
    "Identify all provisions that survive termination and explain how they create ongoing obligations after the agreement ends.",
    "What is the total maximum liability exposure for each party considering direct damages, indemnification, IP claims, and any uncapped obligations?",

    # Multi-document synthesis (7)
    "Across all the contracts in the system, what are the most common termination provisions?",
    "Which contracts have the most restrictive non-compete clauses, and what makes them more restrictive?",
    "Compare the IP ownership models across different agreement types — license vs development vs franchise.",
    "What is the range of royalty rates across all agreements, and what factors seem to influence higher versus lower rates?",
    "Which agreements provide the strongest protections for confidential information, and what specific mechanisms do they use?",
    "Across all contracts, what are the typical cure periods for material breach, and are there outliers?",
    "Compare how different contract types handle force majeure — do franchise agreements differ from license agreements in their approach?",
]


def check_api():
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.status_code == 200:
            print("✅ API is running")
            return True
    except:
        pass
    print("❌ API not running. Start: python -m uvicorn app.main:app --port 8003")
    return False


def run_benchmark():
    print("📊 PipelineIQ Benchmark v2 — 50 Legal Questions")
    print("=" * 60)

    if not check_api():
        return

    # Check documents
    docs = requests.get(f"{API_URL}/documents").json()
    print(f"📄 Documents in DB: {docs['total_documents']}")
    if docs["total_documents"] == 0:
        print("❌ No documents. Run: python scripts/ingest_cuad.py")
        return

    print(f"\n🚀 Running {len(QUESTIONS)} queries sequentially...\n")

    results = []
    start_total = time.time()

    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {question[:50]}...", end=" ")
        start = time.time()

        try:
            resp = requests.post(
                f"{API_URL}/query-optimized",
                json={"query": question},
            )
            elapsed = round((time.time() - start) * 1000, 0)

            if resp.status_code == 200:
                data = resp.json()
                faithfulness = None
                if data.get("evaluation") and "faithfulness" in data["evaluation"]:
                    faithfulness = data["evaluation"]["faithfulness"]

                cache_tag = "CACHE" if data.get("from_cache") else f"{elapsed:.0f}ms"
                print(f"✅ {cache_tag}")

                results.append({
                    "question": question,
                    "latency_ms": elapsed,
                    "faithfulness": faithfulness,
                    "model_used": data.get("model_used", "unknown"),
                    "from_cache": data.get("from_cache", False),
                    "num_sources": data.get("num_sources", 0),
                })
            else:
                print(f"❌ {resp.status_code}")
                results.append({"question": question, "latency_ms": elapsed, "error": resp.status_code})

        except Exception as e:
            print(f"❌ {e}")
            results.append({"question": question, "error": str(e)})

    total_time = round(time.time() - start_total, 1)

    # Print report
    print(f"\n{'=' * 60}")
    print(f"📊 BENCHMARK v2 REPORT")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time}s")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")

    # Latency stats
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r and "error" not in r]
    if latencies:
        print(f"\n⏱️ Latency:")
        print(f"   Min: {min(latencies):.0f}ms")
        print(f"   Avg: {sum(latencies) / len(latencies):.0f}ms")
        print(f"   Max: {max(latencies):.0f}ms")

    # Faithfulness stats
    faithfulness_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    if faithfulness_scores:
        print(f"\n🎯 Faithfulness:")
        print(f"   Min: {min(faithfulness_scores):.3f}")
        print(f"   Avg: {sum(faithfulness_scores) / len(faithfulness_scores):.3f}")
        print(f"   Max: {max(faithfulness_scores):.3f}")
        print(f"   Below 0.7: {sum(1 for s in faithfulness_scores if s < 0.7)}/{len(faithfulness_scores)}")

    # Cache hits
    cache_hits = sum(1 for r in results if r.get("from_cache"))
    print(f"\n💾 Cache hits: {cache_hits}/{len(results)}")

    # Model distribution
    models = {}
    for r in results:
        m = r.get("model_used", "unknown")
        models[m] = models.get(m, 0) + 1
    print(f"\n🤖 Model usage: {models}")

    # Save results
    with open("benchmark_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: benchmark_v2_results.json")


if __name__ == "__main__":
    run_benchmark()
