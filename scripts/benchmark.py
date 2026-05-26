"""
Benchmark script — runs 50 legal questions against the pipeline.
Run: python scripts/benchmark.py
"""

import requests
import time
import json

API_URL = "http://127.0.0.1:8003"

QUESTIONS = [
    # Contract Basics (10)
    "What is the effective date of this agreement?",
    "When does this contract expire?",
    "Who are the parties involved in this agreement?",
    "What is the agreement date?",
    "What type of contract is this?",
    "What is the term of this agreement?",
    "Where was this contract executed?",
    "What is the purpose of this agreement?",
    "Is this agreement binding on successors and assigns?",
    "What are the definitions section key terms?",

    # Termination & Renewal (8)
    "What are the termination provisions?",
    "Can either party terminate for convenience?",
    "What is the notice period for termination?",
    "What are the renewal terms?",
    "Does the contract auto-renew?",
    "What happens upon termination of this agreement?",
    "Are there any survival clauses after termination?",
    "What constitutes a material breach?",

    # Financial Terms (8)
    "What are the payment terms?",
    "Is there a minimum commitment or revenue guarantee?",
    "What are the pricing terms?",
    "Are there any royalty fees mentioned?",
    "What happens to payments upon termination?",
    "Are there any late payment penalties?",
    "What is the compensation structure?",
    "Are there any audit rights for financial records?",

    # Intellectual Property (7)
    "Is there an intellectual property assignment clause?",
    "Who owns the intellectual property created under this agreement?",
    "Are there any licensing rights granted?",
    "What are the IP indemnification terms?",
    "Are there restrictions on using the other party's trademarks?",
    "What happens to IP rights after termination?",
    "Is there a technology transfer clause?",

    # Restrictions & Obligations (9)
    "Are there any non-compete restrictions?",
    "What is the non-solicitation clause?",
    "What are the exclusivity provisions?",
    "What are the confidentiality obligations?",
    "How long do confidentiality obligations last after termination?",
    "What information is considered confidential?",
    "Are there any exceptions to confidentiality?",
    "What are the reporting obligations?",
    "Are there any performance milestones or KPIs?",

    # Liability & Indemnification (5)
    "What is the limitation of liability?",
    "Are there any indemnification provisions?",
    "Is there a cap on damages?",
    "What are the warranty provisions?",
    "Are there any disclaimers of warranty?",

    # Governance (3)
    "What is the governing law and jurisdiction?",
    "How are disputes resolved under this contract?",
    "Is there an arbitration clause?",
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
    print("📊 PipelineIQ Benchmark — 50 Legal Questions")
    print("=" * 60)

    if not check_api():
        return

    # Check if documents are ingested
    docs = requests.get(f"{API_URL}/documents").json()
    print(f"📄 Documents in DB: {docs['total_documents']}")
    if docs["total_documents"] == 0:
        print("❌ No documents ingested. Run: python scripts/ingest_cuad.py")
        return

    print(f"\n🚀 Running {len(QUESTIONS)} queries...\n")

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

                results.append({
                    "question": question,
                    "latency_ms": elapsed,
                    "faithfulness": faithfulness,
                    "model_used": data.get("model_used", "unknown"),
                    "from_cache": data.get("from_cache", False),
                    "num_sources": data.get("num_sources", 0),
                })
                cache_tag = "CACHE" if data.get("from_cache") else f"{elapsed:.0f}ms"
                print(f"✅ {cache_tag}")
            else:
                results.append({
                    "question": question,
                    "latency_ms": elapsed,
                    "error": resp.status_code,
                })
                print(f"❌ {resp.status_code}")

        except Exception as e:
            print(f"❌ {e}")
            results.append({"question": question, "error": str(e)})

    total_time = round(time.time() - start_total, 1)

    # Print report
    print(f"\n{'=' * 60}")
    print(f"📊 BENCHMARK REPORT")
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
        print(f"   Avg: {sum(latencies)/len(latencies):.0f}ms")
        print(f"   Max: {max(latencies):.0f}ms")

    # Faithfulness stats
    faithfulness_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    if faithfulness_scores:
        print(f"\n🎯 Faithfulness:")
        print(f"   Min: {min(faithfulness_scores):.3f}")
        print(f"   Avg: {sum(faithfulness_scores)/len(faithfulness_scores):.3f}")
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

    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
