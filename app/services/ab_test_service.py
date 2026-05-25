from typing import Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.schemas import PipelineConfig, QueryLog


class ABTestService:
    """
    Manages A/B testing between current active config and proposed config.

    Logic:
    - If a proposed (inactive) config exists → A/B test is active
    - Alternates queries between current and proposed config
    - After enough data (10 queries per config) → compare and promote winner
    """

    QUERIES_PER_CONFIG = 10  # Need this many queries before comparing

    def get_test_state(self, db: Session) -> Dict:
        """
        Check if an A/B test is active and return which config to use.
        Returns: {"active": bool, "use_config": dict, "config_version": int}
        """
        # Get current active config
        active_config = (
            db.query(PipelineConfig)
            .filter(PipelineConfig.is_active == True)
            .first()
        )

        # Get latest proposed (inactive) config
        proposed_config = (
            db.query(PipelineConfig)
            .filter(PipelineConfig.is_active == False)
            .order_by(desc(PipelineConfig.version))
            .first()
        )

        # No proposed config → no A/B test, use active or defaults
        if not proposed_config:
            if active_config:
                return {
                    "active": False,
                    "use_config": self._config_to_dict(active_config),
                    "config_version": active_config.version,
                }
            return {
                "active": False,
                "use_config": self._defaults(),
                "config_version": 0,
            }

        # A/B test is active! Decide which config to use this query
        active_version = active_config.version if active_config else 0
        proposed_version = proposed_config.version

        # Count queries for each config version
        active_count = (
            db.query(QueryLog)
            .filter(QueryLog.config_version == active_version)
            .count()
        )
        proposed_count = (
            db.query(QueryLog)
            .filter(QueryLog.config_version == proposed_version)
            .count()
        )

        # Check if we have enough data to decide
        if active_count >= self.QUERIES_PER_CONFIG and proposed_count >= self.QUERIES_PER_CONFIG:
            # Enough data — compare and promote
            winner = self._compare_and_promote(db, active_config, proposed_config)
            return {
                "active": False,
                "use_config": winner["config"],
                "config_version": winner["version"],
                "ab_result": winner,
            }

        # Not enough data yet — alternate between configs
        # Use proposed if it has fewer queries (balance the test)
        if proposed_count <= active_count:
            return {
                "active": True,
                "use_config": self._config_to_dict(proposed_config),
                "config_version": proposed_version,
                "ab_status": f"Testing proposed v{proposed_version} ({proposed_count}/{self.QUERIES_PER_CONFIG})",
            }
        else:
            config = self._config_to_dict(active_config) if active_config else self._defaults()
            return {
                "active": True,
                "use_config": config,
                "config_version": active_version,
                "ab_status": f"Testing control v{active_version} ({active_count}/{self.QUERIES_PER_CONFIG})",
            }

    def _compare_and_promote(self, db: Session, active_config, proposed_config) -> Dict:
        """Compare metrics between two configs and promote the winner."""
        active_version = active_config.version if active_config else 0
        proposed_version = proposed_config.version

        active_metrics = self._get_avg_metrics(db, active_version)
        proposed_metrics = self._get_avg_metrics(db, proposed_version)

        # Score: higher faithfulness is better, lower latency is better
        # Faithfulness weight = 0.7, latency weight = 0.3
        active_score = (
            active_metrics["avg_faithfulness"] * 0.7
            + (1 - min(active_metrics["avg_latency_ms"] / 10000, 1)) * 0.3
        )
        proposed_score = (
            proposed_metrics["avg_faithfulness"] * 0.7
            + (1 - min(proposed_metrics["avg_latency_ms"] / 10000, 1)) * 0.3
        )

        if proposed_score > active_score:
            # Proposed wins — promote it!
            self._promote_config(db, proposed_version)
            winner_version = proposed_version
            winner_label = "proposed"
        else:
            # Active wins — discard proposed
            self._discard_config(db, proposed_version)
            winner_version = active_version
            winner_label = "active"

        winner_config_obj = db.query(PipelineConfig).filter(
            PipelineConfig.version == winner_version
        ).first()

        return {
            "version": winner_version,
            "config": self._config_to_dict(winner_config_obj) if winner_config_obj else self._defaults(),
            "active_metrics": active_metrics,
            "proposed_metrics": proposed_metrics,
            "active_score": round(active_score, 4),
            "proposed_score": round(proposed_score, 4),
            "winner": winner_label,
        }

    def _get_avg_metrics(self, db: Session, config_version: int) -> Dict:
        """Get average metrics for queries that used a specific config."""
        logs = (
            db.query(QueryLog)
            .filter(QueryLog.config_version == config_version)
            .order_by(desc(QueryLog.created_at))
            .limit(self.QUERIES_PER_CONFIG)
            .all()
        )

        if not logs:
            return {"avg_faithfulness": 0, "avg_latency_ms": 99999, "count": 0}

        faithfulness_scores = []
        for log in logs:
            if log.evaluation_scores and "faithfulness" in log.evaluation_scores:
                faithfulness_scores.append(log.evaluation_scores["faithfulness"])

        avg_faithfulness = (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores else 0
        )
        avg_latency = sum(log.latency_ms or 0 for log in logs) / len(logs)

        return {
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "count": len(logs),
        }

    def _promote_config(self, db: Session, version: int):
        """Set a config as active, deactivate all others."""
        db.query(PipelineConfig).update({PipelineConfig.is_active: False})
        db.query(PipelineConfig).filter(
            PipelineConfig.version == version
        ).update({PipelineConfig.is_active: True})
        db.commit()

    def _discard_config(self, db: Session, version: int):
        """Delete the losing proposed config."""
        db.query(PipelineConfig).filter(
            PipelineConfig.version == version
        ).delete()
        db.commit()

    def _config_to_dict(self, config: PipelineConfig) -> Dict:
        return {
            "top_k": config.top_k,
            "rerank_weight": config.rerank_weight,
            "routing_threshold": config.routing_threshold,
            "retry_threshold": config.retry_threshold,
        }

    def _defaults(self) -> Dict:
        return {
            "top_k": 5,
            "rerank_weight": 0.5,
            "routing_threshold": 0.5,
            "retry_threshold": 0.7,
        }


ab_test_service = ABTestService()
