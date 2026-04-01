"""
Dataset pipeline orchestrators.

``DatasetPipeline`` runs stages in list order.
``PipelineOrchestrator`` builds a DAG from stage dependencies and executes in topological order.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import networkx as nx

from .base import PreprocessingStage, StageResult, ValidationResult

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    Composable pipeline that chains preprocessing stages in **declaration order**.

    Does not resolve ``get_dependencies()``; use :class:`PipelineOrchestrator` for DAG order.
    """

    def __init__(self, stages: List[PreprocessingStage]):
        self.stages = stages

    def run(self, data: Any) -> StageResult:
        """Run all stages in sequence, passing output of each to the next."""
        if not self.stages:
            return StageResult(success=True, processed_data=None, stage_name=None)

        current_data = data

        for stage in self.stages:
            start = time.perf_counter()
            result = stage.process(current_data)
            elapsed = time.perf_counter() - start
            result.execution_time = elapsed

            if not result.success:
                logger.error(
                    "Pipeline failed at stage '%s': %s",
                    stage.get_name(),
                    result.error,
                )
                return result

            current_data = result.processed_data
            logger.debug(
                "Stage '%s' completed in %.3fs",
                stage.get_name(),
                elapsed,
            )

        return result

    def validate_all(self, data: Any) -> List[ValidationResult]:
        """Run validation on all stages without processing."""
        results = []
        current_data = data

        for stage in self.stages:
            validation = stage.validate(current_data)
            results.append(validation)

            if not validation.is_valid:
                logger.warning(
                    "Validation failed at stage '%s': %s",
                    stage.get_name(),
                    validation.errors,
                )

        return results

    def get_stage_names(self) -> List[str]:
        """Get ordered list of stage names."""
        return [s.get_name() for s in self.stages]


class PipelineOrchestrator:
    """
    Orchestrator that builds a DAG from :meth:`PreprocessingStage.get_dependencies`
    and executes stages in topological order (Helox-compatible).
    """

    def __init__(self, stages: Optional[List[PreprocessingStage]] = None):
        self.stages: List[PreprocessingStage] = stages or []
        self.dag: Optional[nx.DiGraph] = None
        self.execution_order: List[str] = []
        self.checkpoints: Dict[str, StageResult] = {}

    def add_stage(self, stage: PreprocessingStage) -> None:
        stage_name = stage.get_name()
        existing_names = [s.get_name() for s in self.stages]
        if stage_name in existing_names:
            raise ValueError(f"Stage with name '{stage_name}' already exists")
        self.stages.append(stage)

    def build_dag(self) -> None:
        self.dag = nx.DiGraph()
        for stage in self.stages:
            stage_name = stage.get_name()
            self.dag.add_node(stage_name, stage=stage)
            for dep_name in stage.get_dependencies():
                if dep_name not in self.dag.nodes():
                    raise ValueError(
                        f"Stage '{stage_name}' depends on '{dep_name}', "
                        f"but no stage with that name was added"
                    )
                self.dag.add_edge(dep_name, stage_name)

        if not nx.is_directed_acyclic_graph(self.dag):
            cycles = list(nx.simple_cycles(self.dag))
            raise ValueError(f"Pipeline contains a cycle: {cycles}")

        self.execution_order = list(nx.topological_sort(self.dag))

    def execute(self, initial_data: Any = None) -> StageResult:
        if self.dag is None:
            raise ValueError("DAG not built. Call build_dag() first.")
        if not self.execution_order:
            raise ValueError("Execution order not computed. Call build_dag() first.")

        current_data = initial_data
        result: Optional[StageResult] = None

        for stage_name in self.execution_order:
            stage: PreprocessingStage = self.dag.nodes[stage_name]["stage"]
            start_time = time.time()
            result = stage.process(current_data)
            execution_time = time.time() - start_time

            if result.execution_time is None:
                result.execution_time = execution_time

            self.checkpoints[stage_name] = result

            if not result.success:
                return result
            if result.processed_data:
                current_data = result.processed_data.data

        assert result is not None
        return result
