"""
agents/pipeline_decision.py
-----------------------------
Re-export module for PipelineDecision.

PipelineDecision is defined in data_understanding_agent.py.
This module exists so that other packages (e.g. orchestrator.master_agent)
can do:
    from agents.pipeline_decision import PipelineDecision
"""

from agents.data_understanding_agent import PipelineDecision

__all__ = ["PipelineDecision"]
