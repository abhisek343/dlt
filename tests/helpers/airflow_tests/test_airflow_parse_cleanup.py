from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow import DAG

import dlt
from dlt.common import pendulum
from dlt.helpers.airflow_helper import PipelineTasksGroup
from dlt.pipeline.pipeline import Pipeline


DEFAULT_DATE = pendulum.datetime(2023, 4, 18, tz="UTC")


def _pipeline(name: str) -> Pipeline:
    return dlt.pipeline(
        pipeline_name=name,
        dataset_name=f"{name}_dataset",
        destination=dlt.destinations.duckdb(credentials=":pipeline:"),
    )


def _add_parse_marker(pipeline: Pipeline) -> Path:
    working_dir = Path(pipeline.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    marker = working_dir / "dag-parse.marker"
    marker.write_text("created while parsing the DAG")
    return marker


def test_context_cleans_parse_time_pipeline_and_task_still_runs(tmp_path: Path) -> None:
    with DAG(
        dag_id="dlt_parse_cleanup_execution",
        schedule=None,
        start_date=DEFAULT_DATE,
        catchup=False,
    ) as dag:
        with PipelineTasksGroup(
            "cleanup_group", local_data_folder=str(tmp_path), wipe_local_data=True
        ) as tasks:
            pipeline = _pipeline("parse_cleanup_execution")
            marker = _add_parse_marker(pipeline)
            working_dir = Path(pipeline.working_dir)

            tasks.add_run(
                pipeline,
                [{"id": 1}, {"id": 2}],
                table_name="items",
                decompose="none",
            )
            assert marker.exists()

    assert not working_dir.exists()

    # The pipeline captured by the PythonOperator must still be usable after the
    # parse-time working folder is removed. This exercises the actual task path.
    dag.test()


def test_context_cleans_multiple_registered_pipelines(tmp_path: Path) -> None:
    with DAG(
        dag_id="dlt_parse_cleanup_multiple",
        schedule=None,
        start_date=DEFAULT_DATE,
        catchup=False,
    ):
        with PipelineTasksGroup(
            "cleanup_group", local_data_folder=str(tmp_path), wipe_local_data=True
        ) as tasks:
            add_run_pipeline = _pipeline("parse_cleanup_add_run")
            run_pipeline = _pipeline("parse_cleanup_run")
            add_run_marker = _add_parse_marker(add_run_pipeline)
            run_marker = _add_parse_marker(run_pipeline)

            tasks.add_run(
                add_run_pipeline,
                [{"id": 1}],
                table_name="items",
                decompose="none",
            )
            tasks.run(run_pipeline, [{"id": 2}], table_name="items")

            assert add_run_marker.exists()
            assert run_marker.exists()

    assert not add_run_marker.parent.exists()
    assert not run_marker.parent.exists()


def test_context_cleans_pipeline_when_dag_construction_fails(tmp_path: Path) -> None:
    working_dir = None

    with DAG(
        dag_id="dlt_parse_cleanup_exception",
        schedule=None,
        start_date=DEFAULT_DATE,
        catchup=False,
    ):
        with pytest.raises(RuntimeError, match="stop parsing"):
            with PipelineTasksGroup(
                "cleanup_group", local_data_folder=str(tmp_path), wipe_local_data=True
            ) as tasks:
                pipeline = _pipeline("parse_cleanup_exception")
                marker = _add_parse_marker(pipeline)
                working_dir = marker.parent
                tasks.add_run(
                    pipeline,
                    [{"id": 1}],
                    table_name="items",
                    decompose="none",
                )
                raise RuntimeError("stop parsing")

    assert working_dir is not None
    assert not working_dir.exists()


def test_context_respects_wipe_local_data_false(tmp_path: Path) -> None:
    with DAG(
        dag_id="dlt_parse_cleanup_disabled",
        schedule=None,
        start_date=DEFAULT_DATE,
        catchup=False,
    ):
        with PipelineTasksGroup(
            "cleanup_group", local_data_folder=str(tmp_path), wipe_local_data=False
        ) as tasks:
            pipeline = _pipeline("parse_cleanup_disabled")
            marker = _add_parse_marker(pipeline)
            tasks.add_run(
                pipeline,
                [{"id": 1}],
                table_name="items",
                decompose="none",
            )

    try:
        assert marker.exists()
    finally:
        pipeline._wipe_working_folder()


def test_legacy_add_run_does_not_change_parse_time_lifecycle(tmp_path: Path) -> None:
    with DAG(
        dag_id="dlt_parse_cleanup_legacy",
        schedule=None,
        start_date=DEFAULT_DATE,
        catchup=False,
    ):
        tasks = PipelineTasksGroup(
            "cleanup_group", local_data_folder=str(tmp_path), wipe_local_data=True
        )
        pipeline = _pipeline("parse_cleanup_legacy")
        marker = _add_parse_marker(pipeline)
        tasks.add_run(
            pipeline,
            [{"id": 1}],
            table_name="items",
            decompose="none",
        )

    try:
        # add_run() keeps its historical internal TaskGroup context. Cleanup is
        # opt-in through an explicit outer `with PipelineTasksGroup(...)` block.
        assert marker.exists()
    finally:
        pipeline._wipe_working_folder()
