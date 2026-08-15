import io

from dlt.common.data_writers.writers import DataWriter
from dlt.common.destination import DestinationCapabilitiesContext
from dlt.common.libs.pyarrow import get_nested_column_type_from_py_arrow, pyarrow


def test_object_parquet_writer_preserves_native_nested_json() -> None:
    nested_type = pyarrow.struct(
        [
            pyarrow.field("amount", pyarrow.float64()),
            pyarrow.field("currency", pyarrow.string()),
        ]
    )
    columns = {
        "id": {"name": "id", "data_type": "bigint", "nullable": False},
        "amount_obj": {
            "name": "amount_obj",
            "nullable": True,
            **get_nested_column_type_from_py_arrow(nested_type),
        },
    }
    rows = [{"id": 1, "amount_obj": {"amount": 1.5, "currency": "EUR"}}]

    caps = DestinationCapabilitiesContext.generic_capabilities("parquet")
    caps.supports_nested_types = True

    output = io.BytesIO()
    writer = DataWriter.from_file_format("parquet", "object", output, caps=caps)
    writer.write_all(columns, rows)
    writer.close()

    output.seek(0)
    table = pyarrow.parquet.read_table(output)

    assert table.schema.field("amount_obj").type == nested_type
    assert table.to_pylist() == rows
