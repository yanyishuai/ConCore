import os
import pandas as pd
import json
import time
import sqlite3
from typing import Dict, Any

from tools.context_management.context_handler import append_dataset_metadata, update_context_from_llm
from tools.llm.llm_client import generate_text, strip_json_fences


def extract_metadata_with_llm(filename: str, basic_metadata: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""Analyze this dataset metadata and provide rich context:

FILENAME: {filename}
TYPE: {basic_metadata.get('type', 'unknown')}
COLUMNS: {basic_metadata.get('columns', [])}
ROWS: {basic_metadata.get('rows', 0)}

Respond with ONLY a JSON object containing:
{{
  "description": "Brief description of what this dataset likely contains",
  "potential_analyses": ["list", "of", "possible", "analyses"],
  "key_fields": ["most", "important", "columns"],
  "data_quality_notes": "Any observations about data structure"
}}

Only valid JSON, no additional text."""

    try:
        response_text = strip_json_fences(generate_text(prompt, max_tokens=1024))
        return json.loads(response_text)
    except Exception as e:
        return {
            "description": "Metadata extraction failed",
            "error": str(e)
        }


def handle_file_upload(file, save_dir: str, metadata_path: str, context_path: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    filename = file.filename
    file_path = os.path.join(save_dir, filename)
    file.save(file_path)

    metadata = {
        "filename": filename,
        "file_path": file_path,
        "upload_timestamp": int(time.time()),
        "columns": [],
        "rows": 0,
        "type": None,
        "size_bytes": os.path.getsize(file_path)
    }

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            metadata.update({
                "type": "csv",
                "columns": df.columns.tolist(),
                "rows": len(df),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(3).to_dict(orient="records")
            })

        elif filename.endswith((".xlsx", ".xls")):
            excel_file = pd.ExcelFile(file_path)
            sheets_metadata = []

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_metadata.append({
                    "sheet_name": sheet_name,
                    "columns": df.columns.tolist(),
                    "rows": len(df),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                })

            metadata.update({
                "type": "excel",
                "sheets": sheets_metadata,
                "num_sheets": len(excel_file.sheet_names)
            })

            if sheets_metadata:
                metadata["columns"] = sheets_metadata[0]["columns"]
                metadata["rows"] = sheets_metadata[0]["rows"]

        elif filename.endswith((".db", ".sqlite")):
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            tables_metadata = []
            for table in tables:
                cursor.execute(f"SELECT * FROM {table} LIMIT 0")
                columns = [desc[0] for desc in cursor.description]
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_count = cursor.fetchone()[0]

                tables_metadata.append({
                    "table_name": table,
                    "columns": columns,
                    "rows": row_count
                })

            conn.close()

            metadata.update({
                "type": "database",
                "tables": tables_metadata,
                "num_tables": len(tables)
            })

            if tables_metadata:
                metadata["columns"] = tables_metadata[0]["columns"]
                metadata["rows"] = tables_metadata[0]["rows"]

        elif filename.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    metadata.update({
                        "type": "json",
                        "columns": list(data[0].keys()),
                        "rows": len(data),
                        "sample_data": data[:3]
                    })
                else:
                    metadata.update({
                        "type": "json",
                        "structure": type(data).__name__,
                        "rows": 1
                    })

        else:
            metadata["type"] = "unknown"
            metadata["error"] = "Unsupported file type"

    except Exception as e:
        metadata["type"] = "error"
        metadata["error"] = str(e)

    if metadata["type"] not in ["unknown", "error"]:
        llm_metadata = extract_metadata_with_llm(filename, metadata)
        metadata["llm_insights"] = llm_metadata

        context_text = f"Dataset '{filename}' uploaded: {llm_metadata.get('description', 'No description available')}"
        update_context_from_llm(context_path, context_text, source="file_upload")

    append_dataset_metadata(metadata_path, metadata)

    return metadata
