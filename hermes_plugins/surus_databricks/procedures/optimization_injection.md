## CRITICAL RULES (apply before doing anything else)

### Cell Exclusion
Never count, score, or reference any infrastructure cell that belongs to this agent.
Identify infrastructure cells by the presence of ANY of the following signatures in their source:

    start_time_optimization_agent
    end_time_optimization_agent
    execution_time_optimization_agent
    person_running_optimization_agent
    notebook_python_line_code_count
    notebook_sql_line_code_count
    unique_tables_called_optimization_agent
    opti_agent.metrics

Also exclude cells whose # DBTITLE contains any of:
    notebook analysis, analyze notebook, notebook metadata, run duration,
    push metrics, start time, run analysis, optimization widget, optimization scoring,
    run stats, run close

### Path Detection
Auto-detect the current notebook path via the Databricks context API.
Never hardcode a path, notebook ID, or workspace URL.

### Table Names
All Delta table writes use the universal.opti_agent catalog/schema unless the user explicitly overrides.

| Table                                     | Purpose                          | Written when                                              |
|-------------------------------------------|----------------------------------|-----------------------------------------------------------|
| universal.opti_agent.notebook_stats       | Notebook structural metadata     | First run ever, or when run_stats = "Yes" (upsert)        |
| universal.opti_agent.notebook_runs        | Per-run timing and status log    | Every run (insert on init, update on close)               |
| universal.opti_agent.optimization_scores  | 4-category scores + notes        | Only when run_optimization = "Yes"                        |

---

## CELL ORDER (top to bottom)

1. Widget Cell          -- always first, always runs
2. Stats + Run Init     -- always runs; stats write gated by widget and existing record
3. Optimization Cell    -- runs only when run_optimization = "Yes"
4. Run Close Cell       -- always last, always runs

---

## PHASE 0 -- INSERT WIDGET CELL (absolute first cell)

Insert the following verbatim as the first cell. Do not modify variable names, widget keys, defaults, or labels.
INJECTION RULE: The code blocks below are LITERAL cell contents to be inserted into the target notebook exactly as written. Do not generate your own versions. Your only modifications should be replacing __GENIE_*__ placeholder tokens with computed values from your analysis.

```python
# DBTITLE 0,optimization widget
# ----------------------------------------------------------------------------
#  AGENT WIDGETS -- controls whether optimization scoring and stats refresh run
#  DO NOT EDIT -- managed by Databricks Genie Optimization Agent
# ----------------------------------------------------------------------------

dbutils.widgets.dropdown("run_optimization", "No", ["Yes", "No"], "Run Optimization Scoring?")
dbutils.widgets.dropdown("run_stats",        "No", ["Yes", "No"], "Refresh Notebook Stats?")

run_optimization = dbutils.widgets.get("run_optimization")
run_stats        = dbutils.widgets.get("run_stats")

print(f"Run optimization scoring : {run_optimization}")
print(f"Refresh notebook stats   : {run_stats}")
```

---

## PHASE 1 -- INSERT STATS + RUN INIT CELL (cell #2, immediately after widget cell)

Insert the following verbatim as cell #2. Do not modify any variable names, comments, or logic.
This cell defines run_id, start_time_optimization_agent, notebook_id, tables_found, and all
statistics variables that the optimization cell and run close cell depend on.

```python
# DBTITLE 0,notebook analysis
# ----------------------------------------------------------------------------
#  STATS + RUN INIT -- notebook metadata · run record · start timer
#  DO NOT EDIT -- managed by Databricks Genie Optimization Agent
# ----------------------------------------------------------------------------

import base64, re, string, random, time
from datetime import datetime
from pyspark.sql.types import StructType, StructField, StringType
import requests

# -- Context -----------------------------------------------------------------
ctx              = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token            = ctx.apiToken().get()
host             = ctx.apiUrl().get()
notebook_ws_path = ctx.notebookPath().get()
notebook_id      = ctx.notebookId().get()
person_running_optimization_agent = ctx.userName().get()

# -- Export notebook source --------------------------------------------------
_response   = requests.get(
    f"{host}/api/2.0/workspace/export",
    headers={"Authorization": f"Bearer {token}"},
    params={"path": notebook_ws_path, "format": "SOURCE"}
)
source_code = base64.b64decode(_response.json()["content"]).decode("utf-8")

# -- Workspace metadata ------------------------------------------------------
_obj_meta      = requests.get(
    f"{host}/api/2.0/workspace/get-status",
    headers={"Authorization": f"Bearer {token}"},
    params={"path": notebook_ws_path}
).json()

created_at_ms  = _obj_meta.get("created_at")
modified_at_ms = _obj_meta.get("modified_at")
created_by     = _obj_meta.get("creator_user_name", "unknown")
created_at     = datetime.fromtimestamp(created_at_ms  / 1000).isoformat() if created_at_ms  else None
modified_at    = datetime.fromtimestamp(modified_at_ms / 1000).isoformat() if modified_at_ms else None

# -- Cell parsing ------------------------------------------------------------
_raw_cells = re.split(r'^# COMMAND ----------$', source_code, flags=re.MULTILINE)

_upstream_pattern    = r'(?:FROM|JOIN)\s+[`]?(\w+\.\w+\.\w+)[`]?'
_downstream_pattern  = r'(?:CREATE\s+OR\s+REPLACE\s+(?:TABLE|VIEW)|CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?|INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?|MERGE\s+INTO|UPDATE)\s+[`]?(\w+\.\w+\.\w+)[`]?'
_table_pattern       = r'(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+[`]?(\w+\.\w+\.\w+)[`]?'
_spark_table_pattern = r'spark\.table\(["\']([\w.\s\\]+)["\']\)'
_import_line_pattern = re.compile(r'^\s*from\s+\S+\s+import', re.IGNORECASE)

_agent_sigs = {
    'start_time_optimization_agent', 'end_time_optimization_agent',
    'execution_time_optimization_agent', 'person_running_optimization_agent',
    'notebook_python_line_code_count', 'notebook_sql_line_code_count',
    'unique_tables_called_optimization_agent', 'opti_agent.metrics',
}
_agent_title_kws = [
    'notebook analysis', 'analyze notebook', 'notebook metadata',
    'run duration', 'push metrics', 'start time', 'run analysis',
    'optimization widget', 'optimization scoring', 'run stats', 'run close',
]

_agent_tables = {
    'universal.opti_agent.optimization_scores',
    'universal.opti_agent.notebook_stats',
    'universal.opti_agent.notebook_runs',
}

tables_found      = set()
upstream_tables   = set()
downstream_tables = set()
python_lines  = 0
sql_lines     = 0
total_cells   = 0
sql_cells     = 0
python_cells  = 0
skipped_cells = 0

for _cell_src in _raw_cells:
    _cell_src = _cell_src.strip()
    if not _cell_src:
        continue
    if any(_sig in _cell_src for _sig in _agent_sigs):
        skipped_cells += 1
        continue
    _dbtitle_match = re.search(r'# DBTITLE \d+,(.+)', _cell_src)
    _cell_title    = _dbtitle_match.group(1).strip().lower() if _dbtitle_match else ''
    if any(_kw in _cell_title for _kw in _agent_title_kws):
        skipped_cells += 1
        continue

    _lines  = _cell_src.split('\n')
    _is_sql = any(l.strip().startswith('# MAGIC %sql') for l in _lines)
    _is_md  = any(l.strip().startswith('# MAGIC %md')  for l in _lines)
    _is_py  = not _is_sql and not _is_md

    total_cells += 1
    if _is_sql: sql_cells    += 1
    if _is_py:  python_cells += 1

    for _line in _lines:
        if _import_line_pattern.match(_line):
            continue
        tables_found.update(
            m.lower() for m in re.findall(_table_pattern, _line, re.IGNORECASE)
            if m.lower() not in _agent_tables
        )
        upstream_tables.update(
            m.lower() for m in re.findall(_upstream_pattern, _line, re.IGNORECASE)
            if m.lower() not in _agent_tables
        )
        downstream_tables.update(
            m.lower() for m in re.findall(_downstream_pattern, _line, re.IGNORECASE)
            if m.lower() not in _agent_tables
        )
    for _m in re.findall(_spark_table_pattern, _cell_src):
        _clean = re.sub(r'[\s\\]+', '', _m).lower()
        if _clean.count('.') == 2 and _clean not in _agent_tables:
            tables_found.add(_clean)

    for _line in _lines:
        _stripped = _line.strip()
        if _is_sql and _stripped.startswith('# MAGIC'):
            _stripped = _stripped[7:].strip()
        if not _stripped or _stripped.startswith('#') or _stripped.startswith('%'):
            continue
        if _is_py:    python_lines += 1
        elif _is_sql: sql_lines    += 1

# -- Print notebook statistics -----------------------------------------------
print("\nNotebook Statistics")
print("=" * 40)
print(f"Path          : {notebook_ws_path}")
print(f"ID            : {notebook_id}")
print(f"Created by    : {created_by}")
print(f"Created at    : {created_at}")
print(f"Last modified : {modified_at}")
print(f"Total cells   : {total_cells}  (Python: {python_cells}, SQL: {sql_cells})")
print(f"Python lines  : {python_lines}")
print(f"SQL lines     : {sql_lines}")
print(f"Tables found  : {len(tables_found)}")
for _t in sorted(tables_found):
    print(f"  - {_t}")
print(f"Upstream tables   : {len(upstream_tables)}")
for _t in sorted(upstream_tables):
    print(f"  - {_t}")
print(f"Downstream tables : {len(downstream_tables)}")
for _t in sorted(downstream_tables):
    print(f"  - {_t}")
print(f"Skipped cells : {skipped_cells}")

# -- Write / upsert notebook_stats -------------------------------------------
# Always writes on first visit (no existing record).
# On subsequent runs, only re-runs when run_stats = "Yes", performing an upsert
# so the existing record is updated rather than a duplicate appended.
_stats_exists = spark.sql(
    f"SELECT 1 FROM universal.opti_agent.notebook_stats "
    f"WHERE notebook_id = '{notebook_id}' LIMIT 1"
).collect()

if not _stats_exists or run_stats == "Yes":
    _df_stats = spark.createDataFrame(
        [(
            notebook_ws_path, notebook_id, created_by, created_at, modified_at,
            total_cells, python_cells, sql_cells, python_lines, sql_lines,
            len(tables_found), ", ".join(sorted(tables_found)),
            ", ".join(sorted(upstream_tables)), ", ".join(sorted(downstream_tables)),
            len(upstream_tables), len(downstream_tables)
        )],
        schema=[
            "notebook_path", "notebook_id", "created_by", "created_at", "modified_at",
            "total_cells", "python_cells", "sql_cells", "python_lines", "sql_lines",
            "unique_tables", "table_list",
            "upstream_tables", "downstream_tables",
            "upstream_tables_count", "downstream_tables_count"
        ]
    )
    if not _stats_exists:
        _df_stats.write.mode("append").saveAsTable("universal.opti_agent.notebook_stats")
        print("Notebook stats written (new record).")
    else:
        _upstream_str   = ", ".join(sorted(upstream_tables))
        _downstream_str = ", ".join(sorted(downstream_tables))
        _table_list_str = ", ".join(sorted(tables_found))
        spark.sql(f"""
            UPDATE universal.opti_agent.notebook_stats
            SET    notebook_path            = '{notebook_ws_path}',
                   created_by               = '{created_by}',
                   created_at               = '{created_at}',
                   modified_at              = '{modified_at}',
                   total_cells              = {total_cells},
                   python_cells             = {python_cells},
                   sql_cells                = {sql_cells},
                   python_lines             = {python_lines},
                   sql_lines                = {sql_lines},
                   unique_tables            = {len(tables_found)},
                   table_list               = '{_table_list_str}',
                   upstream_tables          = '{_upstream_str}',
                   downstream_tables        = '{_downstream_str}',
                   upstream_tables_count    = {len(upstream_tables)},
                   downstream_tables_count  = {len(downstream_tables)}
            WHERE  notebook_id = '{notebook_id}'
        """)
        print("Notebook stats updated (run_stats = Yes).")
else:
    print("Notebook stats unchanged (record exists and run_stats = No).")

# -- Run init -- always executes ---------------------------------------------
def _id_gen(size=30, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

run_id    = _id_gen()
start_iso = datetime.now().isoformat()

_agent_widget_keys = {'run_optimization', 'run_stats'}

try:
    notebook_params = {
        n: dbutils.widgets.get(n)
        for n in dbutils.widgets.getAll()
        if n not in _agent_widget_keys
    }
except Exception:
    notebook_params = {}

params_str = ", ".join(f"{k}: {v}" for k, v in notebook_params.items()) or None

print("\nRun Information")
print("=" * 40)
print(f"Run ID     : {run_id}")
print(f"Run by     : {person_running_optimization_agent}")
print(f"Start time : {start_iso}")
print(f"Parameters : {notebook_params or '(none)'}")

# Insert with exit_status = "FAIL" -- the run close cell updates this to SUCCEED
# if and only if all cells between here and the close cell complete without error.
spark.createDataFrame(
    [(run_id, person_running_optimization_agent, start_iso, params_str, notebook_id, "NA", "NA", "FAIL")],
    schema=StructType([
        StructField("run_id",        StringType(), True),
        StructField("run_by",        StringType(), True),
        StructField("start_time",    StringType(), True),
        StructField("parameters",    StringType(), True),
        StructField("notebook_id",   StringType(), True),
        StructField("end_time",      StringType(), True),
        StructField("duration",      StringType(), True),
        StructField("exit_status",   StringType(), True),
    ])
).write.mode("append").saveAsTable("universal.opti_agent.notebook_runs")

# -- Start timer -------------------------------------------------------------
start_time_optimization_agent = time.perf_counter()
```

---

## PHASE 2 -- INSERT OPTIMIZATION CELL (cell #3, immediately after stats + run init cell)

Insert the following verbatim as cell #3. Substitute every __GENIE_*__ placeholder with the
value determined during analysis (see Scoring section below) before inserting.
Every score must be an integer 0-10. Every notes field must be a plain-text string.
The entire body is wrapped in `if run_optimization == "Yes":` so it is always safe to execute.

```python
# DBTITLE 0,optimization scoring
# ----------------------------------------------------------------------------
#  OPTIMIZATION SCORING -- gated by run_optimization widget
#  DO NOT EDIT -- managed by Databricks Genie Optimization Agent
# ----------------------------------------------------------------------------

if run_optimization == "Yes":

    from datetime import date

    _score_readability     = __GENIE_SCORE_READABILITY__      # int 0-10
    _score_maintainability = __GENIE_SCORE_MAINTAINABILITY__  # int 0-10
    _score_performance     = __GENIE_SCORE_PERFORMANCE__      # int 0-10
    _score_reliability     = __GENIE_SCORE_RELIABILITY__      # int 0-10
    _score_total           = _score_readability + _score_maintainability + _score_performance + _score_reliability

    _notes_readability     = """__GENIE_NOTES_READABILITY__"""
    _notes_maintainability = """__GENIE_NOTES_MAINTAINABILITY__"""
    _notes_performance     = """__GENIE_NOTES_PERFORMANCE__"""
    _notes_reliability     = """__GENIE_NOTES_RELIABILITY__"""

    print("Optimization Scores")
    print("=" * 40)
    for _label, _s in [
        ("Readability",     _score_readability),
        ("Maintainability", _score_maintainability),
        ("Performance",     _score_performance),
        ("Reliability",     _score_reliability),
    ]:
        _bar = "█" * _s + "░" * (10 - _s)
        print(f"  {_label:<18} {_bar}  {_s}/10")
    print(f"\n  {'TOTAL':<18}  {_score_total}/40")

    # -- Write optimization_scores -------------------------------------------
    spark.createDataFrame(
        [(
            notebook_id,
            date.today().isoformat(),
            _score_readability,
            _notes_readability,
            _score_maintainability,
            _notes_maintainability,
            _score_performance,
            _notes_performance,
            _score_reliability,
            _notes_reliability,
            _score_total,
        )],
        schema=[
            "notebook_id",
            "score_date",
            "score_readability",
            "notes_readability",
            "score_maintainability",
            "notes_maintainability",
            "score_performance",
            "notes_performance",
            "score_reliability",
            "notes_reliability",
            "score_total",
        ]
    ).write.mode("append").saveAsTable("universal.opti_agent.optimization_scores")

    print("\nOptimization scores written to universal.opti_agent.optimization_scores")

else:
    print("Optimization scoring skipped (run_optimization = No)")
```

---

## PHASE 2 ANALYSIS -- Scoring criteria

Perform this analysis on all user-authored cells only (every cell not excluded by Critical Rules above)
before inserting the optimization cell.

Notes format: 2-5 sentences per category. Cite specific cell content, function names, or patterns
observed. List most impactful issues first. If fully compliant, state what was done well.

Scoring guidance:
  9-10  Fully compliant; best-practice patterns used throughout
  7-8   Minor gaps only; no high-impact issues
  5-6   Several improvements needed; at least one high-impact issue present
  3-4   Multiple anti-patterns; meaningful risk to quality or performance
  0-2   Critical issues; notebook should not be promoted to production as-is

### Category 1 -- Readability (0-10)
- Notebook has a clear title cell and purpose statement
- Logical sections separated with markdown heading cells (imports, config, functions, logic, output)
- A dedicated functions section exists, separate from main execution logic
- Functions and complex transformations have docstrings (inputs, outputs, intent)
- Inline comments explain the "why" on non-obvious logic
- Meaningful variable and DataFrame names -- no df1, df2, tmp, x
- Consistent code style -- spacing, casing, import ordering
- Cells kept short and focused; no sprawling multi-hundred-line cells
- Whitespace used to visually group related logic within cells
- No commented-out or dead code

### Category 2 -- Maintainability (0-10)
- Reusable logic extracted into functions rather than repeated inline
- Functions are small and single-purpose
- No copy-paste blocks that could be parameterised into a loop or helper
- Logic modular enough that adding a new condition or filter requires one change
- Transformations chained or composed in an extensible way
- No hardcoded business logic buried mid-notebook -- rules live in config or constants
- Change history or version notes present, or notebook tracked in version control
- External dependencies and data sources documented (source, owner, expected schema)

### Category 3 -- Performance (0-10)
- Only required columns selected -- no SELECT * or loading unnecessary fields
- Filters applied as early as possible before joins or transforms
- Partition pruning leveraged -- queries target specific partitions
- Broadcast hints used for small-table joins
- Shuffles minimised -- repartition and coalesce used deliberately
- DataFrames cached/persisted where re-used; unpersisted when done
- Native PySpark/Spark SQL functions used instead of Python UDFs
- Actions (collect, count, show) minimised; never called inside loops
- Same data source not read multiple times -- read once, reuse the DataFrame
- Intermediate DataFrames no longer needed are unpersisted or dropped

### Category 4 -- Reliability (0-10)
- Databricks widgets or a config cell used for parameters -- no hardcoded values in logic
- Secrets pulled from Databricks Secret Scope or Key Vault, never hardcoded
- try/except blocks wrap critical steps with meaningful error messages
- Row count and schema assertions present after major reads, joins, or transforms
- Null and duplicate checks performed on key columns before downstream logic
- Notebook is idempotent -- safe to re-run without producing duplicates or incorrect results
- Edge cases handled -- empty DataFrames, unexpected nulls, schema mismatches
- Fallback logic exists for known failure points
- PII and sensitive data not printed in cell outputs or logs

---

## PHASE 3 -- INSERT RUN CLOSE CELL (very last cell of the notebook)

Insert the following verbatim as the absolute last cell.
A run only transitions from FAIL to SUCCEED if every preceding cell completes without error.

```python
# DBTITLE 0,run close
# ----------------------------------------------------------------------------
#  RUN CLOSE -- stop timer · mark run SUCCEED
#  DO NOT EDIT -- managed by Databricks Genie Optimization Agent
# ----------------------------------------------------------------------------

end_time_optimization_agent = time.perf_counter()
total_seconds = round(end_time_optimization_agent - start_time_optimization_agent, 2)
end_iso       = datetime.now().isoformat()

hours   = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = round(total_seconds % 60, 2)

print("Run Duration")
print("=" * 40)
print(f"Total     : {total_seconds}s  ({hours}h {minutes}m {seconds}s)")
print(f"Run ID    : {run_id}")
print(f"Run by    : {person_running_optimization_agent}")
print(f"Start     : {start_iso}")
print(f"End       : {end_iso}")

spark.sql(f"""
    UPDATE universal.opti_agent.notebook_runs
    SET    end_time    = '{end_iso}',
           duration    = '{total_seconds}',
           exit_status = 'SUCCEED'
    WHERE  run_id = '{run_id}'
""")

print("\nRun record closed as SUCCEED.")
```

---

## PHASE 4 -- APPEND MARKDOWN SUMMARY CELL (only when run_optimization = "Yes")

After inserting the run close cell, append one additional %md cell as the final cell.
Populate every {{placeholder}} with values determined during Phase 2 analysis.
Do NOT append this cell if run_optimization = "No".

```
# MAGIC %md
# DBTITLE 0,notebook analysis
# ----------------------------------------------------------------------------
#  OPTIMIZATION REPORT -- generated by Databricks Genie
# ----------------------------------------------------------------------------

## Notebook Optimization Report

**Notebook:** `{{notebook_name}}`
**ID:** `{{notebook_id}}`
**Analysed by:** Databricks Genie Optimization Agent
**Score date:** `{{score_date}}`

---

### Notebook Statistics
| Metric | Value |
|---|---|
| Total cells (excl. agent) | {{total_cells}} (Python: {{python_cells}}, SQL: {{sql_cells}}) |
| Python lines | {{python_lines}} |
| SQL lines | {{sql_lines}} |
| Unique tables referenced | {{unique_tables}} |
| Upstream tables (read) | {{upstream_tables_count}} -- {{upstream_tables}} |
| Downstream tables (write) | {{downstream_tables_count}} -- {{downstream_tables}} |

---

### Optimization Scores
| # | Category | Score | Notes |
|---|---|---|---|
| 1 | Readability | {{score_readability}}/10 | {{notes_readability}} |
| 2 | Maintainability | {{score_maintainability}}/10 | {{notes_maintainability}} |
| 3 | Performance | {{score_performance}}/10 | {{notes_performance}} |
| 4 | Reliability | {{score_reliability}}/10 | {{notes_reliability}} |
| | **TOTAL** | **{{score_total}}/40** | |

---

### Prioritised Recommendations
{{recommendations}}

---
*Generated by Databricks Genie Optimization Agent -- do not edit manually.*
```

---

## EXECUTION ORDER SUMMARY

| Step | Cell    | What is inserted                                                              | Runs when                                              |
|------|---------|-------------------------------------------------------------------------------|--------------------------------------------------------|
| 0    | #1      | Two dropdowns: run_optimization, run_stats (both default No)                  | Always                                                 |
| 1    | #2      | Stats parse; conditional write/upsert to notebook_stats; run record insert to notebook_runs with exit_status = "FAIL"; defines notebook_id and all shared variables | Always; stats write gated by widget + existing record |
| 2    | #3      | Scores + notes substituted into template; full block no-ops if widget = No   | Always inserted; logic runs only when run_optimization = "Yes" |
| 3    | Last    | Timer stop; UPDATE notebook_runs to exit_status = "SUCCEED"                   | Always inserted; only reached if all prior cells succeeded |
| 4    | After close | %md report cell with scores, notes, criteria, recommendations             | Appended only when run_optimization = "Yes"            |

If any mandatory step cannot be completed (e.g. notebook API unavailable), stop immediately
and report the failure. Do not proceed to later steps.

---

## RESPONSE FORMAT

After completing all steps, reply with the following. Omit the optimization block if run_optimization = "No".

  Notebook analysed : <notebook_name>
  Tables detected   : <n> -- <comma-separated list>
  Cells analysed    : <total_cells> (Python: <python_cells>, SQL: <sql_cells>)
  Run ID            : <run_id>

  -- Optimization Scores (only when run_optimization = Yes) --
  Total            : <score_total>/40
  Readability      : <score>/10
  Maintainability  : <score>/10
  Performance      : <score>/10
  Reliability      : <score>/10

  Top 3 recommendations:
    1. <most impactful action>
    2. <second action>
    3. <third action>

  Stats record : <"written (new record)" | "updated (run_stats = Yes)" | "unchanged (run_stats = No)">

  Tables written:
    universal.opti_agent.notebook_runs          -- always
    universal.opti_agent.notebook_stats         -- written/updated or skipped
    universal.opti_agent.optimization_scores    -- written or skipped
