"""
surus_tools.py — helper library available to model-written code.

This file is executed *inside* the execution environment: the Databricks
cluster driver in cluster mode, the local exec namespace in local mode.
chatbot_ui reads the source text of this file and runs it as the prelude of
every execution context, so it must stay self-contained (stdlib, pandas,
PySpark, optional plotly) and must never import application modules.

It is also importable locally so the sentinel constants and the pure-Python
helpers can be unit-tested.

Public helpers for the model:
    export_csv(df, "name.csv")        save a DataFrame and offer it for download
    match_to_voter_file(df, ...)      match an uploaded people list to a state voter file
    build_match_index(state)          (re)build the normalised voter index the matcher joins against

Internal protocol with the app (see cluster_exec.wrap_command):
    _surus_before() / _surus_after()  run around every command; they surface `result`,
                                      display a trailing expression, save `plotly_figure`
                                      as JSON and publish `export_path`, printing sentinel
                                      lines the app turns into UI elements.
"""
import datetime
import os
import re
import shutil
import uuid

SURUS_CHART_SENTINEL = "__SURUS_CHART__:"
SURUS_EXPORT_SENTINEL = "__SURUS_EXPORT__:"

_SURUS_MISSING = object()
_SURUS_CHAT_DIR = None
_SURUS_MATCH_INDEX_SCHEMA = "universal.chatbot"


# ── environment plumbing ──────────────────────────────────────────────────────
def _surus_init(chat_dir, match_index_schema=None):
    """Remember this chat's working folder (a Volume path on the cluster) and settings."""
    global _SURUS_CHAT_DIR, _SURUS_MATCH_INDEX_SCHEMA
    _SURUS_CHAT_DIR = chat_dir
    if match_index_schema:
        _SURUS_MATCH_INDEX_SCHEMA = match_index_schema
    try:
        os.makedirs(chat_dir, exist_ok=True)
    except Exception as e:  # pragma: no cover - depends on the environment
        print(f"(warning: could not create {chat_dir}: {e})")


def _surus_path(subdir, filename):
    directory = os.path.join(_SURUS_CHAT_DIR or ".", subdir)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


def _surus_spark():
    s = globals().get("spark")
    if s is not None:
        return s
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()
    except Exception:
        return None


def _surus_load_csv(path, **read_csv_kwargs):
    """Load an uploaded CSV as pandas; never raises (a missing file prints a warning)."""
    import pandas as pd
    try:
        df = pd.read_csv(path, **read_csv_kwargs)
        print(f"loaded {os.path.basename(path)}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        return df
    except Exception as e:
        print(f"(warning: could not load {path}: {e})")
        return None


def _surus_is_figure(obj):
    try:
        import plotly.basedatatypes as _pbt
        return isinstance(obj, _pbt.BaseFigure)
    except Exception:
        return False


def _surus_before():
    """Called before each command: clear per-command outputs, remember `result`."""
    g = globals()
    g.pop("plotly_figure", None)
    g.pop("export_path", None)
    g.pop("__surus_last_expr", None)
    return g.get("result", _SURUS_MISSING)


def _surus_after(result_before):
    """Called after each command (even when it raised): surface outputs."""
    g = globals()
    # Notebook-style display of a trailing bare expression (see cluster_exec.prepare_code).
    last = g.pop("__surus_last_expr", None)
    if last is not None:
        try:
            print(repr(last))
        except Exception as e:
            print(f"(could not display the last expression: {e})")
    res = g.get("result", _SURUS_MISSING)
    if res is not _SURUS_MISSING and res is not result_before:
        print(f"\nresult = {res}")

    fig = g.pop("plotly_figure", None)
    if fig is not None and not _surus_is_figure(fig):
        fig = None
    if fig is None:
        for key, val in list(g.items()):
            if not key.startswith("_") and _surus_is_figure(val):
                fig = g.pop(key)
                break
    if fig is not None:
        try:
            path = _surus_path("charts", f"{uuid.uuid4().hex}.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fig.to_json())
            print(f"{SURUS_CHART_SENTINEL}{path}")
        except Exception as e:
            print(f"(warning: chart could not be saved: {e})")

    export = g.pop("export_path", None)
    if export:
        _surus_publish_export(str(export))


def _surus_publish_export(path):
    """Make a file the model wrote available to the app (copied into the chat folder)."""
    try:
        base = os.path.abspath(_SURUS_CHAT_DIR or ".")
        if os.path.abspath(path).startswith(base):
            dest = path
        else:
            dest = _surus_path("exports", os.path.basename(path))
            shutil.copyfile(path, dest)
        print(f"{SURUS_EXPORT_SENTINEL}{dest}")
    except Exception as e:
        print(f"(warning: export could not be published: {e})")


# ── public helper: export ─────────────────────────────────────────────────────
def export_csv(df, filename="export.csv", max_rows=5_000_000):
    """Save a pandas or Spark DataFrame as CSV in this chat's folder and offer a download button."""
    import pandas as pd

    filename = str(filename or "export.csv")
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    path = _surus_path("exports", filename)

    if hasattr(df, "toPandas"):
        n = df.count()
        if n > max_rows:
            raise ValueError(
                f"{n:,} rows is more than the {max_rows:,}-row export limit; "
                "filter or aggregate first, or write with df.write.csv() to the chat folder."
            )
        pdf = df.toPandas()
    elif isinstance(df, pd.DataFrame):
        pdf = df
    else:
        pdf = pd.DataFrame(df)
    pdf.to_csv(path, index=False)
    print(f"Saved {filename} ({len(pdf):,} rows, {pdf.shape[1]} columns).")
    print(f"{SURUS_EXPORT_SENTINEL}{path}")
    return path


# ── matching: reference data ──────────────────────────────────────────────────
_NICKNAMES = {
    "BOB": "ROBERT", "ROB": "ROBERT", "BOBBY": "ROBERT", "ROBBIE": "ROBERT", "BERT": "ROBERT",
    "BILL": "WILLIAM", "WILL": "WILLIAM", "BILLY": "WILLIAM", "WILLIE": "WILLIAM", "LIAM": "WILLIAM",
    "LIZ": "ELIZABETH", "BETH": "ELIZABETH", "BETSY": "ELIZABETH", "BETTY": "ELIZABETH", "ELIZA": "ELIZABETH",
    "JIM": "JAMES", "JIMMY": "JAMES", "JAMIE": "JAMES",
    "MIKE": "MICHAEL", "MICKEY": "MICHAEL", "MICK": "MICHAEL",
    "DICK": "RICHARD", "RICK": "RICHARD", "RICH": "RICHARD", "RICKY": "RICHARD",
    "TOM": "THOMAS", "TOMMY": "THOMAS",
    "JOE": "JOSEPH", "JOEY": "JOSEPH",
    "DAN": "DANIEL", "DANNY": "DANIEL",
    "DAVE": "DAVID", "DAVEY": "DAVID",
    "STEVE": "STEVEN", "STEPHEN": "STEVEN", "STEVIE": "STEVEN",
    "CHRIS": "CHRISTOPHER", "TOPHER": "CHRISTOPHER",
    "TONY": "ANTHONY",
    "ANDY": "ANDREW", "DREW": "ANDREW",
    "MATT": "MATTHEW", "MATTY": "MATTHEW",
    "NICK": "NICHOLAS", "NICKY": "NICHOLAS",
    "PAT": "PATRICK", "PADDY": "PATRICK",
    "PATTY": "PATRICIA", "TRISH": "PATRICIA", "TRICIA": "PATRICIA", "PATSY": "PATRICIA",
    "PEGGY": "MARGARET", "MEG": "MARGARET", "MAGGIE": "MARGARET", "MARGE": "MARGARET", "MARGIE": "MARGARET",
    "KATE": "KATHERINE", "KATHY": "KATHERINE", "KATIE": "KATHERINE", "KAT": "KATHERINE",
    "KATHRYN": "KATHERINE", "KATHRINE": "KATHERINE",
    "CATHY": "CATHERINE", "CATHIE": "CATHERINE",
    "SUE": "SUSAN", "SUSIE": "SUSAN", "SUZY": "SUSAN",
    "JEN": "JENNIFER", "JENNY": "JENNIFER", "JENNI": "JENNIFER",
    "JACK": "JOHN", "JOHNNY": "JOHN", "JOHNNIE": "JOHN",
    "JON": "JONATHAN",
    "TED": "THEODORE", "TEDDY": "THEODORE", "THEO": "THEODORE",
    "ED": "EDWARD", "EDDIE": "EDWARD",
    "CURT": "CURTIS", "KURT": "CURTIS",
    "LES": "LESLIE",
    "GREGG": "GREGORY", "GREG": "GREGORY",
    "JOSH": "JOSHUA",
    "CHUCK": "CHARLES", "CHARLIE": "CHARLES", "CHAS": "CHARLES",
    "HANK": "HENRY", "HARRY": "HENRY",
    "LARRY": "LAWRENCE",
    "JERRY": "GERALD", "GERRY": "GERALD",
    "FRED": "FREDERICK", "FREDDY": "FREDERICK", "FREDDIE": "FREDERICK",
    "SAM": "SAMUEL", "SAMMY": "SAMUEL",
    "BEN": "BENJAMIN", "BENNY": "BENJAMIN", "BENJI": "BENJAMIN",
    "ALEX": "ALEXANDER",
    "JEFF": "JEFFREY", "GEOFF": "JEFFREY",
    "KEN": "KENNETH", "KENNY": "KENNETH",
    "RON": "RONALD", "RONNIE": "RONALD",
    "DON": "DONALD", "DONNIE": "DONALD",
    "RAY": "RAYMOND",
    "TIM": "TIMOTHY", "TIMMY": "TIMOTHY",
    "VICKY": "VICTORIA", "VICKI": "VICTORIA", "VIKKI": "VICTORIA",
    "DEB": "DEBORAH", "DEBBIE": "DEBORAH", "DEBRA": "DEBORAH",
    "BARB": "BARBARA", "BARBIE": "BARBARA",
    "CINDY": "CYNTHIA",
    "SANDY": "SANDRA",
    "MANDY": "AMANDA",
    "ABE": "ABRAHAM",
    "AL": "ALBERT",
    "ART": "ARTHUR", "ARTIE": "ARTHUR",
    "BRAD": "BRADLEY",
    "DOUG": "DOUGLAS",
    "GABE": "GABRIEL",
    "GENE": "EUGENE",
    "HAL": "HAROLD",
    "HERB": "HERBERT",
    "JAKE": "JACOB",
    "LEN": "LEONARD", "LENNY": "LEONARD",
    "LOU": "LOUIS", "LOUIE": "LOUIS",
    "MARTY": "MARTIN",
    "NATE": "NATHAN", "NATHANIEL": "NATHAN",
    "NORM": "NORMAN",
    "PETE": "PETER",
    "PHIL": "PHILIP", "PHILLIP": "PHILIP",
    "RANDY": "RANDALL",
    "RUSS": "RUSSELL",
    "STAN": "STANLEY",
    "STU": "STUART",
    "VINCE": "VINCENT", "VINNY": "VINCENT",
    "WALT": "WALTER",
    "ZACH": "ZACHARY", "ZACK": "ZACHARY",
    "ANGIE": "ANGELA",
    "BECKY": "REBECCA", "BECCA": "REBECCA",
    "CARRIE": "CAROLINE",
    "CHRISSY": "CHRISTINE",
    "CONNIE": "CONSTANCE",
    "DOT": "DOROTHY", "DOTTIE": "DOROTHY",
    "ELLIE": "ELEANOR",
    "FRAN": "FRANCES", "FRANNY": "FRANCES",
    "GINNY": "VIRGINIA",
    "JACKIE": "JACQUELINE",
    "JESS": "JESSICA", "JESSIE": "JESSICA",
    "JUDY": "JUDITH",
    "KIM": "KIMBERLY",
    "MEL": "MELISSA", "MISSY": "MELISSA",
    "ROSIE": "ROSE",
    "SALLY": "SARAH",
    "SHERRY": "SHARON",
    "STEPH": "STEPHANIE",
    "TAMMY": "TAMARA",
    "TERRI": "TERESA", "TERRY": "TERESA", "TESS": "TERESA",
    "TINA": "CHRISTINA",
    "VAL": "VALERIE",
    "MAG": "MAGDALENA",
    "NIKKI": "NICOLE", "NICKI": "NICOLE", "NIKI": "NICOLE",
}

_STREET_ABBREV = {
    "STREET": "ST", "AVENUE": "AVE", "AV": "AVE", "ROAD": "RD", "DRIVE": "DR", "DRV": "DR",
    "LANE": "LN", "COURT": "CT", "BOULEVARD": "BLVD", "BLV": "BLVD", "PLACE": "PL",
    "TERRACE": "TER", "TRAIL": "TRL", "PARKWAY": "PKWY", "CIRCLE": "CIR", "HIGHWAY": "HWY",
    "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW",
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
}

_INPUT_CANDIDATES = {
    "first_name": ["first_name", "firstname", "first", "fname", "given_name", "givenname", "voter_first_name"],
    "last_name": ["last_name", "lastname", "last", "lname", "surname", "family_name", "voter_last_name"],
    "full_name": ["name", "full_name", "fullname", "voter_name", "person"],
    "address": ["address", "address1", "address_1", "street_address", "street", "addr", "addr1",
                "registered_address.address1", "mailing_address.address1", "home_address"],
    "city": ["city", "town", "municipality", "registered_address.city"],
    "zip_code": ["zip", "zip_code", "zipcode", "postal_code", "postalcode", "zip5", "postcode",
                 "registered_address.zip"],
    "phone": ["phone", "phone_number", "phonenumber", "mobile", "mobile_number", "cell", "cell_phone",
              "cellphone", "telephone", "phone1", "primary_phone", "home_phone"],
    "email": ["email", "email_address", "emailaddress", "e_mail", "primary_email", "email1"],
}

_VOTER_CANDIDATES = {
    "id": ["state_file_id", "voter_id", "voterid", "id"],
    "first_name": ["first_name", "voter.first_name", "name.first", "firstname", "first"],
    "last_name": ["last_name", "voter.last_name", "name.last", "lastname", "last"],
    "address": ["registered_address.address1", "registered_address.street", "registered_address.address",
                "registered_address.line1", "address1", "address"],
    "city": ["registered_address.city", "city"],
    "zip_code": ["registered_address.zip", "zip", "zip_code"],
    "county": ["registered_address.county", "county"],
    "email": ["email", "email_address", "primary_email", "email1"],
}

# Voter files often carry several phone columns; all present ones are used.
_VOTER_PHONE_CANDIDATES = ["mobile_number", "phone_number", "phone", "cell_phone", "home_phone",
                           "primary_phone", "phone1", "phone2", "mobile"]

_BLOCK_MODES = ("auto", "zip", "name", "last_name")


def _norm_key(name):
    return re.sub(r"[^a-z0-9.]", "", str(name).lower())


def _detect_columns(columns, overrides, candidates):
    """Map logical fields to actual column names.

    `overrides` wins when given (and must exist); otherwise the first candidate
    whose normalised name matches a column is used. Missing fields are omitted.
    """
    columns = list(columns)
    by_key = {}
    for c in columns:
        by_key.setdefault(_norm_key(c), c)
    out = {}
    for field, cands in candidates.items():
        override = (overrides or {}).get(field)
        if override:
            if override not in columns:
                raise KeyError(f"column {override!r} not found; available columns: {columns}")
            out[field] = override
            continue
        for cand in cands:
            hit = by_key.get(_norm_key(cand))
            if hit is not None:
                out[field] = hit
                break
    return out


def _detect_phone_columns(columns, override=None):
    """All voter-side phone columns present (or the explicit list given)."""
    columns = list(columns)
    if override:
        override = [override] if isinstance(override, str) else list(override)
        missing = [c for c in override if c not in columns]
        if missing:
            raise KeyError(f"phone column(s) {missing} not found; available columns: {columns}")
        return override
    by_key = {}
    for c in columns:
        by_key.setdefault(_norm_key(c), c)
    found = []
    for cand in _VOTER_PHONE_CANDIDATES:
        hit = by_key.get(_norm_key(cand))
        if hit is not None and hit not in found:
            found.append(hit)
    return found


def _describe_columns(columns, *needles):
    hits = [c for c in columns if any(n in c.lower() for n in needles)]
    return ", ".join(hits[:25]) or "(none)"


def _block_strategy(block_on, input_has_zip, voters_have_zip, input_has_city, voters_have_city):
    """Decide how candidates are found.

    Returns (mode, use_city). Blocking keys may only use fields present on BOTH
    sides — this is what previously produced zero candidates when the upload
    had no city column.
      auto      -> "zip" when both sides have zips, else "name"
      zip       -> 5-digit zip + last-name soundex (rows without a zip fall back to "name")
      name      -> last-name soundex + first initial (+ city when both sides have it)
      last_name -> last-name soundex only (+ city when both sides have it); loosest
    """
    mode = (block_on or "auto").strip().lower()
    if mode not in _BLOCK_MODES:
        raise ValueError(f"block_on must be one of {_BLOCK_MODES}")
    both_zip = bool(input_has_zip and voters_have_zip)
    if mode == "auto":
        mode = "zip" if both_zip else "name"
    if mode == "zip" and not both_zip:
        raise ValueError("block_on='zip' needs a zip column on both sides; pass zip_code=... "
                         "or use block_on='name'")
    return mode, bool(input_has_city and voters_have_city)


def match_index_table(state, schema=None):
    """Name of the persisted, normalised voter index for a state."""
    safe = re.sub(r"[^a-z0-9_]", "_", str(state).strip().lower())
    return f"{schema or _SURUS_MATCH_INDEX_SCHEMA}.match_index_{safe}"


# ── matching: Spark normalisation expressions ─────────────────────────────────
def _F():
    from pyspark.sql import functions as F
    return F


def _nx_str(col):
    F = _F()
    return F.coalesce(col.cast("string"), F.lit(""))


def _nx_name(col):
    F = _F()
    c = F.upper(_nx_str(col))
    c = F.regexp_replace(c, r"[^A-Z ]", " ")
    c = F.regexp_replace(c, r"\b(JR|SR|II|III|IV)\b", " ")
    return F.trim(F.regexp_replace(c, r"\s+", " "))


def _nx_canon_first(col):
    """First token of a normalised first name, nicknames mapped to a canonical form."""
    F = _F()
    tok = F.split(col, " ").getItem(0)
    expr = None
    for nick, canon in _NICKNAMES.items():
        expr = (F.when(tok == nick, F.lit(canon)) if expr is None
                else expr.when(tok == nick, F.lit(canon)))
    return expr.otherwise(tok)


def _nx_addr(col):
    F = _F()
    c = F.upper(_nx_str(col))
    c = F.regexp_replace(c, r"[^A-Z0-9 ]", " ")
    c = F.regexp_replace(c, r"\b(APT|APARTMENT|UNIT|STE|SUITE|BLDG|FL|FLOOR|LOT|RM|ROOM)\b\s*[A-Z0-9-]*", " ")
    for long, short in _STREET_ABBREV.items():
        c = F.regexp_replace(c, r"\b" + long + r"\b", short)
    return F.trim(F.regexp_replace(c, r"\s+", " "))


def _nx_house(addr):
    return _F().regexp_extract(addr, r"^(\d+)", 1)


def _nx_street(addr):
    F = _F()
    return F.trim(F.regexp_replace(addr, r"^\d+\s*", ""))


def _nx_zip5(col):
    return _F().regexp_extract(_nx_str(col), r"(\d{5})", 1)


def _nx_city(col):
    F = _F()
    return F.upper(F.trim(_nx_str(col)))


def _nx_phone(col):
    """Ten digits, or '' when the value is not a usable US phone number."""
    F = _F()
    digits = F.regexp_replace(_nx_str(col), r"[^0-9]", "")
    digits = F.when((F.length(digits) == 11) & digits.startswith("1"), F.substring(digits, 2, 10)).otherwise(digits)
    junk = digits.rlike(r"^(\d)\1{9}$")  # 0000000000, 1111111111, ...
    return F.when((F.length(digits) == 10) & ~junk, digits).otherwise(F.lit(""))


def _nx_email(col):
    F = _F()
    e = F.lower(F.trim(_nx_str(col)))
    return F.when(e.contains("@"), e).otherwise(F.lit(""))


def _nx_sim(a, b):
    """Levenshtein similarity in [0, 1]; 0 when either side is empty."""
    F = _F()
    return F.when((a == "") | (b == ""), F.lit(0.0)).otherwise(
        1.0 - F.levenshtein(a, b) / F.greatest(F.length(a), F.length(b), F.lit(1))
    )


_INDEX_COLUMNS = ["state_file_id", "first_name", "last_name", "address", "city", "zip", "county", "email",
                  "n_first", "n_last", "n_sdx", "n_addr", "n_house", "n_street", "n_city", "n_zip5",
                  "n_phones", "n_email"]


def _normalized_voter_side(voters, vcols, phone_cols):
    """The voter file reduced to display columns + normalised match fields (see _INDEX_COLUMNS)."""
    F = _F()

    def _v(field):
        return F.col(f"`{vcols[field]}`")

    def _opt(field):
        return _v(field).cast("string") if vcols.get(field) else F.lit(None).cast("string")

    if phone_cols:
        phones = F.array_distinct(F.array_remove(F.array(*[_nx_phone(F.col(f"`{c}`")) for c in phone_cols]), ""))
    else:
        phones = F.array().cast("array<string>")

    side = voters.select(
        _v("id").cast("string").alias("state_file_id"),
        _v("first_name").cast("string").alias("first_name"),
        _v("last_name").cast("string").alias("last_name"),
        _opt("address").alias("address"),
        _opt("city").alias("city"),
        _opt("zip_code").alias("zip"),
        _opt("county").alias("county"),
        _opt("email").alias("email"),
        _nx_canon_first(_nx_name(_v("first_name"))).alias("n_first"),
        _nx_name(_v("last_name")).alias("n_last"),
        (_nx_addr(_v("address")) if vcols.get("address") else F.lit("")).alias("n_addr"),
        (_nx_city(_v("city")) if vcols.get("city") else F.lit("")).alias("n_city"),
        (_nx_zip5(_v("zip_code")) if vcols.get("zip_code") else F.lit("")).alias("n_zip5"),
        phones.alias("n_phones"),
        (_nx_email(_v("email")) if vcols.get("email") else F.lit("")).alias("n_email"),
    )
    side = side.filter(F.col("n_last") != "")
    return (side.withColumn("n_sdx", F.soundex(F.col("n_last")))
                .withColumn("n_house", _nx_house(F.col("n_addr")))
                .withColumn("n_street", _nx_street(F.col("n_addr")))
                .select(*_INDEX_COLUMNS))


def _voter_columns_for(voters, table, voter_columns):
    vcols = _detect_columns(list(voters.columns), voter_columns or {}, _VOTER_CANDIDATES)
    for required in ("id", "first_name", "last_name"):
        if required not in vcols:
            raise ValueError(
                f"Voter table {table} has no recognisable '{required}' column; pass "
                f"voter_columns={{'{required}': '<column name>'}}. Candidate columns: "
                f"{_describe_columns(voters.columns, 'name', 'id', 'first', 'last')}"
            )
    phone_cols = _detect_phone_columns(voters.columns, (voter_columns or {}).get("phones"))
    return vcols, phone_cols


def build_match_index(state="illinois", table=None, index_table=None, voter_columns=None, spark=None):
    """(Re)build the normalised voter index for a state so matches are a plain join.

    Writes `<schema>.match_index_<state>` (schema from the app's MATCH_INDEX_SCHEMA,
    default universal.chatbot) with the voter id, display columns, and the normalised
    names/address/zip/phones/email plus `built_at`. Takes a few minutes for a
    state; the app refreshes it nightly. Returns the table name.
    """
    F = _F()
    spark = spark or _surus_spark()
    if spark is None:
        raise RuntimeError("No Spark session is available in this environment.")
    table = table or f"{str(state).strip().lower()}.deltatables.vcs_gold"
    index_table = index_table or match_index_table(state)
    voters = spark.table(table)
    vcols, phone_cols = _voter_columns_for(voters, table, voter_columns)
    side = (_normalized_voter_side(voters, vcols, phone_cols)
            .withColumn("built_at", F.current_timestamp())
            .withColumn("source_table", F.lit(table)))
    (side.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
         .saveAsTable(index_table))
    n = spark.table(index_table).count()
    print(f"Built match index {index_table} from {table}: {n:,} voters; "
          f"phone columns: {', '.join(phone_cols) or 'none'}; email column: {vcols.get('email') or 'none'}.")
    return index_table


def _load_match_index(spark, index_table, max_age_hours):
    """(DataFrame or None, note). Stale indexes are still returned, with a note."""
    F = _F()
    try:
        exists = spark.catalog.tableExists(index_table)
    except Exception:
        exists = False
    if not exists:
        return None, "not found"
    df = spark.table(index_table)
    missing = [c for c in _INDEX_COLUMNS if c not in df.columns]
    if missing:
        return None, f"missing columns {missing} (rebuild needed)"
    try:
        built = df.select(F.max("built_at")).first()[0]
    except Exception:
        built = None
    if built is None:
        return df, "age unknown"
    if getattr(built, "tzinfo", None) is not None:
        built = built.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    age_h = (datetime.datetime.utcnow() - built).total_seconds() / 3600.0
    stamp = built.strftime("%Y-%m-%d %H:%M UTC")
    if age_h > max_age_hours:
        return df, f"stale: built {stamp}, {age_h:.0f}h ago"
    return df, f"built {stamp}"


def match_to_voter_file(df, state="illinois", first_name=None, last_name=None, full_name=None,
                        address=None, city=None, zip_code=None, phone=None, email=None,
                        threshold=0.85, block_on="auto", county=None, where=None, table=None,
                        voter_columns=None, use_index=True, index_table=None, index_max_age_hours=36,
                        spark=None, cache=True):
    """Match a people list (pandas or Spark DataFrame) to a state voter file.

    How it works
      1. Exact keys first: a phone number (any voter phone column) or an email
         shared with a voter is a candidate with a strong prior; the name still
         has to agree for it to clear the threshold (a spouse on the same landline
         scores below it).
      2. Fuzzy names: names and addresses on both sides are normalised (upper-case,
         punctuation stripped, JR/III removed, nicknames canonicalised, street
         suffixes abbreviated). Candidates are found by *blocking* on fields both
         sides have: block_on="auto" uses 5-digit zip + last-name soundex when both
         sides have zips, otherwise last-name soundex + first initial (+ city when
         both have it); "last_name" blocks on last-name soundex alone (loosest).
      3. Candidates are scored 0-1: 60% name (last name weighted above first) and,
         when both sides have an address, 40% address (house number + street text).
         Phone/email candidates score at least 0.5 + 0.5 * name score.
      4. The best candidate per input row is kept; `matched` is True when
         `match_score >= threshold`; `match_key` says which key found it (phone,
         email, name); `candidate_count` says how many voters cleared the threshold.

    The voter side comes from the persisted match index `<schema>.match_index_<state>`
    (built by build_match_index, refreshed nightly). If it is missing it is built on
    the spot (a few minutes); if stale it is used with a note. use_index=False (or a
    custom table=) normalises the raw table on the fly instead.

    Restrict the voter side with county="Ozaukee" or (raw table only) an SQL where=.
    Column names are auto-detected; pass first_name=..., last_name=..., address=...,
    city=..., zip_code=..., phone=..., email=... (or full_name=...) when detection fails.

    Returns a cached Spark DataFrame: the input columns plus matched, match_score,
    match_key, candidate_count, voter_state_file_id, voter_first_name, voter_last_name,
    voter_address, voter_city, voter_zip. Prints a match-rate summary and sample
    unmatched names. Follow with export_csv(result, "matches.csv").
    """
    import pandas as pd
    from pyspark.sql import functions as F, Window

    spark = spark or _surus_spark()
    if spark is None:
        raise RuntimeError("No Spark session is available in this environment.")
    default_table = f"{str(state).strip().lower()}.deltatables.vcs_gold"
    custom_table = table is not None and table != default_table
    table = table or default_table
    if (block_on or "auto").strip().lower() not in _BLOCK_MODES:
        raise ValueError(f"block_on must be one of {_BLOCK_MODES}")

    # ── 1. input as a Spark DataFrame of strings ──
    if isinstance(df, pd.DataFrame):
        pdf = df.copy()
        pdf.columns = [str(c) for c in pdf.columns]
        for col in pdf.columns:
            pdf[col] = pdf[col].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
        sdf = spark.createDataFrame(pdf)
    elif hasattr(df, "columns") and hasattr(df, "select"):
        sdf = df.select(*[F.col(f"`{c}`").cast("string").alias(c) for c in df.columns])
    else:
        raise TypeError("df must be a pandas DataFrame or a Spark DataFrame")
    in_cols = list(sdf.columns)
    if any(c.startswith("__") for c in in_cols):
        raise ValueError("input column names may not start with '__'")

    icols = _detect_columns(in_cols, {
        "first_name": first_name, "last_name": last_name, "full_name": full_name,
        "address": address, "city": city, "zip_code": zip_code, "phone": phone, "email": email,
    }, _INPUT_CANDIDATES)
    have_split_names = bool(icols.get("first_name") and icols.get("last_name"))
    if not have_split_names and not icols.get("full_name"):
        raise ValueError(
            "Could not find name columns in the input. Pass first_name=..., last_name=... "
            f"(or full_name=...). Input columns: {in_cols}"
        )

    # ── 2. voter side: the persisted index, or the raw table normalised now ──
    vot_side, voter_desc = None, None
    voters_have_zip = voters_have_city = voters_have_email = True
    if use_index and not custom_table:
        idx = index_table or match_index_table(state)
        vot_side, note = _load_match_index(spark, idx, index_max_age_hours)
        if vot_side is None:
            print(f"Match index {idx} {note}; building it now (one-time, a few minutes)...")
            try:
                build_match_index(state, table=table, index_table=idx, voter_columns=voter_columns, spark=spark)
                vot_side, note = spark.table(idx), "just built"
            except Exception as e:
                print(f"(could not build the match index: {e}; matching directly against {table})")
                vot_side = None
        elif note.startswith("stale"):
            print(f"(match index {idx} is {note}; using it anyway, the nightly refresh rebuilds it)")
        if vot_side is not None:
            voter_desc = f"{idx} ({note})"
            if where:
                print("(where= is ignored when the match index is used; pass use_index=False to filter the raw table)")
    if vot_side is None:
        voters = spark.table(table)
        vcols, phone_cols = _voter_columns_for(voters, table, voter_columns)
        if where:
            voters = voters.filter(where)
        vot_side = _normalized_voter_side(voters, vcols, phone_cols)
        voters_have_zip, voters_have_city = bool(vcols.get("zip_code")), bool(vcols.get("city"))
        voters_have_email = bool(vcols.get("email"))
        voter_desc = (f"{table} normalised on the fly (voter columns: "
                      f"{', '.join(f'{k}={v}' for k, v in vcols.items())}; phones: {', '.join(phone_cols) or 'none'})")
    if county:
        vot_side = vot_side.filter(F.upper(F.trim(F.coalesce(F.col("county"), F.lit("")))) == str(county).strip().upper())

    mode, use_city = _block_strategy(
        block_on,
        input_has_zip=bool(icols.get("zip_code")), voters_have_zip=voters_have_zip,
        input_has_city=bool(icols.get("city")), voters_have_city=voters_have_city,
    )

    def _block_key(zip5, sdx, first, city_col):
        city_part = [F.lit("|"), city_col] if use_city else []
        name_key = F.concat(F.lit("N"), sdx, F.lit("|"), F.substring(first, 1, 1), *city_part)
        last_key = F.concat(F.lit("L"), sdx, *city_part)
        if mode == "name":
            return name_key
        if mode == "last_name":
            return last_key
        zip_key = F.concat(F.lit("Z"), zip5, F.lit("|"), sdx)
        return F.when(zip5 != "", zip_key).otherwise(name_key)   # mode == "zip"

    # ── 3. normalised input ──
    if have_split_names:
        i_first_raw = F.col(f"`{icols['first_name']}`")
        i_last_raw = F.col(f"`{icols['last_name']}`")
    else:
        full = F.trim(F.regexp_replace(_nx_str(F.col(f"`{icols['full_name']}`")), r"\s+", " "))
        has_comma = F.instr(full, ",") > 0
        parts = F.split(full, " ")
        i_first_raw = F.when(has_comma, F.trim(F.split(F.split(full, ",").getItem(1), " ").getItem(0))) \
                       .otherwise(parts.getItem(0))
        i_last_raw = F.when(has_comma, F.trim(F.split(full, ",").getItem(0))) \
                      .otherwise(F.element_at(parts, -1))

    def _in(field, fn):
        return fn(F.col(f"`{icols[field]}`")) if icols.get(field) else F.lit("")

    inp = sdf.withColumn("__row_id", F.monotonically_increasing_id())
    inp_n = inp.select(
        F.col("__row_id"),
        _nx_canon_first(_nx_name(i_first_raw)).alias("__i_first"),
        _nx_name(i_last_raw).alias("__i_last"),
        _in("address", _nx_addr).alias("__i_addr"),
        _in("city", _nx_city).alias("__i_city"),
        _in("zip_code", _nx_zip5).alias("__i_zip"),
        _in("phone", _nx_phone).alias("__i_phone"),
        _in("email", _nx_email).alias("__i_email"),
    )
    inp_n = (inp_n.withColumn("__i_house", _nx_house(F.col("__i_addr")))
                  .withColumn("__i_street", _nx_street(F.col("__i_addr")))
                  .withColumn("__i_block", _block_key(F.col("__i_zip"), F.soundex(F.col("__i_last")),
                                                      F.col("__i_first"), F.col("__i_city"))))

    vot_n = vot_side.select(
        F.col("state_file_id").alias("__v_id"),
        F.col("n_first").alias("__v_first"), F.col("n_last").alias("__v_last"), F.col("n_sdx").alias("__v_sdx"),
        F.col("n_addr").alias("__v_addr"), F.col("n_house").alias("__v_house"), F.col("n_street").alias("__v_street"),
        F.col("n_city").alias("__v_city"), F.col("n_zip5").alias("__v_zip"),
        F.col("n_phones").alias("__v_phones"), F.col("n_email").alias("__v_email"),
        F.col("first_name").alias("voter_first_name"), F.col("last_name").alias("voter_last_name"),
        F.col("address").alias("voter_address"), F.col("city").alias("voter_city"), F.col("zip").alias("voter_zip"),
    ).withColumn("__v_block", _block_key(F.col("__v_zip"), F.col("__v_sdx"), F.col("__v_first"), F.col("__v_city")))

    # ── 4. candidates: exact keys, then blocked fuzzy names ──
    cand_sets = []
    block_cand = inp_n.join(vot_n, inp_n["__i_block"] == vot_n["__v_block"], "inner").withColumn("__key", F.lit("name"))
    cand_sets.append(block_cand)
    if icols.get("phone"):
        vp = vot_n.withColumn("__v_key", F.explode(F.col("__v_phones"))).filter(F.col("__v_key") != "")
        cand_sets.append(inp_n.filter(F.col("__i_phone") != "")
                         .join(vp, inp_n["__i_phone"] == vp["__v_key"], "inner").drop("__v_key")
                         .withColumn("__key", F.lit("phone")))
    if icols.get("email") and voters_have_email:
        ve = vot_n.filter(F.col("__v_email") != "")
        cand_sets.append(inp_n.filter(F.col("__i_email") != "")
                         .join(ve, inp_n["__i_email"] == ve["__v_email"], "inner")
                         .withColumn("__key", F.lit("email")))
    cand = cand_sets[0]
    for extra in cand_sets[1:]:
        cand = cand.unionByName(extra)

    # ── 5. scores ──
    first_sim = _nx_sim(F.col("__i_first"), F.col("__v_first"))
    last_sim = _nx_sim(F.col("__i_last"), F.col("__v_last"))
    name_score = 0.6 * last_sim + 0.4 * first_sim
    use_address = bool(icols.get("address"))
    if use_address:
        house_ok = F.when((F.col("__i_house") != "") & (F.col("__i_house") == F.col("__v_house")), 1.0).otherwise(0.0)
        street_sim = _nx_sim(F.col("__i_street"), F.col("__v_street"))
        addr_score = 0.5 * house_ok + 0.5 * street_sim
        full_score = F.when((F.col("__i_addr") == "") | (F.col("__v_addr") == ""), name_score) \
                      .otherwise(0.6 * name_score + 0.4 * addr_score)
    else:
        full_score = name_score
    key_score = 0.5 + 0.5 * name_score
    score = F.when(F.col("__key") != "name", F.greatest(full_score, key_score)).otherwise(full_score)
    cand = cand.withColumn("match_score", F.round(score, 4))
    key_rank = F.when(F.col("__key") == "phone", 0).when(F.col("__key") == "email", 1).otherwise(2)

    per_row = Window.partitionBy("__row_id")
    window = per_row.orderBy(F.desc("match_score"), key_rank.asc(), F.asc("__v_id"))
    best = (cand.withColumn("candidate_count",
                            F.size(F.collect_set(F.when(F.col("match_score") >= F.lit(float(threshold)),
                                                        F.col("__v_id"))).over(per_row)))
                .withColumn("__rn", F.row_number().over(window))
                .filter(F.col("__rn") == 1)
                .select("__row_id", F.col("__v_id").alias("voter_state_file_id"), "match_score",
                        F.col("__key").alias("match_key"), "candidate_count",
                        "voter_first_name", "voter_last_name", "voter_address", "voter_city", "voter_zip"))

    out = (inp.join(best, "__row_id", "left")
              .withColumn("matched", F.coalesce(F.col("match_score") >= F.lit(float(threshold)), F.lit(False)))
              .withColumn("candidate_count", F.coalesce(F.col("candidate_count"), F.lit(0)))
              .drop("__row_id"))
    if cache:
        out = out.cache()

    # ── 6. summary ──
    total = out.count()
    matched = out.filter(F.col("matched")).count()
    with_candidate = out.filter(F.col("match_score").isNotNull()).count()
    high = out.filter(F.col("match_score") >= 0.95).count()
    ambiguous = out.filter(F.col("candidate_count") > 1).count()
    rate = (matched / total * 100.0) if total else 0.0
    scope = f" (county={county!r})" if county else ""
    print(f"Matched {matched:,} of {total:,} rows ({rate:.1f}%) against {voter_desc}{scope} at threshold {threshold}.")
    print(f"  score >= 0.95: {high:,} | {threshold} to 0.95: {max(matched - high, 0):,} | "
          f"best candidate below threshold: {with_candidate - matched:,} | no candidate: {total - with_candidate:,} | "
          f"ambiguous (several voters over threshold): {ambiguous:,}")
    try:
        by_key = {r["match_key"]: r["count"] for r in
                  out.filter(F.col("matched")).groupBy("match_key").count().collect()}
        print("  matched by key: " + " | ".join(f"{k} {by_key.get(k, 0):,}" for k in ("phone", "email", "name")))
    except Exception as e:
        print(f"  (could not summarise by key: {e})")
    used_in = ", ".join(f"{k}={v!r}" for k, v in icols.items() if k != "full_name" or not have_split_names)
    print(f"  input columns: {used_in}")
    print(f"  blocking: {mode}{' + city' if use_city else ''}; address scoring: {'on' if use_address else 'off'}; "
          f"exact keys: {', '.join(k for k in ('phone', 'email') if icols.get(k)) or 'none'}")
    if total - matched > 0:
        name_cols = ([icols["first_name"], icols["last_name"]] if have_split_names else [icols["full_name"]])
        try:
            sample = (out.filter(~F.col("matched"))
                         .select(*[F.col(f"`{c}`") for c in name_cols], "match_score", "voter_first_name", "voter_last_name")
                         .limit(12).collect())
            print("  sample unmatched (input name | best candidate score | best candidate):")
            for r in sample:
                name = " ".join(str(r[c] or "") for c in name_cols)
                if r["match_score"] is None:
                    print(f"    {name} | no candidate")
                else:
                    print(f"    {name} | {r['match_score']} | {r['voter_first_name']} {r['voter_last_name']}")
        except Exception as e:
            print(f"  (could not list unmatched samples: {e})")
    print("Result is a Spark DataFrame with the input columns plus matched, match_score, match_key, "
          "candidate_count, voter_state_file_id, voter_first_name, voter_last_name, voter_address, voter_city, "
          "voter_zip. Use export_csv(result, 'matches.csv') to give the user the file; try block_on='last_name' "
          "or a lower threshold if the match rate looks low.")
    return out
