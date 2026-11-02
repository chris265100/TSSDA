"""Bounded read-only timestamp probes; no values or full-series scans."""


def read_timestamp_sample(session, measurement, *, start, end, limit=64):
    if limit < 2:
        raise ValueError("Timestamp sample limit must be at least two")
    name = measurement["measurement"]
    name = "`" + name.replace("`", "``") + "`"
    sql = (f"SELECT {name} FROM {measurement['device']} "
           f"WHERE {name} IS NOT NULL AND TIME >= {start} AND TIME <= {end} "
           f"ORDER BY TIME ASC LIMIT {int(limit)}")
    dataset = session.execute_query_statement(sql)
    try:
        times = []
        while len(times) < limit and dataset.has_next():
            timestamp = int(dataset.next().get_timestamp())
            if start <= timestamp <= end:
                times.append(timestamp)
        return sorted(set(times))
    finally:
        dataset.close_operation_handle()
