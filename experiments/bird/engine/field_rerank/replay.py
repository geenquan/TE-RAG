def normalize_key_of(table):
    """Simple table-name normalization for internal map keys."""
    return str(table).strip().lower()
