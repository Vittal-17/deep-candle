CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(16) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    target_price DOUBLE PRECISION NOT NULL,
    stop_price DOUBLE PRECISION NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    status VARCHAR(16) NOT NULL,
    exit_reason VARCHAR(16),
    gross_pnl_pct DOUBLE PRECISION,
    net_pnl_pct DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS system_telemetry (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    rss_memory_mb DOUBLE PRECISION NOT NULL,
    open_positions INT NOT NULL,
    total_trades INT NOT NULL,
    win_rate DOUBLE PRECISION,
    cumulative_pnl DOUBLE PRECISION
);
