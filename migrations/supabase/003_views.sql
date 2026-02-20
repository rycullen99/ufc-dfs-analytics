-- Analytical views for Supabase dashboards

-- Fighter performance overview
CREATE OR REPLACE VIEW v_fighter_performance AS
SELECT
    f.canonical_name,
    e.event_date,
    s.salary,
    s.card_section,
    s.ownership_actual,
    s.actual_points,
    s.implied_prob,
    p.predicted_mean AS predicted_ownership,
    ss.sharp_vs_field_delta
FROM fighter_event_snapshot s
JOIN fighter_dim f ON f.canonical_id = s.canonical_id
JOIN event_dim e ON e.date_id = s.date_id
LEFT JOIN ownership_predictions p ON p.date_id = s.date_id AND p.canonical_id = s.canonical_id
LEFT JOIN sharp_signals ss ON ss.date_id = s.date_id AND ss.canonical_id = s.canonical_id
ORDER BY e.event_date DESC, s.salary DESC;

-- Sharp signal summary per event
CREATE OR REPLACE VIEW v_sharp_summary AS
SELECT
    e.event_date,
    e.date_id,
    COUNT(*) AS n_fighters,
    AVG(ss.sharp_vs_field_delta) AS avg_sharp_delta,
    MAX(ss.sharp_vs_field_delta) AS max_sharp_delta,
    MIN(ss.sharp_vs_field_delta) AS min_sharp_delta,
    AVG(ss.sharp_confidence) AS avg_confidence
FROM sharp_signals ss
JOIN event_dim e ON e.date_id = ss.date_id
GROUP BY e.event_date, e.date_id
ORDER BY e.event_date DESC;

-- Ownership prediction accuracy
CREATE OR REPLACE VIEW v_ownership_accuracy AS
SELECT
    e.event_date,
    f.canonical_name,
    s.ownership_actual,
    p.predicted_mean,
    ABS(s.ownership_actual - p.predicted_mean) AS absolute_error,
    s.salary,
    s.card_section
FROM fighter_event_snapshot s
JOIN ownership_predictions p ON p.date_id = s.date_id AND p.canonical_id = s.canonical_id
JOIN fighter_dim f ON f.canonical_id = s.canonical_id
JOIN event_dim e ON e.date_id = s.date_id
WHERE s.ownership_actual IS NOT NULL
ORDER BY e.event_date DESC;
