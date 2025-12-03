DROP TABLE IF EXISTS poems;

CREATE TABLE poems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    author TEXT,
    lines TEXT,
    text TEXT,
    clean_text TEXT,
    lemmatized TEXT,
    word_count INTEGER,
    line_count INTEGER,
    linecount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
